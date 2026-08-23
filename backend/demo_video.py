"""Thread-safe backend playback for uploaded demonstration videos.

The module deliberately has no FastAPI dependency.  HTTP routes are expected
to stream an ``UploadFile.file`` into :meth:`DemoVideoService.write_upload`,
start probing, and translate the typed exceptions below to HTTP responses.
Decoded frames are handed to a synchronous callback from a dedicated worker
thread; production can use that callback to submit the frame to the existing
pipeline and wait until the matching result has been published.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, BinaryIO, Callable, Protocol
import uuid

import cv2
import numpy as np

from config import CONFIG, DemoVideoConfig


logger = logging.getLogger(__name__)


class _H264ExportWriter:
    """Stream annotated BGR frames to a compact, browser-compatible H.264 MP4."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        max_width: int = 0,
    ):
        # A 2688-pixel-wide export is pointless for a slide deck iframe and
        # makes the file too heavy to stream comfortably. Scale on the way out;
        # the overlay is drawn at full resolution first, so nothing is lost
        # beyond the downscale itself.
        self._scale_to: tuple[int, int] | None = None
        if max_width and width > max_width:
            scaled_height = max(2, int(round(height * max_width / width)))
            self._scale_to = (int(max_width), scaled_height - scaled_height % 2)
            width, height = self._scale_to
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise DemoVideoError(
                "FFmpeg is required for H.264 annotated video export"
            )
        command = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{int(width)}x{int(height)}",
            "-r", f"{max(0.1, float(fps)):.6f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(path),
        ]
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt" else 0
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._scale_to is not None and (
            frame.shape[1] != self._scale_to[0] or frame.shape[0] != self._scale_to[1]
        ):
            # INTER_AREA is the right filter for shrinking: it averages the
            # pixels being merged instead of sampling one of them.
            frame = cv2.resize(frame, self._scale_to, interpolation=cv2.INTER_AREA)
        if self._closed or self._process.stdin is None:
            raise DemoVideoError("H.264 export writer is closed")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            error = self._stderr_text()
            raise DemoVideoError(
                f"FFmpeg stopped during H.264 export{': ' + error if error else ''}"
            ) from exc

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        error = self._stderr_text()
        if return_code != 0:
            raise DemoVideoError(
                f"FFmpeg H.264 export failed{': ' + error if error else ''}"
            )

    def _stderr_text(self) -> str:
        if self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return ""


class DemoVideoError(RuntimeError):
    """Base class for expected demo-video failures."""


class DemoVideoValidationError(DemoVideoError):
    pass


class DemoVideoNotFoundError(DemoVideoError):
    pass


class DemoVideoConflictError(DemoVideoError):
    pass


class DemoVideoTooLargeError(DemoVideoValidationError):
    pass


class DemoVideoTimeoutError(DemoVideoError):
    pass


class DemoVideoProcessingDisabledError(DemoVideoError):
    pass


class DemoVideoStatus(str, Enum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    LOADING = "loading"
    READY = "ready"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"
    STOPPED = "stopped"
    FAILED = "failed"
    DELETED = "deleted"


class CaptureLike(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def get(self, prop_id: int) -> float: ...
    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DemoFrameContext:
    """Metadata belonging to exactly one decoded source frame."""

    job_id: str
    run_id: str
    camera_id: str
    frame_index: int
    source_time_sec: float
    source_timestamp: float
    status: str
    mode: str
    playback_mode: str
    export_mode: bool = False
    video_decode_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class DemoVideoJobSnapshot:
    job_id: str
    camera_id: str
    filename: str
    input_path: str
    created_at: float
    updated_at: float
    status: str
    error: str | None
    current_frame: int
    total_frames: int | None
    current_time_sec: float
    duration_sec: float | None
    source_fps: float | None
    width: int | None
    height: int | None
    processing_fps: float
    dropped_frames: int
    pause_requested: bool
    stop_requested: bool
    run_id: str | None
    size_bytes: int
    mime_type: str
    mode: str
    playback_mode: str
    processing_ms: float
    export_status: str
    output_filename: str | None
    # Wall-clock anchor of the current run. Frames enter the pipeline stamped
    # with ``run_started_wall + source_time_sec``, so alerts recorded during the
    # run map back to a position in the clip: ``alert.timestamp - this value``.
    run_started_wall: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DemoVideoStats:
    demo_upload_ms: float = 0.0
    demo_upload_size_bytes: int = 0
    demo_probe_ms: float = 0.0
    demo_decode_ms: float = 0.0
    demo_processing_ms: float = 0.0
    demo_source_fps: float = 0.0
    demo_processing_fps: float = 0.0
    demo_dropped_frames: int = 0
    demo_job_queue_age_ms: float = 0.0
    demo_active_jobs: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(slots=True)
class _Job:
    job_id: str
    camera_id: str
    filename: str
    input_path: Path
    created_at: float
    updated_at: float
    status: DemoVideoStatus
    mime_type: str
    mode: str
    asset_kind: str
    playback_mode: str
    error: str | None = None
    current_frame: int = 0
    total_frames: int | None = None
    current_time_sec: float = 0.0
    duration_sec: float | None = None
    source_fps: float | None = None
    width: int | None = None
    height: int | None = None
    processing_fps: float = 0.0
    dropped_frames: int = 0
    pause_requested: bool = False
    stop_requested: bool = False
    run_id: str | None = None
    size_bytes: int = 0
    processing_ms: float = 0.0
    export_requested: bool = False
    export_status: str = "idle"
    output_path: Path | None = None
    generation: int = 0
    prepared_run: bool = False
    starting: bool = False
    queued_at_monotonic: float = 0.0
    run_started_wall: float = 0.0
    run_started_monotonic: float = 0.0
    paused_total_sec: float = 0.0
    worker: threading.Thread | None = field(default=None, repr=False)
    worker_kind: str | None = None
    read_only_leases: int = 0
    deleting: bool = False


@dataclass(frozen=True, slots=True)
class _ProbeResult:
    total_frames: int | None
    source_fps: float | None
    width: int
    height: int
    duration_sec: float | None


class _CaptureLease:
    """Make timeout and worker cleanup share one idempotent release."""

    def __init__(self):
        self._lock = threading.Lock()
        self._capture: CaptureLike | None = None
        self._released = False

    def assign(self, capture: CaptureLike) -> None:
        should_release = False
        with self._lock:
            if self._released:
                should_release = True
            else:
                self._capture = capture
        if should_release:
            try:
                capture.release()
            except Exception:
                logger.debug("Late demo capture release failed", exc_info=True)

    def release(self) -> None:
        capture = None
        with self._lock:
            if self._released:
                return
            self._released = True
            capture = self._capture
            self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.warning("Demo video capture release failed", exc_info=True)


class DemoVideoInputLease:
    """A small ref-counted lease protecting one uploaded source asset."""

    __slots__ = ("_service", "job_id", "path", "_lock", "_released")

    def __init__(self, service: "DemoVideoService", job_id: str, path: Path):
        self._service = service
        self.job_id = str(job_id)
        self.path = path
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._service._release_input_lease(self.job_id)

    def __enter__(self) -> "DemoVideoInputLease":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()


class _LeasedCapture:
    """Delegate capture operations and release its source lease exactly once."""

    def __init__(self, capture: CaptureLike, lease: DemoVideoInputLease):
        self._capture = capture
        self._lease = lease
        self._release_lock = threading.Lock()
        self._released = False

    def __getattr__(self, name: str):
        return getattr(self._capture, name)

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        try:
            self._capture.release()
        finally:
            self._lease.release()


_MIME_BY_EXTENSION: dict[str, frozenset[str]] = {
    ".mp4": frozenset({"video/mp4", "application/mp4", "application/octet-stream"}),
    ".avi": frozenset({
        "video/x-msvideo",
        "video/avi",
        "video/msvideo",
        "application/octet-stream",
    }),
    ".mov": frozenset({"video/quicktime", "video/mov", "application/octet-stream"}),
    ".mkv": frozenset({"video/x-matroska", "video/mkv", "application/octet-stream"}),
}

_SAFE_FILENAME_RE = re.compile(r"[^0-9A-Za-z_.()\- ]+")

_TRANSITIONS: dict[DemoVideoStatus, frozenset[DemoVideoStatus]] = {
    DemoVideoStatus.UPLOADING: frozenset({
        DemoVideoStatus.UPLOADED,
        DemoVideoStatus.FAILED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.UPLOADED: frozenset({
        DemoVideoStatus.LOADING,
        DemoVideoStatus.STOPPED,
        DemoVideoStatus.FAILED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.LOADING: frozenset({
        DemoVideoStatus.READY,
        DemoVideoStatus.STOPPED,
        DemoVideoStatus.FAILED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.READY: frozenset({
        DemoVideoStatus.PLAYING,
        DemoVideoStatus.STOPPED,
        DemoVideoStatus.FAILED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.PLAYING: frozenset({
        DemoVideoStatus.PAUSED,
        DemoVideoStatus.FINISHED,
        DemoVideoStatus.STOPPED,
        DemoVideoStatus.FAILED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.PAUSED: frozenset({
        DemoVideoStatus.PLAYING,
        DemoVideoStatus.STOPPED,
        DemoVideoStatus.FAILED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.FINISHED: frozenset({
        DemoVideoStatus.READY,
        DemoVideoStatus.STOPPED,
        DemoVideoStatus.FAILED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.STOPPED: frozenset({
        DemoVideoStatus.READY,
        DemoVideoStatus.LOADING,
        DemoVideoStatus.FAILED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.FAILED: frozenset({
        DemoVideoStatus.LOADING,
        DemoVideoStatus.READY,
        DemoVideoStatus.STOPPED,
        DemoVideoStatus.DELETED,
    }),
    DemoVideoStatus.DELETED: frozenset(),
}


class DemoVideoService:
    """Own uploaded jobs and one bounded playback worker per active job.

    All public methods are synchronous and thread-safe.  FastAPI integrations
    should call file/control methods via ``asyncio.to_thread`` so large local
    writes and control waits never occupy the event loop.
    """

    def __init__(
        self,
        config: DemoVideoConfig | None = None,
        *,
        process_frame: Callable[[DemoFrameContext, np.ndarray], Any] | None = None,
        reset_camera: Callable[[str], None] | None = None,
        capture_factory: Callable[[str], CaptureLike] | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        max_size_bytes: int | None = None,
        janitor_interval_seconds: float | None = None,
        start_janitor: bool = True,
    ):
        self.config = config or CONFIG.demo_video
        self._process_frame = process_frame or (lambda _context, _frame: None)
        self._reset_camera = reset_camera or (lambda _camera_id: None)
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._clock = clock
        self._monotonic = monotonic
        configured_limit = int(float(self.config.max_size_mb) * 1024 * 1024)
        self.max_size_bytes = max(
            1,
            int(max_size_bytes) if max_size_bytes is not None else configured_limit,
        )

        self.upload_dir = Path(self.config.upload_dir).expanduser().resolve()
        self.output_dir = Path(self.config.output_dir).expanduser().resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._jobs: dict[str, _Job] = {}
        self._closed = False
        self._janitor_stop = threading.Event()
        self._janitor_thread: threading.Thread | None = None
        self._helper_threads: set[threading.Thread] = set()
        # Limit concurrent read-only metadata decoders.
        self._read_only_probe_slot = threading.BoundedSemaphore(1)

        self._upload_ms = 0.0
        self._upload_size_bytes = 0
        self._probe_ms = 0.0
        self._decode_ms = 0.0
        self._processing_ms = 0.0
        self._source_fps = 0.0
        self._processing_fps = 0.0
        self._dropped_frames = 0
        self._queue_age_ms = 0.0

        ttl = max(0.05, float(self.config.job_ttl_seconds))
        self._janitor_interval = max(
            0.02,
            float(janitor_interval_seconds)
            if janitor_interval_seconds is not None
            else min(60.0, max(1.0, ttl / 4.0)),
        )
        self.cleanup_orphaned_directories()
        if start_janitor:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise DemoVideoConflictError("Demo video service is closed")
            if self._janitor_thread is not None and self._janitor_thread.is_alive():
                return
            self._janitor_stop.clear()
            thread = threading.Thread(
                target=self._janitor_loop,
                name="demo-video-janitor",
                daemon=True,
            )
            self._janitor_thread = thread
            thread.start()

    @staticmethod
    def _safe_filename(filename: str) -> str:
        raw = str(filename or "").replace("\x00", "").strip()
        # Treat both POSIX and Windows separators consistently on every host.
        base = raw.replace("\\", "/").rsplit("/", 1)[-1].strip()
        base = _SAFE_FILENAME_RE.sub("_", base)
        base = base.strip(" .")
        if not base or base in {".", ".."}:
            raise DemoVideoValidationError("Video filename is empty or unsafe")
        if len(base) > 180:
            suffix = Path(base).suffix
            base = f"{Path(base).stem[: max(1, 180 - len(suffix))]}{suffix}"
        return base

    def validate_upload(self, filename: str, content_type: str | None) -> str:
        safe = self._safe_filename(filename)
        extension = Path(safe).suffix.lower()
        allowed_extensions = {
            str(item).lower() if str(item).startswith(".") else f".{str(item).lower()}"
            for item in self.config.allowed_extensions
        }
        if extension not in allowed_extensions:
            raise DemoVideoValidationError(
                f"Unsupported video extension: {extension or '(none)'}"
            )
        mime = str(content_type or "").split(";", 1)[0].strip().lower()
        allowed_mimes = _MIME_BY_EXTENSION.get(extension, frozenset())
        if not mime or mime not in allowed_mimes:
            raise DemoVideoValidationError(
                f"Unsupported MIME type for {extension}: {mime or '(none)'}"
            )
        return safe

    @staticmethod
    def _validate_camera_id(camera_id: str | None) -> str:
        value = str(camera_id or "demo_upload").strip()
        if not value or len(value) > 128 or any(ord(char) < 32 for char in value):
            raise DemoVideoValidationError("camera_id is empty or invalid")
        return value

    @staticmethod
    def _normalize_mode(mode: str | None) -> str:
        value = str(mode or "site").strip().lower()
        if value not in {"site", "checkpoint"}:
            raise DemoVideoValidationError("mode must be 'site' or 'checkpoint'")
        return value

    def _normalize_playback_mode(self, playback_mode: str | None) -> str:
        value = str(playback_mode or self.config.playback_mode).strip().lower()
        if value not in {"realtime", "fast"}:
            raise DemoVideoValidationError(
                "playback_mode must be 'realtime' or 'fast'"
            )
        return value

    @staticmethod
    def _inside(root: Path, candidate: Path) -> bool:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            return False

    def _job_path(self, job_id: str, filename: str) -> Path:
        directory = (self.upload_dir / job_id).resolve()
        target = (directory / filename).resolve()
        if not self._inside(self.upload_dir, directory) or not self._inside(directory, target):
            raise DemoVideoValidationError("Resolved upload path escapes upload root")
        return target

    def create_upload(
        self,
        filename: str,
        content_type: str,
        *,
        camera_id: str = "demo_upload",
        mode: str = "site",
        playback_mode: str | None = None,
        asset_kind: str = "demo",
    ) -> DemoVideoJobSnapshot:
        safe_filename = self.validate_upload(filename, content_type)
        normalized_camera = self._validate_camera_id(camera_id)
        normalized_mode = self._normalize_mode(mode)
        normalized_playback = self._normalize_playback_mode(playback_mode)
        normalized_kind = str(asset_kind or "demo").strip().lower()
        if normalized_kind not in {"demo", "distance"}:
            raise DemoVideoValidationError("asset_kind is invalid")
        now = float(self._clock())

        with self._condition:
            self._ensure_open_locked()
            while True:
                job_id = uuid.uuid4().hex
                if job_id not in self._jobs:
                    break
            target = self._job_path(job_id, safe_filename)
            target.parent.mkdir(parents=False, exist_ok=False)
            job = _Job(
                job_id=job_id,
                camera_id=normalized_camera,
                filename=safe_filename,
                input_path=target,
                created_at=now,
                updated_at=now,
                status=DemoVideoStatus.UPLOADING,
                mime_type=str(content_type).split(";", 1)[0].strip().lower(),
                mode=normalized_mode,
                asset_kind=normalized_kind,
                playback_mode=normalized_playback,
            )
            self._jobs[job_id] = job
            self._condition.notify_all()
            return self._snapshot_locked(job)

    # Alias useful to route code which calls the first phase "begin".
    begin_upload = create_upload

    def write_upload(
        self,
        job_id: str,
        source: BinaryIO,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> DemoVideoJobSnapshot:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        with self._condition:
            job = self._get_locked(job_id)
            if job.status is not DemoVideoStatus.UPLOADING:
                raise DemoVideoConflictError(
                    f"Upload cannot be written while job is {job.status.value}"
                )
            if job.worker is not None and job.worker.is_alive():
                raise DemoVideoConflictError("This upload is already being written")
            job.worker = threading.current_thread()
            job.worker_kind = "upload"
            job.stop_requested = False
            target = job.input_path

        started = self._monotonic()
        size = 0
        try:
            with open(target, "xb") as destination:
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    with self._condition:
                        current = self._jobs.get(str(job_id))
                        if (
                            current is None
                            or current.status is not DemoVideoStatus.UPLOADING
                            or current.stop_requested
                            or self._closed
                        ):
                            raise DemoVideoConflictError(
                                "Upload was cancelled while being written"
                            )
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise DemoVideoValidationError(
                            "Upload stream returned non-binary data"
                        )
                    size += len(chunk)
                    if size > self.max_size_bytes:
                        raise DemoVideoTooLargeError(
                            f"Video exceeds {self.max_size_bytes} byte limit"
                        )
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if size <= 0:
                raise DemoVideoValidationError("Uploaded video is empty")
        except Exception as exc:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove partial demo upload %s", target)
            with self._condition:
                current = self._jobs.get(str(job_id))
                if current is not None and current.status is DemoVideoStatus.UPLOADING:
                    self._transition_locked(
                        current,
                        DemoVideoStatus.FAILED,
                        error=self._error_text(exc),
                    )
                if (
                    current is not None
                    and current.worker is threading.current_thread()
                ):
                    current.worker = None
                    current.worker_kind = None
                    self._condition.notify_all()
            logger.warning(
                "Demo video job failed during upload %s: %s",
                job_id,
                self._error_text(exc),
            )
            raise

        elapsed_ms = max(0.0, (self._monotonic() - started) * 1000.0)
        with self._condition:
            job = self._get_locked(job_id)
            if (
                job.status is not DemoVideoStatus.UPLOADING
                or job.stop_requested
                or self._closed
            ):
                target.unlink(missing_ok=True)
                if job.status is DemoVideoStatus.UPLOADING:
                    self._transition_locked(
                        job,
                        DemoVideoStatus.FAILED,
                        error="DemoVideoConflictError: Upload was cancelled",
                    )
                if job.worker is threading.current_thread():
                    job.worker = None
                    job.worker_kind = None
                    self._condition.notify_all()
                raise DemoVideoConflictError("Upload was cancelled while being written")
            job.size_bytes = size
            if job.worker is threading.current_thread():
                job.worker = None
                job.worker_kind = None
            self._upload_ms = elapsed_ms
            self._upload_size_bytes = size
            self._transition_locked(job, DemoVideoStatus.UPLOADED)
            logger.info("Demo video job uploaded: %s", job.job_id)
            return self._snapshot_locked(job)

    store_upload = write_upload

    def upload_stream(
        self,
        filename: str,
        content_type: str,
        source: BinaryIO,
        *,
        camera_id: str = "demo_upload",
        mode: str = "site",
        playback_mode: str | None = None,
        asset_kind: str = "demo",
        chunk_size: int = 1024 * 1024,
    ) -> DemoVideoJobSnapshot:
        job = self.create_upload(
            filename,
            content_type,
            camera_id=camera_id,
            mode=mode,
            playback_mode=playback_mode,
            asset_kind=asset_kind,
        )
        return self.write_upload(job.job_id, source, chunk_size=chunk_size)

    def start_probe(self, job_id: str) -> DemoVideoJobSnapshot:
        return self._start_probe(
            job_id,
            reserve_worker_capacity=True,
            allow_processing_disabled=False,
            release_read_only_slot=False,
        )

    probe = start_probe

    def start_read_only_probe(self, job_id: str) -> DemoVideoJobSnapshot:
        """Probe one asset outside playback capacity, with a separate limit."""

        if not self._read_only_probe_slot.acquire(blocking=False):
            raise DemoVideoConflictError("Read-only video probe capacity is full")
        try:
            return self._start_probe(
                job_id,
                reserve_worker_capacity=False,
                allow_processing_disabled=True,
                release_read_only_slot=True,
            )
        except BaseException:
            self._read_only_probe_slot.release()
            raise

    def _start_probe(
        self,
        job_id: str,
        *,
        reserve_worker_capacity: bool,
        allow_processing_disabled: bool,
        release_read_only_slot: bool,
    ) -> DemoVideoJobSnapshot:
        if not bool(self.config.processing_enabled) and not allow_processing_disabled:
            raise DemoVideoProcessingDisabledError(
                "Demo video processing is disabled"
            )
        with self._condition:
            self._ensure_open_locked()
            job = self._get_locked(job_id)
            if job.status not in {
                DemoVideoStatus.UPLOADED,
                DemoVideoStatus.FAILED,
                DemoVideoStatus.STOPPED,
            }:
                raise DemoVideoConflictError(
                    f"Video cannot be probed while job is {job.status.value}"
                )
            if not job.input_path.is_file():
                raise DemoVideoValidationError("Uploaded video file is missing")
            if reserve_worker_capacity:
                self._reserve_worker_locked(job)
            elif job.worker is not None and job.worker.is_alive():
                raise DemoVideoConflictError("This job already has an active worker")
            job.generation += 1
            generation = job.generation
            job.error = None
            job.stop_requested = False
            job.queued_at_monotonic = self._monotonic()
            self._transition_locked(job, DemoVideoStatus.LOADING)
            thread = threading.Thread(
                target=(
                    self._probe_supervisor_with_read_only_slot
                    if release_read_only_slot
                    else self._probe_supervisor
                ),
                args=(job.job_id, generation),
                name=f"demo-video-probe-{job.job_id[:8]}",
                daemon=True,
            )
            job.worker = thread
            job.worker_kind = (
                "read_only_probe" if release_read_only_slot else "probe"
            )
            thread.start()
            return self._snapshot_locked(job)

    def _probe_supervisor_with_read_only_slot(
        self,
        job_id: str,
        generation: int,
    ) -> None:
        try:
            self._probe_supervisor(job_id, generation)
        finally:
            self._read_only_probe_slot.release()

    def _probe_supervisor(self, job_id: str, generation: int) -> None:
        started = self._monotonic()
        done = threading.Event()
        lease = _CaptureLease()
        outcome: list[_ProbeResult] = []
        errors: list[BaseException] = []

        with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job.generation != generation:
                return
            path = str(job.input_path)
            self._queue_age_ms = max(
                0.0,
                (self._monotonic() - job.queued_at_monotonic) * 1000.0,
            )

        def do_probe() -> None:
            capture: CaptureLike | None = None
            try:
                capture = self._capture_factory(path)
                if capture is None:
                    raise DemoVideoValidationError("VideoCapture was not created")
                lease.assign(capture)
                if not bool(capture.isOpened()):
                    raise DemoVideoValidationError("Video cannot be opened")

                fps = self._positive_float(capture.get(cv2.CAP_PROP_FPS))
                frame_count_value = self._positive_float(
                    capture.get(cv2.CAP_PROP_FRAME_COUNT)
                )
                total_frames = (
                    max(1, int(round(frame_count_value)))
                    if frame_count_value is not None else None
                )
                width_value = self._positive_float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height_value = self._positive_float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

                ok, first_frame = capture.read()
                if not ok or first_frame is None or not isinstance(first_frame, np.ndarray) or first_frame.size == 0:
                    raise DemoVideoValidationError(
                        "Video is corrupt or contains no decodable frames"
                    )
                frame_height, frame_width = first_frame.shape[:2]
                width = int(round(width_value)) if width_value is not None else int(frame_width)
                height = int(round(height_value)) if height_value is not None else int(frame_height)
                if width < 1 or height < 1:
                    width, height = int(frame_width), int(frame_height)
                duration = (
                    float(total_frames) / fps
                    if total_frames is not None and fps is not None else None
                )
                outcome.append(_ProbeResult(
                    total_frames=total_frames,
                    source_fps=fps,
                    width=width,
                    height=height,
                    duration_sec=duration,
                ))
            except BaseException as exc:
                errors.append(exc)
            finally:
                lease.release()
                done.set()

        helper = threading.Thread(
            target=do_probe,
            name=f"demo-video-probe-io-{job_id[:8]}",
            daemon=True,
        )
        with self._condition:
            self._helper_threads.add(helper)
        helper.start()

        timeout = max(0.05, float(self.config.probe_timeout_seconds))
        watchdog_deadline = time.monotonic() + timeout
        timed_out = False
        cancelled = False
        while not done.wait(timeout=0.02):
            with self._condition:
                current = self._jobs.get(job_id)
                cancelled = bool(
                    current is None
                    or current.generation != generation
                    or current.stop_requested
                    or self._closed
                )
            if cancelled:
                lease.release()
                break
            if time.monotonic() >= watchdog_deadline:
                timed_out = True
                lease.release()
                break

        with self._condition:
            elapsed_ms = max(0.0, (self._monotonic() - started) * 1000.0)
            self._probe_ms = elapsed_ms
            job = self._jobs.get(job_id)
            if job is not None and job.generation == generation:
                if cancelled or job.stop_requested:
                    if job.status is DemoVideoStatus.LOADING:
                        self._transition_locked(job, DemoVideoStatus.STOPPED)
                elif timed_out:
                    error = DemoVideoTimeoutError(
                        f"Video probe exceeded {timeout:g} seconds"
                    )
                    self._transition_locked(
                        job,
                        DemoVideoStatus.FAILED,
                        error=self._error_text(error),
                    )
                    logger.warning(
                        "Demo video job failed during probe %s: %s",
                        job_id,
                        job.error,
                    )
                elif errors or not outcome:
                    error = errors[0] if errors else DemoVideoValidationError(
                        "Video probe returned no metadata"
                    )
                    self._transition_locked(
                        job,
                        DemoVideoStatus.FAILED,
                        error=self._error_text(error),
                    )
                    logger.warning(
                        "Demo video job failed during probe %s: %s",
                        job_id,
                        job.error,
                    )
                else:
                    result = outcome[0]
                    job.total_frames = result.total_frames
                    job.source_fps = result.source_fps
                    job.width = result.width
                    job.height = result.height
                    job.duration_sec = result.duration_sec
                    self._source_fps = float(result.source_fps or 0.0)
                    self._transition_locked(job, DemoVideoStatus.READY)
                    logger.info("Demo video job ready: %s", job_id)

        # The state timeout is bounded, but the supervisor remains the job's
        # active worker until the underlying call really exits.  This prevents
        # DELETE from unlinking a file still touched by an uncooperative codec.
        if helper.is_alive():
            done.wait()
        helper.join()
        with self._condition:
            self._helper_threads.discard(helper)
            job = self._jobs.get(job_id)
            if (
                job is not None
                and job.generation == generation
                and job.status is DemoVideoStatus.STOPPED
                and outcome
                and not errors
            ):
                # A normal stop may race a nearly completed probe. Preserve
                # the metadata so this same upload remains restartable.
                result = outcome[0]
                job.total_frames = result.total_frames
                job.source_fps = result.source_fps
                job.width = result.width
                job.height = result.height
                job.duration_sec = result.duration_sec
                self._source_fps = float(result.source_fps or 0.0)
            if (
                job is not None
                and job.generation == generation
                and job.worker is threading.current_thread()
            ):
                job.worker = None
                job.worker_kind = None
            self._condition.notify_all()

    def play(self, job_id: str) -> DemoVideoJobSnapshot:
        with self._condition:
            job = self._get_locked(job_id)
            job.export_requested = False
            status = job.status
        if status is DemoVideoStatus.PAUSED:
            return self.resume(job_id)
        if status in {DemoVideoStatus.FINISHED, DemoVideoStatus.STOPPED}:
            return self.restart(job_id, autoplay=True)
        if status is not DemoVideoStatus.READY:
            raise DemoVideoConflictError(
                f"Video cannot play while job is {status.value}"
            )

        self._prepare_run(job_id, force_new=False)
        return self._start_prepared_run(job_id)

    def _prepare_run(self, job_id: str, *, force_new: bool) -> None:
        if not bool(self.config.processing_enabled):
            raise DemoVideoProcessingDisabledError(
                "Demo video processing is disabled"
            )
        with self._condition:
            self._ensure_open_locked()
            job = self._get_locked(job_id)
            if job.status is not DemoVideoStatus.READY:
                raise DemoVideoConflictError(
                    f"Run cannot be prepared while job is {job.status.value}"
                )
            if job.starting:
                raise DemoVideoConflictError("A run is already being started")
            if job.prepared_run and not force_new:
                return
            self._assert_camera_available_locked(job)
            self._assert_worker_capacity_locked(job)
            job.starting = True
            job.generation += 1
            generation = job.generation
            run_id = uuid.uuid4().hex
            camera_id = job.camera_id

        try:
            self._reset_camera(camera_id)
        except Exception as exc:
            with self._condition:
                job = self._jobs.get(job_id)
                if job is not None and job.generation == generation:
                    job.starting = False
                    self._transition_locked(
                        job,
                        DemoVideoStatus.FAILED,
                        error=f"Camera reset failed: {self._error_text(exc)}",
                    )
            logger.warning(
                "Demo video job failed during camera reset %s: %s",
                job_id,
                self._error_text(exc),
            )
            raise DemoVideoError(f"Camera reset failed: {exc}") from exc

        with self._condition:
            job = self._get_locked(job_id)
            if job.generation != generation or job.status is not DemoVideoStatus.READY:
                job.starting = False
                raise DemoVideoConflictError("Job changed while run was being prepared")
            job.run_id = run_id
            job.prepared_run = True
            job.starting = False
            job.stop_requested = False
            job.pause_requested = False
            job.error = None
            self._condition.notify_all()

    def _start_prepared_run(self, job_id: str) -> DemoVideoJobSnapshot:
        with self._condition:
            self._ensure_open_locked()
            job = self._get_locked(job_id)
            if job.status is not DemoVideoStatus.READY or not job.prepared_run or not job.run_id:
                raise DemoVideoConflictError("Job has no prepared run")
            self._assert_camera_available_locked(job)
            self._reserve_worker_locked(job)
            generation = job.generation
            job.prepared_run = False
            job.stop_requested = False
            job.pause_requested = False
            job.queued_at_monotonic = self._monotonic()
            job.run_started_wall = float(self._clock()) - job.current_time_sec
            job.run_started_monotonic = self._monotonic() - job.current_time_sec
            job.paused_total_sec = 0.0
            self._transition_locked(job, DemoVideoStatus.PLAYING)
            thread = threading.Thread(
                target=self._playback_worker,
                args=(job.job_id, generation, job.run_id),
                name=f"demo-video-play-{job.job_id[:8]}",
                daemon=True,
            )
            job.worker = thread
            job.worker_kind = "playback"
            thread.start()
            logger.info("Demo video job playing: %s", job.job_id)
            return self._snapshot_locked(job)

    def pause(
        self,
        job_id: str,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> DemoVideoJobSnapshot:
        with self._condition:
            job = self._get_locked(job_id)
            if job.status is DemoVideoStatus.PAUSED:
                return self._snapshot_locked(job)
            if job.status is not DemoVideoStatus.PLAYING:
                raise DemoVideoConflictError(
                    f"Video cannot pause while job is {job.status.value}"
                )
            job.pause_requested = True
            job.updated_at = float(self._clock())
            self._condition.notify_all()
            if wait:
                limit = (
                    float(self.config.control_timeout_seconds)
                    if timeout is None else max(0.0, float(timeout))
                )
                reached = self._condition.wait_for(
                    lambda: (
                        job.status is not DemoVideoStatus.PLAYING
                        or job.job_id not in self._jobs
                    ),
                    timeout=limit,
                )
                if not reached:
                    raise DemoVideoTimeoutError(
                        "Pause was requested but the current frame is still processing"
                    )
            current = self._jobs.get(job_id)
            if current is None:
                raise DemoVideoNotFoundError(f"Unknown demo video job: {job_id}")
            return self._snapshot_locked(current)

    def resume(self, job_id: str) -> DemoVideoJobSnapshot:
        with self._condition:
            self._ensure_open_locked()
            job = self._get_locked(job_id)
            if job.status is DemoVideoStatus.PLAYING and not job.pause_requested:
                return self._snapshot_locked(job)
            if job.status not in {DemoVideoStatus.PAUSED, DemoVideoStatus.PLAYING}:
                raise DemoVideoConflictError(
                    f"Video cannot resume while job is {job.status.value}"
                )
            job.pause_requested = False
            if job.status is DemoVideoStatus.PAUSED:
                self._assert_camera_available_locked(job)
                self._transition_locked(job, DemoVideoStatus.PLAYING)
                logger.info("Demo video job resumed: %s", job.job_id)
            self._condition.notify_all()
            return self._snapshot_locked(job)

    def restart(
        self,
        job_id: str,
        *,
        autoplay: bool = True,
        timeout: float | None = None,
    ) -> DemoVideoJobSnapshot:
        with self._condition:
            job = self._get_locked(job_id)
            if job.status in {
                DemoVideoStatus.UPLOADING,
                DemoVideoStatus.UPLOADED,
                DemoVideoStatus.LOADING,
                DemoVideoStatus.DELETED,
            }:
                raise DemoVideoConflictError(
                    f"Video cannot restart while job is {job.status.value}"
                )
            has_worker = job.worker is not None and job.worker.is_alive()
        if has_worker:
            self.stop(job_id, timeout=timeout)

        with self._condition:
            job = self._get_locked(job_id)
            if job.worker is not None and job.worker.is_alive():
                raise DemoVideoConflictError("Previous worker is still active")
            if job.width is None or job.height is None:
                raise DemoVideoConflictError(
                    "Video metadata is unavailable; probe the upload first"
                )
            if job.status is not DemoVideoStatus.READY:
                self._transition_locked(job, DemoVideoStatus.READY)
            job.current_frame = 0
            job.current_time_sec = 0.0
            job.processing_fps = 0.0
            job.processing_ms = 0.0
            job.dropped_frames = 0
            job.error = None
            job.run_id = None
            job.prepared_run = False
            job.stop_requested = False
            job.pause_requested = False

        self._prepare_run(job_id, force_new=True)
        if autoplay:
            return self._start_prepared_run(job_id)
        return self.get(job_id)

    def export(self, job_id: str) -> DemoVideoJobSnapshot:
        """Run every source frame through inference and write annotated MP4."""
        self.restart(job_id, autoplay=False)
        with self._condition:
            job = self._get_locked(job_id)
            output_dir = (self.output_dir / job.job_id).resolve()
            if not self._inside(self.output_dir, output_dir):
                raise DemoVideoValidationError("Invalid demo output path")
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(job.filename).stem or "video"
            safe_stem = self._safe_filename(stem)
            job.output_path = output_dir / f"{safe_stem}_perimetr_ai.mp4"
            job.output_path.unlink(missing_ok=True)
            job.export_requested = True
            job.export_status = "processing"
            job.playback_mode = "fast"
            job.updated_at = float(self._clock())
        return self._start_prepared_run(job_id)

    def stop(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> DemoVideoJobSnapshot:
        with self._condition:
            job = self._get_locked(job_id)
            if job.status is DemoVideoStatus.DELETED:
                raise DemoVideoNotFoundError(f"Unknown demo video job: {job_id}")
            worker = job.worker
            if worker is None or not worker.is_alive():
                if job.status is not DemoVideoStatus.STOPPED:
                    self._transition_locked(job, DemoVideoStatus.STOPPED)
                job.stop_requested = True
                job.pause_requested = False
                logger.info("Demo video job stopped: %s", job.job_id)
                return self._snapshot_locked(job)
            job.stop_requested = True
            job.pause_requested = False
            # Invalidate probe/playback commits that race with this request.
            generation = job.generation
            self._condition.notify_all()

        limit = (
            float(self.config.control_timeout_seconds)
            if timeout is None else max(0.0, float(timeout))
        )
        if worker is not threading.current_thread():
            worker.join(timeout=limit)
        if worker.is_alive():
            raise DemoVideoTimeoutError(
                "Demo video worker did not stop before the control timeout"
            )

        with self._condition:
            job = self._get_locked(job_id)
            if job.generation != generation or job.worker is not worker:
                # Do not clear state belonging to a newer worker generation.
                return self._snapshot_locked(job)
            if job.status is not DemoVideoStatus.STOPPED:
                self._transition_locked(job, DemoVideoStatus.STOPPED)
            job.worker = None
            job.worker_kind = None
            job.stop_requested = True
            job.pause_requested = False
            logger.info("Demo video job stopped: %s", job.job_id)
            return self._snapshot_locked(job)

    def output_path_for(self, job_id: str) -> Path:
        with self._condition:
            job = self._get_locked(job_id)
            if job.export_status != "ready" or job.output_path is None:
                raise DemoVideoConflictError("Annotated video export is not ready")
            path = job.output_path.resolve()
            if not self._inside(self.output_dir, path) or not path.is_file():
                raise DemoVideoNotFoundError("Annotated video export was not found")
            return path

    def input_path_for(self, job_id: str) -> Path:
        """Return a validated path for metadata-only callers.

        Code which keeps using the file after this method returns must instead
        hold :meth:`acquire_input_lease` (or use :meth:`open_input_capture`).
        """

        with self._condition:
            job = self._get_locked(job_id)
            if job.status is DemoVideoStatus.UPLOADING:
                raise DemoVideoConflictError("Video upload is not complete")
            path = job.input_path.resolve()
            if not self._inside(self.upload_dir, path):
                raise DemoVideoValidationError("Video source escapes upload root")
            if not path.is_file():
                raise DemoVideoNotFoundError("Uploaded video source was not found")
            return path

    def asset_kind_for(self, job_id: str) -> str:
        """Return the immutable internal purpose assigned during upload."""

        with self._condition:
            return self._get_locked(job_id).asset_kind

    def acquire_input_lease(self, job_id: str) -> DemoVideoInputLease:
        """Pin a source file against DELETE, TTL cleanup and shutdown cleanup."""

        with self._condition:
            self._ensure_open_locked()
            job = self._get_locked(job_id)
            if job.status is DemoVideoStatus.UPLOADING:
                raise DemoVideoConflictError("Video upload is not complete")
            path = job.input_path.resolve()
            if not self._inside(self.upload_dir, path):
                raise DemoVideoValidationError("Video source escapes upload root")
            if not path.is_file():
                raise DemoVideoNotFoundError("Uploaded video source was not found")
            job.read_only_leases += 1
            job.updated_at = float(self._clock())
            self._condition.notify_all()
            return DemoVideoInputLease(self, job.job_id, path)

    def _release_input_lease(self, job_id: str) -> None:
        with self._condition:
            job = self._jobs.get(str(job_id))
            if job is None:
                return
            job.read_only_leases = max(0, int(job.read_only_leases) - 1)
            job.updated_at = float(self._clock())
            self._condition.notify_all()

    def open_input_capture(self, job_id: str) -> CaptureLike:
        """Open an independent source decoder for read-only random access."""

        lease = self.acquire_input_lease(job_id)
        try:
            capture = self._capture_factory(str(lease.path))
            if capture is None:
                raise DemoVideoValidationError("VideoCapture was not created")
            return _LeasedCapture(capture, lease)
        except BaseException:
            lease.release()
            raise

    def delete(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> DemoVideoJobSnapshot:
        limit = (
            float(self.config.control_timeout_seconds)
            if timeout is None else max(0.0, float(timeout))
        )
        deadline = self._monotonic() + limit
        with self._condition:
            job = self._get_locked(job_id)
            job_dir = job.input_path.parent.resolve()
            output_dir = (self.output_dir / job.job_id).resolve()
            if not self._inside(self.upload_dir, job_dir):
                raise DemoVideoValidationError("Job directory escapes upload root")
            if not self._inside(self.output_dir, output_dir):
                raise DemoVideoValidationError("Output directory escapes output root")
            # Block new leases while the asset is being removed.
            job.deleting = True
            generation = job.generation
            job.stop_requested = True
            job.pause_requested = False
            job.prepared_run = False
            worker = job.worker
            self._condition.notify_all()

        if worker is not None and worker.is_alive():
            if worker is threading.current_thread():
                with self._condition:
                    current = self._jobs.get(str(job_id))
                    if current is not None:
                        current.deleting = False
                        self._condition.notify_all()
                raise DemoVideoConflictError("A video worker cannot delete its own job")
            worker.join(timeout=max(0.0, deadline - self._monotonic()))
            if worker.is_alive():
                with self._condition:
                    current = self._jobs.get(str(job_id))
                    if current is not None:
                        current.deleting = False
                        self._condition.notify_all()
                raise DemoVideoTimeoutError(
                    "Demo video worker did not stop before the delete timeout"
                )

        with self._condition:
            job = self._jobs.get(str(job_id))
            if job is None or job.status is DemoVideoStatus.DELETED:
                raise DemoVideoNotFoundError(f"Unknown demo video job: {job_id}")
            if job.generation != generation:
                job.deleting = False
                self._condition.notify_all()
                raise DemoVideoConflictError("Job generation changed during delete")
            if job.worker is not None and job.worker is not worker and job.worker.is_alive():
                job.deleting = False
                self._condition.notify_all()
                raise DemoVideoConflictError("Cannot delete a video in use")
            if job.worker is worker and (worker is None or not worker.is_alive()):
                job.worker = None
                job.worker_kind = None
            leases_released = self._condition.wait_for(
                lambda: job.read_only_leases == 0,
                timeout=max(0.0, deadline - self._monotonic()),
            )
            if not leases_released:
                job.deleting = False
                self._condition.notify_all()
                raise DemoVideoTimeoutError(
                    "Video source is still being streamed or decoded"
                )

        try:
            self._remove_path(job_dir)
            self._remove_path(output_dir)
        except BaseException:
            with self._condition:
                current = self._jobs.get(str(job_id))
                if current is not None:
                    current.deleting = False
                    self._condition.notify_all()
            raise

        with self._condition:
            job = self._jobs.get(str(job_id))
            if job is None:
                raise DemoVideoNotFoundError(f"Unknown demo video job: {job_id}")
            self._transition_locked(job, DemoVideoStatus.DELETED)
            snapshot = self._snapshot_locked(job)
            self._jobs.pop(job_id, None)
            self._condition.notify_all()
            logger.info("Demo video job deleted: %s", job_id)
            return snapshot

    def get(self, job_id: str) -> DemoVideoJobSnapshot:
        with self._condition:
            return self._snapshot_locked(self._get_locked(job_id))

    snapshot = get

    def list_jobs(self) -> list[DemoVideoJobSnapshot]:
        with self._condition:
            return [
                self._snapshot_locked(job)
                for job in sorted(self._jobs.values(), key=lambda item: item.created_at)
            ]

    def wait_for_status(
        self,
        job_id: str,
        statuses: DemoVideoStatus | str | set[DemoVideoStatus | str],
        *,
        timeout: float = 5.0,
    ) -> DemoVideoJobSnapshot:
        if isinstance(statuses, (DemoVideoStatus, str)):
            raw_statuses: set[DemoVideoStatus | str] = {statuses}
        else:
            raw_statuses = set(statuses)
        wanted = {
            status if isinstance(status, DemoVideoStatus) else DemoVideoStatus(str(status))
            for status in raw_statuses
        }
        with self._condition:
            self._get_locked(job_id)
            reached = self._condition.wait_for(
                lambda: (
                    job_id not in self._jobs
                    or self._jobs[job_id].status in wanted
                ),
                timeout=max(0.0, float(timeout)),
            )
            if not reached:
                raise DemoVideoTimeoutError(
                    f"Job did not reach {sorted(status.value for status in wanted)}"
                )
            return self._snapshot_locked(self._get_locked(job_id))

    def stats(self) -> dict[str, int | float]:
        with self._condition:
            active = sum(
                1
                for job in self._jobs.values()
                if job.starting
                or (job.worker is not None and job.worker.is_alive())
                or job.status in {DemoVideoStatus.LOADING, DemoVideoStatus.PLAYING, DemoVideoStatus.PAUSED}
            )
            return DemoVideoStats(
                demo_upload_ms=round(self._upload_ms, 3),
                demo_upload_size_bytes=int(self._upload_size_bytes),
                demo_probe_ms=round(self._probe_ms, 3),
                demo_decode_ms=round(self._decode_ms, 3),
                demo_processing_ms=round(self._processing_ms, 3),
                demo_source_fps=round(self._source_fps, 3),
                demo_processing_fps=round(self._processing_fps, 3),
                demo_dropped_frames=int(self._dropped_frames),
                demo_job_queue_age_ms=round(self._queue_age_ms, 3),
                demo_active_jobs=active,
            ).to_dict()

    def _playback_worker(self, job_id: str, generation: int, run_id: str) -> None:
        capture: CaptureLike | None = None
        writer = None
        export_path: Path | None = None
        export_completed = False
        try:
            with self._condition:
                job = self._jobs.get(job_id)
                if job is None or job.generation != generation or job.run_id != run_id:
                    return
                path = str(job.input_path)
                self._queue_age_ms = max(
                    0.0,
                    (self._monotonic() - job.queued_at_monotonic) * 1000.0,
                )

            capture = self._capture_factory(path)
            if capture is None or not bool(capture.isOpened()):
                raise DemoVideoValidationError("Video cannot be opened for playback")

            with self._condition:
                job = self._get_locked(job_id)
                export_enabled = bool(job.export_requested)
                export_path = job.output_path
                width = int(job.width or 0)
                height = int(job.height or 0)
                output_fps = float(job.source_fps or 25.0)
            if export_enabled:
                if export_path is None or width <= 0 or height <= 0:
                    raise DemoVideoValidationError("Video export metadata is unavailable")
                writer = _H264ExportWriter(
                    export_path,
                    width,
                    height,
                    output_fps,
                    max_width=int(getattr(self.config, "export_max_width", 0)),
                )

            processed_count = 0
            paused_at: float | None = None
            while True:
                with self._condition:
                    job = self._jobs.get(job_id)
                    if job is None or job.generation != generation or job.run_id != run_id:
                        return
                    if job.stop_requested or self._closed:
                        if job.status in {DemoVideoStatus.PLAYING, DemoVideoStatus.PAUSED}:
                            self._transition_locked(job, DemoVideoStatus.STOPPED)
                        return

                    if job.pause_requested:
                        if job.status is DemoVideoStatus.PLAYING:
                            self._transition_locked(job, DemoVideoStatus.PAUSED)
                            logger.info("Demo video job paused: %s", job_id)
                        if paused_at is None:
                            paused_at = self._monotonic()
                        self._condition.wait_for(
                            lambda: (
                                self._closed
                                or job.stop_requested
                                or not job.pause_requested
                                or job.generation != generation
                            )
                        )
                        if paused_at is not None:
                            paused_for = max(0.0, self._monotonic() - paused_at)
                            job.paused_total_sec += paused_for
                            job.run_started_monotonic += paused_for
                            paused_at = None
                        continue

                    index = int(job.current_frame)
                    fps = float(job.source_fps or 25.0)
                    source_time = index / max(fps, 0.1)
                    playback_mode = job.playback_mode
                    due_at = job.run_started_monotonic + source_time
                    wait_seconds = due_at - self._monotonic()
                    if playback_mode == "realtime" and wait_seconds > 0.0:
                        self._condition.wait(timeout=wait_seconds)
                        continue

                    if playback_mode == "realtime":
                        elapsed_source = max(
                            0.0,
                            self._monotonic() - job.run_started_monotonic,
                        )
                        frames_behind = int(
                            max(0.0, elapsed_source - source_time) * fps
                        )
                    else:
                        frames_behind = 0

                if frames_behind > 1:
                    skipped, reached_eof = self._skip_frames(
                        capture,
                        frames_behind - 1,
                        job_id,
                        generation,
                        run_id,
                    )
                    if skipped:
                        with self._condition:
                            job = self._jobs.get(job_id)
                            if job is None or job.generation != generation or job.run_id != run_id:
                                return
                            job.current_frame += skipped
                            job.dropped_frames += skipped
                            job.current_time_sec = job.current_frame / max(
                                float(job.source_fps or 25.0),
                                0.1,
                            )
                            job.updated_at = float(self._clock())
                            self._dropped_frames += skipped
                            self._condition.notify_all()
                    if reached_eof:
                        with self._condition:
                            job = self._jobs.get(job_id)
                            if job is not None and job.generation == generation and job.run_id == run_id:
                                self._transition_locked(job, DemoVideoStatus.FINISHED)
                                export_completed = bool(job.export_requested)
                                logger.info("Demo video job finished: %s", job_id)
                        return

                decode_started = self._monotonic()
                ok, frame = capture.read()
                decode_ms = max(0.0, (self._monotonic() - decode_started) * 1000.0)
                with self._condition:
                    self._decode_ms = decode_ms
                if not ok or frame is None:
                    with self._condition:
                        job = self._jobs.get(job_id)
                        if job is not None and job.generation == generation and job.run_id == run_id:
                            self._transition_locked(job, DemoVideoStatus.FINISHED)
                            export_completed = bool(job.export_requested)
                            logger.info("Demo video job finished: %s", job_id)
                    return
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    raise DemoVideoValidationError("Decoder returned an invalid frame")

                with self._condition:
                    job = self._jobs.get(job_id)
                    if job is None or job.generation != generation or job.run_id != run_id:
                        return
                    frame_index = int(job.current_frame)
                    fps = float(job.source_fps or 25.0)
                    source_time = frame_index / max(fps, 0.1)
                    context = DemoFrameContext(
                        job_id=job_id,
                        run_id=run_id,
                        camera_id=job.camera_id,
                        frame_index=frame_index,
                        source_time_sec=source_time,
                        source_timestamp=job.run_started_wall + source_time,
                        status=DemoVideoStatus.PLAYING.value,
                        mode=job.mode,
                        playback_mode=job.playback_mode,
                        export_mode=bool(job.export_requested),
                        video_decode_ms=decode_ms,
                    )

                processing_started = self._monotonic()
                rendered_frame = self._process_frame(context, frame)
                if writer is not None:
                    if not isinstance(rendered_frame, np.ndarray) or rendered_frame.size == 0:
                        raise DemoVideoError("Inference did not return an annotated frame")
                    if rendered_frame.shape[1] != width or rendered_frame.shape[0] != height:
                        rendered_frame = cv2.resize(rendered_frame, (width, height))
                    writer.write(rendered_frame)
                processing_ms = max(
                    0.0,
                    (self._monotonic() - processing_started) * 1000.0,
                )
                processed_count += 1

                with self._condition:
                    job = self._jobs.get(job_id)
                    if job is None or job.generation != generation or job.run_id != run_id:
                        return
                    # Commit only the context that was actually processed.  A
                    # generation change can therefore never advance a new run.
                    if job.current_frame != frame_index:
                        return
                    job.current_frame = frame_index + 1
                    job.current_time_sec = job.current_frame / max(fps, 0.1)
                    job.processing_ms = processing_ms
                    active_elapsed = max(
                        1e-6,
                        self._monotonic() - job.run_started_monotonic,
                    )
                    job.processing_fps = processed_count / active_elapsed
                    job.updated_at = float(self._clock())
                    self._processing_ms = processing_ms
                    self._processing_fps = job.processing_fps
                    self._source_fps = float(job.source_fps or 0.0)
                    if (
                        job.total_frames is not None
                        and job.current_frame >= job.total_frames
                    ):
                        self._transition_locked(job, DemoVideoStatus.FINISHED)
                        if job.export_requested:
                            export_completed = True
                        logger.info("Demo video job finished: %s", job_id)
                        return
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                job = self._jobs.get(job_id)
                if (
                    job is not None
                    and job.generation == generation
                    and job.run_id == run_id
                    and job.status is not DemoVideoStatus.DELETED
                ):
                    self._transition_locked(
                        job,
                        DemoVideoStatus.FAILED,
                        error=self._error_text(exc),
                    )
                    if job.export_requested:
                        job.export_status = "failed"
                    logger.exception("Demo video job failed: %s", job_id)
        finally:
            writer_error: Exception | None = None
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    writer_error = exc
                    export_completed = False
                    logger.exception("Demo H.264 finalization failed: %s", job_id)
            if writer_error is not None:
                with self._condition:
                    job = self._jobs.get(job_id)
                    if job is not None and job.generation == generation:
                        self._transition_locked(
                            job,
                            DemoVideoStatus.FAILED,
                            error=self._error_text(writer_error),
                        )
                        job.export_status = "failed"
            elif export_completed:
                with self._condition:
                    job = self._jobs.get(job_id)
                    if job is not None and job.generation == generation:
                        job.export_status = "ready"
                        job.updated_at = float(self._clock())
                        self._condition.notify_all()
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    logger.warning(
                        "Demo playback capture release failed for %s",
                        job_id,
                        exc_info=True,
                    )
            if export_path is not None and not export_completed:
                try:
                    export_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove incomplete demo export %s", export_path)
            with self._condition:
                job = self._jobs.get(job_id)
                if (
                    job is not None
                    and job.generation == generation
                    and job.run_id == run_id
                    and job.worker is threading.current_thread()
                ):
                    job.worker = None
                    job.worker_kind = None
                self._condition.notify_all()

    def _skip_frames(
        self,
        capture: CaptureLike,
        count: int,
        job_id: str,
        generation: int,
        run_id: str,
    ) -> tuple[int, bool]:
        skipped = 0
        grab = getattr(capture, "grab", None)
        for _ in range(max(0, int(count))):
            with self._condition:
                job = self._jobs.get(job_id)
                if (
                    job is None
                    or job.generation != generation
                    or job.run_id != run_id
                    or job.stop_requested
                    or job.pause_requested
                    or self._closed
                ):
                    return skipped, False
                if (
                    job.total_frames is not None
                    and job.current_frame + skipped >= job.total_frames
                ):
                    return skipped, True
            if callable(grab):
                ok = bool(grab())
            else:
                ok, _discarded = capture.read()
            if not ok:
                return skipped, True
            skipped += 1
        return skipped, False

    def cleanup_expired(self, *, now: float | None = None) -> int:
        current = float(self._clock() if now is None else now)
        ttl = max(0.0, float(self.config.job_ttl_seconds))
        with self._condition:
            expired = [
                job.job_id
                for job in self._jobs.values()
                if current - job.updated_at >= ttl
                and not job.starting
                and not job.deleting
                and job.read_only_leases == 0
                and (job.worker is None or not job.worker.is_alive())
                and job.status not in {
                    DemoVideoStatus.LOADING,
                    DemoVideoStatus.PLAYING,
                    DemoVideoStatus.PAUSED,
                    DemoVideoStatus.DELETED,
                }
            ]
        removed = 0
        for job_id in expired:
            try:
                self.delete(job_id)
                removed += 1
            except DemoVideoNotFoundError:
                continue
            except Exception:
                logger.exception("Expired demo video cleanup failed: %s", job_id)
        return removed

    def cleanup_orphaned_directories(self, *, now: float | None = None) -> int:
        current = float(self._clock() if now is None else now)
        ttl = max(0.0, float(self.config.job_ttl_seconds))
        removed = 0
        for root in (self.upload_dir, self.output_dir):
            try:
                children = list(root.iterdir())
            except FileNotFoundError:
                continue
            for child in children:
                try:
                    age = current - child.stat().st_mtime
                    if age < ttl:
                        continue
                    resolved = child.resolve()
                    if not self._inside(root, resolved):
                        continue
                    self._remove_path(resolved)
                    removed += 1
                except FileNotFoundError:
                    continue
                except OSError:
                    logger.warning(
                        "Could not remove orphaned demo path %s",
                        child,
                        exc_info=True,
                    )
        return removed

    def close(self, *, timeout: float | None = None) -> None:
        with self._condition:
            self._closed = True
            self._janitor_stop.set()
            workers = [
                job.worker
                for job in self._jobs.values()
                if job.worker is not None and job.worker.is_alive()
            ]
            for job in self._jobs.values():
                job.stop_requested = True
                job.pause_requested = False
            self._condition.notify_all()
            janitor = self._janitor_thread

        limit = (
            None
            if timeout is None
            else max(0.0, float(timeout))
        )
        deadline = None if limit is None else self._monotonic() + limit
        for thread in workers:
            if thread is threading.current_thread():
                continue
            remaining = None if deadline is None else max(0.0, deadline - self._monotonic())
            thread.join(timeout=remaining)
        if janitor is not None and janitor is not threading.current_thread():
            remaining = None if deadline is None else max(0.0, deadline - self._monotonic())
            janitor.join(timeout=remaining)

        with self._condition:
            helpers = list(self._helper_threads)
        for helper in helpers:
            if helper is threading.current_thread():
                continue
            remaining = None if deadline is None else max(0.0, deadline - self._monotonic())
            helper.join(timeout=remaining)

        alive = [thread.name for thread in workers if thread.is_alive()]
        alive.extend(thread.name for thread in helpers if thread.is_alive())
        if alive:
            raise DemoVideoTimeoutError(
                f"Demo video workers did not stop: {', '.join(alive)}"
            )
        with self._condition:
            leases_released = self._condition.wait_for(
                lambda: all(job.read_only_leases == 0 for job in self._jobs.values()),
                timeout=(
                    None
                    if deadline is None
                    else max(0.0, deadline - self._monotonic())
                ),
            )
            leased_jobs = [
                job.job_id
                for job in self._jobs.values()
                if job.read_only_leases > 0
            ]
        if not leases_released:
            raise DemoVideoTimeoutError(
                "Demo video source leases did not close: " + ", ".join(leased_jobs)
            )
        # Remove process-local demo files after all workers have stopped.
        with self._condition:
            cleanup_paths = [
                (job.input_path.parent.resolve(),
                 (self.output_dir / job.job_id).resolve())
                for job in self._jobs.values()
                if job.status is not DemoVideoStatus.DELETED
            ]
        for upload_path, output_path in cleanup_paths:
            try:
                if self._inside(self.upload_dir, upload_path):
                    self._remove_path(upload_path)
                if self._inside(self.output_dir, output_path):
                    self._remove_path(output_path)
            except OSError:
                logger.warning(
                    "Could not clean demo video files during shutdown",
                    exc_info=True,
                )

    def _janitor_loop(self) -> None:
        while not self._janitor_stop.wait(self._janitor_interval):
            try:
                self.cleanup_expired()
            except Exception:
                logger.exception("Demo video TTL janitor failed")

    def _assert_camera_available_locked(self, requested: _Job) -> None:
        for other in self._jobs.values():
            if other.job_id == requested.job_id or other.camera_id != requested.camera_id:
                continue
            if other.worker_kind == "read_only_probe":
                continue
            active = (
                other.starting
                or (other.worker is not None and other.worker.is_alive())
                or other.status in {
                    DemoVideoStatus.LOADING,
                    DemoVideoStatus.PLAYING,
                    DemoVideoStatus.PAUSED,
                }
            )
            if active:
                raise DemoVideoConflictError(
                    f"camera_id {requested.camera_id!r} is used by job {other.job_id}"
                )

    def _assert_worker_capacity_locked(self, requested: _Job) -> None:
        active = sum(
            1
            for job in self._jobs.values()
            if job.job_id != requested.job_id
            and job.worker_kind != "read_only_probe"
            and (
                job.starting
                or (job.worker is not None and job.worker.is_alive())
                or job.status in {
                    DemoVideoStatus.LOADING,
                    DemoVideoStatus.PLAYING,
                    DemoVideoStatus.PAUSED,
                }
            )
        )
        if active >= max(1, int(self.config.max_pending_jobs)):
            raise DemoVideoConflictError("Demo video worker capacity is full")

    def _reserve_worker_locked(self, job: _Job) -> None:
        if job.worker is not None and job.worker.is_alive():
            raise DemoVideoConflictError("This job already has an active worker")
        self._assert_camera_available_locked(job)
        self._assert_worker_capacity_locked(job)

    def _transition_locked(
        self,
        job: _Job,
        status: DemoVideoStatus,
        *,
        error: str | None = None,
    ) -> None:
        status = DemoVideoStatus(status)
        if status is not job.status:
            if status not in _TRANSITIONS[job.status]:
                raise DemoVideoConflictError(
                    f"Invalid demo video transition {job.status.value} -> {status.value}"
                )
            job.status = status
        job.error = error
        job.updated_at = float(self._clock())
        self._condition.notify_all()

    def _snapshot_locked(self, job: _Job) -> DemoVideoJobSnapshot:
        return DemoVideoJobSnapshot(
            job_id=job.job_id,
            camera_id=job.camera_id,
            filename=job.filename,
            input_path=str(job.input_path),
            created_at=float(job.created_at),
            updated_at=float(job.updated_at),
            status=job.status.value,
            error=job.error,
            current_frame=int(job.current_frame),
            total_frames=job.total_frames,
            current_time_sec=float(job.current_time_sec),
            duration_sec=job.duration_sec,
            source_fps=job.source_fps,
            width=job.width,
            height=job.height,
            processing_fps=float(job.processing_fps),
            dropped_frames=int(job.dropped_frames),
            pause_requested=bool(job.pause_requested),
            stop_requested=bool(job.stop_requested),
            run_id=job.run_id,
            size_bytes=int(job.size_bytes),
            mime_type=job.mime_type,
            mode=job.mode,
            playback_mode=job.playback_mode,
            processing_ms=float(job.processing_ms),
            export_status=job.export_status,
            output_filename=(job.output_path.name if job.output_path else None),
            run_started_wall=float(job.run_started_wall),
        )

    def _get_locked(self, job_id: str, *, allow_deleting: bool = False) -> _Job:
        job = self._jobs.get(str(job_id))
        if job is None or job.status is DemoVideoStatus.DELETED:
            raise DemoVideoNotFoundError(f"Unknown demo video job: {job_id}")
        if job.deleting and not allow_deleting:
            raise DemoVideoConflictError(f"Demo video job is being deleted: {job_id}")
        return job

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise DemoVideoConflictError("Demo video service is closed")

    @staticmethod
    def _positive_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0.0 else None

    @staticmethod
    def _error_text(exc: BaseException) -> str:
        text = f"{type(exc).__name__}: {exc}".strip()
        return text[:500]

    @staticmethod
    def _remove_path(path: Path) -> None:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


__all__ = [
    "DemoFrameContext",
    "DemoVideoConflictError",
    "DemoVideoError",
    "DemoVideoJobSnapshot",
    "DemoVideoNotFoundError",
    "DemoVideoProcessingDisabledError",
    "DemoVideoService",
    "DemoVideoStats",
    "DemoVideoStatus",
    "DemoVideoTimeoutError",
    "DemoVideoTooLargeError",
    "DemoVideoValidationError",
]
