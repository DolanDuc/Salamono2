"""Rolling evidence clips for confirmed safety events.

The recorder keeps a small JPEG-compressed ring buffer per camera.  When an
alert is confirmed it snapshots the preceding frames and collects a short
period of subsequent frames.  MP4 decoding/encoding and disk I/O are performed
by one bounded background writer so an alarm cannot freeze the frame-ingest
request.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

from config import CONFIG, EvidenceConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EncodedFrame:
    timestamp: float
    jpeg: bytes


@dataclass
class _PendingClip:
    camera_id: str
    alert_id: str
    end_timestamp: float
    filepath: str
    url: str
    frames: list[_EncodedFrame] = field(default_factory=list)


@dataclass(frozen=True)
class _EvidenceJob:
    """Immutable snapshot handed to the background MP4 writer."""

    filepath: str
    frames: tuple[_EncodedFrame, ...]
    final: bool


class EvidenceRecorder:
    # Evidence is normally generated rarely.  This limit protects the process
    # from an alert storm while still leaving ample room for separate cameras
    # and for a preliminary/final job for the same alert.
    _MAX_WRITE_JOBS = 16

    def __init__(self, config: EvidenceConfig | None = None):
        self.cfg = config or CONFIG.evidence
        self.enabled = bool(self.cfg.enabled)
        self.base_dir = self.cfg.clips_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._buffers: dict[str, deque[_EncodedFrame]] = defaultdict(
            lambda: deque(maxlen=max(1, int(self.cfg.max_buffer_frames)))
        )
        self._pending: dict[str, _PendingClip] = {}
        self._last_sample: dict[str, float] = {}
        self._lock = threading.RLock()
        self._closed = False

        self._write_jobs: deque[_EvidenceJob] = deque()
        self._write_condition = threading.Condition()
        self._writer_stopping = False
        self._writer_active = False
        self._dropped_write_jobs = 0
        self._writer_thread: threading.Thread | None = None
        if self.enabled:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="evidence-writer",
                daemon=True,
            )
            self._writer_thread.start()

    def _encode(self, frame: np.ndarray) -> bytes | None:
        # Camera footage is 2688 px wide, which makes a six-second evidence clip
        # heavier than the source recording it was cut from. Nobody reviews an
        # incident at full sensor resolution; scale on the way into the buffer so
        # both memory and the written file stay reasonable.
        max_width = int(getattr(self.cfg, "max_width", 0))
        if max_width and frame.shape[1] > max_width:
            height = max(2, int(round(frame.shape[0] * max_width / frame.shape[1])))
            frame = cv2.resize(
                frame, (max_width, height - height % 2), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.jpeg_quality)],
        )
        return bytes(buf) if ok else None

    def should_sample(self, camera_id: str, timestamp: float) -> bool:
        """Return whether the next frame would enter the evidence ring.

        This cheap preflight lets the caller postpone expensive backend
        annotation until the recorder's own (usually much lower) sample rate.
        :meth:`push` still repeats the check under the same lock, so this is an
        optimization hint rather than a correctness requirement.
        """

        if not self.enabled:
            return False
        interval = 1.0 / max(float(self.cfg.sample_fps), 0.1)
        with self._lock:
            if self._closed:
                return False
            last = self._last_sample.get(camera_id, float("-inf"))
            return timestamp - last >= interval

    def push(self, camera_id: str, frame: np.ndarray, timestamp: float) -> float:
        """Sample a frame and return the synchronous JPEG encode time in ms."""

        if not self.enabled or frame is None or frame.size == 0:
            return 0.0
        interval = 1.0 / max(float(self.cfg.sample_fps), 0.1)
        with self._lock:
            if self._closed:
                return 0.0
            last = self._last_sample.get(camera_id, float("-inf"))
            if timestamp - last < interval:
                return 0.0
            encode_started = time.perf_counter()
            encoded = self._encode(frame)
            encode_ms = (time.perf_counter() - encode_started) * 1000.0
            if encoded is None:
                return encode_ms
            item = _EncodedFrame(timestamp=timestamp, jpeg=encoded)
            self._last_sample[camera_id] = timestamp
            self._buffers[camera_id].append(item)

            finished: list[str] = []
            for key, pending in self._pending.items():
                if pending.camera_id != camera_id:
                    continue
                if not pending.frames or pending.frames[-1].timestamp < timestamp:
                    pending.frames.append(item)
                if timestamp >= pending.end_timestamp:
                    self._enqueue_write(pending, final=True)
                    finished.append(key)
            for key in finished:
                self._pending.pop(key, None)
            return encode_ms

    def trigger(self, camera_id: str, alert_id: str, timestamp: float) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            if self._closed:
                return None
            ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{alert_id}.mp4"
            filepath = os.path.join(self.base_dir, filename)
            url = f"/static/clips/{filename}"
            start = timestamp - max(0.0, float(self.cfg.pre_seconds))
            frames = [f for f in self._buffers[camera_id] if f.timestamp >= start]
            pending = _PendingClip(
                camera_id=camera_id,
                alert_id=alert_id,
                end_timestamp=timestamp + max(0.0, float(self.cfg.post_seconds)),
                filepath=filepath,
                url=url,
                frames=list(frames),
            )
            # Encode once after the post-event window. Previously an immediate
            # preliminary H.264 job and a second final job competed with live
            # YOLO/MediaPipe work. reset_camera()/close() still finalize the
            # frames collected so far if the stream ends before this window.
            if self.cfg.post_seconds > 0:
                self._pending[alert_id] = pending
            else:
                self._enqueue_write(pending, final=True)
            return url

    def reset_camera(self, camera_id: str) -> None:
        """Drop pre-event history while safely finalizing pending clips."""
        with self._lock:
            self._buffers.pop(camera_id, None)
            self._last_sample.pop(camera_id, None)
            for key, pending in list(self._pending.items()):
                if pending.camera_id != camera_id:
                    continue
                self._enqueue_write(pending, final=True)
                self._pending.pop(key, None)

    def _enqueue_write(self, pending: _PendingClip, *, final: bool) -> bool:
        if not pending.frames:
            return False
        job = _EvidenceJob(
            filepath=pending.filepath,
            frames=tuple(pending.frames),
            final=final,
        )
        with self._write_condition:
            if self._writer_stopping:
                return False

            # If a preliminary snapshot has not started yet, replace it with
            # the richer final snapshot rather than writing the same path
            # twice.  Never replace a queued final snapshot with an older one.
            for index, existing in enumerate(self._write_jobs):
                if existing.filepath != job.filepath:
                    continue
                should_replace = (
                    (job.final and not existing.final)
                    or len(job.frames) >= len(existing.frames)
                )
                if should_replace and not (existing.final and not job.final):
                    self._write_jobs[index] = job
                self._write_condition.notify()
                return True

            if len(self._write_jobs) >= self._MAX_WRITE_JOBS:
                # Prefer dropping an obsolete preliminary write.  If an alert
                # storm filled the queue with final clips, discard the oldest
                # queued job so live ingest remains bounded and the newest
                # evidence is retained.
                drop_index = next(
                    (
                        index
                        for index, queued in enumerate(self._write_jobs)
                        if not queued.final
                    ),
                    0,
                )
                del self._write_jobs[drop_index]
                self._dropped_write_jobs += 1

            self._write_jobs.append(job)
            self._write_condition.notify()
            return True

    def _writer_loop(self) -> None:
        while True:
            with self._write_condition:
                while not self._write_jobs and not self._writer_stopping:
                    self._write_condition.wait()
                if not self._write_jobs and self._writer_stopping:
                    return
                job = self._write_jobs.popleft()
                self._writer_active = True

            try:
                self._write_job(job)
            except Exception:
                # A corrupt frame or a transient filesystem error must not
                # terminate the only writer and strand subsequent clips.
                logger.exception("Evidence MP4 write failed: %s", job.filepath)
            finally:
                with self._write_condition:
                    self._writer_active = False
                    self._write_condition.notify_all()

    def _write_job(self, job: _EvidenceJob) -> None:
        if not job.frames:
            return
        decoded: list[np.ndarray] = []
        for item in job.frames:
            arr = np.frombuffer(item.jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                decoded.append(frame)
        if not decoded:
            return

        height, width = decoded[0].shape[:2]
        # VideoWriter exposes a partially-written destination while it is
        # running.  Finish a uniquely named MP4 beside the target and publish
        # it atomically, so the static-file route sees either the previous
        # preliminary clip or the complete replacement.
        temp_path = (
            f"{job.filepath}.{threading.get_ident()}."
            f"{uuid.uuid4().hex}.tmp.mp4"
        )
        ffmpeg = self.cfg.ffmpeg_binary or shutil.which("ffmpeg")
        if not ffmpeg:
            logger.error("Evidence clip not created: FFmpeg is unavailable (PATH/EVIDENCE_FFMPEG_BINARY)")
            return
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(max(1.0, float(self.cfg.sample_fps))), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", temp_path,
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
            )
            assert process.stdin is not None
            for frame in decoded:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            returncode = process.wait()
            if returncode != 0:
                logger.error("Evidence H.264 encoding failed: %s", stderr.strip())
                return
        except (OSError, BrokenPipeError) as exc:
            logger.error("Evidence H.264 encoding failed: %s", exc)
            return

        try:
            if os.path.getsize(temp_path) > 0:
                os.replace(temp_path, job.filepath)
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._closed:
                # A shutdown before post_seconds elapsed still persists every
                # post frame collected so far.
                for pending in list(self._pending.values()):
                    self._enqueue_write(pending, final=True)
                self._pending.clear()
                self._closed = True

        with self._write_condition:
            if not self._writer_stopping:
                self._writer_stopping = True
                self._write_condition.notify_all()

        writer_thread = self._writer_thread
        if writer_thread is not None and writer_thread is not threading.current_thread():
            # The writer exits only after draining all queued jobs.
            writer_thread.join()
