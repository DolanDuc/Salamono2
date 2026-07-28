"""Nagrywanie surowego strumienia RTSP z kamer przemysłowych po Ethernecie.

Port narzędzia `camrecord.exe` (Piotr, Windows/C++/FFmpeg) na wieloplatformowy
Python, wpięty w backend perimetr. Różnice względem oryginału:

- Działa na Macu/Linuksie/Windows (subprocess ffmpeg, nie skompilowany .exe).
- **Nie nadpisuje** — każde nagranie ma znacznik czasu w nazwie
  (`cam<id>_YYYY-MM-DD_HHMMSS.mp4`), więc kolejna sesja nie kasuje poprzedniej.
  To była główna wada `camrecord` (zawsze pisał do `_streams/cam<id>_stream.mp4`).
- Sterowane przyciskiem z panelu (start/stop), nie z CLI.

Zapis to **stream copy** (`-c copy`) — surowy strumień bez re-enkodowania,
dokładnie jak w oryginale (`avcodec_parameters_copy` + `av_write_frame`).
Zero utraty jakości, minimalne CPU.

Binarka ffmpeg: systemowa (`ffmpeg` w PATH) jeśli jest, inaczej statyczna
z pakietu `imageio-ffmpeg` (pip) — żeby nie zmuszać do instalacji Homebrew.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse


def resolve_ffmpeg() -> str:
    """Zwraca ścieżkę do binarki ffmpeg — systemowa, inaczej z imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - zależne od środowiska
        raise RuntimeError(
            "Nie znaleziono ffmpeg. Zainstaluj `brew install ffmpeg` "
            "albo `pip install imageio-ffmpeg`."
        ) from e


def _mask(url: str) -> str:
    """Ukrywa hasło w URL-u RTSP do wyświetlenia w UI/logach."""
    try:
        p = urlparse(url)
        if p.password:
            netloc = p.hostname or ""
            if p.port:
                netloc += f":{p.port}"
            user = p.username or ""
            return url.replace(f"{user}:{p.password}@", f"{user}:***@")
    except Exception:
        pass
    return url


@dataclass
class Camera:
    id: str
    name: str
    rtsp_url: str
    brand: str = ""

    def public(self) -> dict:
        """Reprezentacja bez hasła — do zwrotu na front."""
        return {
            "id": self.id,
            "name": self.name,
            "brand": self.brand,
            "rtsp_url_masked": _mask(self.rtsp_url),
        }


def load_cameras_from(path: str) -> list[Camera]:
    """Wczytuje listę kamer z pliku JSON (współdzielone przez recorder i mostek detekcji)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    cams = raw.get("cameras", raw) if isinstance(raw, dict) else raw
    out: list[Camera] = []
    for c in cams:
        out.append(Camera(
            id=str(c["id"]),
            name=c.get("name", str(c["id"])),
            rtsp_url=c["rtsp_url"],
            brand=c.get("brand", ""),
        ))
    return out


class _Recording:
    def __init__(self, camera: Camera, path: str, proc: subprocess.Popen, log_path: str):
        self.camera = camera
        self.path = path
        self.proc = proc
        self.log_path = log_path
        self.started_at = datetime.now()

    def elapsed_sec(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

    def size_bytes(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def is_running(self) -> bool:
        return self.proc.poll() is None


class RecorderManager:
    """Trzyma po jednym procesie ffmpeg na kamerę. Thread-safe."""

    def __init__(self, recordings_dir: str, cameras_path: str):
        self.recordings_dir = recordings_dir
        self.cameras_path = cameras_path
        self._active: dict[str, _Recording] = {}
        self._lock = threading.Lock()
        os.makedirs(recordings_dir, exist_ok=True)
        os.makedirs(os.path.join(recordings_dir, "_logs"), exist_ok=True)
        self._ffmpeg = resolve_ffmpeg()

    # ---- konfiguracja kamer -------------------------------------------

    def load_cameras(self) -> list[Camera]:
        return load_cameras_from(self.cameras_path)

    def _camera(self, camera_id: str) -> Camera | None:
        for cam in self.load_cameras():
            if cam.id == camera_id:
                return cam
        return None

    # ---- sterowanie nagrywaniem ---------------------------------------

    def _ffmpeg_cmd(self, camera: Camera, out_path: str) -> list[str]:
        cmd = [self._ffmpeg, "-loglevel", "warning"]
        # TCP dla RTSP — bardziej niezawodne po Ethernecie niż domyślne UDP
        # (oryginalny camrecord ciągnął przez UDP i gubił pakiety).
        if camera.rtsp_url.lower().startswith("rtsp://"):
            cmd += ["-rtsp_transport", "tcp", "-timeout", "5000000"]
        cmd += [
            "-i", camera.rtsp_url,
            "-c", "copy",             # surowy stream, bez re-enkodowania
            # +faststart przepisywałby cały plik na stopie (wolne przy wielkich
            # nagraniach); moov na końcu wystarcza do lokalnego przeglądu.
            "-f", "mp4",
            "-y", out_path,
        ]
        return cmd

    def probe(self, camera_id: str) -> dict:
        """Testuje połączenie z kamerą — pobiera 1 klatkę i czyta parametry.

        Odpowiada na pytanie „czy kamera jest podłączona i widoczna po
        Ethernecie?" zanim zaczniesz nagrywać. Zwraca reachable + rozdzielczość
        i fps, albo czytelny błąd (zły IP / hasło / kabel / podsieć).
        """
        camera = self._camera(camera_id)
        if camera is None:
            raise KeyError(f"Nieznana kamera: {camera_id}")

        cmd = [self._ffmpeg, "-loglevel", "info"]
        if camera.rtsp_url.lower().startswith("rtsp://"):
            cmd += ["-rtsp_transport", "tcp", "-timeout", "5000000"]
        cmd += ["-i", camera.rtsp_url, "-frames:v", "1", "-f", "null", "-"]

        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        except subprocess.TimeoutExpired:
            return {
                "camera_id": camera_id, "reachable": False,
                "error": "Timeout — kamera nie odpowiada (sprawdź kabel/PoE, IP i podsieć).",
            }

        err = p.stderr or ""
        reachable = p.returncode == 0
        info: dict = {"camera_id": camera_id, "reachable": reachable}
        # Wyciągnij rozdzielczość i fps z linii „Stream #0:0: Video: h264 ... 1920x1080 ... 25 fps".
        import re
        m = re.search(r"(\d{2,5})x(\d{2,5})", err)
        if m:
            info["width"], info["height"] = int(m.group(1)), int(m.group(2))
        fps = re.search(r"([\d.]+)\s*fps", err)
        if fps:
            info["fps"] = float(fps.group(1))
        codec = re.search(r"Video:\s*([a-z0-9]+)", err)
        if codec:
            info["codec"] = codec.group(1)
        if not reachable:
            # Zwięzła diagnoza z ostatnich linii ffmpeg.
            tail = [ln for ln in err.strip().splitlines() if ln.strip()][-2:]
            info["error"] = " / ".join(tail) or f"ffmpeg exit {p.returncode}"
        return info

    def start(self, camera_id: str) -> dict:
        with self._lock:
            existing = self._active.get(camera_id)
            if existing and existing.is_running():
                raise RuntimeError(f"Kamera {camera_id} już nagrywa.")

            camera = self._camera(camera_id)
            if camera is None:
                raise KeyError(f"Nieznana kamera: {camera_id}")

            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            fname = f"cam{camera.id}_{ts}.mp4"
            out_path = os.path.join(self.recordings_dir, fname)
            log_path = os.path.join(self.recordings_dir, "_logs", f"{fname}.log")

            log_f = open(log_path, "wb")
            proc = subprocess.Popen(
                self._ffmpeg_cmd(camera, out_path),
                stdin=subprocess.PIPE,   # 'q' = łagodne zamknięcie (zapis moov)
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            self._active[camera_id] = _Recording(camera, out_path, proc, log_path)
            return {"camera_id": camera_id, "filename": fname, "status": "recording"}

    def stop(self, camera_id: str) -> dict:
        with self._lock:
            rec = self._active.get(camera_id)
            if rec is None or not rec.is_running():
                self._active.pop(camera_id, None)
                raise RuntimeError(f"Kamera {camera_id} nie nagrywa.")
            proc = rec.proc

        # Łagodne zamknięcie: 'q' na stdin → ffmpeg dopisuje trailer mp4.
        try:
            if proc.stdin:
                proc.stdin.write(b"q")
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        with self._lock:
            rec = self._active.pop(camera_id, None)
        size = rec.size_bytes() if rec else 0
        return {
            "camera_id": camera_id,
            "filename": os.path.basename(rec.path) if rec else None,
            "size_bytes": size,
            "status": "stopped",
        }

    def status(self) -> list[dict]:
        out = []
        with self._lock:
            items = list(self._active.items())
        for cam_id, rec in items:
            running = rec.is_running()
            if not running:
                # ffmpeg padł sam (np. zerwany link) — sprzątamy wpis.
                with self._lock:
                    self._active.pop(cam_id, None)
            out.append({
                "camera_id": cam_id,
                "filename": os.path.basename(rec.path),
                "recording": running,
                "elapsed_sec": round(rec.elapsed_sec(), 1),
                "size_bytes": rec.size_bytes(),
                "exit_code": rec.proc.returncode,
            })
        return out

    def recordings(self) -> list[dict]:
        out = []
        if not os.path.isdir(self.recordings_dir):
            return out
        for name in os.listdir(self.recordings_dir):
            if not name.lower().endswith(".mp4"):
                continue
            path = os.path.join(self.recordings_dir, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            out.append({
                "filename": name,
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            })
        out.sort(key=lambda r: r["modified"], reverse=True)
        return out

    def shutdown(self) -> None:
        """Zatrzymuje wszystkie nagrania (przy wyłączaniu serwera)."""
        for cam_id in list(self._active.keys()):
            try:
                self.stop(cam_id)
            except Exception:
                pass
