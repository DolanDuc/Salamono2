"""Mostek RTSP → detekcja + kalibracja ArUco.

Ciągnie klatki z kamer przemysłowych po RTSP i przepuszcza je przez DOKŁADNIE
ten sam pipeline co ścieżka HTTP z telefonu (`backend/routes/ingest.py`):
detekcja YOLO, wykrywanie markerów ArUco, fuzja multi-camera do wspólnego
układu metrycznego, alarmy, broadcast po WebSocket.

Dzięki temu kamery RTSP (Hikvision/Axis) trafiają na żywo do systemu, a nie
tylko się nagrywają. Kalibracja ArUco włącza się sama — pipeline sprawdza
`calibration_store.get(camera_id)`; wystarczy skalibrować kamerę na stronie
„Kalibracja" tymi samymi markerami co pozostałe.

Architektura współbieżności: klatki grabuje wątek per kamera (OpenCV
VideoCapture blokuje), ale samo przetwarzanie (`_handle_site`) jest wrzucane
na pętlę asyncio przez `run_coroutine_threadsafe` — więc leci SERYJNIE, tak
samo jak żądania HTTP. Zero nowych wyścigów na `world_state`.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from types import SimpleNamespace

import cv2

from backend.recorder import Camera, load_cameras_from


class _BridgeState:
    def __init__(self, camera: Camera, mode: str):
        self.camera = camera
        self.mode = mode
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.frames = 0
        self.started_at = time.time()
        self.last_frame_at = 0.0
        self.error: str | None = None
        self.connected = False


class DetectionBridge:
    """Po jednym wątku-czytniku RTSP na kamerę. Przetwarzanie na pętli asyncio."""

    def __init__(self, app, cameras_path: str, target_fps: float = 4.0):
        self.app = app
        self.cameras_path = cameras_path
        self.target_fps = target_fps
        self._states: dict[str, _BridgeState] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Ostatni moment, gdy fuzja połączyła tę samą osobę z >1 kamery
        # = kamery patrzą na ten sam punkt z różnych stron (kalibracja działa).
        self._overlap: dict | None = None
        # RTSP po TCP dla OpenCV/FFmpeg (niezawodniej po Ethernecie niż UDP)
        # + timeout 5 s, żeby nieosiągalna kamera nie wisiała w nieskończoność.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000",
        )

    def _camera(self, camera_id: str) -> Camera | None:
        for cam in load_cameras_from(self.cameras_path):
            if cam.id == camera_id:
                return cam
        return None

    # ---- sterowanie ---------------------------------------------------

    def start(self, camera_id: str, mode: str = "site") -> dict:
        # Zapamiętaj pętlę asyncio (start leci z handlera HTTP = z pętli).
        self._loop = asyncio.get_running_loop()
        with self._lock:
            st = self._states.get(camera_id)
            if st and st.thread and st.thread.is_alive():
                raise RuntimeError(f"Detekcja live już działa dla kamery {camera_id}.")
            camera = self._camera(camera_id)
            if camera is None:
                raise KeyError(f"Nieznana kamera: {camera_id}")
            state = _BridgeState(camera, mode)
            t = threading.Thread(target=self._worker, args=(state,), daemon=True)
            state.thread = t
            self._states[camera_id] = state
            t.start()
        return {"camera_id": camera_id, "mode": mode, "status": "live"}

    def stop(self, camera_id: str) -> dict:
        with self._lock:
            st = self._states.get(camera_id)
            if st is None or not (st.thread and st.thread.is_alive()):
                self._states.pop(camera_id, None)
                raise RuntimeError(f"Detekcja live nie działa dla kamery {camera_id}.")
            st.stop_event.set()
            thread = st.thread
        thread.join(timeout=5)
        with self._lock:
            self._states.pop(camera_id, None)
        return {"camera_id": camera_id, "status": "stopped"}

    def status(self) -> list[dict]:
        out = []
        with self._lock:
            items = list(self._states.items())
        for cam_id, st in items:
            alive = bool(st.thread and st.thread.is_alive())
            if not alive:
                with self._lock:
                    self._states.pop(cam_id, None)
            out.append({
                "camera_id": cam_id,
                "live": alive,
                "mode": st.mode,
                "connected": st.connected,
                "frames": st.frames,
                "elapsed_sec": round(time.time() - st.started_at, 1),
                "error": st.error,
            })
        return out

    def overlap(self, fresh_sec: float = 3.0) -> dict | None:
        """Zwraca info o nakładaniu się kamer, jeśli świeże (inaczej None)."""
        ov = self._overlap
        if ov and (time.time() - ov["at"]) <= fresh_sec:
            return ov
        return None

    def shutdown(self) -> None:
        for cam_id in list(self._states.keys()):
            try:
                self.stop(cam_id)
            except Exception:
                pass

    # ---- wątek-czytnik ------------------------------------------------

    def _worker(self, state: _BridgeState) -> None:
        url = state.camera.rtsp_url
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # trzymaj tylko najświeższą klatkę
        except Exception:
            pass
        if not cap.isOpened():
            state.error = "Nie udało się otworzyć strumienia (kabel/IP/hasło?)."
            cap.release()
            return

        state.connected = True
        min_interval = 1.0 / max(self.target_fps, 0.1)
        last_proc = 0.0
        fails = 0

        while not state.stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                # Pojedyncze zgubione klatki się zdarzają — nie migaj od razu.
                if fails > 10:
                    state.connected = False
                    state.error = "Utrata sygnału ze strumienia."
                if fails > 50:
                    break
                time.sleep(0.1)
                continue
            fails = 0
            state.connected = True

            now = time.monotonic()
            if now - last_proc < min_interval:
                continue  # flush bufora, trzymaj niską latencję
            last_proc = now

            if self._loop is None:
                break
            fut = asyncio.run_coroutine_threadsafe(
                self._process(frame, state.camera.id, state.mode), self._loop,
            )
            try:
                fut.result(timeout=30)  # backpressure: nie wyprzedzaj przetwarzania
                state.frames += 1
                state.last_frame_at = time.time()
                state.error = None
            except Exception as e:  # noqa: BLE001
                state.error = f"Błąd przetwarzania: {e}"

        cap.release()
        state.connected = False

    # ---- przetwarzanie na pętli asyncio -------------------------------

    async def _process(self, frame, camera_id: str, mode: str):
        # Import tu, żeby uniknąć cyklu importów przy starcie.
        from backend.routes.ingest import _handle_checkpoint, _handle_site

        t0 = time.monotonic()
        now = time.time()
        req = SimpleNamespace(app=self.app)
        if mode == "checkpoint" and self.app.state.ppe_detector is not None:
            result = await _handle_checkpoint(req, frame, now, t0, camera_id)
        else:
            result = await _handle_site(req, frame, now, t0, camera_id,
                                        zones_only=(mode == "zones"))

        # Sygnał „kamery patrzą na to samo z różnych stron": fuzja połączyła
        # tę samą osobę z ≥2 kamer (world_persons z listą cameras > 1).
        world_persons = getattr(result, "world_persons", None) or []
        multi = [wp for wp in world_persons if len(getattr(wp, "cameras", [])) >= 2]
        if multi:
            cams = sorted({c for wp in multi for c in wp.cameras})
            self._overlap = {"cameras": cams, "persons": len(multi), "at": time.time()}

        await self.app.state.ws_manager.broadcast_json(result.model_dump())
        return result
