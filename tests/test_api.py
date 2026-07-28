import sys
import os
import time
from unittest.mock import MagicMock

import numpy as np
import cv2
import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.detector import Detection
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.marker_detector import MarkerDetector
from backend.ws_manager import ConnectionManager
from backend.frame_store import FrameStore
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore
from config import CONFIG


def _mock_detector():
    det = MagicMock()
    det.detect.return_value = [
        Detection(class_id=0, class_name="person", category="person",
                  box=(100, 100, 200, 300), confidence=0.85),
        Detection(class_id=7, class_name="truck", category="vehicle",
                  box=(400, 100, 600, 300), confidence=0.92),
    ]
    return det


@pytest.fixture(autouse=True)
def init_app_state(tmp_path):
    app.state.detector = _mock_detector()
    app.state.danger_detector = DangerDetector()
    app.state.temporal_filter = TemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.ws_manager = ConnectionManager()
    app.state.frame_store = FrameStore(str(tmp_path / "flagged"))
    app.state.alert_store = AlertStore(str(tmp_path / "alerts.jsonl"))
    app.state.zone_store = ZoneStore(str(tmp_path / "zones.json"))
    app.state.zone_detector = ZoneBreachDetector()
    app.state.zone_temporal_filter = ZoneTemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.marker_detector = MarkerDetector()
    app.state.calibration_store = CalibrationStore(str(tmp_path / "calibration.json"))
    app.state.marker_zone_cache = {}
    app.state.debug_inject_person = None
    app.state.ppe_detector = None
    app.state.ppe_checker = None
    app.state.frame_counter = 0
    app.state.start_time = time.time()


def _make_test_jpeg(width=640, height=480):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 300), (0, 255, 0), -1)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_stats():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_frames_processed" in data
        assert "uptime_seconds" in data
        assert data["total_frames_processed"] == 0


@pytest.mark.asyncio
async def test_post_frame():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        resp = await client.post(
            "/api/frame",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "test", "timestamp": "1000.0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["frame_id"] == 1
        assert len(data["detections"]) == 2
        assert data["frame_jpeg_b64"] != ""
        assert data["processing_ms"] > 0
        categories = {d["category"] for d in data["detections"]}
        assert "person" in categories
        assert "vehicle" in categories


@pytest.mark.asyncio
async def test_post_frame_no_dangers_far_apart():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        resp = await client.post(
            "/api/frame",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "test", "timestamp": "1000.0"},
        )
        data = resp.json()
        assert len(data["active_dangers"]) == 0
        assert len(data["confirmed_alerts"]) == 0


@pytest.mark.asyncio
async def test_overlap_stationary_vehicle_no_alarm():
    """Osoba nachodzi na STOJĄCY pojazd (ten sam box) → BRAK alarmu.

    Bramkowanie ruchem: zbliżenie do nieruchomego pojazdu jest normalne.
    """
    det = _mock_detector()
    det.detect.return_value = [
        Detection(class_id=0, class_name="person", category="person",
                  box=(150, 150, 250, 300), confidence=0.85),
        Detection(class_id=7, class_name="truck", category="vehicle",
                  box=(100, 100, 300, 300), confidence=0.92),
    ]
    app.state.detector = det

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        for i in range(4):  # kilka klatek, pojazd cały czas stoi
            resp = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={"camera_id": "test_stat", "timestamp": str(1000.0 + i)},
            )
        data = resp.json()
        assert len(data["active_dangers"]) == 0
        assert len(data["confirmed_alerts"]) == 0


@pytest.mark.asyncio
async def test_overlap_moving_vehicle_alarms():
    """Osoba nachodzi na JADĄCY pojazd (box się przesuwa) → alarm."""
    person = Detection(class_id=0, class_name="person", category="person",
                       box=(150, 150, 250, 300), confidence=0.85)
    det = _mock_detector()
    # Pojazd przesuwa się w prawo między klatkami (jazda).
    det.detect.side_effect = [
        [person, Detection(class_id=7, class_name="truck", category="vehicle",
                           box=(100 + i * 30, 100, 300 + i * 30, 300),
                           confidence=0.92)]
        for i in range(5)
    ]
    app.state.detector = det

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        seen_danger = False
        for i in range(5):
            resp = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={"camera_id": "test_move", "timestamp": str(2000.0 + i)},
            )
            if len(resp.json()["active_dangers"]) > 0:
                seen_danger = True
        assert seen_danger, "jadący pojazd blisko osoby powinien dać alarm"


@pytest.mark.asyncio
async def test_alerts_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/alerts")
        assert resp.status_code == 200
        assert resp.json() == []


@pytest.mark.asyncio
async def test_frame_counter_increments():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        for i in range(3):
            resp = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={"camera_id": "test"},
            )
            assert resp.json()["frame_id"] == i + 1
