import os
import sys
import time
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.detector import Detection
from backend.frame_store import FrameStore
from backend.main import app
from backend.marker_detector import MarkerDetector
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore
from config import CONFIG


SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
RIGHT_HALF = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]


def _mock_detector(detections=None):
    det = MagicMock()
    det.detect.return_value = detections or []
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
        required=1, cooldown_sec=0.5,
    )
    app.state.marker_detector = MarkerDetector()
    app.state.calibration_store = CalibrationStore(str(tmp_path / "calibration.json"))
    app.state.marker_zone_cache = {}
    app.state.debug_inject_person = None
    app.state.ppe_detector = None
    app.state.ppe_checker = None
    app.state.frame_counter = 0
    app.state.start_time = time.time()


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_get_zones_empty():
    async with _client() as c:
        r = await c.get("/api/zones/cam_new")
        assert r.status_code == 200
        assert r.json() == {"camera_id": "cam_new", "zones": []}


@pytest.mark.asyncio
async def test_put_and_get_zones_roundtrip():
    async with _client() as c:
        payload = {"zones": [{
            "name": "Wykop A", "severity": "DANGER", "polygon": SQUARE,
        }]}
        r = await c.put("/api/zones/cam_a", json=payload)
        assert r.status_code == 200
        saved = r.json()["zones"]
        assert len(saved) == 1
        assert saved[0]["name"] == "Wykop A"
        assert saved[0]["id"]  # id auto-generated

        r2 = await c.get("/api/zones/cam_a")
        assert r2.status_code == 200
        assert r2.json()["zones"][0]["name"] == "Wykop A"


@pytest.mark.asyncio
async def test_put_replaces_existing():
    async with _client() as c:
        await c.put("/api/zones/cam_r", json={"zones": [{
            "name": "Old", "severity": "DANGER", "polygon": SQUARE,
        }]})
        r = await c.put("/api/zones/cam_r", json={"zones": [
            {"name": "New1", "severity": "WARNING", "polygon": SQUARE},
            {"name": "New2", "severity": "DANGER", "polygon": SQUARE},
        ]})
        names = [z["name"] for z in r.json()["zones"]]
        assert names == ["New1", "New2"]


@pytest.mark.asyncio
async def test_delete_camera_zones():
    async with _client() as c:
        await c.put("/api/zones/cam_d", json={"zones": [{
            "name": "X", "severity": "DANGER", "polygon": SQUARE,
        }]})
        r = await c.delete("/api/zones/cam_d")
        assert r.status_code == 200
        assert r.json()["zones"] == []
        r2 = await c.get("/api/zones/cam_d")
        assert r2.json()["zones"] == []


@pytest.mark.asyncio
async def test_reject_polygon_with_too_few_vertices():
    async with _client() as c:
        r = await c.put("/api/zones/cam_bad", json={"zones": [{
            "name": "Bad", "severity": "DANGER",
            "polygon": [[0.1, 0.1], [0.5, 0.5]],
        }]})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_world_zone_roundtrip_site():
    async with _client() as c:
        payload = {"zones": [{
            "name": "Wykop world", "severity": "DANGER",
            "coordinate_space": "world",
            "polygon": [[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]],
        }]}
        r = await c.put("/api/zones/_site", json=payload)
        assert r.status_code == 200
        saved = r.json()["zones"][0]
        assert saved["coordinate_space"] == "world"
        assert saved["polygon"] == [[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]]

        r2 = await c.get("/api/zones/_site")
        assert r2.json()["zones"][0]["coordinate_space"] == "world"


@pytest.mark.asyncio
async def test_world_zone_accepts_metres_beyond_unit_range():
    # Metre coordinates >1 must NOT be rejected by the 0..1 image check.
    async with _client() as c:
        r = await c.put("/api/zones/_site", json={"zones": [{
            "name": "Daleka", "severity": "WARNING",
            "coordinate_space": "world",
            "polygon": [[-2.0, 0.0], [10.0, 0.0], [10.0, 8.5]],
        }]})
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_world_zone_rejects_out_of_range_metres():
    async with _client() as c:
        r = await c.put("/api/zones/_site", json={"zones": [{
            "name": "Bad", "severity": "DANGER",
            "coordinate_space": "world",
            "polygon": [[0.0, 0.0], [999.0, 0.0], [999.0, 5.0]],
        }]})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_world_zone_rejects_marker_ids():
    async with _client() as c:
        r = await c.put("/api/zones/_site", json={"zones": [{
            "name": "Bad", "severity": "DANGER",
            "coordinate_space": "world",
            "polygon": [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]],
            "marker_ids": [10, 20, 30, 40],
        }]})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_reject_unknown_coordinate_space():
    async with _client() as c:
        r = await c.put("/api/zones/cam_bad", json={"zones": [{
            "name": "Bad", "severity": "DANGER",
            "coordinate_space": "galactic", "polygon": SQUARE,
        }]})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_image_zone_still_rejects_metres():
    # Regression: default image zones keep the 0..1 validation.
    async with _client() as c:
        r = await c.put("/api/zones/cam_img", json={"zones": [{
            "name": "Bad", "severity": "DANGER",
            "polygon": [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]],
        }]})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_reject_polygon_out_of_bounds():
    async with _client() as c:
        r = await c.put("/api/zones/cam_bad", json={"zones": [{
            "name": "Bad", "severity": "DANGER",
            "polygon": [[0.1, 0.1], [1.5, 0.5], [0.5, 0.5]],
        }]})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_reject_invalid_severity():
    async with _client() as c:
        r = await c.put("/api/zones/cam_bad", json={"zones": [{
            "name": "Bad", "severity": "CRITICAL", "polygon": SQUARE,
        }]})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_list_all_cameras():
    async with _client() as c:
        await c.put("/api/zones/cam_x", json={"zones": [{
            "name": "X", "severity": "DANGER", "polygon": SQUARE,
        }]})
        await c.put("/api/zones/cam_y", json={"zones": [{
            "name": "Y", "severity": "WARNING", "polygon": SQUARE,
        }]})
        r = await c.get("/api/zones")
        cams = r.json()["cameras"]
        assert set(cams.keys()) == {"cam_x", "cam_y"}


def _jpeg(width=640, height=480):
    import cv2
    import numpy as np
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_frame_ingest_reports_zone_breach():
    # Mock detector to return one person in the right half of frame
    person = Detection(class_id=0, class_name="person", category="person",
                       box=(450, 280, 550, 460), confidence=0.9)
    app.state.detector = _mock_detector([person])

    async with _client() as c:
        await c.put("/api/zones/cam_int", json={"zones": [{
            "name": "Wykop", "severity": "DANGER", "polygon": RIGHT_HALF,
        }]})
        r = await c.post(
            "/api/frame",
            files={"image": ("f.jpg", _jpeg(), "image/jpeg")},
            data={"camera_id": "cam_int", "mode": "site",
                  "timestamp": "1000.0"},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["active_zone_breaches"]) == 1
        # required=1 in the fixture → also confirmed on first frame
        assert len(data["confirmed_zone_breaches"]) == 1
        breach = data["confirmed_zone_breaches"][0]
        assert breach["zone_name"] == "Wykop"
        assert breach["severity"] == "DANGER"


@pytest.mark.asyncio
async def test_frame_ingest_no_breach_when_person_outside_zone():
    person = Detection(class_id=0, class_name="person", category="person",
                       box=(50, 280, 150, 460), confidence=0.9)  # left half
    app.state.detector = _mock_detector([person])

    async with _client() as c:
        await c.put("/api/zones/cam_int2", json={"zones": [{
            "name": "Right", "severity": "DANGER", "polygon": RIGHT_HALF,
        }]})
        r = await c.post(
            "/api/frame",
            files={"image": ("f.jpg", _jpeg(), "image/jpeg")},
            data={"camera_id": "cam_int2", "mode": "site",
                  "timestamp": "1000.0"},
        )
        data = r.json()
        assert data["active_zone_breaches"] == []
        assert data["confirmed_zone_breaches"] == []
