"""Multi-camera fusion end-to-end through /api/frame.

Dwie kamery ze znanymi homografiami (piksel/100 = metr) patrzą na tę samą
osobę — po serii klatek musi powstać DOKŁADNIE JEDEN alarm world-zone,
a world_persons ma jedną sfuzowaną pozycję z obu kamer.
"""
import os
import sys
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.alert_storage import AlertStore
from backend.calibration import Calibration, CalibrationStore
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.detector import Detection
from backend.frame_store import FrameStore
from backend.main import app
from backend.marker_detector import MarkerDetector
from backend.world_state import (
    WorldState,
    WorldZoneDetector,
    WorldZoneTemporalFilter,
)
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import Zone, ZoneStore
from config import CONFIG, WorldConfig

# pixel * 0.01 = metre → foot at (200, 200) px lands at (2.0, 2.0) m
SCALE_H = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]
# same plane but shifted 10 cm in x — a slightly different viewpoint
SHIFT_H = [[0.01, 0.0, 0.1], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]

WORLD_ZONE = Zone(
    id="wz1", name="Wykop world", severity="DANGER",
    coordinate_space="world",
    polygon=[[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]],
)

# person bbox whose foot point (bottom-center) is at (200, 200) px
PERSON = Detection(class_id=0, class_name="person", category="person",
                   box=(160, 80, 240, 200), confidence=0.9)


def _calibration(camera_id: str, homography) -> Calibration:
    return Calibration(camera_id=camera_id, marker_ids=[10, 20, 30, 40],
                       width_m=3.0, height_m=3.0, homography=homography)


@pytest.fixture(autouse=True)
def init_app_state(tmp_path):
    det = MagicMock()
    det.detect.side_effect = lambda _f: [PERSON]
    app.state.detector = det
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
    app.state.zone_temporal_filter = ZoneTemporalFilter(required=1,
                                                        cooldown_sec=0.5)
    app.state.marker_detector = MarkerDetector()
    app.state.calibration_store = CalibrationStore(
        str(tmp_path / "calibration.json"))
    app.state.world_state = WorldState(WorldConfig())
    app.state.world_zone_detector = WorldZoneDetector()
    app.state.world_zone_filter = WorldZoneTemporalFilter(required=3,
                                                          cooldown_sec=60.0)
    app.state.marker_zone_cache = {}
    app.state.debug_inject_person = None
    app.state.ppe_detector = None
    app.state.ppe_checker = None
    app.state.frame_counter = 0
    app.state.start_time = time.time()


def _blank_jpeg(w=640, h=640):
    frame = np.full((h, w, 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


async def _post_frame(c, camera_id):
    return await c.post(
        "/api/frame",
        files={"image": ("f.jpg", _blank_jpeg(), "image/jpeg")},
        data={"camera_id": camera_id},
    )


def _seed_calibrations_and_zone():
    app.state.calibration_store.set(_calibration("cam_a", SCALE_H))
    app.state.calibration_store.set(_calibration("cam_b", SHIFT_H))
    app.state.zone_store.replace("_site", [WORLD_ZONE])


@pytest.mark.asyncio
async def test_two_cameras_one_fused_person_one_alarm():
    _seed_calibrations_and_zone()
    async with _client() as c:
        last = None
        for _ in range(4):  # alternating frames fill the 3-frame streak
            await _post_frame(c, "cam_a")
            last = await _post_frame(c, "cam_b")
        body = last.json()
        assert body["camera_id"] == "cam_b"
        # one fused person sourced from both cameras
        assert len(body["world_persons"]) == 1
        assert set(body["world_persons"][0]["cameras"]) == {"cam_a", "cam_b"}
        assert body["world_zones"][0]["id"] == "wz1"
        # exactly ONE alarm record despite 8 frames from 2 cameras
        records = [r for r in app.state.alert_store.query(limit=100)
                   if r.kind == "world_zone_breach"]
        assert len(records) == 1
        assert records[0].camera_id == "site"
        assert set(records[0].details["cameras"]) == {"cam_a", "cam_b"}


@pytest.mark.asyncio
async def test_no_world_output_without_calibration():
    app.state.zone_store.replace("_site", [WORLD_ZONE])
    async with _client() as c:
        r = await _post_frame(c, "cam_nocal")
        body = r.json()
        assert body["world_persons"] == []
        assert body["world_zones"] == []
        assert body["confirmed_world_breaches"] == []


@pytest.mark.asyncio
async def test_no_alarm_when_person_outside_world_zone():
    _seed_calibrations_and_zone()
    # foot at (550, 550) px → (5.5, 5.5) m — outside the 1..4 x 1..3 zone
    app.state.detector.detect.side_effect = lambda _f: [Detection(
        class_id=0, class_name="person", category="person",
        box=(510, 430, 590, 550), confidence=0.9,
    )]
    async with _client() as c:
        for _ in range(4):
            await _post_frame(c, "cam_a")
            last = await _post_frame(c, "cam_b")
        assert len(last.json()["world_persons"]) == 1
        records = [r for r in app.state.alert_store.query(limit=100)
                   if r.kind == "world_zone_breach"]
        assert records == []


@pytest.mark.asyncio
async def test_world_state_endpoint_and_calibration_list():
    _seed_calibrations_and_zone()
    async with _client() as c:
        r = await c.get("/api/calibration")
        assert r.status_code == 200
        cams = {c_["camera_id"] for c_ in r.json()["calibrations"]}
        assert cams == {"cam_a", "cam_b"}

        await _post_frame(c, "cam_a")
        rs = await c.get("/api/world/state")
        assert rs.status_code == 200
        body = rs.json()
        assert len(body["persons"]) == 1
        assert body["zones"][0]["id"] == "wz1"
        assert {cm["camera_id"] for cm in body["cameras"]} == {"cam_a", "cam_b"}


@pytest.mark.asyncio
async def test_per_camera_debug_inject():
    app.state.detector.detect.side_effect = lambda _f: []
    async with _client() as c:
        r = await c.post("/api/debug/inject-person", json={
            "box_norm": [0.4, 0.4, 0.6, 0.8], "frames": 2,
            "camera_id": "cam_a",
        })
        assert r.json()["status"] == "armed"
        # cam_b gets nothing, cam_a gets the synthetic person
        rb = await _post_frame(c, "cam_b")
        assert [d for d in rb.json()["detections"]
                if d["category"] == "person"] == []
        ra = await _post_frame(c, "cam_a")
        persons = [d for d in ra.json()["detections"]
                   if d["category"] == "person"]
        assert len(persons) == 1
