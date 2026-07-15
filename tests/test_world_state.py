import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.world_state import (
    WorldObservation,
    WorldState,
    WorldZoneDetector,
    WorldZoneTemporalFilter,
)
from backend.zones_store import Zone
from config import WorldConfig


CFG = WorldConfig(assoc_threshold_m=0.7, id_match_threshold_m=0.9,
                  obs_ttl_sec=1.5)

WORLD_ZONE = Zone(
    name="Wykop", severity="DANGER", coordinate_space="world",
    polygon=[[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]],
)


def _obs(cam, x, y, conf=0.9, ts=100.0):
    return WorldObservation(camera_id=cam, x_m=x, y_m=y, confidence=conf,
                            ts=ts)


class TestWorldStateFusion:
    def test_two_cameras_close_fuse_to_one(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0, conf=0.9)], now=100.0)
        ws.update("cam_b", [_obs("cam_b", 2.3, 2.0, conf=0.6)], now=100.1)
        fused = ws.fuse(now=100.2)
        assert len(fused) == 1
        assert set(fused[0].cameras) == {"cam_a", "cam_b"}
        # confidence-weighted mean pulled toward the higher-confidence obs
        assert 2.0 < fused[0].x_m < 2.15
        assert fused[0].confidence == 0.9

    def test_two_cameras_far_stay_separate(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0)], now=100.0)
        ws.update("cam_b", [_obs("cam_b", 2.9, 2.0)], now=100.0)
        fused = ws.fuse(now=100.1)
        assert len(fused) == 2

    def test_same_camera_never_fuses_with_itself(self):
        # Two people close together seen by ONE camera stay two people.
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0),
                            _obs("cam_a", 2.4, 2.0)], now=100.0)
        fused = ws.fuse(now=100.1)
        assert len(fused) == 2

    def test_stale_camera_dropped_after_ttl(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0)], now=100.0)
        ws.update("cam_b", [_obs("cam_b", 5.0, 5.0)], now=101.4)
        fused = ws.fuse(now=101.6)  # cam_a is 1.6s old > TTL 1.5
        assert len(fused) == 1
        assert fused[0].cameras == ["cam_b"]

    def test_update_replaces_camera_slot(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0)], now=100.0)
        ws.update("cam_a", [], now=100.3)  # person left the frame
        assert ws.fuse(now=100.4) == []

    def test_fused_id_stable_across_small_movement(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0)], now=100.0)
        first = ws.fuse(now=100.0)[0]
        ws.update("cam_a", [_obs("cam_a", 2.3, 2.1)], now=100.4)
        second = ws.fuse(now=100.4)[0]
        assert second.fused_id == first.fused_id

    def test_fused_id_new_after_jump(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0)], now=100.0)
        first = ws.fuse(now=100.0)[0]
        ws.update("cam_a", [_obs("cam_a", 8.0, 8.0)], now=100.4)
        second = ws.fuse(now=100.4)[0]
        assert second.fused_id != first.fused_id

    def test_prev_id_used_at_most_once(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0)], now=100.0)
        ws.fuse(now=100.0)
        # Two people now both near the old position — only one inherits.
        ws.update("cam_a", [_obs("cam_a", 2.1, 2.0),
                            _obs("cam_a", 1.9, 2.0)], now=100.4)
        fused = ws.fuse(now=100.4)
        assert len({p.fused_id for p in fused}) == 2


class TestWorldZoneDetector:
    def test_breach_inside_zone(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0)], now=100.0)
        fused = ws.fuse(now=100.0)
        events = WorldZoneDetector().evaluate(fused, [WORLD_ZONE])
        assert len(events) == 1
        assert events[0].severity == "DANGER"

    def test_no_breach_outside_zone(self):
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 8.0, 8.0)], now=100.0)
        fused = ws.fuse(now=100.0)
        assert WorldZoneDetector().evaluate(fused, [WORLD_ZONE]) == []

    def test_ignores_image_space_and_inactive_zones(self):
        image_zone = Zone(name="Img", polygon=[[0.1, 0.1], [0.9, 0.1],
                                               [0.9, 0.9]])
        inactive = WORLD_ZONE.model_copy(update={"active": False})
        ws = WorldState(CFG)
        ws.update("cam_a", [_obs("cam_a", 2.0, 2.0)], now=100.0)
        fused = ws.fuse(now=100.0)
        assert WorldZoneDetector().evaluate(fused, [image_zone, inactive]) == []


class TestWorldZoneTemporalFilter:
    def _event(self, fused_id="p1"):
        from backend.world_state import FusedPerson, WorldZoneBreachEvent
        person = FusedPerson(fused_id=fused_id, x_m=2.0, y_m=2.0,
                             confidence=0.9, cameras=["cam_a", "cam_b"])
        return WorldZoneBreachEvent(zone=WORLD_ZONE, person=person,
                                    severity="DANGER", frame_timestamp=0.0)

    def test_confirms_after_required_streak(self):
        f = WorldZoneTemporalFilter(required=3, cooldown_sec=8.0)
        assert f.update([self._event()], now=1.0) == []
        assert f.update([self._event()], now=1.3) == []
        confirmed = f.update([self._event()], now=1.6)
        assert len(confirmed) == 1
        assert confirmed[0].confirmed is True

    def test_cooldown_blocks_repeat_alert(self):
        f = WorldZoneTemporalFilter(required=1, cooldown_sec=8.0)
        assert len(f.update([self._event()], now=1.0)) == 1
        assert f.update([self._event()], now=2.0) == []
        assert len(f.update([self._event()], now=10.0)) == 1

    def test_same_person_two_cameras_one_alert(self):
        # Dedupe is structural: both cameras produce ONE fused person,
        # so the filter sees one key regardless of source count.
        f = WorldZoneTemporalFilter(required=1, cooldown_sec=8.0)
        confirmed = f.update([self._event()], now=1.0)
        assert len(confirmed) == 1

    def test_streak_resets_when_person_leaves(self):
        f = WorldZoneTemporalFilter(required=3, cooldown_sec=8.0)
        f.update([self._event()], now=1.0)
        f.update([self._event()], now=1.3)
        f.update([], now=1.6)  # person left — streak dies
        assert f.update([self._event()], now=1.9) == []
