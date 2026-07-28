from dataclasses import dataclass
from collections import defaultdict

from backend.detector import Detection
from config import CONFIG, DangerConfig


@dataclass
class DangerEvent:
    rule_name: str
    person: Detection
    hazard: Detection
    distance_px: float
    overlap_iou: float
    severity: str  # "WARNING" or "DANGER"
    frame_timestamp: float
    confirmed: bool = False


def bbox_iou(a: tuple, b: tuple) -> float:
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_min_distance(a: tuple, b: tuple) -> float:
    dx = max(0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0, max(a[1] - b[3], b[1] - a[3]))
    return (dx**2 + dy**2) ** 0.5


class DangerDetector:
    def __init__(self, config: DangerConfig | None = None):
        self.cfg = config or CONFIG.danger

    def evaluate(
        self,
        detections: list[Detection],
        frame_timestamp: float = 0.0,
        dangerous_vehicle_boxes: set | None = None,
    ) -> list[DangerEvent]:
        """dangerous_vehicle_boxes: gdy podane, alarm liczą TYLKO pojazdy z tego
        zbioru (ruchome/w oknie podtrzymania). None = stara logika (wszystkie)."""
        persons = [d for d in detections if d.category == "person"]
        vehicles = [d for d in detections if d.category == "vehicle"]
        if dangerous_vehicle_boxes is not None:
            vehicles = [v for v in vehicles if tuple(v.box) in dangerous_vehicle_boxes]
        events = []
        for p in persons:
            for v in vehicles:
                iou = bbox_iou(p.box, v.box)
                dist = bbox_min_distance(p.box, v.box)
                if iou > self.cfg.overlap_iou:
                    events.append(DangerEvent(
                        rule_name="person_vehicle_overlap",
                        person=p, hazard=v,
                        distance_px=0.0, overlap_iou=iou,
                        severity="DANGER",
                        frame_timestamp=frame_timestamp,
                    ))
                elif dist < self.cfg.proximity_px:
                    events.append(DangerEvent(
                        rule_name="person_near_vehicle",
                        person=p, hazard=v,
                        distance_px=dist, overlap_iou=0.0,
                        severity="WARNING",
                        frame_timestamp=frame_timestamp,
                    ))
        return events


class TemporalFilter:
    def __init__(self, required: int = 3, cooldown_sec: float = 10.0):
        self.required = required
        self.cooldown_sec = cooldown_sec
        self._streak: dict[str, int] = defaultdict(int)
        self._last_alert_time: dict[str, float] = {}

    @staticmethod
    def _pair_key(event: DangerEvent) -> str:
        def center(box):
            return ((box[0] + box[2]) // 64, (box[1] + box[3]) // 64)
        pc = center(event.person.box)
        vc = center(event.hazard.box)
        return f"{event.rule_name}_{pc}_{vc}"

    def update(self, events: list[DangerEvent], now: float) -> list[DangerEvent]:
        current_keys = set()
        confirmed = []
        for e in events:
            key = self._pair_key(e)
            current_keys.add(key)
            self._streak[key] += 1
            if self._streak[key] >= self.required:
                last = self._last_alert_time.get(key, float("-inf"))
                if now - last >= self.cooldown_sec:
                    e.confirmed = True
                    confirmed.append(e)
                    self._last_alert_time[key] = now

        dead = [k for k in self._streak if k not in current_keys]
        for k in dead:
            del self._streak[k]
        return confirmed
