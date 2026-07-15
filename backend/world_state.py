"""Multi-camera ground-plane fusion.

Każda kamera skalibrowana na TE SAME 4 markery ArUco ma homografię do
wspólnej płaszczyzny metrycznej (backend/calibration.py). Foot point osoby
(bottom-center bboxa) rzutujemy przez Calibration.project() → (x, y) w
metrach. Ten moduł trzyma ostatnie obserwacje per kamera, klastruje je
między kamerami (osoba widziana z 2 kątów = 1 punkt) i ewaluuje strefy
world-space (Zone.coordinate_space == "world", klucz "_site") RAZ — więc
alarm z dwóch kamer to jeden alarm, a pozycja jest średnią ważoną.

Kamery są niezsynchronizowane (streamy z telefonów, skew ~0.5 s) — TTL
liczymy po czasie SERWERA, nie po timestampach z telefonu.
"""
from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from backend.zone_rules import point_in_polygon
from backend.zones_store import Zone
from config import CONFIG, WorldConfig


@dataclass
class WorldObservation:
    """One person seen by one camera, projected to the world plane."""
    camera_id: str
    x_m: float
    y_m: float
    confidence: float
    ts: float                 # server receive time
    box: tuple = ()           # image bbox (x1, y1, x2, y2) — for thumbnails


@dataclass
class FusedPerson:
    fused_id: str
    x_m: float
    y_m: float
    confidence: float         # max over sources
    cameras: list[str] = field(default_factory=list)


@dataclass
class WorldZoneBreachEvent:
    zone: Zone
    person: FusedPerson
    severity: str
    frame_timestamp: float
    confirmed: bool = False


class WorldState:
    """Latest per-camera observations + cross-camera fusion.

    update() podmienia obserwacje danej kamery (ostatnia klatka wygrywa),
    fuse() zrzuca przeterminowane kamery, klastruje greedy po odległości
    (max 1 obserwacja per kamera w klastrze) i utrzymuje stabilne fused_id
    przez nearest-match do poprzedniego wyniku fuzji.
    """

    def __init__(self, config: WorldConfig | None = None):
        self.cfg = config or CONFIG.world
        self._lock = threading.Lock()
        self._by_camera: dict[str, list[WorldObservation]] = {}
        self._camera_seen: dict[str, float] = {}
        self._prev_fused: list[FusedPerson] = []

    def update(self, camera_id: str, observations: list[WorldObservation],
               now: float) -> None:
        with self._lock:
            self._by_camera[camera_id] = list(observations)
            self._camera_seen[camera_id] = now

    def fuse(self, now: float) -> list[FusedPerson]:
        with self._lock:
            fresh: list[WorldObservation] = []
            for cam, seen in list(self._camera_seen.items()):
                if now - seen > self.cfg.obs_ttl_sec:
                    del self._camera_seen[cam]
                    self._by_camera.pop(cam, None)
                    continue
                fresh.extend(self._by_camera.get(cam, []))

            clusters = self._cluster(fresh)
            fused = [self._merge(c) for c in clusters]
            self._assign_ids(fused)
            self._prev_fused = fused
            return list(fused)

    def _cluster(self, obs: list[WorldObservation]) -> list[list[WorldObservation]]:
        """Greedy: highest-confidence seeds first; obserwacja dołącza do
        klastra gdy jest bliżej niż assoc_threshold_m od jego seeda i klaster
        nie ma jeszcze obserwacji z jej kamery."""
        threshold = self.cfg.assoc_threshold_m
        pending = sorted(obs, key=lambda o: -o.confidence)
        clusters: list[list[WorldObservation]] = []
        for o in pending:
            placed = False
            for cluster in clusters:
                seed = cluster[0]
                if any(c.camera_id == o.camera_id for c in cluster):
                    continue
                if _dist(seed.x_m, seed.y_m, o.x_m, o.y_m) <= threshold:
                    cluster.append(o)
                    placed = True
                    break
            if not placed:
                clusters.append([o])
        return clusters

    @staticmethod
    def _merge(cluster: list[WorldObservation]) -> FusedPerson:
        total_w = sum(o.confidence for o in cluster) or 1e-9
        x = sum(o.x_m * o.confidence for o in cluster) / total_w
        y = sum(o.y_m * o.confidence for o in cluster) / total_w
        return FusedPerson(
            fused_id="",  # assigned in _assign_ids
            x_m=x, y_m=y,
            confidence=max(o.confidence for o in cluster),
            cameras=sorted({o.camera_id for o in cluster}),
        )

    def _assign_ids(self, fused: list[FusedPerson]) -> None:
        """Nearest-match do poprzedniego ticku → stabilne ID (klucz dla
        filtra temporalnego i cooldownu). Każde poprzednie ID użyte raz."""
        available = list(self._prev_fused)
        for p in fused:
            best, best_d = None, self.cfg.id_match_threshold_m
            for prev in available:
                d = _dist(p.x_m, p.y_m, prev.x_m, prev.y_m)
                if d <= best_d:
                    best, best_d = prev, d
            if best is not None:
                p.fused_id = best.fused_id
                available.remove(best)
            else:
                p.fused_id = uuid.uuid4().hex[:8]


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class WorldZoneDetector:
    """Zone breach na sfuzowanych pozycjach — polygon i punkt w metrach."""

    def evaluate(
        self,
        fused_persons: list[FusedPerson],
        zones: list[Zone],
        frame_timestamp: float = 0.0,
    ) -> list[WorldZoneBreachEvent]:
        events: list[WorldZoneBreachEvent] = []
        for p in fused_persons:
            for z in zones:
                if not z.active or z.coordinate_space != "world":
                    continue
                if len(z.polygon) < 3:
                    continue
                if point_in_polygon(p.x_m, p.y_m, z.polygon):
                    events.append(WorldZoneBreachEvent(
                        zone=z,
                        person=p,
                        severity=z.severity,
                        frame_timestamp=frame_timestamp,
                    ))
        return events


class WorldZoneTemporalFilter:
    """Jak ZoneTemporalFilter, ale klucz = (zone_id, fused_id) — stabilne ID
    z fuzji zamiast komórki bboxa, więc dedupe między kamerami jest
    strukturalny."""

    def __init__(self, required: int = 3, cooldown_sec: float = 8.0):
        self.required = required
        self.cooldown_sec = cooldown_sec
        self._streak: dict[str, int] = defaultdict(int)
        self._last_alert_time: dict[str, float] = {}

    @staticmethod
    def _key(event: WorldZoneBreachEvent) -> str:
        return f"{event.zone.id}_{event.person.fused_id}"

    def update(self, events: list[WorldZoneBreachEvent],
               now: float) -> list[WorldZoneBreachEvent]:
        current_keys = set()
        confirmed: list[WorldZoneBreachEvent] = []
        for e in events:
            k = self._key(e)
            current_keys.add(k)
            self._streak[k] += 1
            if self._streak[k] >= self.required:
                last = self._last_alert_time.get(k, float("-inf"))
                if now - last >= self.cooldown_sec:
                    e.confirmed = True
                    confirmed.append(e)
                    self._last_alert_time[k] = now
        dead = [k for k in self._streak if k not in current_keys]
        for k in dead:
            del self._streak[k]
        return confirmed
