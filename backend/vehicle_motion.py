"""Śledzenie ruchu pojazdów — bramkowanie alarmu „osoba przy pojeździe".

Zbliżenie się do **nieruchomego** pojazdu jest normalne (zaparkowana koparka).
Alarm ma sens tylko przy pojeździe, który się **porusza**. Ten moduł śledzi
pojazdy między klatkami (per kamera) i wykrywa ruch trzema sygnałami:

1. **Przesunięcie bboxa** — środek jedzie (pojazd się przemieszcza).
2. **Zmiana rozmiaru bboxa** — dojazd do/od kamery, obrót korpusu.
3. **Ruch wewnętrzny** — różnica pikseli w obrębie pojazdu między klatkami:
   łapie osprzęt (łyżka koparki) poruszający się przy nieruchomym korpusie.

Stan „niebezpieczny" utrzymuje się `hold_sec` sekund po ostatnim ruchu — pojazd,
który stanął na kilkanaście sekund, dalej jest traktowany jako niebezpieczny.

Uwaga: model YOLO (COCO) rozpoznaje generyczne pojazdy (car/bus/truck), nie
„koparkę". Wykrywanie ruchu jest jednak niezależne od klasy — działa dla
dowolnego pojazdu, więc łyżka koparki liczona przez ruch wewnętrzny działa bez
dedykowanego modelu. Osobny „rejestr pojazdów budowlanych" to przyszłe
rozszerzenie (własny wytrenowany model klas sprzętu).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.danger_rules import bbox_iou
from backend.detector import Detection


@dataclass
class _Track:
    box: tuple
    gray_roi: np.ndarray | None
    last_motion_ts: float
    last_seen_ts: float


class VehicleMotionTracker:
    """Per-kamera śledzenie pojazdów + ocena, które są ruchome/niebezpieczne."""

    _ROI = (32, 32)

    def __init__(self, cfg=None):
        if cfg is None:
            from config import CONFIG
            cfg = CONFIG.vehicle_motion
        self.cfg = cfg
        self._tracks: dict[str, list[_Track]] = {}

    def _roi_gray(self, frame: np.ndarray, box: tuple) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        try:
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return cv2.resize(g, self._ROI)
        except cv2.error:
            return None

    def _is_moving(self, track: _Track, new_box: tuple,
                   new_roi: np.ndarray | None) -> bool:
        ox = (track.box[0] + track.box[2]) / 2.0
        oy = (track.box[1] + track.box[3]) / 2.0
        nx = (new_box[0] + new_box[2]) / 2.0
        ny = (new_box[1] + new_box[3]) / 2.0
        diag = (((new_box[2] - new_box[0]) ** 2
                 + (new_box[3] - new_box[1]) ** 2) ** 0.5) or 1.0
        if (((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5) / diag >= self.cfg.displacement_frac:
            return True

        ow, oh = track.box[2] - track.box[0], track.box[3] - track.box[1]
        nw, nh = new_box[2] - new_box[0], new_box[3] - new_box[1]
        if ow > 0 and oh > 0:
            if (abs(nw - ow) / ow >= self.cfg.size_change_frac
                    or abs(nh - oh) / oh >= self.cfg.size_change_frac):
                return True

        if track.gray_roi is not None and new_roi is not None:
            diff = cv2.absdiff(track.gray_roi, new_roi)
            changed = float((diff > self.cfg.pixel_diff_thresh).mean())
            if changed >= self.cfg.internal_motion_frac:
                return True
        return False

    def update(self, camera_id: str, vehicles: list[Detection],
               frame: np.ndarray, now: float) -> set[tuple]:
        """Zwraca zbiór boxów pojazdów aktualnie 'niebezpiecznych' (ruch < hold_sec temu)."""
        tracks = self._tracks.setdefault(camera_id, [])
        used: set[int] = set()
        dangerous: set[tuple] = set()

        for v in vehicles:
            roi = self._roi_gray(frame, v.box)
            best_idx, best_iou = -1, 0.0
            for i, t in enumerate(tracks):
                if i in used:
                    continue
                iou = bbox_iou(v.box, t.box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_idx >= 0 and best_iou >= self.cfg.match_iou:
                t = tracks[best_idx]
                used.add(best_idx)
                if self._is_moving(t, v.box, roi):
                    t.last_motion_ts = now
                t.box = tuple(v.box)
                if roi is not None:
                    t.gray_roi = roi
                t.last_seen_ts = now
            else:
                # Nowy pojazd — nie wiadomo, czy jedzie. Domyślnie NIEruchomy
                # (bez alarmu), dopóki nie zaobserwujemy ruchu. To realizuje
                # regułę: zbliżenie do stojącego pojazdu jest OK.
                t = _Track(box=tuple(v.box), gray_roi=roi,
                           last_motion_ts=float("-inf"), last_seen_ts=now)
                tracks.append(t)

            if now - t.last_motion_ts <= self.cfg.hold_sec:
                dangerous.add(tuple(v.box))

        keep_for = max(self.cfg.hold_sec, 5.0)
        self._tracks[camera_id] = [t for t in tracks
                                   if now - t.last_seen_ts <= keep_for]
        return dangerous
