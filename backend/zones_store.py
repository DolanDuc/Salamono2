import json
import os
import threading
import time
import uuid

from pydantic import BaseModel, Field


class Zone(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "Strefa"
    severity: str = "DANGER"                  # WARNING or DANGER
    polygon: list[list[float]] = []           # [[x, y], ...] normalized 0-1
    # Optional: if non-empty, this zone is *defined by ArUco markers*.
    # Polygon is recomputed in each frame from marker centres (ordered by
    # the list here, so [10, 20, 30, 40] gives TL → TR → BR → BL).
    # `polygon` field is used as fallback cache written to disk.
    marker_ids: list[int] = []
    # "image" — polygon normalized 0..1 in camera frame (default, legacy).
    # "world" — polygon in metres in the shared calibration plane
    # (multi-camera site zones, stored under camera key "_site").
    coordinate_space: str = "image"
    active: bool = True
    created_at: float = Field(default_factory=time.time)


class ZoneStore:
    """JSON-file persistence for zones grouped by camera_id."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._data: dict[str, list[Zone]] = self._load()

    def _load(self) -> dict[str, list[Zone]]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        out: dict[str, list[Zone]] = {}
        for cam_id, zone_list in raw.items():
            out[cam_id] = [Zone.model_validate(z) for z in zone_list]
        return out

    def _flush(self) -> None:
        serializable = {
            cam_id: [z.model_dump() for z in zones]
            for cam_id, zones in self._data.items()
        }
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def for_camera(self, camera_id: str) -> list[Zone]:
        with self._lock:
            return [z.model_copy() for z in self._data.get(camera_id, [])]

    def all_cameras(self) -> dict[str, list[Zone]]:
        with self._lock:
            return {
                cam_id: [z.model_copy() for z in zones]
                for cam_id, zones in self._data.items()
            }

    def replace(self, camera_id: str, zones: list[Zone]) -> list[Zone]:
        with self._lock:
            self._data[camera_id] = list(zones)
            self._flush()
            return [z.model_copy() for z in self._data[camera_id]]

    def clear(self, camera_id: str) -> None:
        with self._lock:
            self._data.pop(camera_id, None)
            self._flush()
