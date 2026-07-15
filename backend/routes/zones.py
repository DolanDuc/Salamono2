from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.zones_store import Zone

router = APIRouter()


class ZoneIn(BaseModel):
    id: str | None = None
    name: str = "Strefa"
    severity: str = "DANGER"
    polygon: list[list[float]] = Field(default_factory=list)
    marker_ids: list[int] = Field(default_factory=list)
    coordinate_space: str = "image"
    active: bool = True


class ZonesPayload(BaseModel):
    zones: list[ZoneIn]


def _validate_polygon(poly: list[list[float]]) -> None:
    if len(poly) < 3:
        raise HTTPException(400, "polygon requires >= 3 vertices")
    for pt in poly:
        if len(pt) != 2:
            raise HTTPException(400, "polygon vertex must be [x, y]")
        x, y = pt
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise HTTPException(400, "polygon vertices must be normalized 0..1")


def _validate_world_polygon(poly: list[list[float]]) -> None:
    """World zones: vertices in metres in the shared calibration plane."""
    if len(poly) < 3:
        raise HTTPException(400, "polygon requires >= 3 vertices")
    for pt in poly:
        if len(pt) != 2:
            raise HTTPException(400, "polygon vertex must be [x, y]")
        x, y = pt
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise HTTPException(400, "polygon vertices must be numbers")
        if not (-50.0 <= x <= 50.0 and -50.0 <= y <= 50.0):
            raise HTTPException(
                400, "world polygon vertices must be within -50..50 metres")


def _validate_marker_ids(ids: list[int]) -> None:
    if len(ids) < 3:
        raise HTTPException(400, "marker_ids requires >= 3 IDs to form a polygon")
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "marker_ids must be unique")
    for mid in ids:
        if not (0 <= int(mid) < 1000):
            raise HTTPException(400, "marker_ids must be non-negative integers < 1000")


@router.get("/zones")
async def list_all_zones(request: Request):
    store = request.app.state.zone_store
    return {"cameras": {
        cam_id: [z.model_dump() for z in zones]
        for cam_id, zones in store.all_cameras().items()
    }}


@router.get("/zones/{camera_id}")
async def list_zones(request: Request, camera_id: str):
    store = request.app.state.zone_store
    return {"camera_id": camera_id,
            "zones": [z.model_dump() for z in store.for_camera(camera_id)]}


@router.put("/zones/{camera_id}")
async def set_zones(request: Request, camera_id: str, payload: ZonesPayload):
    store = request.app.state.zone_store
    valid_severities = {"WARNING", "DANGER"}
    zones: list[Zone] = []
    for z_in in payload.zones:
        if z_in.coordinate_space not in ("image", "world"):
            raise HTTPException(400, "coordinate_space must be 'image' or 'world'")
        if z_in.coordinate_space == "world":
            # World zones live in metres on the shared calibration plane —
            # marker-resolved polygons don't apply there.
            if z_in.marker_ids:
                raise HTTPException(
                    400, "marker_ids not supported for world zones")
            _validate_world_polygon(z_in.polygon)
        # Marker-defined zones don't need a polygon up front — backend
        # resolves it from live marker detections each frame.
        elif z_in.marker_ids:
            _validate_marker_ids(z_in.marker_ids)
        else:
            _validate_polygon(z_in.polygon)
        if z_in.severity not in valid_severities:
            raise HTTPException(400, f"severity must be one of {valid_severities}")
        z = Zone(
            name=z_in.name.strip() or "Strefa",
            severity=z_in.severity,
            polygon=z_in.polygon,
            marker_ids=list(z_in.marker_ids),
            coordinate_space=z_in.coordinate_space,
            active=z_in.active,
        )
        if z_in.id:
            z.id = z_in.id
        zones.append(z)
    saved = store.replace(camera_id, zones)
    return {"camera_id": camera_id,
            "zones": [z.model_dump() for z in saved]}


@router.delete("/zones/{camera_id}")
async def clear_zones(request: Request, camera_id: str):
    store = request.app.state.zone_store
    store.clear(camera_id)
    return {"camera_id": camera_id, "zones": []}
