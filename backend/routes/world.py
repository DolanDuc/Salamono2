"""World-state endpoint — fused multi-camera site view.

GET /api/world/state
    Snapshot for the top-down site map: fused person positions (metres),
    world-space zones ("_site"), and calibrated cameras. The live path is
    the /ws/live frames (world_persons / world_zones fields); this endpoint
    is the polling/debug fallback.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/world/state")
async def world_state(request: Request):
    fused = request.app.state.world_state.fuse(time.time())
    site_zones = [z for z in request.app.state.zone_store.for_camera("_site")
                  if z.coordinate_space == "world" and z.active]
    cals = request.app.state.calibration_store.all()
    return {
        "persons": [{
            "fused_id": p.fused_id,
            "x_m": round(p.x_m, 2),
            "y_m": round(p.y_m, 2),
            "confidence": round(p.confidence, 3),
            "cameras": list(p.cameras),
        } for p in fused],
        "zones": [{
            "id": z.id,
            "name": z.name,
            "severity": z.severity,
            "polygon": z.polygon,
        } for z in site_zones],
        "cameras": [{
            "camera_id": c.camera_id,
            "width_m": c.width_m,
            "height_m": c.height_m,
        } for c in cals],
    }
