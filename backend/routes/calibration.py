"""Calibration endpoints.

POST /api/calibration/{camera_id}
    Body (multipart):
        image        : JPEG/PNG frame containing all 4 markers
        marker_ids   : "id_tl,id_tr,id_br,id_bl" (comma-separated int)
        width_m      : real width of reference area
        height_m     : real height of reference area

GET /api/calibration/{camera_id}
DELETE /api/calibration/{camera_id}
"""
from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.calibration import calibrate_from_markers
from backend.models import CalibrationOut

router = APIRouter()


def _to_out(cal) -> CalibrationOut:
    return CalibrationOut(
        camera_id=cal.camera_id,
        marker_ids=cal.marker_ids,
        width_m=cal.width_m,
        height_m=cal.height_m,
        created_at=cal.created_at,
    )


@router.post("/calibration/{camera_id}", response_model=CalibrationOut)
async def create_calibration(
    request: Request,
    camera_id: str,
    image: UploadFile = File(...),
    marker_ids: str = Form(...),
    width_m: float = Form(...),
    height_m: float = Form(...),
):
    try:
        parsed_ids = [int(x.strip()) for x in marker_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "marker_ids must be a comma-separated list of ints")
    if len(parsed_ids) != 4:
        raise HTTPException(400, "Need exactly 4 marker IDs (TL, TR, BR, BL)")

    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    detector = request.app.state.marker_detector
    detections = detector.detect(frame)
    try:
        cal = calibrate_from_markers(
            camera_id=camera_id,
            detections=detections,
            marker_ids=parsed_ids,
            width_m=width_m,
            height_m=height_m,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    request.app.state.calibration_store.set(cal)
    return _to_out(cal)


@router.get("/calibration")
async def list_calibrations(request: Request):
    """All calibrated cameras — the site map view uses this to label
    cameras and pick the reference-rectangle scale."""
    cals = request.app.state.calibration_store.all()
    return {"calibrations": [_to_out(c).model_dump() for c in cals]}


@router.get("/calibration/{camera_id}", response_model=CalibrationOut)
async def get_calibration(request: Request, camera_id: str):
    cal = request.app.state.calibration_store.get(camera_id)
    if cal is None:
        raise HTTPException(404, "No calibration for this camera")
    return _to_out(cal)


@router.delete("/calibration/{camera_id}")
async def delete_calibration(request: Request, camera_id: str):
    removed = request.app.state.calibration_store.clear(camera_id)
    if not removed:
        raise HTTPException(404, "No calibration for this camera")
    return {"status": "deleted", "camera_id": camera_id}
