"""Endpointy mostka RTSP → detekcja + kalibracja ArUco.

Uruchamia/zatrzymuje live-detekcję dla kamery RTSP (kamera trafia do tego
samego pipeline'u co telefon: YOLO + ArUco + fuzja multi-camera). Osobne od
nagrywania — możesz nagrywać i puszczać live jednocześnie.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class DetectSelect(BaseModel):
    camera_id: str
    mode: str = "site"


def _bridge(request: Request):
    b = getattr(request.app.state, "detect_bridge", None)
    if b is None:
        raise HTTPException(503, "Mostek detekcji niedostępny.")
    return b


@router.get("/detect/status")
async def status(request: Request):
    return {"active": _bridge(request).status()}


@router.get("/detect/overlap")
async def overlap(request: Request):
    """Czy kamery patrzą na ten sam punkt z różnych stron (fuzja ArUco)."""
    ov = _bridge(request).overlap()
    return {"overlap": ov}


@router.post("/detect/start")
async def start(request: Request, payload: DetectSelect):
    try:
        return _bridge(request).start(payload.camera_id, payload.mode)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.post("/detect/stop")
async def stop(request: Request, payload: DetectSelect):
    try:
        return _bridge(request).stop(payload.camera_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
