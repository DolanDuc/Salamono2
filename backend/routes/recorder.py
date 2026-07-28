"""Endpointy nagrywania surowego strumienia RTSP.

Kamery konfiguruje się w `data/cameras.json` (poza repo — patrz
`cameras.example.json`). Nagrania lądują w `data/recordings/` z nazwą
zawierającą znacznik czasu, więc kolejne sesje się nie nadpisują.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class CameraSelect(BaseModel):
    camera_id: str


def _mgr(request: Request):
    mgr = getattr(request.app.state, "recorder", None)
    if mgr is None:
        raise HTTPException(503, "Recorder niedostępny (brak ffmpeg?).")
    return mgr


@router.get("/recorder/cameras")
async def list_cameras(request: Request):
    return {"cameras": [c.public() for c in _mgr(request).load_cameras()]}


@router.get("/recorder/status")
async def status(request: Request):
    return {"active": _mgr(request).status()}


@router.get("/recorder/recordings")
async def recordings(request: Request):
    return {"recordings": _mgr(request).recordings()}


@router.post("/recorder/test")
async def test_connection(request: Request, payload: CameraSelect):
    """Sprawdza czy kamera jest osiągalna po Ethernecie (pobiera 1 klatkę)."""
    try:
        return _mgr(request).probe(payload.camera_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/recorder/start")
async def start(request: Request, payload: CameraSelect):
    try:
        return _mgr(request).start(payload.camera_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.post("/recorder/stop")
async def stop(request: Request, payload: CameraSelect):
    try:
        return _mgr(request).stop(payload.camera_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
