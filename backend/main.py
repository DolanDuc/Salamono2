import base64
import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.detector import Detector, PPE_CATEGORIES
from backend.frame_store import FrameStore
from backend.marker_detector import MarkerDetector
from backend.models import StatsOut
from backend.ppe_rules import PPEChecker
from backend.routes import alerts, calibration, debug, ingest, pair, ws, zones
from backend.world_state import (
    WorldState,
    WorldZoneDetector,
    WorldZoneTemporalFilter,
)
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore
from config import CONFIG

ALERTS_LOG_PATH = os.path.join(CONFIG.flagged_frames_dir, "..", "alerts.jsonl")
ZONES_PATH = os.path.join(CONFIG.flagged_frames_dir, "..", "zones.json")
CALIBRATION_PATH = os.path.join(CONFIG.flagged_frames_dir, "..", "calibration.json")

PANEL_PASSWORD = os.getenv("PANEL_PASSWORD", "")
DEMO_TOKEN = os.getenv("DEMO_TOKEN", "")
DEMO_COOKIE = "perimetr_demo"
# 12-hour cookie so a pitch can run without re-auth even after tab reloads.
DEMO_COOKIE_MAX_AGE = 12 * 60 * 60

PUBLIC_PATHS = {"/api/health"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()
    app.state.danger_detector = DangerDetector()
    app.state.temporal_filter = TemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.ppe_detector = None
    app.state.ppe_checker = None
    ppe_path = CONFIG.ppe.model_path
    if os.path.exists(ppe_path):
        try:
            app.state.ppe_detector = Detector(
                model_name=ppe_path,
                categories=PPE_CATEGORIES,
                confidence=CONFIG.ppe.confidence,
            )
            app.state.ppe_checker = PPEChecker()
            print(f"[startup] PPE checkpoint mode ready ({ppe_path})")
        except Exception as e:
            print(f"[startup] PPE model failed to load: {e}")
    else:
        print(f"[startup] PPE model not found at {ppe_path}; checkpoint mode disabled")
    app.state.ws_manager = ConnectionManager()
    app.state.frame_store = FrameStore()
    app.state.alert_store = AlertStore(ALERTS_LOG_PATH)
    app.state.zone_store = ZoneStore(ZONES_PATH)
    app.state.zone_detector = ZoneBreachDetector()
    app.state.zone_temporal_filter = ZoneTemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.marker_detector = MarkerDetector()
    app.state.calibration_store = CalibrationStore(CALIBRATION_PATH)
    # Multi-camera ground-plane fusion (see backend/world_state.py).
    app.state.world_state = WorldState()
    app.state.world_zone_detector = WorldZoneDetector()
    app.state.world_zone_filter = WorldZoneTemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    # In-memory cache of last-seen polygon per marker-defined zone.
    # Format: {(camera_id, zone_id): (polygon_normalized, last_seen_ts)}
    app.state.marker_zone_cache = {}
    # Debug: inject a synthetic person detection into the next N frames.
    # None when idle. See backend/routes/debug.py.
    app.state.debug_inject_person = None
    app.state.frame_counter = 0
    app.state.start_time = time.time()
    yield


app = FastAPI(title="Perimetr", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def frame_headers(request: Request, call_next):
    """Allow iframe embedding from any origin (Adam wkleja panel do pitch
    HTML). CSP frame-ancestors * jest nowoczesną wersją X-Frame-Options
    ALLOWALL i honorują ją Chrome / Firefox / Safari."""
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", "frame-ancestors *")
    return response


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not PANEL_PASSWORD or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    # 1) Demo token via query param — sets a cookie so subsequent asset
    #    requests (style.css, tokens/msbp.css, app.js…) pass through.
    if DEMO_TOKEN and request.query_params.get("demo") == DEMO_TOKEN:
        response = await call_next(request)
        response.set_cookie(
            key=DEMO_COOKIE,
            value=DEMO_TOKEN,
            max_age=DEMO_COOKIE_MAX_AGE,
            httponly=True,
            secure=True,
            samesite="none",     # required for cross-origin iframe embeds
        )
        return response

    # 2) Demo cookie set earlier in the same session — silent pass.
    if DEMO_TOKEN and request.cookies.get(DEMO_COOKIE) == DEMO_TOKEN:
        return await call_next(request)

    # 3) Classic HTTP basic auth.
    header = request.headers.get("Authorization", "")
    if header.startswith("Basic "):
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8", "ignore")
            _, _, pw = decoded.partition(":")
            if secrets.compare_digest(pw, PANEL_PASSWORD):
                return await call_next(request)
        except Exception:
            pass

    return Response(
        status_code=401,
        content="Unauthorized",
        headers={"WWW-Authenticate": 'Basic realm="Perimetr"'},
    )

app.include_router(ingest.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(zones.router, prefix="/api")
app.include_router(calibration.router, prefix="/api")
app.include_router(debug.router, prefix="/api")
app.include_router(pair.router, prefix="/api")
app.include_router(ws.router)


@app.get("/api/stats", response_model=StatsOut)
async def get_stats():
    elapsed = time.time() - app.state.start_time
    fps = app.state.frame_counter / elapsed if elapsed > 0 else 0
    return StatsOut(
        total_frames_processed=app.state.frame_counter,
        total_alerts=app.state.alert_store.count(),
        uptime_seconds=round(elapsed, 1),
        current_fps=round(fps, 2),
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/modes")
async def modes():
    return {
        "modes": ["site"] + (["checkpoint"] if app.state.ppe_detector else []),
        "checkpoint_available": app.state.ppe_detector is not None,
    }


os.makedirs(CONFIG.flagged_frames_dir, exist_ok=True)

app.mount("/static/flagged",
          StaticFiles(directory=CONFIG.flagged_frames_dir),
          name="flagged")

app.mount("/phone",
          StaticFiles(directory="phone", html=True),
          name="phone")

app.mount("/",
          StaticFiles(directory="frontend", html=True),
          name="frontend")
