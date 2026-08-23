import asyncio
import base64
from concurrent.futures import Future
import logging
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.calibration import Calibration
from backend.camera_registry import CameraRegistry
from backend.danger_rules import DangerEvent, DynamicSafetyZone
from backend.detector import Detection, Detector
from backend.frame_processor import FrameJob
from backend.marker_detector import MarkerDetection
from backend.performance_profiler import FramePerformanceTrace
from backend.preview_encoder import EncodedPreview
from backend.posture_detector import (
    POSE_LANDMARK_COUNT,
    POSTURE_CONNECTIONS,
    POSTURE_DRAW_MIN_VISIBILITY,
    PostureAssessment,
    transform_cached_landmarks,
)
from backend.models import (
    ActiveZoneOut,
    AlarmRecord,
    AlertOut,
    AlertSeverity,
    DetectionOut,
    DynamicSafetyZoneOut,
    FrameAcceptedOut,
    FrameResultOut,
    MarkerDetectionOut,
    PersonDistanceOut,
    PPECheckOut,
    PostureAssessmentOut,
    WorkerIdentificationOut,
    UnidentifiedWorkerOut,
    ZoneBreachOut,
)
from backend.ppe_rules import PPEEvent
from backend.worker_identification import WorkerIdentity, UnidentifiedWorkerEvent
from backend.worker_store import WorkerRecord
from backend.zone_rules import ZoneBreachEvent, signed_distance_to_polygon
from config import CONFIG

SITE_RULE_DESCRIPTIONS = {
    "person_vehicle_overlap": "Osoba w obrysie pojazdu lub maszyny",
    "person_vehicle_danger_zone": "Osoba w krytycznej strefie maszyny",
    "person_near_vehicle": "Osoba w ostrzegawczej strefie maszyny",
}

MAX_FRAME_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_DECODED_FRAME_BYTES = 12 * 1024 * 1024
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FramePayload:
    """Immutable metadata retained beside a detached background frame."""

    mode: str
    received_at: float
    preview_jpeg_bytes: bytes
    preview_jpeg_b64: str
    preview_jpeg_future: Future[EncodedPreview] | None = None
    performance_trace: FramePerformanceTrace | None = None
    render_annotated: bool = False


def _profiled_detection(
    detector,
    frame: np.ndarray,
    trace: FramePerformanceTrace | None,
    *,
    img_size: int | None = None,
) -> list[Detection]:
    profiled = getattr(detector, "detect_profiled", None)
    if not isinstance(detector, Detector) or not callable(profiled):
        started = time.perf_counter()
        detections = detector.detect(frame)
        if trace is not None:
            trace.set(
                "yolo_inference_ms",
                (time.perf_counter() - started) * 1000.0,
            )
        return detections

    detections, timing = profiled(frame, img_size=img_size)
    if trace is not None:
        trace.set("yolo_preprocess_ms", timing.preprocess_ms)
        trace.set("yolo_inference_ms", timing.inference_ms)
        trace.set("yolo_postprocess_ms", timing.postprocess_ms)
    return detections


def _identity_for_person(
    person: Detection | DetectionOut,
    identities: list[WorkerIdentity],
) -> WorkerIdentity | None:
    track_id = getattr(person, "track_id", None)
    if track_id is not None:
        for identity in identities:
            if getattr(identity, "track_id", None) == track_id:
                return identity
    box = tuple(person.box)
    for identity in identities:
        if tuple(identity.person.box) == box:
            return identity
    return None


def _worker_id_for_person(
    person: Detection | DetectionOut,
    identities: list[WorkerIdentity],
) -> str | None:
    identity = _identity_for_person(person, identities)
    return identity.worker_id if identity is not None and identity.alert_eligible else None


def _identify_workers(
    request: Request,
    camera_id: str,
    frame: np.ndarray,
    persons: list[Detection],
    timestamp: float,
    performance_trace: FramePerformanceTrace | None = None,
    *,
    wait_for_current_frame: bool = False,
) -> tuple[list[Detection], list[WorkerIdentity], list[Detection]]:
    """Schedule asynchronous ID work and return current cached identities.

    The third item contains only tracks for which at least one background scan
    has completed.  It prevents the unidentified-worker policy from firing
    while a newly observed person is still waiting for its first ArUco pass.
    Embedded/test applications without the background worker retain the
    previous synchronous identifier contract.
    """
    worker = getattr(request.app.state, "worker_id_worker", None)
    identifier = getattr(request.app.state, "worker_identifier", None)
    if worker is not None:
        try:
            submit_started = time.perf_counter()
            if wait_for_current_frame:
                submit = getattr(worker, "submit_for_export", worker.submit_latest)
            else:
                submit = getattr(
                    worker,
                    "submit_borrowed_latest",
                    worker.submit_latest,
                )
            tracked_persons = submit(
                camera_id,
                frame,
                persons,
                timestamp,
            )
            if wait_for_current_frame:
                wait_for_result = getattr(worker, "wait_for_result", None)
                if callable(wait_for_result):
                    wait_for_result(
                        camera_id,
                        after_timestamp=float(np.nextafter(timestamp, -np.inf)),
                        timeout=CONFIG.demo_video.control_timeout_seconds,
                    )
            if performance_trace is not None:
                performance_trace.set(
                    "worker_id_submit_ms",
                    (time.perf_counter() - submit_started) * 1000.0,
                )
            identities = worker.current_identities(
                camera_id,
                tracked_persons,
                timestamp,
            )
            scanned = set(worker.scanned_track_ids(camera_id))
            monitor_persons = [
                person for person in tracked_persons
                if person.track_id is not None and person.track_id in scanned
            ]
            return tracked_persons, identities, monitor_persons
        except Exception:
            # A long-lived worker failure must not turn frame ingestion into a
            # synchronous ArUco fallback or stop the video pipeline.
            logger.exception(
                "Worker ID scheduling failed for camera %s",
                camera_id,
            )
            return persons, [], []

    if identifier is not None and identifier.available:
        try:
            submit_started = time.perf_counter()
            identities = identifier.process(
                camera_id,
                frame,
                persons,
                timestamp,
            )
            if performance_trace is not None:
                performance_trace.set(
                    "worker_id_submit_ms",
                    (time.perf_counter() - submit_started) * 1000.0,
                )
        except Exception:
            logger.exception(
                "Synchronous Worker ID fallback failed for camera %s",
                camera_id,
            )
            identities = []
        return persons, identities, persons
    return persons, [], []


def _worker_identification_available(request: Request) -> bool:
    identifier = getattr(request.app.state, "worker_identifier", None)
    if identifier is None or not identifier.available:
        return False
    worker = getattr(request.app.state, "worker_id_worker", None)
    if worker is None:
        return True
    try:
        return bool(getattr(worker.stats(), "thread_alive", True))
    except Exception:
        return False


def _worker_profiles(
    profile_resolver,
    identities: list[WorkerIdentity],
) -> dict[str, WorkerRecord]:
    if profile_resolver is None or not identities:
        return {}
    return profile_resolver.get_many(
        identity.worker_id for identity in identities
    )


def _attach_worker_profile(
    record: AlarmRecord,
    profile: WorkerRecord | None,
) -> AlarmRecord:
    """Snapshot profile fields so historical alerts remain auditable.

    ``worker_id`` stays the canonical filter key.  Names are deliberately
    copied only when the scanned identifier is registered; unknown QR tags
    keep working exactly as before.
    """
    if profile is not None:
        record.details.update({
            "worker_first_name": profile.first_name,
            "worker_last_name": profile.last_name,
            "worker_full_name": profile.full_name,
            "worker_position": profile.position,
            "worker_department": profile.department,
        })
    return record


def _attach_identity_snapshot(
    record: AlarmRecord,
    identity: WorkerIdentity | None,
    profile: WorkerRecord | None,
) -> AlarmRecord:
    """Freeze alert-safe identity metadata at event creation time."""
    if identity is None or not identity.alert_eligible or profile is None:
        record.details.pop("worker_id", None)
        record.details.update({
            "worker": None,
            "identity_status": "unidentified",
            "identity_snapshot_at": record.timestamp,
        })
        if identity is not None:
            record.details.update({
                "track_id": identity.track_id,
                "marker_id": identity.marker_id,
                "marker_age_sec": identity.marker_age_sec,
                "identity_confidence": identity.confidence,
            })
        return record
    record.details.update({
        "worker_id": identity.worker_id,
        "marker_id": identity.marker_id,
        "identity_status": identity.identity_status,
        "last_marker_seen_at": record.timestamp - identity.marker_age_sec,
        "marker_age_sec": identity.marker_age_sec,
        "identity_confidence": identity.confidence,
        "identity_snapshot_at": record.timestamp,
        "worker": {
            "worker_id": profile.worker_id,
            "first_name": profile.first_name,
            "last_name": profile.last_name,
        },
    })
    return _attach_worker_profile(record, profile)


def _site_record(
    alert: AlertOut,
    camera_id: str,
    worker_id: str | None = None,
    clip_url: str | None = None,
) -> AlarmRecord:
    details = {
        "distance_px": alert.distance_px,
        "distance_m": alert.distance_m,
        "calibrated": alert.calibrated,
        "overlap_iou": alert.overlap_iou,
        "person_confidence": alert.person.confidence,
        "hazard_confidence": alert.hazard.confidence,
        "person_box": alert.person.box,
        "hazard_box": alert.hazard.box,
        "hazard_class": alert.hazard.class_name,
    }
    if worker_id:
        details["worker_id"] = worker_id
    return AlarmRecord(
        id=alert.id,
        timestamp=alert.timestamp,
        mode="site",
        kind="site_hazard",
        severity=alert.severity,
        rule_name=alert.rule_name,
        description=SITE_RULE_DESCRIPTIONS.get(alert.rule_name, alert.rule_name),
        camera_id=camera_id,
        thumbnail_url=alert.frame_thumbnail_url,
        clip_url=clip_url,
        details=details,
    )


def _zone_record(
    breach: ZoneBreachOut,
    camera_id: str,
    worker_id: str | None = None,
    clip_url: str | None = None,
) -> AlarmRecord:
    details = {
        "zone_id": breach.zone_id,
        "zone_name": breach.zone_name,
        "inside": breach.inside,
        "distance_px": breach.distance_px,
        "distance_m": breach.distance_m,
        "calibrated": breach.calibrated,
        "person_confidence": breach.person.confidence,
        "person_box": breach.person.box,
    }
    if worker_id:
        details["worker_id"] = worker_id
    return AlarmRecord(
        id=breach.id,
        timestamp=breach.timestamp,
        mode="site",
        kind="zone_breach" if breach.rule_name == "zone_breach" else "zone_approach",
        severity=breach.severity,
        rule_name=breach.rule_name,
        description=(
            f"Wejscie w strefe: {breach.zone_name}"
            if breach.rule_name == "zone_breach"
            else f"Zblizanie do strefy: {breach.zone_name}"
        ),
        camera_id=camera_id,
        thumbnail_url=breach.frame_thumbnail_url,
        clip_url=clip_url,
        details=details,
    )


def _ppe_record(
    check: PPECheckOut,
    camera_id: str,
    worker_id: str | None = None,
    clip_url: str | None = None,
) -> AlarmRecord:
    missing_pretty = {"hardhat": "kask", "vest": "kamizelka"}
    missing_labels = [missing_pretty.get(m, m) for m in check.missing]
    desc = "Brak PPE: " + " + ".join(missing_labels) if missing_labels else "PPE OK"
    details = {
        "missing": check.missing,
        "has_hardhat": check.has_hardhat,
        "has_vest": check.has_vest,
        "person_confidence": check.person.confidence,
        "person_box": check.person.box,
    }
    if worker_id:
        details["worker_id"] = worker_id
    return AlarmRecord(
        id=check.id,
        timestamp=check.timestamp,
        mode="checkpoint",
        kind="ppe_missing",
        severity=check.severity,
        rule_name="missing_" + "_".join(check.missing) if check.missing else "ppe_ok",
        description=desc,
        camera_id=camera_id,
        thumbnail_url=check.frame_thumbnail_url,
        clip_url=clip_url,
        details=details,
    )


POSTURE_SIGNAL_DESCRIPTIONS = {
    "repeated_body_sway": "powtarzalne kołysanie tułowia",
    "unstable_trajectory": "niestabilny tor ruchu",
    "irregular_step_pattern": "nieregularny wzorzec kroku",
    "upper_body_instability": "niestabilność górnej części ciała",
    "sudden_balance_loss": "nagła utrata równowagi",
    "possible_fall": "możliwy upadek",
    "hand_to_mouth_pattern": "powtarzalny gest ręka–usta",
    "ml_fall_down": "TCN+GRU: upadek",
    "ml_lying_down": "TCN+GRU: pozycja leżąca",
    "fall_suspected": "podejrzenie upadku",
    "fall_detected": "potwierdzony upadek",
    "person_on_ground": "osoba na ziemi",
    "unstable_movement": "utrzymujący się niestabilny ruch",
    "smoking_detected": "WYKRYTO PALENIE",
}


POSTURE_COORDINATION_SIGNALS = frozenset({
    "repeated_body_sway",
    "unstable_trajectory",
    "irregular_step_pattern",
    "upper_body_instability",
    "sudden_balance_loss",
})


def posture_signal_category(signals: list[str] | set[str]) -> str:
    """Classify posture signals once, for both alerts and the frame overlay.

    Returns one of ``fall``, ``lying``, ``unstable``, ``smoking`` or
    ``coordination``. Keeping this in one place stops the highlight drawn on
    the video from drifting away from the alert written to the history.
    """
    signals = set(signals)
    if signals & {"fall_detected", "fall_suspected", "possible_fall", "ml_fall_down"}:
        return "fall"
    if signals & {"person_on_ground", "ml_lying_down"}:
        return "lying"
    if "unstable_movement" in signals:
        return "unstable"
    if "smoking_detected" in signals or (
        "hand_to_mouth_pattern" in signals
        and not POSTURE_COORDINATION_SIGNALS & signals
    ):
        return "smoking"
    return "coordination"


def _posture_record(
    assessment: PostureAssessmentOut,
    camera_id: str,
    worker_id: str | None = None,
    clip_url: str | None = None,
) -> AlarmRecord:
    signal_labels = [
        POSTURE_SIGNAL_DESCRIPTIONS.get(signal, signal)
        for signal in assessment.signals
    ]
    category = posture_signal_category(assessment.signals)
    is_fall = category == "fall"
    is_lying = category == "lying"
    is_unstable = category == "unstable"
    is_smoking = category == "smoking"
    if is_fall:
        kind = "fall_detected"
        rule_name = (
            "fall_detected"
            if "fall_detected" in assessment.signals
            else "fall_suspected"
            if "fall_suspected" in assessment.signals
            else "ml_fall_down"
            if "ml_fall_down" in assessment.signals
            else "possible_fall"
        )
        desc = "Wykryto sekwencję upadku pracownika"
    elif is_lying:
        kind = "person_on_ground"
        rule_name = "person_on_ground"
        desc = "Wykryto utrzymującą się pozycję na ziemi — wymagana weryfikacja"
    elif is_unstable:
        kind = "posture_anomaly"
        rule_name = "unstable_movement"
        desc = "Wykryto utrzymujący się niestabilny wzorzec ruchu"
    elif is_smoking:
        kind = "smoking_gesture"
        rule_name = "smoking_detected"
        desc = "WYKRYTO PALENIE"
    else:
        kind = "posture_anomaly"
        rule_name = "coordination_anomaly"
        desc = "Nietypowy wzorzec koordynacji ruchowej"
        if signal_labels:
            desc += ": " + ", ".join(signal_labels[:2])
    details = {
        "track_id": assessment.track_id,
        "risk_score": assessment.risk_score,
        "status": assessment.status,
        "signals": assessment.signals,
        "metrics": assessment.metrics,
        "pose_confidence": assessment.pose_confidence,
        "history_seconds": assessment.history_seconds,
        "behavior_label": assessment.behavior_label,
        "behavior_confidence": assessment.behavior_confidence,
        "behavior_probabilities": assessment.behavior_probabilities,
        "behavior_valid_ratio": assessment.behavior_valid_ratio,
        "behavior_window_seconds": assessment.behavior_window_seconds,
        "behavior_inference_ms": assessment.behavior_inference_ms,
        "safety_label": assessment.safety_label,
        "safety_confidence": assessment.safety_confidence,
        "safety_probabilities": assessment.safety_probabilities,
        "torso_quality": assessment.torso_quality,
        "upper_body_quality": assessment.upper_body_quality,
        "lower_body_quality": assessment.lower_body_quality,
        "visible_ratio": assessment.visible_ratio,
        "learned_event_type": assessment.learned_event_type,
        "learned_event_reason": assessment.learned_event_reason,
        "person_confidence": assessment.person.confidence,
        "person_box": assessment.person.box,
        "interpretation": "requires_human_verification",
    }
    if worker_id:
        details["worker_id"] = worker_id
    return AlarmRecord(
        id=assessment.id,
        timestamp=assessment.timestamp,
        mode="site",
        kind=kind,
        severity=assessment.severity,
        rule_name=rule_name,
        description=desc,
        camera_id=camera_id,
        thumbnail_url=assessment.frame_thumbnail_url,
        clip_url=clip_url,
        details=details,
    )


def _unidentified_record(
    event_out: UnidentifiedWorkerOut,
    camera_id: str,
    mode: str,
    clip_url: str | None = None,
) -> AlarmRecord:
    return AlarmRecord(
        id=event_out.id,
        timestamp=event_out.timestamp,
        mode=mode,
        kind="unidentified_worker",
        severity=event_out.severity,
        rule_name="worker_id_missing",
        description="Osoba bez rozpoznanego identyfikatora QR",
        camera_id=camera_id,
        thumbnail_url=event_out.frame_thumbnail_url,
        clip_url=clip_url,
        details={
            "person_box": event_out.person.box,
            "person_confidence": event_out.person.confidence,
            "interpretation": "visible_or_cached_worker_tag_not_found",
        },
    )


router = APIRouter()

COLOR_PERSON = (0, 255, 0)
COLOR_VEHICLE = (255, 136, 0)
COLOR_DANGER = (0, 0, 255)
COLOR_WARNING = (0, 165, 255)
COLOR_HARDHAT = (255, 255, 0)
COLOR_VEST = (0, 255, 255)
COLOR_OK = (0, 200, 0)
COLOR_ZONE_WARN = (0, 200, 255)
COLOR_ZONE_DANGER = (60, 60, 255)
COLOR_MARKER = (255, 0, 255)
COLOR_MARKER_ACTIVE = (0, 255, 255)
COLOR_POSTURE = (255, 200, 0)
COLOR_POSTURE_OBSERVE = (255, 255, 0)


def _live_preview_jpeg_bytes(
    raw: bytes,
    frame: np.ndarray,
) -> bytes:
    """Return JPEG bytes for the unannotated live background.

    Phone/browser capture already uploads JPEG, so retaining those bytes
    avoids a second lossy compression pass.  Other OpenCV-readable formats
    are normalized because the response contract and panel data URL both
    explicitly declare ``image/jpeg``.
    """
    if raw.startswith(b"\xff\xd8"):
        jpeg = raw
    else:
        encoded, jpeg_buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if not encoded:
            return b""
        jpeg = jpeg_buffer.tobytes()
    return jpeg


def _live_preview_jpeg(
    raw: bytes,
    frame: np.ndarray,
) -> tuple[bytes, str]:
    jpeg = _live_preview_jpeg_bytes(raw, frame)
    return jpeg, base64.b64encode(jpeg).decode() if jpeg else ""


def _live_preview_jpeg_b64(raw: bytes, frame: np.ndarray) -> str:
    """Backward-compatible helper used by focused route tests."""

    return _live_preview_jpeg(raw, frame)[1]


# How long a marker-defined zone remembers its last polygon after all
# corner markers stop being visible. Keeps the zone alive across brief
# occlusions instead of flickering off for a frame or two.
MARKER_ZONE_CACHE_TTL = 2.0


def _resolve_marker_zones(zones, markers, camera_id: str,
                          cache: dict, now: float,
                          frame_w: int, frame_h: int):
    """Return zones with their polygon field filled in from marker positions.

    - Regular (polygon) zones pass through unchanged.
    - Marker-defined zones: if all corner markers visible → recompute polygon
      from their centres and refresh cache. If some missing → fall back to
      last-seen polygon if fresh (< MARKER_ZONE_CACHE_TTL), otherwise drop
      the zone from this frame.
    """
    by_id = {m.marker_id: m for m in markers}
    resolved = []
    for z in zones:
        if not z.marker_ids:
            resolved.append(z)
            continue
        centres = []
        all_visible = True
        for mid in z.marker_ids:
            m = by_id.get(int(mid))
            if m is None:
                all_visible = False
                break
            cx, cy = m.center
            centres.append([cx / frame_w, cy / frame_h])
        cache_key = (camera_id, z.id)
        if all_visible:
            cache[cache_key] = (centres, now)
            polygon = centres
        else:
            cached = cache.get(cache_key)
            if cached and (now - cached[1]) < MARKER_ZONE_CACHE_TTL:
                polygon = cached[0]
            else:
                continue
        z_copy = z.model_copy()
        z_copy.polygon = polygon
        resolved.append(z_copy)
    return resolved


def _marker_to_out(m: MarkerDetection) -> MarkerDetectionOut:
    return MarkerDetectionOut(
        marker_id=m.marker_id,
        corners=[[float(x), float(y)] for x, y in m.corners],
        center=[float(m.center[0]), float(m.center[1])],
    )


def _scheduled_markers(
    request: Request,
    frame: np.ndarray,
    camera_id: str,
    zones,
    performance_trace: FramePerformanceTrace | None = None,
) -> list[MarkerDetection]:
    """Run ArUco only when the current camera can use its result.

    Tests and small embedding applications which construct ``app.state``
    manually remain compatible: without a scheduler we retain the historical
    direct-detector behaviour.
    """
    scheduler = getattr(request.app.state, "marker_scheduler", None)
    started = time.perf_counter()
    if scheduler is None:
        markers = request.app.state.marker_detector.detect(frame)
        if performance_trace is not None:
            performance_trace.set(
                "marker_detection_ms",
                (time.perf_counter() - started) * 1000.0,
            )
        return markers
    marker_zones_active = any(
        zone.active and len(zone.marker_ids) >= 3
        for zone in zones
    )
    def record_timing(elapsed_ms: float) -> None:
        if performance_trace is not None:
            performance_trace.set("marker_detection_ms", elapsed_ms)

    markers = scheduler.process(
        camera_id,
        frame,
        marker_zones_active=marker_zones_active,
        timing_callback=record_timing,
    )
    return markers


def _worker_to_out(
    identity: WorkerIdentity,
    profile: WorkerRecord | None = None,
) -> WorkerIdentificationOut:
    return WorkerIdentificationOut(
        worker_id=identity.worker_id,
        source=identity.source,
        track_id=getattr(identity, "track_id", None),
        confidence=getattr(identity, "confidence", None),
        person_box=list(identity.person.box),
        tag_polygon=[[round(x, 1), round(y, 1)] for x, y in identity.tag_polygon],
        cached=identity.cached,
        identity_status=identity.identity_status,
        identity_confidence=identity.confidence,
        marker_age_sec=identity.marker_age_sec,
        marker_id=identity.marker_id,
        alert_eligible=identity.alert_eligible,
        registered=profile is not None,
        first_name=profile.first_name if profile is not None else None,
        last_name=profile.last_name if profile is not None else None,
        full_name=profile.full_name if profile is not None else None,
        position=profile.position if profile is not None else None,
        department=profile.department if profile is not None else None,
    )


def _unidentified_to_out(
    event: UnidentifiedWorkerEvent,
    event_id: str,
    thumb_url: str | None = None,
) -> UnidentifiedWorkerOut:
    return UnidentifiedWorkerOut(
        id=event_id,
        severity=AlertSeverity(event.severity),
        person=_det_to_out(event.person),
        timestamp=event.frame_timestamp,
        frame_thumbnail_url=thumb_url,
    )


def _dynamic_zone_to_out(
    zone: DynamicSafetyZone,
    frame_w: int,
    frame_h: int,
) -> DynamicSafetyZoneOut:
    polygon = [
        [
            round(float(x) / max(frame_w, 1), 6),
            round(float(y) / max(frame_h, 1), 6),
        ]
        for x, y in zone.polygon_px
    ]
    return DynamicSafetyZoneOut(
        zone_id=zone.zone_id,
        severity=AlertSeverity(zone.severity),
        hazard=_det_to_out(zone.hazard),
        polygon=polygon,
        threshold_m=(round(zone.threshold_m, 2) if zone.threshold_m is not None else None),
        threshold_px=(round(zone.threshold_px, 1) if zone.threshold_px is not None else None),
        calibrated=zone.calibrated,
    )


def _person_distances(
    persons: list[Detection],
    zones: list,
    frame_w: int,
    frame_h: int,
    calibration: Calibration | None,
) -> list[PersonDistanceOut]:
    """Return the nearest configured-zone boundary for every person."""
    active_zones = [z for z in zones if z.active and len(z.polygon) >= 3]
    if not active_zones:
        return []
    out: list[PersonDistanceOut] = []
    for p in persons:
        x1, y1, x2, y2 = p.box
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)
        candidates = []
        for zone in active_zones:
            polygon_px = [
                (float(x) * frame_w, float(y) * frame_h)
                for x, y in zone.polygon
            ]
            distance_px = signed_distance_to_polygon((foot_x, foot_y), polygon_px)
            distance_m = None
            if calibration is not None:
                foot_m = calibration.project(
                    foot_x, foot_y, frame_w, frame_h,
                )
                polygon_m = [
                    calibration.project(x, y, frame_w, frame_h)
                    for x, y in polygon_px
                ]
                distance_m = signed_distance_to_polygon(foot_m, polygon_m)
            ranking = abs(distance_m) if distance_m is not None else abs(distance_px)
            candidates.append((ranking, zone, distance_px, distance_m))
        if not candidates:
            continue
        _ranking, zone, distance_px, distance_m = min(candidates, key=lambda item: item[0])
        signed = distance_m if distance_m is not None else distance_px
        out.append(PersonDistanceOut(
            person_box=[int(x1), int(y1), int(x2), int(y2)],
            zone_id=zone.id,
            zone_name=zone.name,
            distance_px=round(distance_px, 1),
            distance_m=(round(distance_m, 2) if distance_m is not None else None),
            inside=signed <= 0.0,
            calibrated=distance_m is not None,
        ))
    return out


def _annotate_markers(frame: np.ndarray,
                      markers: list[MarkerDetection],
                      calibration_ids: set[int]) -> np.ndarray:
    if not markers:
        return frame
    out = frame
    for m in markers:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in m.corners],
                       dtype=np.int32)
        color = COLOR_MARKER_ACTIVE if m.marker_id in calibration_ids else COLOR_MARKER
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2,
                      lineType=cv2.LINE_AA)
        cx, cy = m.center
        cv2.putText(out, f"ID {m.marker_id}",
                    (int(cx) - 20, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out


def _annotate_person_distances(frame: np.ndarray,
                               distances: list[PersonDistanceOut]) -> np.ndarray:
    if not distances:
        return frame
    out = frame
    for d in distances:
        x1, y1, x2, y2 = d.person_box
        if d.distance_m is not None:
            amount = abs(d.distance_m)
            unit = "m"
        else:
            amount = abs(d.distance_px or 0.0)
            unit = "px"
        if d.inside:
            label = f"{d.zone_name}: WEWNATRZ ({amount:.1f} {unit})"
            color = COLOR_DANGER
        else:
            label = f"{d.zone_name}: {amount:.1f} {unit}"
            color = COLOR_OK if amount > 1.5 and unit == "m" else COLOR_WARNING
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # Draw slightly below the person bbox top so it doesn't collide with
        # the class label rendered above the bbox by _annotate_site.
        text_x = x1
        text_y = min(y2 + th + 8, frame.shape[0] - 4)
        cv2.rectangle(out, (text_x, text_y - th - 4),
                      (text_x + tw + 6, text_y + 2), color, -1)
        cv2.putText(out, label, (text_x + 3, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def _det_to_out(d: Detection) -> DetectionOut:
    return DetectionOut(
        class_id=d.class_id,
        class_name=d.class_name,
        category=d.category,
        box=list(d.box),
        confidence=round(d.confidence, 3),
    )


def _event_to_alert(e: DangerEvent, alert_id: str,
                     thumb_url: str | None = None) -> AlertOut:
    return AlertOut(
        id=alert_id,
        rule_name=e.rule_name,
        severity=AlertSeverity(e.severity),
        person=_det_to_out(e.person),
        hazard=_det_to_out(e.hazard),
        distance_px=round(e.distance_px, 1),
        distance_m=(round(e.distance_m, 2) if e.distance_m is not None else None),
        calibrated=e.calibrated,
        overlap_iou=round(e.overlap_iou, 3),
        timestamp=e.frame_timestamp,
        frame_thumbnail_url=thumb_url,
    )


def _zone_to_out(e: ZoneBreachEvent, breach_id: str,
                  thumb_url: str | None = None) -> ZoneBreachOut:
    return ZoneBreachOut(
        id=breach_id,
        zone_id=e.zone.id,
        zone_name=e.zone.name,
        severity=AlertSeverity(e.severity),
        rule_name=e.rule_name,
        inside=e.inside,
        distance_px=(round(e.distance_px, 1) if e.distance_px is not None else None),
        distance_m=(round(e.distance_m, 2) if e.distance_m is not None else None),
        calibrated=e.calibrated,
        person=_det_to_out(e.person),
        timestamp=e.frame_timestamp,
        frame_thumbnail_url=thumb_url,
    )


def _ppe_to_out(e: PPEEvent, check_id: str,
                thumb_url: str | None = None) -> PPECheckOut:
    return PPECheckOut(
        id=check_id,
        severity=AlertSeverity(e.severity),
        missing=e.missing,
        has_hardhat=e.hardhat is not None,
        has_vest=e.vest is not None,
        person=_det_to_out(e.person),
        timestamp=e.frame_timestamp,
        frame_thumbnail_url=thumb_url,
    )


def _posture_to_out(
    assessment: PostureAssessment,
    assessment_id: str,
    thumb_url: str | None = None,
) -> PostureAssessmentOut:
    pose_landmarks: list[list[float]] = []
    pose = assessment.landmarks
    if pose is not None and pose.ndim == 2 and pose.shape[1] >= 4:
        landmark_count = min(POSE_LANDMARK_COUNT, pose.shape[0])
        for idx in range(landmark_count):
            x, y, z, visibility = pose[idx, :4]
            if not np.isfinite((x, y, z, visibility)).all():
                # Preserve landmark indices (connections refer to them) while
                # ensuring the JSON payload never contains NaN/Infinity.
                x, y, z, visibility = 0.0, 0.0, 0.0, 0.0
            pose_landmarks.append([
                round(float(x), 6),
                round(float(y), 6),
                round(float(z), 6),
                round(float(visibility), 3),
            ])

    return PostureAssessmentOut(
        id=assessment_id,
        track_id=assessment.track_id,
        severity=AlertSeverity(assessment.severity),
        status=assessment.status,
        risk_score=round(assessment.risk_score, 3),
        signals=list(assessment.signals),
        metrics=dict(assessment.metrics),
        pose_confidence=round(assessment.pose_confidence, 3),
        history_seconds=round(assessment.history_seconds, 2),
        behavior_label=assessment.behavior_label,
        behavior_confidence=round(assessment.behavior_confidence, 4),
        behavior_probabilities={
            label: round(float(probability), 4)
            for label, probability in assessment.behavior_probabilities.items()
        },
        behavior_valid_ratio=round(assessment.behavior_valid_ratio, 4),
        behavior_window_seconds=round(assessment.behavior_window_seconds, 3),
        behavior_inference_ms=round(assessment.behavior_inference_ms, 3),
        safety_label=assessment.safety_label,
        safety_confidence=round(assessment.safety_confidence, 4),
        safety_probabilities={
            label: round(float(probability), 4)
            for label, probability in assessment.safety_probabilities.items()
        },
        torso_quality=round(assessment.torso_quality, 4),
        upper_body_quality=round(assessment.upper_body_quality, 4),
        lower_body_quality=round(assessment.lower_body_quality, 4),
        visible_ratio=round(assessment.visible_ratio, 4),
        secondary_behavior_probabilities={
            label: round(float(probability), 4)
            for label, probability in assessment.secondary_behavior_probabilities.items()
        },
        secondary_behavior_valid_ratio=round(
            assessment.secondary_behavior_valid_ratio, 4
        ),
        secondary_behavior_window_seconds=round(
            assessment.secondary_behavior_window_seconds, 3
        ),
        secondary_behavior_inference_ms=round(
            assessment.secondary_behavior_inference_ms, 3
        ),
        learned_event_type=assessment.learned_event_type,
        learned_event_reason=assessment.learned_event_reason,
        person=_det_to_out(assessment.person),
        pose_landmarks=pose_landmarks,
        timestamp=assessment.frame_timestamp,
        confirmed=assessment.confirmed,
        frame_thumbnail_url=thumb_url,
    )


def _annotate_posture(
    frame: np.ndarray,
    assessments: list[PostureAssessment],
) -> np.ndarray:
    if not assessments:
        return frame
    out = frame
    h, w = out.shape[:2]
    for assessment in assessments:
        if assessment.severity == "DANGER":
            color = COLOR_DANGER
        elif assessment.severity == "WARNING":
            color = COLOR_WARNING
        else:
            color = COLOR_POSTURE_OBSERVE

        pose = assessment.landmarks
        if pose is not None and pose.ndim == 2 and pose.shape[1] >= 4:
            points: dict[int, tuple[int, int]] = {}
            landmark_count = min(POSE_LANDMARK_COUNT, pose.shape[0])
            for idx in range(landmark_count):
                x, y, _, visibility = pose[idx, :4]
                if (
                    visibility < POSTURE_DRAW_MIN_VISIBILITY
                    or not np.isfinite((x, y, visibility)).all()
                    or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
                ):
                    continue
                points[idx] = (
                    min(w - 1, max(0, int(round(float(x) * (w - 1))))),
                    min(h - 1, max(0, int(round(float(y) * (h - 1))))),
                )

            for a, b in POSTURE_CONNECTIONS:
                if a not in points or b not in points:
                    continue
                cv2.line(out, points[a], points[b], color, 2, cv2.LINE_AA)
            for point in points.values():
                cv2.circle(out, point, 3, color, -1, cv2.LINE_AA)

        label = assessment.behavior_label or "analyzing"
        x1, y1, x2, _ = assessment.person.box
        text_y = max(18, y1 - 24)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (x1, text_y - th - 5),
                      (min(w - 1, x1 + tw + 6), text_y + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        if assessment.confirmed:
            cv2.rectangle(out, (x1, y1), (x2, assessment.person.box[3]), color, 3)
    return out


def _annotate_zones(frame: np.ndarray, zones: list,
                    active_breaches: list[ZoneBreachEvent]) -> np.ndarray:
    if not zones:
        return frame
    h, w = frame.shape[:2]
    breached_ids = {b.zone.id for b in active_breaches if b.inside}
    approach_ids = {b.zone.id for b in active_breaches if not b.inside}
    out = frame
    overlay = frame.copy()
    for z in zones:
        if not z.active or len(z.polygon) < 3:
            continue
        pts = np.array(
            [[int(round(x * w)), int(round(y * h))] for x, y in z.polygon],
            dtype=np.int32,
        )
        is_danger = z.severity == "DANGER"
        base_color = COLOR_ZONE_DANGER if is_danger else COLOR_ZONE_WARN
        breached = z.id in breached_ids
        approaching = z.id in approach_ids
        cv2.fillPoly(overlay, [pts], base_color)
        alpha = 0.35 if breached else (0.23 if approaching else 0.15)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        border = COLOR_DANGER if breached else (COLOR_WARNING if approaching else base_color)
        thickness = 3 if breached else 2
        cv2.polylines(out, [pts], isClosed=True, color=border,
                      thickness=thickness, lineType=cv2.LINE_AA)
        label_x, label_y = int(pts[0][0]), max(20, int(pts[0][1]) - 8)
        label = z.name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (label_x, label_y - th - 4),
                      (label_x + tw + 6, label_y + 2), border, -1)
        cv2.putText(out, label, (label_x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def _annotate_zone_distances(
    frame: np.ndarray,
    events: list[ZoneBreachEvent],
) -> np.ndarray:
    """Draw the nearest static-zone distance next to each affected person."""
    if not events:
        return frame
    out = frame
    by_person: dict[tuple[int, int, int, int], ZoneBreachEvent] = {}
    for event in events:
        key = tuple(event.person.box)
        current = by_person.get(key)
        current_distance = (
            abs(current.distance_m) if current and current.distance_m is not None
            else abs(current.distance_px) if current and current.distance_px is not None
            else float("inf")
        )
        event_distance = (
            abs(event.distance_m) if event.distance_m is not None
            else abs(event.distance_px) if event.distance_px is not None
            else float("inf")
        )
        if current is None or event.inside or event_distance < current_distance:
            by_person[key] = event

    for event in by_person.values():
        x1, _y1, _x2, y2 = event.person.box
        if event.distance_m is not None:
            amount = abs(event.distance_m)
            unit = "m"
        else:
            amount = abs(event.distance_px or 0.0)
            unit = "px"
        if event.inside:
            label = f"{event.zone.name}: WEWNATRZ ({amount:.1f} {unit})"
            color = COLOR_DANGER
        else:
            label = f"{event.zone.name}: {amount:.1f} {unit}"
            color = COLOR_WARNING
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        text_y = min(y2 + th + 28, out.shape[0] - 4)
        cv2.rectangle(out, (x1, text_y - th - 4),
                      (min(out.shape[1] - 1, x1 + tw + 6), text_y + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def _annotate_dynamic_safety_zones(
    frame: np.ndarray,
    zones: list[DynamicSafetyZone],
) -> np.ndarray:
    if not zones:
        return frame
    out = frame
    # Draw the warning ring as well as the danger one: on the recorded demo the
    # approach is the part that explains what the system is doing, and without
    # it a person turning orange has no visible cause.
    for zone in sorted(zones, key=lambda z: z.severity == "DANGER"):
        if zone.severity not in {"DANGER", "WARNING"}:
            continue
        if len(zone.polygon_px) < 3:
            continue
        pts = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in zone.polygon_px],
            dtype=np.int32,
        )
        color = COLOR_DANGER if zone.severity == "DANGER" else COLOR_WARNING
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        alpha = 0.10
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        x, y = zone.hazard.box[0], max(18, zone.hazard.box[1] - 32)
        if zone.threshold_m is not None:
            label = f"{zone.severity} {zone.threshold_m:.1f} m"
        else:
            label = f"{zone.severity} strefa dynamiczna"
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 2, cv2.LINE_AA)
    return out


def _annotate_worker_ids(
    frame: np.ndarray,
    identities: list[WorkerIdentity],
    profiles: dict[str, WorkerRecord] | None = None,
) -> np.ndarray:
    if not identities:
        return frame
    out = frame
    profiles = profiles or {}
    worker_color = (128, 30, 82)  # frontend rgba(82, 30, 128, 0.92), in BGR
    for identity in identities:
        x1, y1, _, _ = identity.person.box
        profile = profiles.get(identity.worker_id)
        label = f"ID {identity.worker_id}"
        if profile is not None and profile.full_name:
            label += f" · {profile.full_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        text_top = min(max(0, y1 + 3), max(0, out.shape[0] - th - 8))
        text_y = text_top + th + 4
        cv2.rectangle(
            out,
            (x1, text_top),
            (min(out.shape[1] - 1, x1 + tw + 6), text_y + 2),
            worker_color,
            -1,
        )
        cv2.putText(out, label, (x1 + 3, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if identity.tag_polygon and not identity.cached:
            pts = np.asarray(
                [[int(round(x)), int(round(y))] for x, y in identity.tag_polygon],
                dtype=np.int32,
            )
            cv2.polylines(out, [pts], True, worker_color, 2, cv2.LINE_AA)
    return out


def _annotate_unidentified(
    frame: np.ndarray,
    events: list[UnidentifiedWorkerEvent],
) -> np.ndarray:
    if not events:
        return frame
    out = frame
    for event in events:
        x1, y1, x2, y2 = event.person.box
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_WARNING, 3)
        label = "ID BRAK"
        cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + 70, y1), COLOR_WARNING, -1)
        cv2.putText(out, label, (x1 + 4, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# Anyone the system has something to say about gets a translucent fill and a
# thicker border in the alert colour; everyone else keeps the plain green
# outline. Orange warns, red means danger — the convention a site manager
# already reads without a legend.
PERSON_HIGHLIGHT_ALPHA = 0.30

PersonHighlights = dict[tuple[int, int, int, int], tuple[tuple[int, int, int], str]]

# reason -> (precedence, colour, on-frame label); lower precedence wins.
# Labels avoid Polish diacritics because OpenCV's Hershey fonts cannot draw them.
_PERSON_HIGHLIGHT_RULES: dict[str, tuple[int, tuple[int, int, int], str]] = {
    "fall": (10, COLOR_DANGER, "UPADEK"),
    "lying": (20, COLOR_DANGER, "OSOBA NA ZIEMI"),
    "danger_zone": (30, COLOR_DANGER, "STREFA DANGER"),
    "ppe": (40, COLOR_DANGER, "BRAK PPE"),
    "smoking": (50, COLOR_DANGER, "PALENIE"),
    "warning_zone": (60, COLOR_WARNING, "ZBLIZANIE"),
}


def _person_highlights(
    *,
    dangers: list[DangerEvent] = (),
    zone_breaches: list = (),
    posture_assessments: list[PostureAssessment] = (),
    ppe_events: list[PPEEvent] = (),
) -> PersonHighlights:
    """Pick one colour per person: the most serious thing said about them."""
    best: dict[tuple[int, int, int, int], tuple[int, tuple[int, int, int], str]] = {}

    def offer(box, reason: str) -> None:
        rule = _PERSON_HIGHLIGHT_RULES.get(reason)
        if rule is None:
            return
        key = tuple(int(v) for v in box)
        current = best.get(key)
        if current is None or rule[0] < current[0]:
            best[key] = rule

    for event in dangers:
        offer(
            event.person.box,
            "danger_zone" if event.severity == "DANGER" else "warning_zone",
        )
    for breach in zone_breaches:
        offer(
            breach.person.box,
            "danger_zone" if breach.severity == "DANGER" else "warning_zone",
        )
    for assessment in posture_assessments:
        category = posture_signal_category(assessment.signals)
        if category in {"fall", "lying", "smoking"}:
            offer(assessment.person.box, category)
    for event in ppe_events:
        if event.missing:
            offer(event.person.box, "ppe")

    return {box: (colour, label) for box, (_, colour, label) in best.items()}


def _draw_person_highlights(
    out: np.ndarray,
    highlights: PersonHighlights,
) -> np.ndarray:
    if not highlights:
        return out
    # One frame copy for every fill, not one per person: at 2688x1520 the copy
    # costs more than the drawing.
    overlay = out.copy()
    for (x1, y1, x2, y2), (colour, _label) in highlights.items():
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(
        overlay, PERSON_HIGHLIGHT_ALPHA, out, 1.0 - PERSON_HIGHLIGHT_ALPHA, 0, out
    )
    height = out.shape[0]
    for (x1, y1, x2, y2), (colour, label) in highlights.items():
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Below the box: the detection label already sits above it.
        baseline = min(height - 4, y2 + th + 10)
        cv2.rectangle(
            out, (x1, baseline - th - 8), (x1 + tw + 10, baseline), colour, -1
        )
        cv2.putText(out, label, (x1 + 5, baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _annotate_site(frame: np.ndarray, detections: list[Detection],
                   raw_dangers: list[DangerEvent],
                   confirmed: list[DangerEvent],
                   highlights: PersonHighlights | None = None) -> np.ndarray:
    out = frame.copy()
    highlights = highlights or {}

    for d in detections:
        x1, y1, x2, y2 = d.box
        color = COLOR_PERSON if d.category == "person" else COLOR_VEHICLE
        if d.category == "person" and tuple(int(v) for v in d.box) in highlights:
            # The highlight pass draws this person's box in the alert colour.
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for e in raw_dangers:
        pc = ((e.person.box[0] + e.person.box[2]) // 2,
              (e.person.box[1] + e.person.box[3]) // 2)
        hc = ((e.hazard.box[0] + e.hazard.box[2]) // 2,
              (e.hazard.box[1] + e.hazard.box[3]) // 2)
        color = COLOR_DANGER if e.severity == "DANGER" else COLOR_WARNING
        cv2.line(out, pc, hc, color, 1, cv2.LINE_AA)

    out = _draw_person_highlights(out, highlights)

    if confirmed:
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), COLOR_DANGER, -1)
        cv2.putText(out, f"ALARM — {len(confirmed)} danger(s)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    return out


def _annotate_ppe(frame: np.ndarray, detections: list[Detection],
                  events: list[PPEEvent],
                  highlights: PersonHighlights | None = None) -> np.ndarray:
    out = frame.copy()
    highlights = highlights or {}

    for d in detections:
        x1, y1, x2, y2 = d.box
        if d.category == "person":
            if tuple(int(v) for v in d.box) in highlights:
                continue
            color = COLOR_PERSON
        elif d.category == "hardhat":
            color = COLOR_HARDHAT
        elif d.category == "vest":
            color = COLOR_VEST
        else:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    out = _draw_person_highlights(out, highlights)

    banner_color = None
    banner_text = None
    for e in events:
        if not e.confirmed:
            continue
        if e.missing:
            banner_color = COLOR_DANGER
            banner_text = "MISSING: " + " + ".join(m.upper() for m in e.missing)
        else:
            banner_color = COLOR_OK
            banner_text = "PPE OK"

    if banner_color and banner_text:
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), banner_color, -1)
        cv2.putText(out, banner_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out


def _align_posture_to_current_people(
    assessments: list[PostureAssessment],
    persons: list[Detection],
    *,
    frame_width: int,
    frame_height: int,
    now: float,
    source_timestamp: float,
) -> list[PostureAssessment]:
    """Move a completed background pose onto the freshest YOLO person box."""
    if not assessments or not persons:
        return []
    remaining = list(persons)
    aligned: list[PostureAssessment] = []
    for assessment in assessments:
        if not remaining:
            break
        old_box = assessment.person.box
        old_cx = (old_box[0] + old_box[2]) / 2.0
        old_cy = (old_box[1] + old_box[3]) / 2.0
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                ((remaining[index].box[0] + remaining[index].box[2]) / 2.0 - old_cx) ** 2
                + ((remaining[index].box[1] + remaining[index].box[3]) / 2.0 - old_cy) ** 2
            ),
        )
        person = remaining.pop(best_index)
        landmarks = assessment.landmarks
        if landmarks is not None:
            landmarks = transform_cached_landmarks(
                landmarks,
                old_box,
                person.box,
                old_frame_width=assessment.frame_width or frame_width,
                old_frame_height=assessment.frame_height or frame_height,
                new_frame_width=frame_width,
                new_frame_height=frame_height,
            )
        metrics = dict(assessment.metrics)
        metrics["worker_result_age_ms"] = max(0.0, (now - source_timestamp) * 1000.0)
        aligned.append(replace(
            assessment,
            person=person,
            landmarks=landmarks,
            frame_timestamp=now,
            frame_width=frame_width,
            frame_height=frame_height,
            metrics=metrics,
        ))
    return aligned


def _runtime_options(request: Request, camera_id: str):
    store = getattr(request.app.state, "runtime_processing_store", None)
    if store is None:
        from backend.runtime_options import RuntimeProcessingOptions
        return RuntimeProcessingOptions(updated_at=time.time())
    return store.get(camera_id)


def _handle_site(
    request: Request,
    frame: np.ndarray,
    now: float,
    t0: float,
    camera_id: str,
    preview_jpeg_b64: str,
    performance_trace: FramePerformanceTrace | None = None,
    rendered_frames: list[np.ndarray] | None = None,
) -> FrameResultOut:
    detector = request.app.state.detector
    danger_detector = request.app.state.danger_detector
    temporal_filter = request.app.state.temporal_filter
    frame_store = request.app.state.frame_store
    alert_store = request.app.state.alert_store
    evidence_recorder = getattr(request.app.state, "evidence_recorder", None)
    zone_store = request.app.state.zone_store
    zone_detector = request.app.state.zone_detector
    zone_temporal_filter = request.app.state.zone_temporal_filter
    calibration_store = request.app.state.calibration_store
    posture_manager = getattr(request.app.state, "posture_manager", None)
    unidentified_monitor = getattr(request.app.state, "unidentified_worker_monitor", None)
    worker_store = getattr(request.app.state, "worker_store", None)
    worker_profile_cache = getattr(
        request.app.state,
        "worker_profile_cache",
        None,
    )
    runtime = _runtime_options(request, camera_id)

    detector_ran = runtime.requires_detector("site")
    yolo_size_controller = getattr(
        request.app.state,
        "yolo_size_controller",
        None,
    )
    selected_yolo_size = None
    if detector_ran and yolo_size_controller is not None:
        selected_yolo_size = yolo_size_controller.select_size(
            camera_id,
            backend_supports_adaptive_size=bool(
                getattr(detector, "supports_adaptive_size", True)
            ),
        )
    detections = (
        _profiled_detection(
            detector,
            frame,
            performance_trace,
            img_size=selected_yolo_size,
        )
        if detector_ran else []
    )

    # Debug: inject a synthetic person into the detection list for N frames.
    inj = request.app.state.debug_inject_person
    if inj and inj.get("remaining", 0) > 0:
        fh_i, fw_i = frame.shape[:2]
        x1n, y1n, x2n, y2n = inj["box_norm"]
        detections.append(Detection(
            class_id=0,
            class_name="person",
            category="person",
            box=(int(x1n * fw_i), int(y1n * fh_i),
                 int(x2n * fw_i), int(y2n * fh_i)),
            confidence=float(inj["confidence"]),
        ))
        inj["remaining"] -= 1
        if inj["remaining"] <= 0:
            request.app.state.debug_inject_person = None

    fh, fw = frame.shape[:2]
    needs_zone_geometry = runtime.zones or runtime.distances or runtime.markers
    zones = zone_store.for_camera(camera_id) if needs_zone_geometry else []
    marker_zones_active = bool(
        (runtime.zones or runtime.distances)
        and any(
            zone.active and len(zone.marker_ids or []) >= 3
            for zone in zones
        )
    )
    marker_scheduler = getattr(request.app.state, "marker_scheduler", None)
    calibration_marker_sampling = bool(
        marker_scheduler is not None
        and marker_scheduler.calibration_session_active(camera_id)
    )
    marker_processing_required = bool(
        runtime.markers
        or marker_zones_active
        or calibration_marker_sampling
    )
    markers = (
        _scheduled_markers(
            request,
            frame,
            camera_id,
            zones,
            performance_trace,
        )
        if marker_processing_required else []
    )
    stored_calibration = calibration_store.get(camera_id)
    calibration = (
        stored_calibration
        if stored_calibration is not None
        and stored_calibration.is_compatible(fw, fh)
        else None
    )
    persons = [d for d in detections if d.category == "person"]
    latest_detection_store = getattr(
        request.app.state,
        "latest_detection_store",
        None,
    )
    if detector_ran and latest_detection_store is not None:
        latest_detection_store.put(camera_id, now, fw, fh, detections)
    if performance_trace is not None:
        performance_trace.has_person = bool(persons)

    worker_identities: list[WorkerIdentity] = []
    monitor_persons = persons
    if runtime.worker_id and persons:
        persons, worker_identities, monitor_persons = _identify_workers(
            request,
            camera_id,
            frame,
            persons,
            now,
            performance_trace,
            wait_for_current_frame=rendered_frames is not None,
        )
    worker_profiles = _worker_profiles(
        worker_profile_cache or worker_store,
        worker_identities,
    )
    unidentified_events = (
        unidentified_monitor.update(
            camera_id,
            "site",
            monitor_persons,
            worker_identities,
            now,
        )
        if runtime.worker_id and unidentified_monitor is not None else []
    )

    # Proximity and static-zone rules are independently switchable at runtime.
    if runtime.distances:
        raw_dangers = danger_detector.evaluate(
            detections, now, calibration, frame_w=fw, frame_h=fh,
        )
        confirmed = temporal_filter.update(raw_dangers, now, camera_id)
        dynamic_safety_zones = danger_detector.dynamic_zones(
            detections, fw, fh, calibration,
        )
    else:
        raw_dangers = []
        confirmed = []
        dynamic_safety_zones = []

    if marker_zones_active:
        zones = _resolve_marker_zones(
            zones, markers, camera_id,
            request.app.state.marker_zone_cache,
            now, fw, fh,
        )
    if runtime.zones:
        raw_zone_breaches = zone_detector.evaluate(
            detections, zones, fw, fh, now, calibration,
        )
        confirmed_zone_breaches = zone_temporal_filter.update(
            raw_zone_breaches,
            now,
            camera_id,
        )
    else:
        raw_zone_breaches = []
        confirmed_zone_breaches = []
    person_distances = (
        _person_distances(persons, zones, fw, fh, calibration)
        if runtime.distances and zones else []
    )

    posture_assessments: list[PostureAssessment] = []
    if (
        runtime.posture
        and persons
        and posture_manager is not None
        and posture_manager.available
    ):
        posture_worker = getattr(request.app.state, "posture_worker", None)
        if posture_worker is None:
            # Compatibility path for small embedded deployments and tests.
            # Production initializes a dedicated worker in ``main.py``.
            posture_result = posture_manager.process(camera_id, frame, persons, now)
            posture_assessments = posture_result.assessments
        else:
            # MediaPipe owns a private frame copy and never blocks the main
            # detector pipeline. Its one-slot queue always replaces stale work.
            if rendered_frames is not None:
                submit = getattr(
                    posture_worker,
                    "submit_for_export",
                    posture_worker.submit_latest,
                )
            else:
                submit = getattr(
                    posture_worker,
                    "submit_borrowed_latest",
                    posture_worker.submit_latest,
                )
            submit(camera_id, frame, persons, now)
            if rendered_frames is not None:
                wait_for_result = getattr(posture_worker, "wait_for_result", None)
                waited_snapshot = (
                    wait_for_result(
                        camera_id,
                        after_timestamp=float(np.nextafter(now, -np.inf)),
                        timeout=CONFIG.demo_video.control_timeout_seconds,
                    )
                    if callable(wait_for_result) else posture_worker.get_latest(camera_id)
                )
                posture_snapshot = waited_snapshot or posture_worker.get_latest(camera_id)
            else:
                posture_snapshot = posture_worker.get_latest(camera_id)
            if posture_snapshot is not None and posture_snapshot.result is not None:
                posture_assessments = _align_posture_to_current_people(
                    posture_snapshot.result.assessments,
                    persons,
                    frame_width=fw,
                    frame_height=fh,
                    now=now,
                    source_timestamp=posture_snapshot.frame_timestamp,
                )

                # A completed background result can be reused over multiple
                # YOLO frames. Persist its confirmed alert only once while
                # retaining the cached landmarks for a stable live skeleton.
                seen = getattr(
                    request.app.state,
                    "posture_confirmations_seen",
                    None,
                )
                if seen is None:
                    seen = {}
                    request.app.state.posture_confirmations_seen = seen
                result_key = (
                    posture_snapshot.frame_timestamp,
                    posture_snapshot.completed_at,
                )
                if seen.get(camera_id) == result_key:
                    posture_assessments = [
                        replace(assessment, confirmed=False)
                        if assessment.confirmed else assessment
                        for assessment in posture_assessments
                    ]
                else:
                    seen[camera_id] = result_key

    if detector_ran and yolo_size_controller is not None:
        yolo_size_controller.observe(
            camera_id,
            detections,
            posture_assessments,
        )

    annotated: np.ndarray | None = None
    overlay_ms = 0.0

    def _evidence_frame() -> np.ndarray:
        # The live panel draws metadata overlays itself.  Backend rasterization
        # is therefore needed only for sampled evidence and alert thumbnails.
        # Keeping it lazy removes a full-resolution copy/draw from ordinary
        # frames while preserving annotated evidence when an event occurs.
        nonlocal annotated, overlay_ms
        if annotated is not None:
            return annotated
        overlay_started = time.perf_counter()
        out = frame.copy()
        if runtime.zones:
            out = _annotate_zones(out, zones, raw_zone_breaches)
        if runtime.distances:
            out = _annotate_dynamic_safety_zones(out, dynamic_safety_zones)
        # Raw events, not confirmed ones: the highlight should stay on the
        # person for as long as the situation lasts, while the alert history
        # still only gets the temporally confirmed events.
        out = _annotate_site(
            out,
            detections,
            raw_dangers if runtime.distances else [],
            confirmed if runtime.distances else [],
            _person_highlights(
                dangers=raw_dangers if runtime.distances else [],
                zone_breaches=raw_zone_breaches if runtime.zones else [],
                posture_assessments=posture_assessments if runtime.posture else [],
            ),
        )
        if runtime.markers or marker_zones_active:
            calibration_ids = (
                set(stored_calibration.marker_ids) if stored_calibration else set()
            )
            out = _annotate_markers(out, markers, calibration_ids)
        if runtime.distances:
            out = _annotate_person_distances(out, person_distances)
        if runtime.posture:
            out = _annotate_posture(out, posture_assessments)
        if runtime.worker_id:
            out = _annotate_worker_ids(out, worker_identities, worker_profiles)
            out = _annotate_unidentified(out, unidentified_events)
        annotated = out
        overlay_ms += (time.perf_counter() - overlay_started) * 1000.0
        return out

    if evidence_recorder is not None and (
        runtime.distances or runtime.zones or runtime.posture or runtime.worker_id
    ):
        wants_sample = getattr(evidence_recorder, "should_sample", None)
        if not callable(wants_sample) or wants_sample(camera_id, now):
            evidence_jpeg_ms = evidence_recorder.push(
                camera_id,
                _evidence_frame(),
                now,
            )
            if performance_trace is not None and isinstance(
                evidence_jpeg_ms,
                (int, float),
            ):
                performance_trace.add("jpeg_encode_ms", evidence_jpeg_ms)

    alert_outs = []
    for evt in confirmed:
        alert_id = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(_evidence_frame(), alert_id)
        clip_url = evidence_recorder.trigger(camera_id, alert_id, now) if evidence_recorder else None
        alert = _event_to_alert(evt, alert_id, thumb_url)
        alert_outs.append(alert)
        identity = _identity_for_person(evt.person, worker_identities)
        worker_id = identity.worker_id if identity is not None and identity.alert_eligible else None
        record = _site_record(alert, camera_id, worker_id, clip_url)
        alert_store.append(_attach_identity_snapshot(record, identity, worker_profiles.get(worker_id)))

    zone_breach_outs = []
    for zevt in confirmed_zone_breaches:
        breach_id = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(_evidence_frame(), breach_id)
        clip_url = evidence_recorder.trigger(camera_id, breach_id, now) if evidence_recorder else None
        breach_out = _zone_to_out(zevt, breach_id, thumb_url)
        zone_breach_outs.append(breach_out)
        identity = _identity_for_person(zevt.person, worker_identities)
        worker_id = identity.worker_id if identity is not None and identity.alert_eligible else None
        record = _zone_record(breach_out, camera_id, worker_id, clip_url)
        alert_store.append(_attach_identity_snapshot(record, identity, worker_profiles.get(worker_id)))

    posture_outs = [
        _posture_to_out(assessment, f"active-{assessment.track_id}")
        for assessment in posture_assessments
    ]
    confirmed_posture_outs = []
    for assessment in posture_assessments:
        if not assessment.confirmed:
            continue
        posture_id = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(_evidence_frame(), posture_id)
        clip_url = evidence_recorder.trigger(camera_id, posture_id, now) if evidence_recorder else None
        posture_out = _posture_to_out(assessment, posture_id, thumb_url)
        confirmed_posture_outs.append(posture_out)
        identity = _identity_for_person(assessment.person, worker_identities)
        worker_id = identity.worker_id if identity is not None and identity.alert_eligible else None
        record = _posture_record(posture_out, camera_id, worker_id, clip_url)
        alert_store.append(_attach_identity_snapshot(record, identity, worker_profiles.get(worker_id)))

    unidentified_outs = []
    for event in unidentified_events:
        event_id = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(_evidence_frame(), event_id)
        clip_url = evidence_recorder.trigger(camera_id, event_id, now) if evidence_recorder else None
        event_out = _unidentified_to_out(event, event_id, thumb_url)
        unidentified_outs.append(event_out)
        alert_store.append(_unidentified_record(event_out, camera_id, "site", clip_url))

    active_outs = [_event_to_alert(event, "active") for event in raw_dangers]
    active_zone_outs = [_zone_to_out(event, "active") for event in raw_zone_breaches]
    active_zones_out = [
        ActiveZoneOut(
            id=zone.id,
            name=zone.name,
            severity=zone.severity,
            polygon=zone.polygon,
            marker_ids=list(zone.marker_ids or []),
            warning_distance_m=zone.warning_distance_m,
            warning_distance_px=zone.warning_distance_px,
        )
        for zone in zones
        if runtime.zones and zone.active and zone.polygon and len(zone.polygon) >= 3
    ]

    if performance_trace is not None:
        performance_trace.set("overlay_ms", overlay_ms)

    if rendered_frames is not None:
        rendered_frames.append(_evidence_frame())

    processing_ms = (time.monotonic() - t0) * 1000
    request.app.state.frame_counter += 1

    return FrameResultOut(
        frame_id=request.app.state.frame_counter,
        timestamp=now,
        camera_id=camera_id,
        mode="site",
        detections=[_det_to_out(detection) for detection in detections] if runtime.boxes else [],
        active_dangers=active_outs,
        confirmed_alerts=alert_outs,
        posture_assessments=posture_outs,
        confirmed_posture_alerts=confirmed_posture_outs,
        posture_available=bool(runtime.posture and posture_manager and posture_manager.available),
        behavior_classifier_available=bool(
            runtime.posture and posture_manager and getattr(posture_manager, "behavior_available", False)
        ),
        active_zone_breaches=active_zone_outs,
        confirmed_zone_breaches=zone_breach_outs,
        markers=(
            [_marker_to_out(marker) for marker in markers]
            if runtime.markers else []
        ),
        worker_identifications=[
            _worker_to_out(identity, worker_profiles.get(identity.worker_id))
            for identity in worker_identities
        ],
        unidentified_workers=unidentified_outs,
        worker_identification_available=bool(
            runtime.worker_id and _worker_identification_available(request)
        ),
        dynamic_safety_zones=[
            _dynamic_zone_to_out(zone, fw, fh) for zone in dynamic_safety_zones
        ],
        person_distances=person_distances,
        active_zones=active_zones_out,
        calibration_active=calibration is not None,
        frame_jpeg_b64=preview_jpeg_b64,
        processing_ms=max(0.1, round(processing_ms, 1)),
        runtime_options={key: bool(value) for key, value in runtime.to_dict().items() if key != "updated_at"},
        detector_ran=detector_ran,
    )


def _handle_checkpoint(
    request: Request,
    frame: np.ndarray,
    now: float,
    t0: float,
    camera_id: str,
    preview_jpeg_b64: str,
    performance_trace: FramePerformanceTrace | None = None,
    rendered_frames: list[np.ndarray] | None = None,
) -> FrameResultOut:
    detector = request.app.state.ppe_detector
    checker = request.app.state.ppe_checker
    frame_store = request.app.state.frame_store
    alert_store = request.app.state.alert_store
    evidence_recorder = getattr(request.app.state, "evidence_recorder", None)
    unidentified_monitor = getattr(request.app.state, "unidentified_worker_monitor", None)
    worker_store = getattr(request.app.state, "worker_store", None)
    worker_profile_cache = getattr(
        request.app.state,
        "worker_profile_cache",
        None,
    )
    runtime = _runtime_options(request, camera_id)

    detector_ran = runtime.requires_detector("checkpoint")
    detections = (
        _profiled_detection(detector, frame, performance_trace)
        if detector_ran else []
    )
    persons = [d for d in detections if d.category == "person"]
    latest_detection_store = getattr(
        request.app.state,
        "latest_detection_store",
        None,
    )
    if detector_ran and latest_detection_store is not None:
        latest_detection_store.put(
            camera_id,
            now,
            int(frame.shape[1]),
            int(frame.shape[0]),
            detections,
        )
    if performance_trace is not None:
        performance_trace.has_person = bool(persons)
    worker_identities: list[WorkerIdentity] = []
    monitor_persons = persons
    if runtime.worker_id and persons:
        persons, worker_identities, monitor_persons = _identify_workers(
            request,
            camera_id,
            frame,
            persons,
            now,
            performance_trace,
            wait_for_current_frame=rendered_frames is not None,
        )
    worker_profiles = _worker_profiles(
        worker_profile_cache or worker_store,
        worker_identities,
    )
    unidentified_events = (
        unidentified_monitor.update(
            camera_id,
            "checkpoint",
            monitor_persons,
            worker_identities,
            now,
        )
        if runtime.worker_id and unidentified_monitor is not None else []
    )

    if runtime.ppe:
        events = checker.evaluate(detections, frame_h=frame.shape[0],
                                  frame_timestamp=now)
        confirmed = checker.confirm(events, now, camera_id)
    else:
        events = []
        confirmed = []
    annotated: np.ndarray | None = None
    overlay_ms = 0.0

    def _evidence_frame() -> np.ndarray:
        nonlocal annotated, overlay_ms
        if annotated is not None:
            return annotated
        overlay_started = time.perf_counter()
        out = _annotate_ppe(
            frame,
            detections if runtime.ppe else [],
            confirmed if runtime.ppe else [],
            _person_highlights(ppe_events=events if runtime.ppe else []),
        )
        if runtime.worker_id:
            out = _annotate_worker_ids(out, worker_identities, worker_profiles)
            out = _annotate_unidentified(out, unidentified_events)
        annotated = out
        overlay_ms += (time.perf_counter() - overlay_started) * 1000.0
        return out

    if evidence_recorder is not None and (runtime.ppe or runtime.worker_id):
        wants_sample = getattr(evidence_recorder, "should_sample", None)
        if not callable(wants_sample) or wants_sample(camera_id, now):
            evidence_jpeg_ms = evidence_recorder.push(
                camera_id,
                _evidence_frame(),
                now,
            )
            if performance_trace is not None and isinstance(
                evidence_jpeg_ms,
                (int, float),
            ):
                performance_trace.add("jpeg_encode_ms", evidence_jpeg_ms)

    check_outs = []
    for event in confirmed:
        check_id = uuid.uuid4().hex[:8]
        # Successful checkpoint checks are returned to the UI but do not need
        # persistent evidence. Save thumbnail/clip only for actual violations.
        thumb_url = (
            frame_store.save(_evidence_frame(), check_id)
            if event.missing else None
        )
        clip_url = (
            evidence_recorder.trigger(camera_id, check_id, now)
            if event.missing and evidence_recorder else None
        )
        check_out = _ppe_to_out(event, check_id, thumb_url)
        check_outs.append(check_out)
        if event.missing:
            identity = _identity_for_person(event.person, worker_identities)
            worker_id = identity.worker_id if identity is not None and identity.alert_eligible else None
            record = _ppe_record(check_out, camera_id, worker_id, clip_url)
            alert_store.append(_attach_identity_snapshot(record, identity, worker_profiles.get(worker_id)))

    unidentified_outs = []
    for event in unidentified_events:
        event_id = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(_evidence_frame(), event_id)
        clip_url = evidence_recorder.trigger(camera_id, event_id, now) if evidence_recorder else None
        event_out = _unidentified_to_out(event, event_id, thumb_url)
        unidentified_outs.append(event_out)
        alert_store.append(_unidentified_record(event_out, camera_id, "checkpoint", clip_url))

    if performance_trace is not None:
        performance_trace.set("overlay_ms", overlay_ms)

    if rendered_frames is not None:
        rendered_frames.append(_evidence_frame())

    processing_ms = (time.monotonic() - t0) * 1000
    request.app.state.frame_counter += 1

    return FrameResultOut(
        frame_id=request.app.state.frame_counter,
        timestamp=now,
        camera_id=camera_id,
        mode="checkpoint",
        detections=[_det_to_out(detection) for detection in detections] if runtime.boxes else [],
        ppe_checks=check_outs,
        worker_identifications=[
            _worker_to_out(identity, worker_profiles.get(identity.worker_id))
            for identity in worker_identities
        ],
        unidentified_workers=unidentified_outs,
        worker_identification_available=bool(
            runtime.worker_id and _worker_identification_available(request)
        ),
        frame_jpeg_b64=preview_jpeg_b64,
        processing_ms=max(0.1, round(processing_ms, 1)),
        runtime_options={key: bool(value) for key, value in runtime.to_dict().items() if key != "updated_at"},
        detector_ran=detector_ran,
    )


def _with_calibration_status(
    app,
    result: FrameResultOut,
    frame: np.ndarray,
) -> FrameResultOut:
    stored_calibration = app.state.calibration_store.get(result.camera_id)
    calibration_warning = (
        stored_calibration.compatibility_warning(
            int(frame.shape[1]),
            int(frame.shape[0]),
        )
        if stored_calibration is not None else None
    )
    return result.model_copy(update={
        # A stored record remains visible after a format/orientation change;
        # only metric projection is disabled for an incompatible frame.
        "calibration_active": stored_calibration is not None,
        "calibration_valid": (
            stored_calibration is not None and calibration_warning is None
        ),
        "calibration_warning": calibration_warning,
    })


def create_decoded_frame_job(
    app,
    *,
    frame: np.ndarray,
    camera_id: str,
    timestamp: float,
    mode: str = "site",
    jpeg_quality: int = 85,
    video_decode_ms: float | None = None,
    performance_source: str = "demo",
    render_annotated: bool = False,
) -> FrameJob:
    """Adapt an already-decoded backend frame to the existing pipeline.

    Live camera uploads still use :func:`receive_frame`.  Demo-video workers
    call this adapter so their decoded NumPy frame follows the exact same
    inference, latest-frame, camera-registry and WebSocket path without an
    artificial browser JPEG upload.
    """

    profiler = getattr(app.state, "performance_profiler", None)
    performance_trace = (
        profiler.new_trace(
            source=performance_source,
            video_decode_ms=video_decode_ms,
        )
        if profiler is not None else None
    )
    prepare_started = time.perf_counter()

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("decoded frame must be a non-empty NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("decoded frame must use BGR three-channel layout")
    camera_id = str(camera_id).strip()
    if not camera_id:
        raise ValueError("camera_id must not be empty")

    quality = min(100, max(1, int(jpeg_quality)))
    preview_encoder = getattr(app.state, "preview_encoder", None)
    preview_jpeg_future: Future[EncodedPreview] | None = None
    jpeg_bytes = b""
    jpeg_ms = 0.0
    if preview_encoder is not None:
        # JPEG is CPU work and YOLO is predominantly GPU work. Run both from
        # the same immutable decoded frame, then join only at publication.
        preview_jpeg_future = preview_encoder.submit(frame, quality)
    else:
        jpeg_started = time.perf_counter()
        encoded, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not encoded:
            raise ValueError("decoded frame could not be encoded for preview")
        jpeg_bytes = buffer.tobytes()
        jpeg_ms = (time.perf_counter() - jpeg_started) * 1000.0
        if performance_trace is not None:
            performance_trace.set("jpeg_encode_ms", jpeg_ms)
    received_at = time.time()
    effective_mode = (
        "checkpoint"
        if mode == "checkpoint" and getattr(app.state, "ppe_detector", None) is not None
        else "site"
    )

    latest_frame_store = getattr(app.state, "latest_frame_store", None)
    if latest_frame_store is not None:
        stored = latest_frame_store.set(
            camera_id,
            frame,
            float(timestamp),
            effective_mode,
            received_at=received_at,
            image_bytes=jpeg_bytes or None,
            content_type="image/jpeg" if jpeg_bytes else None,
        )
        if not stored:
            raise ValueError("decoded frame exceeds the latest-frame store limit")

    job = FrameJob(
        camera_id=camera_id,
        frame=frame,
        timestamp=float(timestamp),
        payload=_FramePayload(
            mode=effective_mode,
            received_at=received_at,
            preview_jpeg_bytes=jpeg_bytes,
            preview_jpeg_b64="",
            preview_jpeg_future=preview_jpeg_future,
            performance_trace=performance_trace,
            render_annotated=bool(render_annotated),
        ),
    )
    _record_frame_received(
        app,
        camera_id=camera_id,
        timestamp=float(timestamp),
        received_at=received_at,
        mode=effective_mode,
        frame=frame,
        processing_pending=True,
    )
    if performance_trace is not None:
        performance_trace.set(
            "frame_prepare_ms",
            max(
                0.0,
                (time.perf_counter() - prepare_started) * 1000.0 - jpeg_ms,
            ),
        )
        performance_trace.mark_queued()
    return job


def process_frame_job(
    app,
    job: FrameJob,
    *,
    rendered_frames: list[np.ndarray] | None = None,
) -> FrameResultOut:
    """Heavy inference callback used by :class:`LatestFrameProcessor`.

    The generic processor invokes this function through ``asyncio.to_thread``.
    A process-wide model lock also protects deployments where a legacy
    synchronous client and background phone ingest briefly overlap.
    """

    payload = job.payload
    if not isinstance(payload, _FramePayload):
        raise TypeError("frame job payload has an unsupported shape")

    request_context = SimpleNamespace(app=app)
    processing_started = time.monotonic()
    performance_trace = payload.performance_trace
    analysis_lock = getattr(app.state, "analysis_lock", None)
    lock_context = analysis_lock if analysis_lock is not None else nullcontext()
    render_sink = rendered_frames if payload.render_annotated else None
    with lock_context:
        if performance_trace is not None:
            performance_trace.set(
                "queue_age_ms",
                (time.perf_counter() - performance_trace.queued_at) * 1000.0,
            )
        if (
            payload.mode == "checkpoint"
            and app.state.ppe_detector is not None
        ):
            result = _handle_checkpoint(
                request_context,
                job.frame,
                job.timestamp,
                processing_started,
                job.camera_id,
                payload.preview_jpeg_b64,
                performance_trace,
                render_sink,
            )
        else:
            result = _handle_site(
                request_context,
                job.frame,
                job.timestamp,
                processing_started,
                job.camera_id,
                payload.preview_jpeg_b64,
                performance_trace,
                render_sink,
            )
        result = _with_calibration_status(app, result, job.frame)
        return result


def reset_camera_temporal_state(
    app,
    camera_id: str,
    *,
    timeout: float | None = 10.0,
) -> None:
    """Reset one source run without changing its runtime layer switches."""

    posture_worker = getattr(app.state, "posture_worker", None)
    posture_manager = getattr(app.state, "posture_manager", None)
    if posture_worker is not None:
        if not posture_worker.reset_camera(camera_id, timeout=timeout):
            raise TimeoutError(
                f"posture worker did not release camera {camera_id!r}"
            )
    elif posture_manager is not None:
        posture_manager.reset_camera(camera_id)

    yolo_size_controller = getattr(app.state, "yolo_size_controller", None)
    if yolo_size_controller is not None:
        yolo_size_controller.reset_camera(camera_id)

    latest_detection_store = getattr(app.state, "latest_detection_store", None)
    if latest_detection_store is not None:
        latest_detection_store.remove(camera_id)

    worker_id_worker = getattr(app.state, "worker_id_worker", None)
    if worker_id_worker is not None:
        worker_id_worker.reset_camera(camera_id)

    analysis_lock = getattr(app.state, "analysis_lock", None)
    lock_context = analysis_lock if analysis_lock is not None else nullcontext()
    with lock_context:
        for name in (
            "temporal_filter",
            "zone_temporal_filter",
            "ppe_checker",
            "unidentified_worker_monitor",
        ):
            component = getattr(app.state, name, None)
            reset = getattr(component, "reset_camera", None)
            if callable(reset):
                reset(camera_id)

        seen = getattr(app.state, "posture_confirmations_seen", None)
        if isinstance(seen, dict):
            seen.pop(camera_id, None)
        marker_cache = getattr(app.state, "marker_zone_cache", None)
        if isinstance(marker_cache, dict):
            for key in [
                key for key in marker_cache
                if isinstance(key, tuple) and key and key[0] == camera_id
            ]:
                marker_cache.pop(key, None)

    for name in ("marker_scheduler", "evidence_recorder"):
        component = getattr(app.state, name, None)
        reset = getattr(component, "reset_camera", None)
        if callable(reset):
            reset(camera_id)

    latest = getattr(app.state, "latest_frame_store", None)
    remove_latest = getattr(latest, "remove", None)
    if callable(remove_latest):
        remove_latest(camera_id)
    registry = getattr(app.state, "camera_registry", None)
    if registry is not None:
        registry.pop(camera_id, None)


def _record_frame_received(
    app,
    *,
    camera_id: str,
    timestamp: float,
    received_at: float,
    mode: str,
    frame: np.ndarray,
    processing_pending: bool,
) -> None:
    registry = getattr(app.state, "camera_registry", None)
    if registry is None:
        registry = CameraRegistry(max_cameras=32)
        app.state.camera_registry = registry
    previous = dict(registry.get(camera_id, {}))
    stored_calibration = app.state.calibration_store.get(camera_id)
    calibration_warning = (
        stored_calibration.compatibility_warning(
            int(frame.shape[1]),
            int(frame.shape[0]),
        )
        if stored_calibration is not None else None
    )
    previous.update({
        "camera_id": camera_id,
        # Freshness always follows server receipt time, never a phone clock.
        "last_seen": received_at,
        "captured_at": timestamp,
        "mode": mode,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "frames_received": int(previous.get("frames_received", 0)) + 1,
        "processing_pending": processing_pending,
        "processing_error": None,
        "calibration_active": stored_calibration is not None,
        "calibration_valid": (
            stored_calibration is not None and calibration_warning is None
        ),
        "calibration_warning": calibration_warning,
    })
    registry[camera_id] = previous


def _record_frame_processed(
    app,
    job: FrameJob,
    result: FrameResultOut,
) -> bool:
    payload = job.payload
    if not isinstance(payload, _FramePayload):
        return False
    registry = getattr(app.state, "camera_registry", None)
    if registry is None:
        registry = CameraRegistry(max_cameras=32)
        app.state.camera_registry = registry
    previous = dict(registry.get(job.camera_id, {}))
    is_current_receipt = (
        payload.received_at
        >= float(previous.get("last_seen", payload.received_at) or 0.0)
    )
    is_newest_processed = (
        payload.received_at
        >= float(
            previous.get(
                "last_processed_received_at",
                payload.received_at,
            )
            or 0.0
        )
    )
    previous.update({
        "camera_id": job.camera_id,
        "frames": int(previous.get("frames", 0)) + 1,
    })
    if is_newest_processed:
        previous.update({
            "last_processed_at": time.time(),
            "last_processed_received_at": payload.received_at,
            "processed_captured_at": job.timestamp,
            "processed_mode": result.mode,
            "processing_ms": result.processing_ms,
            "processing_error": None,
        })
        if is_current_receipt:
            previous.update({
                "processing_pending": False,
                "calibration_active": result.calibration_active,
                "calibration_valid": result.calibration_valid,
                "calibration_warning": result.calibration_warning,
            })
    registry[job.camera_id] = previous
    return is_newest_processed


async def record_frame_failure(
    app,
    job: FrameJob,
    exc: BaseException,
) -> None:
    """Clear a current pending flag and retain bounded operational evidence."""

    payload = job.payload
    if not isinstance(payload, _FramePayload):
        return
    if payload.performance_trace is not None:
        payload.performance_trace.finish()
    registry = getattr(app.state, "camera_registry", None)
    if registry is None:
        registry = CameraRegistry(max_cameras=32)
        app.state.camera_registry = registry
    previous = dict(registry.get(job.camera_id, {}))
    is_current_receipt = (
        payload.received_at
        >= float(previous.get("last_seen", payload.received_at) or 0.0)
    )
    is_newest_failure = (
        payload.received_at
        >= float(
            previous.get(
                "last_failed_received_at",
                payload.received_at,
            )
            or 0.0
        )
    )
    if is_newest_failure:
        error_text = f"{type(exc).__name__}: {exc}"
        previous.update({
            "camera_id": job.camera_id,
            "last_failed_at": time.time(),
            "last_failed_received_at": payload.received_at,
            "processing_error": error_text[:500],
        })
        if is_current_receipt:
            previous["processing_pending"] = False
        registry[job.camera_id] = previous
    logger.error(
        "Frame processing failed for camera %s",
        job.camera_id,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


async def publish_frame_result(
    app,
    job: FrameJob,
    result: FrameResultOut,
    extra_metadata: dict | None = None,
) -> None:
    """Publish processed metadata and the matching JPEG on the event loop."""

    payload = job.payload
    performance_trace = (
        payload.performance_trace
        if isinstance(payload, _FramePayload) else None
    )
    if not _record_frame_processed(app, job, result):
        # A newer result was already published for this camera (possible
        # during a short async→sync compatibility transition).
        if performance_trace is not None:
            performance_trace.set("websocket_send_ms", 0.0)
            performance_trace.finish()
        return
    manager = app.state.ws_manager
    data = result.model_dump()
    if extra_metadata:
        data.update(extra_metadata)
    send_started: float | None = None
    try:
        jpeg_bytes = (
            payload.preview_jpeg_bytes
            if isinstance(payload, _FramePayload) else None
        )
        if (
            isinstance(payload, _FramePayload)
            and payload.preview_jpeg_future is not None
        ):
            encoded_preview = await asyncio.wrap_future(
                payload.preview_jpeg_future
            )
            jpeg_bytes = encoded_preview.jpeg_bytes
            if performance_trace is not None:
                performance_trace.set(
                    "jpeg_encode_ms",
                    encoded_preview.encode_ms,
                )
        send_started = time.perf_counter()
        if hasattr(manager, "broadcast_frame"):
            await manager.broadcast_frame(data, jpeg_bytes)
        else:
            # Small integrations and older tests may expose only this method.
            await manager.broadcast_json(data)
    finally:
        if performance_trace is not None:
            performance_trace.set(
                "websocket_send_ms",
                (
                    (time.perf_counter() - send_started) * 1000.0
                    if send_started is not None else 0.0
                ),
            )
            performance_trace.finish()


@router.post(
    "/frame",
    response_model=FrameResultOut,
    responses={
        202: {
            "model": FrameAcceptedOut,
            "description": "Frame stored in the bounded background queue",
        },
    },
)
async def receive_frame(
    request: Request,
    image: UploadFile = File(...),
    camera_id: str = Form(default="cam_default"),
    timestamp: float = Form(default=None),
    mode: str = Form(default="site"),
    include_frame: bool = Form(default=True),
    async_processing: bool = Form(default=False),
):
    profiler = getattr(request.app.state, "performance_profiler", None)
    performance_trace = (
        profiler.new_trace(source="phone") if profiler is not None else None
    )
    prepare_started = time.perf_counter()
    raw = await image.read(MAX_FRAME_UPLOAD_BYTES + 1)
    if len(raw) > MAX_FRAME_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"Frame upload exceeds {MAX_FRAME_UPLOAD_BYTES} bytes",
        )
    arr = np.frombuffer(raw, np.uint8)
    frame = await asyncio.to_thread(cv2.imdecode, arr, cv2.IMREAD_COLOR)
    if frame is None:
        return FrameResultOut(
            frame_id=0, timestamp=0, camera_id=camera_id, mode=mode,
            detections=[], frame_jpeg_b64="", processing_ms=0,
        )
    if frame.nbytes > MAX_DECODED_FRAME_BYTES:
        raise HTTPException(
            413,
            "Decoded frame is too large: "
            f"{frame.nbytes} bytes exceeds {MAX_DECODED_FRAME_BYTES}",
        )

    # Device timestamps describe the captured event, but freshness/online
    # checks must use the server clock: phone clocks can drift or be spoofed.
    received_at = time.time()
    now = timestamp if timestamp is not None else received_at
    latest_frame_store = getattr(request.app.state, "latest_frame_store", None)
    if latest_frame_store is not None:
        stored = await asyncio.to_thread(
            latest_frame_store.set,
            camera_id,
            frame,
            now,
            mode,
            received_at=received_at,
            image_bytes=raw,
            content_type=image.content_type or "application/octet-stream",
        )
        if not stored:
            # Never advertise/process a camera frame that calibration could
            # not retain; otherwise /api/cameras and /from-latest disagree.
            raise HTTPException(
                413,
                "Frame exceeds the latest-frame store limit and was rejected",
            )
    # Preserve the uploaded JPEG; overlays are rendered by the client.
    jpeg_started = time.perf_counter()
    if async_processing:
        # Publish JPEG bytes directly to binary WebSocket clients.
        preview_jpeg_bytes = await asyncio.to_thread(
            _live_preview_jpeg_bytes,
            raw,
            frame,
        )
        preview_jpeg_b64 = ""
    else:
        preview_jpeg_bytes, preview_jpeg_b64 = await asyncio.to_thread(
            _live_preview_jpeg,
            raw,
            frame,
        )
    jpeg_ms = (time.perf_counter() - jpeg_started) * 1000.0
    if performance_trace is not None:
        performance_trace.set("jpeg_encode_ms", jpeg_ms)
    effective_mode = (
        "checkpoint"
        if mode == "checkpoint" and request.app.state.ppe_detector is not None
        else "site"
    )
    job = FrameJob(
        camera_id=camera_id,
        frame=frame,
        timestamp=now,
        payload=_FramePayload(
            mode=effective_mode,
            received_at=received_at,
            preview_jpeg_bytes=preview_jpeg_bytes,
            preview_jpeg_b64=preview_jpeg_b64,
            performance_trace=performance_trace,
        ),
    )

    if async_processing:
        processor = getattr(request.app.state, "frame_processor", None)
        if processor is None:
            _record_frame_received(
                request.app,
                camera_id=camera_id,
                timestamp=now,
                received_at=received_at,
                mode=effective_mode,
                frame=frame,
                processing_pending=False,
            )
            raise HTTPException(
                503,
                "Background frame processor is not initialized",
            )

        submission = await processor.submit(job)
        if performance_trace is not None:
            performance_trace.set(
                "frame_prepare_ms",
                max(
                    0.0,
                    (time.perf_counter() - prepare_started) * 1000.0 - jpeg_ms,
                ),
            )
            performance_trace.mark_queued()
            if submission.replaced:
                performance_trace.dropped_frames += 1
                profiler.record_drop("replaced_pending")
        _record_frame_received(
            request.app,
            camera_id=camera_id,
            timestamp=now,
            received_at=received_at,
            mode=effective_mode,
            frame=frame,
            processing_pending=submission.accepted,
        )
        if not submission.accepted:
            if performance_trace is not None:
                performance_trace.dropped_frames += 1
                profiler.record_drop(submission.reason or "rejected")
                performance_trace.finish(has_person=None)
            raise HTTPException(
                503,
                f"Background frame processor rejected the frame: "
                f"{submission.reason or 'capacity unavailable'}",
            )
        stats = processor.stats
        accepted = FrameAcceptedOut(
            camera_id=camera_id,
            timestamp=now,
            mode=effective_mode,
            queue_depth=submission.queue_depth,
            replaced_pending=submission.replaced,
            dropped_frames=stats.replaced + stats.dropped,
        )
        return JSONResponse(
            status_code=202,
            content=accepted.model_dump(mode="json"),
        )

    if performance_trace is not None:
        performance_trace.set(
            "frame_prepare_ms",
            max(
                0.0,
                (time.perf_counter() - prepare_started) * 1000.0 - jpeg_ms,
            ),
        )
        performance_trace.mark_queued()
    _record_frame_received(
        request.app,
        camera_id=camera_id,
        timestamp=now,
        received_at=received_at,
        mode=effective_mode,
        frame=frame,
        processing_pending=True,
    )
    try:
        result = await asyncio.to_thread(process_frame_job, request.app, job)
    except Exception as exc:
        await record_frame_failure(request.app, job, exc)
        raise HTTPException(500, "Frame processing failed") from exc
    await publish_frame_result(request.app, job, result)
    if include_frame:
        return result

    # The phone already owns the uploaded pixels and only needs detections and
    # alerts from this HTTP response.  The complete frame is still broadcast to
    # the live panel over WebSocket, so omitting this duplicate base64 payload
    # substantially reduces response bandwidth at higher capture rates.
    return result.model_copy(update={"frame_jpeg_b64": ""})
