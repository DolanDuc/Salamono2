import base64
import time
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from backend.calibration import Calibration
from backend.danger_rules import DangerEvent
from backend.detector import Detection
from backend.marker_detector import MarkerDetection
from backend.models import (
    ActiveZoneOut,
    AlarmRecord,
    AlertOut,
    AlertSeverity,
    DetectionOut,
    FrameResultOut,
    MarkerDetectionOut,
    PersonDistanceOut,
    PPECheckOut,
    ZoneBreachOut,
)
from backend.ppe_rules import PPEEvent
from backend.zone_rules import ZoneBreachEvent

SITE_RULE_DESCRIPTIONS = {
    "person_vehicle_overlap": "Osoba w strefie pojazdu",
    "person_near_vehicle": "Osoba blisko pojazdu",
}


def _site_record(alert: AlertOut, camera_id: str) -> AlarmRecord:
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
        details={
            "distance_px": alert.distance_px,
            "overlap_iou": alert.overlap_iou,
            "person_confidence": alert.person.confidence,
            "hazard_confidence": alert.hazard.confidence,
            "person_box": alert.person.box,
            "hazard_box": alert.hazard.box,
            "hazard_class": alert.hazard.class_name,
        },
    )


def _zone_record(breach: ZoneBreachOut, camera_id: str) -> AlarmRecord:
    return AlarmRecord(
        id=breach.id,
        timestamp=breach.timestamp,
        mode="site",
        kind="zone_breach",
        severity=breach.severity,
        rule_name="zone_" + breach.zone_id,
        description=f"Wejscie w strefe: {breach.zone_name}",
        camera_id=camera_id,
        thumbnail_url=breach.frame_thumbnail_url,
        details={
            "zone_id": breach.zone_id,
            "zone_name": breach.zone_name,
            "person_confidence": breach.person.confidence,
            "person_box": breach.person.box,
        },
    )


def _ppe_record(check: PPECheckOut, camera_id: str) -> AlarmRecord:
    missing_pretty = {"hardhat": "kask", "vest": "kamizelka"}
    missing_labels = [missing_pretty.get(m, m) for m in check.missing]
    if missing_labels:
        desc = "Brak PPE: " + " + ".join(missing_labels)
    else:
        desc = "PPE OK"
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
        details={
            "missing": check.missing,
            "has_hardhat": check.has_hardhat,
            "has_vest": check.has_vest,
            "person_confidence": check.person.confidence,
            "person_box": check.person.box,
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
    dbg_lines = []
    dbg_lines.append(
        f"[resolver] zones={len(zones)} markers_ids={list(by_id.keys())} "
        f"frame={frame_w}x{frame_h}"
    )
    for z in zones:
        if not z.marker_ids:
            resolved.append(z)
            dbg_lines.append(f"[resolver] PASS-THROUGH id={z.id} name={z.name!r} "
                             f"regular polygon_len={len(z.polygon)}")
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
            dbg_lines.append(
                f"[resolver] RESOLVED id={z.id} name={z.name!r} "
                f"want={list(z.marker_ids)} centres_len={len(centres)}"
            )
        else:
            cached = cache.get(cache_key)
            if cached and (now - cached[1]) < MARKER_ZONE_CACHE_TTL:
                polygon = cached[0]
                dbg_lines.append(
                    f"[resolver] CACHE-HIT id={z.id} want={list(z.marker_ids)} "
                    f"age={now - cached[1]:.2f}s"
                )
            else:
                dbg_lines.append(
                    f"[resolver] DROPPED id={z.id} name={z.name!r} "
                    f"want={list(z.marker_ids)} visible={list(by_id.keys())} "
                    f"cache_present={cache_key in cache}"
                )
                continue
        try:
            z_copy = z.model_copy()
            z_copy.polygon = polygon
            resolved.append(z_copy)
            dbg_lines.append(
                f"[resolver] APPENDED id={z_copy.id} polygon_len={len(z_copy.polygon)} "
                f"active={z_copy.active}"
            )
        except Exception as exc:
            dbg_lines.append(f"[resolver] COPY-FAIL id={z.id} exc={exc!r}")
    # Only log when there's a marker-zone involved, otherwise Railway logs
    # explode at 10fps × normal traffic.
    if any(z.marker_ids for z in zones):
        print("\n".join(dbg_lines), flush=True)
    return resolved


def _marker_to_out(m: MarkerDetection) -> MarkerDetectionOut:
    return MarkerDetectionOut(
        marker_id=m.marker_id,
        corners=[[float(x), float(y)] for x, y in m.corners],
        center=[float(m.center[0]), float(m.center[1])],
    )


def _person_distances(persons: list[Detection],
                      calibration: Calibration | None,
                      ) -> list[PersonDistanceOut]:
    if calibration is None:
        return []
    out: list[PersonDistanceOut] = []
    for p in persons:
        x1, y1, x2, y2 = p.box
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)
        d = calibration.distance_to_boundary_m(foot_x, foot_y)
        out.append(PersonDistanceOut(
            person_box=[int(x1), int(y1), int(x2), int(y2)],
            distance_m=round(d, 2),
            inside=d <= 0.0,
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
        if d.inside:
            label = f"WEWNATRZ ({abs(d.distance_m):.1f} m od granicy)"
            color = COLOR_DANGER
        else:
            label = f"{d.distance_m:.1f} m od strefy"
            color = COLOR_OK if d.distance_m > 1.5 else COLOR_WARNING
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


def _annotate_zones(frame: np.ndarray, zones: list,
                    active_breaches: list[ZoneBreachEvent]) -> np.ndarray:
    if not zones:
        return frame
    h, w = frame.shape[:2]
    breached_ids = {b.zone.id for b in active_breaches}
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
        cv2.fillPoly(overlay, [pts], base_color)
        alpha = 0.35 if breached else 0.15
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        border = COLOR_DANGER if breached else base_color
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


def _annotate_site(frame: np.ndarray, detections: list[Detection],
                   raw_dangers: list[DangerEvent],
                   confirmed: list[DangerEvent]) -> np.ndarray:
    out = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d.box
        color = COLOR_PERSON if d.category == "person" else COLOR_VEHICLE
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

    for e in confirmed:
        overlay = out.copy()
        cv2.rectangle(overlay,
                      (e.person.box[0], e.person.box[1]),
                      (e.person.box[2], e.person.box[3]),
                      COLOR_DANGER, -1)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

    if confirmed:
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), COLOR_DANGER, -1)
        cv2.putText(out, f"ALARM — {len(confirmed)} danger(s)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    return out


def _annotate_ppe(frame: np.ndarray, detections: list[Detection],
                  events: list[PPEEvent]) -> np.ndarray:
    out = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d.box
        if d.category == "person":
            color = COLOR_PERSON
        elif d.category == "hardhat":
            color = COLOR_HARDHAT
        elif d.category == "vest":
            color = COLOR_VEST
        else:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    banner_color = None
    banner_text = None
    for e in events:
        if not e.confirmed:
            continue
        color = COLOR_DANGER if e.missing else COLOR_OK
        overlay = out.copy()
        cv2.rectangle(overlay,
                      (e.person.box[0], e.person.box[1]),
                      (e.person.box[2], e.person.box[3]),
                      color, -1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
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


async def _handle_site(request: Request, frame: np.ndarray, now: float,
                        t0: float, camera_id: str) -> FrameResultOut:
    # DEBUG canary — if this line never appears in Railway logs even though
    # POST /api/frame returns 200, then Railway is running a stale image
    # that predates the resolver code.
    print(f"[site] handling frame cam={camera_id} at {now:.1f}", flush=True)
    detector = request.app.state.detector
    danger_detector = request.app.state.danger_detector
    temporal_filter = request.app.state.temporal_filter
    frame_store = request.app.state.frame_store
    alert_store = request.app.state.alert_store
    zone_store = request.app.state.zone_store
    zone_detector = request.app.state.zone_detector
    zone_temporal_filter = request.app.state.zone_temporal_filter
    marker_detector = request.app.state.marker_detector
    calibration_store = request.app.state.calibration_store

    detections = detector.detect(frame)
    raw_dangers = danger_detector.evaluate(detections, now)
    confirmed = temporal_filter.update(raw_dangers, now)

    zones = zone_store.for_camera(camera_id)
    fh, fw = frame.shape[:2]

    markers = marker_detector.detect(frame)
    # Marker-defined zones get their polygon recomputed from live markers
    # before running breach evaluation.
    zones = _resolve_marker_zones(
        zones, markers, camera_id,
        request.app.state.marker_zone_cache,
        now, fw, fh,
    )
    raw_zone_breaches = zone_detector.evaluate(
        detections, zones, fw, fh, now,
    )
    confirmed_zone_breaches = zone_temporal_filter.update(raw_zone_breaches, now)

    calibration = calibration_store.get(camera_id)
    persons = [d for d in detections if d.category == "person"]
    person_distances = _person_distances(persons, calibration)

    annotated = _annotate_zones(frame, zones, raw_zone_breaches)
    annotated = _annotate_site(annotated, detections, raw_dangers, confirmed)
    calibration_ids = set(calibration.marker_ids) if calibration else set()
    annotated = _annotate_markers(annotated, markers, calibration_ids)
    annotated = _annotate_person_distances(annotated, person_distances)

    alert_outs = []
    for evt in confirmed:
        alert_id = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(annotated, alert_id)
        alert = _event_to_alert(evt, alert_id, thumb_url)
        alert_outs.append(alert)
        alert_store.append(_site_record(alert, camera_id))

    zone_breach_outs = []
    for zevt in confirmed_zone_breaches:
        bid = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(annotated, bid)
        bout = _zone_to_out(zevt, bid, thumb_url)
        zone_breach_outs.append(bout)
        alert_store.append(_zone_record(bout, camera_id))

    active_outs = [_event_to_alert(e, "active") for e in raw_dangers]
    active_zone_outs = [_zone_to_out(z, "active") for z in raw_zone_breaches]

    # Ship the resolved zones so the frontend can draw marker-zone polygons
    # (their stored polygon is empty; the live one only exists in memory).
    active_zones_out = [
        ActiveZoneOut(
            id=z.id,
            name=z.name,
            severity=z.severity,
            polygon=z.polygon,
            marker_ids=list(z.marker_ids or []),
        )
        for z in zones
        if z.active and z.polygon and len(z.polygon) >= 3
    ]

    _, jpeg_buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(jpeg_buf).decode()

    processing_ms = (time.monotonic() - t0) * 1000
    request.app.state.frame_counter += 1

    return FrameResultOut(
        frame_id=request.app.state.frame_counter,
        timestamp=now,
        mode="site",
        detections=[_det_to_out(d) for d in detections],
        active_dangers=active_outs,
        confirmed_alerts=alert_outs,
        active_zone_breaches=active_zone_outs,
        confirmed_zone_breaches=zone_breach_outs,
        markers=[_marker_to_out(m) for m in markers],
        person_distances=person_distances,
        active_zones=active_zones_out,
        calibration_active=calibration is not None,
        frame_jpeg_b64=b64,
        processing_ms=round(processing_ms, 1),
    )


async def _handle_checkpoint(request: Request, frame: np.ndarray, now: float,
                              t0: float, camera_id: str) -> FrameResultOut:
    detector = request.app.state.ppe_detector
    checker = request.app.state.ppe_checker
    frame_store = request.app.state.frame_store
    alert_store = request.app.state.alert_store

    detections = detector.detect(frame)
    events = checker.evaluate(detections, frame_h=frame.shape[0],
                              frame_timestamp=now)
    confirmed = checker.confirm(events, now)
    annotated = _annotate_ppe(frame, detections, confirmed)

    check_outs = []
    for evt in confirmed:
        cid = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(annotated, cid)
        out = _ppe_to_out(evt, cid, thumb_url)
        check_outs.append(out)
        if evt.missing:
            alert_store.append(_ppe_record(out, camera_id))

    _, jpeg_buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(jpeg_buf).decode()

    processing_ms = (time.monotonic() - t0) * 1000
    request.app.state.frame_counter += 1

    return FrameResultOut(
        frame_id=request.app.state.frame_counter,
        timestamp=now,
        mode="checkpoint",
        detections=[_det_to_out(d) for d in detections],
        ppe_checks=check_outs,
        frame_jpeg_b64=b64,
        processing_ms=round(processing_ms, 1),
    )


@router.post("/frame", response_model=FrameResultOut)
async def receive_frame(
    request: Request,
    image: UploadFile = File(...),
    camera_id: str = Form(default="cam_default"),
    timestamp: float = Form(default=None),
    mode: str = Form(default="site"),
):
    t0 = time.monotonic()

    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return FrameResultOut(
            frame_id=0, timestamp=0, mode=mode,
            detections=[], frame_jpeg_b64="", processing_ms=0,
        )

    now = timestamp or time.time()

    if mode == "checkpoint" and request.app.state.ppe_detector is not None:
        result = await _handle_checkpoint(request, frame, now, t0, camera_id)
    else:
        result = await _handle_site(request, frame, now, t0, camera_id)

    await request.app.state.ws_manager.broadcast_json(result.model_dump())
    return result
