#!/usr/bin/env python3
"""Run the recorded site clips through a Perimetr instance and freeze the result.

The demo UI must not depend on live inference: on a laptop at a pitch there is
no camera, no GPU and no tolerance for a model that decides to miss the one
event everybody came to see. So every clip is processed once, ahead of time,
and what ships is the annotated video plus the alerts the model actually
raised, with each alert pinned to a second of that clip.

Frames decoded by the demo pipeline enter inference stamped with
``run_started_wall + position in the clip``, and the alert rules copy that
stamp onto the records they emit. Subtracting the run anchor therefore yields
the exact position in the clip -- no polling, no clock correlation.

Each clip gets its own camera_id so its detection profile is independent: the
gate clips run PPE and worker identification, the smoking and fall clips run
pose and behaviour, the machine clips run the dynamic danger zone. Nothing
computes what its clip does not need.

Usage:
    python tools/bake_demo_clips.py --base-url https://host --token TOKEN \
        --clips-dir ~/Movies/Perimetr/pociete --out-dir frontend-demo
    python tools/bake_demo_clips.py ... --only palenie --only upadki
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import urllib.error
import urllib.parse
import urllib.request


# ---------------------------------------------------------------- clip profiles

@dataclass(frozen=True)
class ClipProfile:
    """One recorded clip and the detection modules its scenario needs."""

    key: str
    filename: str
    title: str
    mode: str = "site"
    modules: dict[str, bool] = field(default_factory=dict)

    @property
    def camera_id(self) -> str:
        return f"demo_{self.key}"


_ALL_MODULES = ("boxes", "posture", "zones", "markers", "distances", "worker_id", "ppe")

_RECORDING_STAMP = re.compile(r"^(?P<camera>[a-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{6})")


def _recording_origin(filename: str) -> tuple[str, str]:
    """Pull camera and recording time out of the recorder's file name.

    The panel shows both next to every clip and every history entry, so they
    have to come from the recording itself rather than be typed in twice.
    """
    match = _RECORDING_STAMP.match(filename)
    if not match:
        return "cam", ""
    date = match.group("date")
    clock = match.group("time")
    day, month, year = date[8:], date[5:7], date[:4]
    return match.group("camera"), f"{day}.{month}.{year} {clock[:2]}:{clock[2:4]}"


def _profile(**enabled: bool) -> dict[str, bool]:
    """Explicitly disable everything not asked for; defaults are all-on."""
    return {name: bool(enabled.get(name, False)) for name in _ALL_MODULES}


CLIPS: tuple[ClipProfile, ...] = (
    ClipProfile(
        key="bramka_01",
        filename="cam111_2026-07-30_123349__00m25s-01m00s__bramka_bez_ppe.mp4",
        title="Bramka wejściowa — brak kasku i kamizelki",
        mode="checkpoint",
        modules=_profile(boxes=True, ppe=True, worker_id=True),
    ),
    ClipProfile(
        key="bramka_02",
        filename="cam111_2026-07-30_123349__01m08s-01m38s__bramka_bez_ppe.mp4",
        title="Bramka wejściowa — kolejne wejścia bez PPE",
        mode="checkpoint",
        modules=_profile(boxes=True, ppe=True, worker_id=True),
    ),
    ClipProfile(
        key="palenie",
        filename="cam111_2026-07-30_123349__02m30s-03m00s__palenie.mp4",
        title="Palenie na terenie budowy",
        modules=_profile(boxes=True, posture=True),
    ),
    ClipProfile(
        key="upadki",
        filename="cam111_2026-07-30_123705__01m30s-02m15s__upadki.mp4",
        title="Upadek pracownika",
        modules=_profile(boxes=True, posture=True),
    ),
    ClipProfile(
        key="pose_i_id",
        filename="cam111_2026-07-30_123705__02m35s-03m00s__dwie_osoby_pose_i_id.mp4",
        title="Dwie osoby — pozycje i identyfikacja",
        modules=_profile(boxes=True, posture=True, worker_id=True, markers=True),
    ),
    ClipProfile(
        key="koparka_statyczna",
        filename="cam111_2026-07-30_125000__00m00s-00m40s__statyczna_koparka_strefa.mp4",
        title="Strefa wokół koparki — wiele osób w kadrze",
        modules=_profile(boxes=True, zones=True, distances=True),
    ),
    ClipProfile(
        key="ruchomy_cel",
        filename="cam111_2026-07-30_125349__00m24s-00m55s__ruchomy_cel_strefa.mp4",
        title="Strefa ruchomej maszyny",
        modules=_profile(boxes=True, zones=True, distances=True),
    ),
    ClipProfile(
        key="osoba_na_koparce",
        filename="cam111_2026-07-30_125349__01m08s-01m35s__osoba_na_koparce_w_strefie.mp4",
        title="Osoba na stopniu koparki w strefie",
        modules=_profile(boxes=True, zones=True, distances=True),
    ),
    ClipProfile(
        key="barierka",
        filename="cam111_2026-07-30_125646__00m29s-01m00s__osoby_przy_barierce.mp4",
        title="Osoby przy barierce",
        modules=_profile(boxes=True, zones=True, distances=True),
    ),
    ClipProfile(
        key="kalibracja",
        filename="cam111_2026-07-30_130156__00m00s-00m46s__widok_z_gory_kalibracja.mp4",
        title="Widok z góry — kalibracja placu",
        modules=_profile(boxes=True, markers=True),
    ),
    ClipProfile(
        key="fragment",
        filename="cam111_2026-07-30_123051__01m24s-02m14s__fragment.mp4",
        title="Plac budowy — obraz ciągły",
        modules=_profile(boxes=True),
    ),
)


# ------------------------------------------------------------------ http client

def _default_ssl_context() -> ssl.SSLContext:
    """Some python.org builds ship without a trust store; fall back to certifi."""
    context = ssl.create_default_context()
    if context.cert_store_stats().get("x509_ca"):
        return context
    try:
        import certifi
    except ImportError:
        return context
    return ssl.create_default_context(cafile=certifi.where())


class Api:
    """Minimal client for the Perimetr API, authenticated by the demo token."""

    def __init__(self, base_url: str, token: str, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.ssl_context = _default_ssl_context()

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        query = {"demo": self.token, **(params or {})}
        return f"{self.base}{path}?{urllib.parse.urlencode(query)}"

    def _send(self, request: urllib.request.Request, timeout: float | None = None):
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout, context=self.ssl_context) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"{exc.code} {request.get_method()} {request.full_url.split('?')[0]}: {detail}") from exc
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload

    def get(self, path: str, params: dict[str, Any] | None = None):
        return self._send(urllib.request.Request(self._url(path, params)))

    def get_bytes(self, path: str) -> bytes:
        with urllib.request.urlopen(self._url(path), timeout=self.timeout, context=self.ssl_context) as response:
            return response.read()

    def post(self, path: str, body: Any = None, method: str = "POST"):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        return self._send(urllib.request.Request(self._url(path), data=data, headers=headers, method=method))

    def upload(self, path: str, file_path: Path, fields: dict[str, str]):
        """Multipart upload; the clips are ~100 MB so the body is streamed as one buffer."""
        boundary = "----perimetr-bake-boundary"
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
            )
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; "
            f"filename=\"{file_path.name}\"\r\nContent-Type: video/mp4\r\n\r\n".encode()
        )
        parts.append(file_path.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        body = b"".join(parts)
        request = urllib.request.Request(
            self._url(path),
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            method="POST",
        )
        return self._send(request, timeout=max(self.timeout, 600.0))


# ----------------------------------------------------------------------- baking

TERMINAL_STATUSES = {"finished", "stopped", "failed", "error"}


def bake_clip(api: Api, clip: ClipProfile, clips_dir: Path, out_dir: Path, poll: float) -> dict[str, Any]:
    source = clips_dir / clip.filename
    if not source.is_file():
        raise FileNotFoundError(f"Missing clip: {source}")

    print(f"\n=== {clip.key}: {clip.title}")
    print(f"    modules: {', '.join(k for k, v in clip.modules.items() if v) or 'none'}")

    api.post(f"/api/runtime/{clip.camera_id}", clip.modules, method="PATCH")

    size_mb = source.stat().st_size / (1024 * 1024)
    print(f"    upload {size_mb:.0f} MB ...", flush=True)
    started = time.monotonic()
    job = api.upload(
        "/api/demo-videos",
        source,
        {"camera_id": clip.camera_id, "mode": clip.mode, "playback_mode": "fast"},
    )
    job_id = job["job_id"]
    print(f"    job {job_id} ({time.monotonic() - started:.0f} s, {job.get('total_frames')} frames)")

    # export() restarts the job and runs every frame through inference while
    # writing the annotated MP4, so it replaces a separate play/export pair.
    snapshot = api.post(f"/api/demo-videos/{job_id}/export")
    anchor = float(snapshot.get("run_started_wall") or 0.0)
    if not anchor:
        raise RuntimeError(
            "Instance does not expose run_started_wall — deploy the demo branch, "
            "otherwise alerts cannot be pinned to a position in the clip."
        )

    processing_started = time.monotonic()
    while True:
        time.sleep(poll)
        snapshot = api.get(f"/api/demo-videos/{job_id}")
        status = str(snapshot.get("status", ""))
        export_status = str(snapshot.get("export_status", ""))
        current = float(snapshot.get("current_time_sec") or 0.0)
        duration = snapshot.get("duration_sec")
        elapsed = time.monotonic() - processing_started
        print(
            f"    {status}/{export_status} — {current:.1f}s"
            f"{f' / {duration:.1f}s' if duration else ''}"
            f" (wall {elapsed / 60:.1f} min, {snapshot.get('processing_fps', 0):.2f} fps)",
            flush=True,
        )
        if status == "failed" or export_status == "failed":
            raise RuntimeError(f"Job failed: {snapshot.get('error') or 'unknown reason'}")
        if export_status == "ready" or status in TERMINAL_STATUSES:
            break

    alerts = [
        record
        for record in (api.get("/api/alerts", {"limit": 500, "since": anchor - 0.5}) or [])
        if record.get("camera_id") == clip.camera_id
    ]
    events = sorted(
        (
            {
                "t": round(max(0.0, float(record["timestamp"]) - anchor), 2),
                "severity": record["severity"],
                "kind": record["kind"],
                "rule": record["rule_name"],
                "description": record["description"],
            }
            for record in alerts
        ),
        key=lambda event: event["t"],
    )
    print(f"    {len(events)} alerts: " + (", ".join(f"{e['t']}s {e['kind']}" for e in events[:6]) or "none"))

    video_name = f"{clip.key}.mp4"
    if snapshot.get("export_status") == "ready":
        (out_dir / "clips").mkdir(parents=True, exist_ok=True)
        (out_dir / "clips" / video_name).write_bytes(api.get_bytes(f"/api/demo-videos/{job_id}/output"))
        print(f"    saved clips/{video_name}")
    else:
        video_name = None
        print("    WARNING: no annotated export, falling back to the source clip")

    api.post(f"/api/demo-videos/{job_id}", method="DELETE")

    camera, recorded = _recording_origin(clip.filename)
    return {
        "key": clip.key,
        "title": clip.title,
        "camera": camera,
        "recorded": recorded,
        "src": f"clips/{video_name}" if video_name else f"clips/{clip.filename}",
        "duration_sec": snapshot.get("duration_sec"),
        "modules": [name for name, on in clip.modules.items() if on],
        "mode": clip.mode,
        "alerts": events,
    }


# Kinds that stay in bake-report.json but never reach the panel:
#  - posture_anomaly: heuristic gait rules fire almost continuously on ordinary
#    site movement — a worker bending over pipes reads as "unstable trajectory".
#  - unidentified_worker: fires for everyone without an ArUco tag, which on
#    these recordings is everyone, and QR identification is not what the demo
#    is showing.
MUTED_KINDS = ("posture_anomaly", "unidentified_worker")


def dedupe_alerts(
    alerts: list[dict[str, Any]],
    window: float,
    muted: tuple[str, ...] = MUTED_KINDS,
) -> list[dict[str, Any]]:
    """Collapse repeats of the same alert kind within `window` seconds.

    A person standing near a working excavator re-triggers the proximity rule
    for as long as they stand there — 85 alerts on one 40-second clip. Every one
    is true, but a history panel scrolling past 85 identical lines shows noise
    where the point is that the system caught the hazard. The full set stays in
    bake-report.json; this only thins what the demo lists.
    """
    alerts = [a for a in alerts if a["kind"] not in muted]
    if window <= 0:
        return alerts
    kept: list[dict[str, Any]] = []
    last_seen: dict[str, float] = {}
    for alert in alerts:
        kind = alert["kind"]
        if kind not in last_seen or alert["t"] - last_seen[kind] >= window:
            kept.append(alert)
            last_seen[kind] = alert["t"]
    return kept


def write_clips_js(
    results: list[dict[str, Any]],
    out_dir: Path,
    dedupe_window: float,
    muted: tuple[str, ...] = MUTED_KINDS,
) -> Path:
    """Emit the demo's clip manifest, keeping the shape demo.js already reads."""
    by_key = {clip.key: clip for clip in CLIPS}
    entries = []
    for result in results:
        if not result.get("recorded"):
            # Reports written before the origin fields existed, or rebuilt with
            # --from-report: recover them from the clip definition.
            source = by_key.get(result["key"])
            if source is not None:
                camera, recorded = _recording_origin(source.filename)
                result = {**result, "camera": camera, "recorded": recorded}
        alerts = ",\n".join(
            "      { t: %s, severity: %s, kind: %s, rule: %s, description: %s }"
            % (
                event["t"],
                json.dumps(event["severity"]),
                json.dumps(event["kind"]),
                json.dumps(event["rule"]),
                json.dumps(event["description"], ensure_ascii=False),
            )
            for event in dedupe_alerts(result["alerts"], dedupe_window, muted)
        )
        entries.append(
            "  {\n"
            f"    id: {json.dumps(result['key'])},\n"
            f"    title: {json.dumps(result['title'], ensure_ascii=False)},\n"
            f"    src: {json.dumps(result['src'])},\n"
            f"    camera: {json.dumps(result.get('camera', 'cam'))},\n"
            f"    recorded: {json.dumps(result.get('recorded', ''))},\n"
            f"    duration: {result.get('duration_sec') or 0},\n"
            f"    alerts: [\n{alerts}\n    ],\n"
            "  }"
        )
    # index.html loads this with a plain <script> tag, so it must stay a
    # classic script defining the global demo.js reads — not an ES module.
    body = (
        "// Generated by tools/bake_demo_clips.py — do not edit by hand.\n"
        "// Alert times come from the model itself: each one is the second of the\n"
        "// clip at which Perimetr raised it during the offline run.\n"
        "const DEMO_CLIPS = [\n" + ",\n".join(entries) + ",\n];\n"
    )
    path = out_dir / "clips.js"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True, help="DEMO_TOKEN of the target instance")
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[], help="Bake only these clip keys (repeatable)")
    parser.add_argument("--poll", type=float, default=15.0, help="Seconds between progress checks")
    parser.add_argument(
        "--dedupe-window", type=float, default=5.0,
        help="Collapse repeats of one alert kind within this many seconds (0 disables)",
    )
    parser.add_argument(
        "--keep-kind", action="append", default=[],
        help=f"Show an otherwise muted alert kind ({', '.join(MUTED_KINDS)})",
    )
    parser.add_argument(
        "--from-report", action="store_true",
        help="Rebuild clips.js from an existing bake-report.json without reprocessing video",
    )
    args = parser.parse_args()

    selected = [clip for clip in CLIPS if not args.only or clip.key in args.only]
    if not selected:
        print(f"No clip matches {args.only}; known keys: {', '.join(c.key for c in CLIPS)}", file=sys.stderr)
        return 2

    muted = tuple(k for k in MUTED_KINDS if k not in args.keep_kind)

    if args.from_report:
        report = json.loads((args.out_dir / "bake-report.json").read_text())
        manifest = write_clips_js(report, args.out_dir, args.dedupe_window, muted)
        shown = sum(len(dedupe_alerts(c["alerts"], args.dedupe_window, muted)) for c in report)
        total = sum(len(c["alerts"]) for c in report)
        print(f"{len(report)} clips, {shown} of {total} alerts listed → {manifest}")
        return 0

    api = Api(args.base_url, args.token)
    health = api.get("/api/readiness")
    print("Instance:", args.base_url)
    components = health.get("components", {})
    for name in ("ppe_checkpoint", "posture_analysis", "learned_behavior_classifier",
                 "secondary_smoking_phone_classifier", "metric_calibration"):
        print(f"  {'ok ' if components.get(name) else 'OFF'} {name}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for clip in selected:
        try:
            results.append(bake_clip(api, clip, args.clips_dir, args.out_dir, args.poll))
        except Exception as exc:  # one bad clip must not discard the rest
            print(f"    ERROR {clip.key}: {exc}", file=sys.stderr)

    if not results:
        return 1

    (args.out_dir / "bake-report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = write_clips_js(results, args.out_dir, args.dedupe_window, muted)
    total = sum(len(result["alerts"]) for result in results)
    shown = sum(len(dedupe_alerts(r["alerts"], args.dedupe_window, muted)) for r in results)
    print(f"\n{len(results)} clips, {shown} of {total} alerts listed → {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
