#!/usr/bin/env python3
"""Cut a self-contained clip for every alert out of the annotated recording.

The backend can write evidence clips while it analyses, but that ties the clips
to a full re-run of the model and to whatever sampling rate the recorder used.
Everything needed to produce them afterwards is already on disk: the annotated
video and, for each alert, the second of that video where it fired. Cutting
here keeps the event clip identical in frame rate and appearance to the
recording it came from, and takes seconds instead of minutes per clip.

ffmpeg comes from imageio-ffmpeg, so no system package is required.

Usage:
    python tools/cut_event_clips.py --dir frontend-demo
    python tools/cut_event_clips.py --dir frontend-demo --pre 2 --post 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def ffmpeg_binary() -> str:
    """Prefer a system ffmpeg, fall back to the one bundled with imageio."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit(
            "Brak ffmpeg. Zainstaluj: pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def cut(ffmpeg: str, source: Path, start: float, duration: float, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            # -ss before -i seeks fast; re-encoding keeps the cut frame-accurate
            # instead of snapping to the previous keyframe.
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    ffmpeg: {result.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="Directory holding bake-report.json and clips/")
    parser.add_argument("--pre", type=float, default=2.0, help="Seconds of run-up before the alert")
    parser.add_argument("--post", type=float, default=5.0, help="Seconds kept after the alert")
    parser.add_argument("--only", action="append", default=[], help="Limit to these clip keys")
    parser.add_argument(
        "--dedupe-window", type=float, default=5.0,
        help="Match the manifest's collapsing so no clip is cut for an alert nobody sees",
    )
    parser.add_argument(
        "--skip-kind", action="append", default=None,
        help="Alert kinds not worth a clip (default: the kinds muted in the manifest)",
    )
    args = parser.parse_args()

    report_path = args.dir / "bake-report.json"
    if not report_path.is_file():
        print(f"Brak {report_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).parent))
    from bake_demo_clips import MUTED_KINDS, dedupe_alerts
    if args.skip_kind is None:
        args.skip_kind = list(MUTED_KINDS)

    ffmpeg = ffmpeg_binary()
    report = json.loads(report_path.read_text())
    written = 0

    for clip in report:
        if args.only and clip["key"] not in args.only:
            continue
        source = args.dir / clip["src"]
        if not source.is_file():
            print(f"{clip['key']}: brak {source}, pomijam")
            continue

        duration = float(clip.get("duration_sec") or 0.0)
        # Only alerts the panel actually lists are worth a file.
        wanted = dedupe_alerts(clip["alerts"], args.dedupe_window, tuple(args.skip_kind))
        print(f"\n{clip['key']}: {len(wanted)} zdarzeń")
        for alert in wanted:
            start = max(0.0, float(alert["t"]) - args.pre)
            end = float(alert["t"]) + args.post
            if duration:
                end = min(end, duration)
            length = max(0.5, end - start)
            name = f"{clip['key']}_{alert['t']:07.2f}".replace(".", "_") + ".mp4"
            target = args.dir / "clips" / "events" / name
            if cut(ffmpeg, source, start, length, target):
                alert["clip"] = f"clips/events/{name}"
                written += 1
                print(f"    {alert['t']:6.2f}s → {name} ({target.stat().st_size / 1048576:.1f} MB)")

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{written} klipów zdarzeń w {args.dir / 'clips' / 'events'}")
    print("Manifest przebuduj: bake_demo_clips.py --from-report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
