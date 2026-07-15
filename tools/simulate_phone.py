"""
Simulate a phone camera by reading a video file and POSTing frames to the backend.

Usage:
    python tools/simulate_phone.py --video data/sample_videos/sample.mp4
    python tools/simulate_phone.py --video sample.mp4 --server http://localhost:8000 --fps 2
"""
import argparse
import sys
import time

import cv2
import requests


def main():
    parser = argparse.ArgumentParser(description="Phone camera simulator")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="Backend URL")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frames per second to send")
    parser.add_argument("--loop", action="store_true",
                        help="Loop video continuously")
    parser.add_argument("--camera-id", default="simulator",
                        help="Camera ID to send")
    parser.add_argument("--timestamp-offset", type=float, default=0.0,
                        help="Seconds added to sent timestamps — simulates "
                             "phone clock skew (fusion must tolerate it, "
                             "TTL runs on server time)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open '{args.video}'")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(src_fps / args.fps))
    interval = 1.0 / args.fps

    print(f"Video: {args.video} ({total} frames @ {src_fps:.1f}fps)")
    print(f"Sending to {args.server} at {args.fps} fps (skip={skip})")
    print(f"Loop: {args.loop}")
    print()

    frame_idx = 0
    sent = 0
    total_alerts = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                print("--- Looping video ---")
                continue
            else:
                break

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        t0 = time.monotonic()
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        try:
            resp = requests.post(
                f"{args.server}/api/frame",
                files={"image": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
                data={
                    "camera_id": args.camera_id,
                    "timestamp": str(time.time() + args.timestamp_offset),
                },
                timeout=10,
            )
            data = resp.json()
            n_det = len(data.get("detections", []))
            n_active = len(data.get("active_dangers", []))
            n_confirmed = len(data.get("confirmed_alerts", []))
            proc_ms = data.get("processing_ms", 0)
            total_alerts += n_confirmed
            sent += 1

            status = ""
            if n_confirmed > 0:
                status = " *** ALARM ***"
            elif n_active > 0:
                status = " (danger pending)"

            print(f"  [{sent:4d}] Frame {frame_idx:5d} | "
                  f"Dets: {n_det:2d} | "
                  f"Active: {n_active} | "
                  f"Alerts: {n_confirmed} | "
                  f"Proc: {proc_ms:6.1f}ms | "
                  f"Total alerts: {total_alerts}{status}")

        except requests.RequestException as e:
            print(f"  [{sent:4d}] Frame {frame_idx:5d} | ERROR: {e}")

        elapsed = time.monotonic() - t0
        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    print(f"\nDone. Sent {sent} frames, {total_alerts} total alerts.")


if __name__ == "__main__":
    main()
