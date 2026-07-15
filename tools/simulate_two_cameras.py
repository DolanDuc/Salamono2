"""Deterministyczna próba generalna multi-camera fusion (bez sprzętu).

Syntetyczny plac: 4 markery ArUco (4x4_50, ID 10/20/30/40) w rogach
obszaru referencyjnego 3x3 m, oglądane przez DWIE wirtualne kamery
o różnych, znanych homografiach. Skrypt:

  1. renderuje klatkę kalibracyjną per kamera i kalibruje obie
     przez POST /api/calibration/{cam},
  2. zakłada world-strefę "_site" (1..4 x 1..3 m) przez PUT /api/zones/_site,
  3. prowadzi osobę po zaplanowanej ścieżce PRZEZ strefę — pozycja
     rzutowana do pikseli każdej kamery i wstrzykiwana per kamera
     przez /api/debug/inject-person,
  4. sprawdza: 1 sfuzowana osoba z 2 kamer + DOKŁADNIE JEDEN alarm
     world_zone_breach (dedupe działa).

Użycie:
    python tools/simulate_two_cameras.py [--server http://localhost:8000]
"""
import argparse
import sys
import time

import cv2
import numpy as np
import requests

MARKER_IDS = [10, 20, 30, 40]          # TL, TR, BR, BL
REF_W, REF_H = 3.0, 3.0                # metry
MARKER_SIZE_M = 0.25
FRAME_W, FRAME_H = 960, 720

# world(m) -> pixels. Kamera A: czysta skala; kamera B: przesunięta
# + delikatny człon perspektywiczny (inny punkt widzenia).
H_CAM = {
    "cam_a": np.array([[150.0, 0.0, 150.0],
                       [0.0, 150.0, 100.0],
                       [0.0, 0.0, 1.0]]),
    "cam_b": np.array([[140.0, 20.0, 220.0],
                       [-10.0, 145.0, 130.0],
                       [0.0, 0.02, 1.0]]),
}

# Ścieżka osoby (metry): podchodzi, wchodzi w strefę 1..4 x 1..3, wychodzi.
WALK = [(0.2, 2.0), (0.7, 2.0), (1.5, 2.0), (2.0, 2.0), (2.5, 2.0),
        (3.0, 2.0), (3.5, 2.0), (4.5, 2.0), (5.0, 2.0)]
FRAMES_PER_STEP = 2                    # filtr temporalny wymaga serii


def world_to_px(cam: str, x_m: float, y_m: float) -> tuple[float, float]:
    v = H_CAM[cam] @ np.array([x_m, y_m, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def render_calibration_frame(cam: str) -> bytes:
    """Biała klatka z 4 markerami wwarpowanymi w perspektywę kamery."""
    frame = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # środki markerów w rogach obszaru referencyjnego
    centres_m = [(0.0, 0.0), (REF_W, 0.0), (REF_W, REF_H), (0.0, REF_H)]
    half = MARKER_SIZE_M / 2.0
    for mid, (cx, cy) in zip(MARKER_IDS, centres_m):
        marker = cv2.aruco.generateImageMarker(dictionary, mid, 200)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        # narożniki markera w metrach (TL, TR, BR, BL wokół środka)
        world_corners = np.array([
            [cx - half, cy - half], [cx + half, cy - half],
            [cx + half, cy + half], [cx - half, cy + half],
        ], dtype=np.float64)
        px_corners = np.array(
            [world_to_px(cam, wx, wy) for wx, wy in world_corners],
            dtype=np.float32)
        src = np.array([[0, 0], [200, 0], [200, 200], [0, 200]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, px_corners)
        warped = cv2.warpPerspective(
            marker_bgr, M, (FRAME_W, FRAME_H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        mask = cv2.warpPerspective(
            np.full(marker_bgr.shape[:2], 255, dtype=np.uint8), M,
            (FRAME_W, FRAME_H))
        frame[mask > 0] = warped[mask > 0]
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return jpeg.tobytes()


def blank_jpeg() -> bytes:
    frame = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", frame)
    return jpeg.tobytes()


def person_box_norm(cam: str, x_m: float, y_m: float) -> list[float]:
    """bbox osoby w znormalizowanych coords: foot point w (x_m, y_m)."""
    fx, fy = world_to_px(cam, x_m, y_m)
    box_w, box_h = 60.0, 160.0
    x1 = max(0.0, min(1.0, (fx - box_w / 2) / FRAME_W))
    x2 = max(0.0, min(1.0, (fx + box_w / 2) / FRAME_W))
    y2 = max(0.0, min(1.0, fy / FRAME_H))
    y1 = max(0.0, min(1.0, (fy - box_h) / FRAME_H))
    if x1 >= x2 or y1 >= y2:
        return []
    return [x1, y1, x2, y2]


def main():
    parser = argparse.ArgumentParser(description="Two-camera fusion rehearsal")
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()
    s = args.server.rstrip("/")

    # 1. Kalibracja obu kamer na te same markery
    for cam in H_CAM:
        r = requests.post(
            f"{s}/api/calibration/{cam}",
            files={"image": ("cal.jpg", render_calibration_frame(cam),
                             "image/jpeg")},
            data={"marker_ids": ",".join(map(str, MARKER_IDS)),
                  "width_m": str(REF_W), "height_m": str(REF_H)},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"[FAIL] kalibracja {cam}: {r.status_code} {r.text}")
            sys.exit(1)
        print(f"[ok] kalibracja {cam}")

    # 2. World-strefa
    r = requests.put(f"{s}/api/zones/_site", json={"zones": [{
        "name": "Wykop (world)", "severity": "DANGER",
        "coordinate_space": "world",
        "polygon": [[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]],
    }]}, timeout=10)
    if r.status_code != 200:
        print(f"[FAIL] strefa _site: {r.status_code} {r.text}")
        sys.exit(1)
    print("[ok] strefa _site (1..4 x 1..3 m)")

    baseline = len(requests.get(
        f"{s}/api/alerts", params={"kind": "world_zone_breach", "limit": 100},
        timeout=10).json())

    # 3. Spacer przez strefę
    fused_seen_two_cams = False
    for step, (x_m, y_m) in enumerate(WALK):
        for _ in range(FRAMES_PER_STEP):
            last = None
            for cam in H_CAM:
                box = person_box_norm(cam, x_m, y_m)
                if box:
                    requests.post(f"{s}/api/debug/inject-person", json={
                        "box_norm": box, "frames": 1, "camera_id": cam,
                    }, timeout=10)
                last = requests.post(
                    f"{s}/api/frame",
                    files={"image": ("f.jpg", blank_jpeg(), "image/jpeg")},
                    data={"camera_id": cam}, timeout=30,
                ).json()
            time.sleep(0.05)
        persons = last.get("world_persons", [])
        cams = persons[0]["cameras"] if persons else []
        if len(cams) >= 2:
            fused_seen_two_cams = True
        breaches = last.get("confirmed_world_breaches", [])
        print(f"  step {step} ({x_m:.1f}, {y_m:.1f}) m | "
              f"fused: {len(persons)} (kamery: {','.join(cams) or '—'})"
              f"{' *** WORLD ALARM ***' if breaches else ''}")

    # 4. Werdykt
    records = requests.get(
        f"{s}/api/alerts", params={"kind": "world_zone_breach", "limit": 100},
        timeout=10).json()
    new = len(records) - baseline
    print()
    print(f"Fuzja z 2 kamer widziana: {fused_seen_two_cams}")
    print(f"Nowe alarmy world_zone_breach: {new} (oczekiwane: 1)")
    if fused_seen_two_cams and new == 1:
        print("[PASS] multi-camera fusion działa: 1 osoba, 1 alarm, 2 kamery")
        sys.exit(0)
    print("[FAIL] sprawdź kalibrację/progi (WORLD_ASSOC_THRESHOLD_M)")
    sys.exit(1)


if __name__ == "__main__":
    main()
