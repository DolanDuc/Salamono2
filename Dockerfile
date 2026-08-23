FROM python:3.11-slim

WORKDIR /app

# ffmpeg is not optional: annotated demo exports and evidence clips both shell
# out to it, and without it those jobs fail at the first frame.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libegl1 libgles2 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Detector, PPE and MediaPipe models all ship in the repository (models/,
# ppe.pt), so the image no longer downloads anything at build time.

COPY config.py .
COPY ppe.pt .
COPY backend/ backend/
COPY frontend/ frontend/
COPY phone/ phone/
COPY pose_event/ pose_event/
COPY tools/ tools/
COPY models/ models/
# Ground-plane calibration for the Distance module and the ArUco worker
# registry — without these the metric thresholds silently fall back to pixels.
COPY distance_assets/ distance_assets/
COPY data/ data/

RUN mkdir -p data/flagged_frames data/event_clips data/sample_videos

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
