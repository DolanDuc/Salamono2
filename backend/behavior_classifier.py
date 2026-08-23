"""Runtime adapter for compatible occlusion-aware TCN+GRU pose-event models.

The classifier consumes a four-second history of MediaPipe landmarks for one
tracked person.  It preserves the public fields used by the existing Perimetr
API and additionally exposes the learned safety head and body-part quality
metrics used by the event controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np

from pose_event.runtime import PoseEventRuntime
from pose_event.v32_behavior_runtime import V32BehaviorRuntime


@dataclass(frozen=True)
class BehaviorPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    valid_ratio: float
    window_seconds: float
    inference_ms: float
    safety_label: str
    safety_confidence: float
    safety_probabilities: dict[str, float]
    torso_quality: float
    upper_body_quality: float
    lower_body_quality: float
    visible_ratio: float
    secondary_probabilities: dict[str, float]
    secondary_valid_ratio: float
    secondary_window_seconds: float
    secondary_inference_ms: float


def _record_peaks(peaks: dict[str, float], probabilities: dict[str, float]) -> None:
    for label, value in probabilities.items():
        score = float(value)
        if score > peaks.get(label, 0.0):
            peaks[label] = score


class BehaviorClassifier:
    """Load and run a compatible pose-event action/safety checkpoint.

    ``predict_history`` is kept for drop-in compatibility with the previous behavior-classifier adapter.  The checkpoint defines the action/safety class
    names, feature normalization, FPS and window length, so deployment cannot
    silently drift from training.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        feature_fps: float = 15.0,
        min_valid_ratio: float = 0.45,
        min_window_coverage: float = 0.70,
        max_sample_gap_seconds: float = 0.50,
        secondary_enabled: bool = False,
        secondary_model_path: str | Path | None = None,
        secondary_device: str | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Behavior model not found: {path}")

        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"PyTorch is unavailable: {exc}") from exc

        requested = (device or "auto").strip().lower()
        if requested in {"", "auto"}:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"

        self.model_path = str(path)
        self.device = requested
        # Alert thresholds are only meaningful next to the scores the model
        # actually produces. Without this, a clip that raises no alert is
        # indistinguishable from one the classifier never scored at all.
        self.prediction_count = 0
        self.peak_probabilities: dict[str, float] = {}
        self.secondary_peak_probabilities: dict[str, float] = {}
        self.feature_fps = float(feature_fps)
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_window_coverage = float(min_window_coverage)
        self.max_sample_gap_seconds = float(max_sample_gap_seconds)

        self.runtime = PoseEventRuntime(
            path,
            device=requested,
            min_valid_ratio=self.min_valid_ratio,
            max_sample_gap_seconds=self.max_sample_gap_seconds,
        )
        if abs(self.runtime.fps - self.feature_fps) > 1e-3:
            raise ValueError(
                f"Checkpoint FPS={self.runtime.fps}, configured "
                f"feature_fps={self.feature_fps}"
            )

        self.labels = list(self.runtime.action_classes)
        self.class_names = list(self.runtime.action_classes)
        self.safety_labels = list(self.runtime.safety_classes)
        self.window_frames = int(self.runtime.frame_count)
        self.secondary_runtime: V32BehaviorRuntime | None = None
        self.secondary_error: str | None = None
        if secondary_enabled and secondary_model_path:
            try:
                self.secondary_runtime = V32BehaviorRuntime(
                    secondary_model_path,
                    device=secondary_device or requested,
                    min_valid_ratio=self.min_valid_ratio,
                    max_sample_gap_seconds=self.max_sample_gap_seconds,
                )
                if abs(self.secondary_runtime.fps - self.feature_fps) > 1e-3:
                    raise ValueError(
                        f"Secondary checkpoint FPS={self.secondary_runtime.fps}, "
                        f"configured feature_fps={self.feature_fps}"
                    )
                if self.secondary_runtime.frame_count != self.runtime.frame_count:
                    raise ValueError(
                        "Primary/secondary pose windows differ: "
                        f"{self.runtime.frame_count} != "
                        f"{self.secondary_runtime.frame_count}"
                    )
            except Exception as exc:
                self.secondary_runtime = None
                self.secondary_error = str(exc)
        self.warmup_ms = self._warmup(torch)

    @property
    def target_window_seconds(self) -> float:
        return float(self.runtime.window_seconds)

    def _warmup(self, torch_module) -> float:
        started = time.perf_counter()
        with torch_module.inference_mode():
            x = torch_module.zeros(
                (1, self.runtime.frame_count, len(self.runtime.mean)),
                dtype=torch_module.float32,
                device=self.runtime.device,
            )
            mask = torch_module.ones(
                (1, self.runtime.frame_count),
                dtype=torch_module.bool,
                device=self.runtime.device,
            )
            self.runtime.model(x, mask)
            if self.runtime.device.type == "cuda":
                torch_module.cuda.synchronize(self.runtime.device)
        return (time.perf_counter() - started) * 1000.0

    def _coverage_ok(
        self,
        history: Sequence[tuple[float, np.ndarray]],
    ) -> bool:
        if len(history) < 2:
            return False
        timestamps = sorted(float(item[0]) for item in history)
        observed = max(0.0, timestamps[-1] - timestamps[0])
        required = self.target_window_seconds * self.min_window_coverage
        return observed + 1e-6 >= required

    def predict_history(
        self,
        history: Sequence[tuple[float, np.ndarray]],
    ) -> BehaviorPrediction | None:
        items = list(history)
        if not self._coverage_ok(items):
            return None

        timestamps: list[float] = []
        poses: list[np.ndarray] = []
        for timestamp, pose in items:
            arr = np.asarray(pose, dtype=np.float32)
            if arr.shape not in ((33, 4), (33, 5)):
                continue
            if not np.isfinite(float(timestamp)):
                continue
            timestamps.append(float(timestamp))
            poses.append(arr.copy())
        if len(timestamps) < 2:
            return None

        prediction = self.runtime.predict_history(timestamps, poses)
        if prediction is None:
            return None

        secondary_probabilities: dict[str, float] = {}
        secondary_valid_ratio = 0.0
        secondary_window_seconds = 0.0
        secondary_inference_ms = 0.0
        if self.secondary_runtime is not None:
            try:
                secondary = self.secondary_runtime.predict_history(timestamps, poses)
            except Exception as exc:
                self.secondary_error = str(exc)
                secondary = None
            if secondary is not None:
                secondary_probabilities = dict(secondary.probabilities)
                secondary_valid_ratio = float(secondary.valid_ratio)
                secondary_window_seconds = float(secondary.window_seconds)
                secondary_inference_ms = float(secondary.inference_ms)

        self.prediction_count += 1
        _record_peaks(self.peak_probabilities, prediction.action_probabilities)
        _record_peaks(self.secondary_peak_probabilities, secondary_probabilities)

        return BehaviorPrediction(
            label=prediction.action_label,
            confidence=prediction.action_confidence,
            probabilities=dict(prediction.action_probabilities),
            valid_ratio=prediction.valid_ratio,
            window_seconds=prediction.window_seconds,
            inference_ms=prediction.inference_ms,
            safety_label=prediction.safety_label,
            safety_confidence=prediction.safety_confidence,
            safety_probabilities=dict(prediction.safety_probabilities),
            torso_quality=prediction.torso_quality,
            upper_body_quality=prediction.upper_body_quality,
            lower_body_quality=prediction.lower_body_quality,
            visible_ratio=prediction.visible_ratio,
            secondary_probabilities=secondary_probabilities,
            secondary_valid_ratio=secondary_valid_ratio,
            secondary_window_seconds=secondary_window_seconds,
            secondary_inference_ms=secondary_inference_ms,
        )

    def predict(
        self,
        history: Iterable[tuple[float, np.ndarray]],
    ) -> BehaviorPrediction | None:
        return self.predict_history(list(history))

    classify = predict
    __call__ = predict
