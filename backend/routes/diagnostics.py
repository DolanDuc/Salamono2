"""Read-only operational diagnostics for pipeline performance."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


@router.get("/performance")
async def performance_diagnostics(request: Request):
    profiler = getattr(request.app.state, "performance_profiler", None)
    if profiler is None:
        raise HTTPException(503, "Performance profiler is not initialized")
    payload = profiler.snapshot()

    detector = getattr(request.app.state, "detector", None)
    status = getattr(detector, "status", None)
    payload["detector"] = status() if callable(status) else {
        "backend": "ultralytics",
        "model": type(detector).__name__ if detector is not None else None,
    }
    adaptive = getattr(request.app.state, "yolo_size_controller", None)
    adaptive_status = getattr(adaptive, "status", None)
    if callable(adaptive_status):
        payload["adaptive_yolo"] = adaptive_status()

    processor = getattr(request.app.state, "frame_processor", None)
    if processor is not None:
        stats = processor.stats
        payload["frame_processor"] = {
            "submitted": stats.submitted,
            "replaced": stats.replaced,
            "dropped": stats.dropped,
            "processed": stats.processed,
            "failed": stats.failed,
            "queue_depth": processor.queue_depth,
        }
    posture_manager = getattr(request.app.state, "posture_manager", None)
    posture_worker = getattr(request.app.state, "posture_worker", None)
    if posture_manager is not None:
        classifier = getattr(posture_manager, "behavior_classifier", None)
        payload["posture"] = {
            "available": bool(getattr(posture_manager, "available", False)),
            "unavailable_reason": getattr(
                posture_manager,
                "unavailable_reason",
                None,
            ),
            "sample_fps": float(posture_manager.cfg.sample_fps),
            "optical_flow_scale": float(
                getattr(posture_manager.cfg, "optical_flow_scale", 1.0)
            ),
            "behavior_available": bool(
                getattr(posture_manager, "behavior_available", False)
            ),
            "secondary_behavior_available": bool(
                getattr(posture_manager, "secondary_behavior_available", False)
            ),
            "secondary_behavior_error": getattr(
                posture_manager, "secondary_behavior_unavailable_reason", None
            ),
            "behavior_stride": int(
                posture_manager.cfg.behavior_inference_stride_samples
            ),
            "behavior_warmup_ms": (
                round(float(classifier.warmup_ms), 3)
                if classifier is not None
                and hasattr(classifier, "warmup_ms") else None
            ),
            "worker_busy": bool(posture_worker and posture_worker.busy),
            "worker_pending": int(
                posture_worker.pending_count if posture_worker else 0
            ),
            # Highest score each behaviour class reached so far, so a run that
            # raised no alert can be told apart from one the classifier never
            # scored — and so thresholds can be set against real numbers.
            "predictions": int(getattr(classifier, "prediction_count", 0)),
            "peak_probabilities": {
                label: round(value, 4)
                for label, value in sorted(
                    getattr(classifier, "peak_probabilities", {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "secondary_peak_probabilities": {
                label: round(value, 4)
                for label, value in sorted(
                    getattr(classifier, "secondary_peak_probabilities", {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }
    worker_id_worker = getattr(request.app.state, "worker_id_worker", None)
    if worker_id_worker is not None:
        worker_stats = worker_id_worker.stats()
        payload["worker_id"] = (
            asdict(worker_stats)
            if is_dataclass(worker_stats) else worker_stats
        )
    demo_service = getattr(request.app.state, "demo_video_service", None)
    if demo_service is not None:
        payload["demo_video"] = demo_service.stats()
    return payload


@router.post("/performance/reset")
async def reset_performance_diagnostics(request: Request):
    profiler = getattr(request.app.state, "performance_profiler", None)
    if profiler is None:
        raise HTTPException(503, "Performance profiler is not initialized")
    profiler.reset()
    return {"status": "reset", "performance": profiler.snapshot()}
