import os
from dataclasses import dataclass, field


@dataclass
class YOLOConfig:
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"
    img_size: int = 640


@dataclass
class DangerConfig:
    proximity_px: int = 50
    overlap_iou: float = 0.01
    consecutive_frames_required: int = 3
    cooldown_seconds: float = 10.0


@dataclass
class IngestConfig:
    max_frame_size_bytes: int = 2_000_000
    target_fps: float = 10.0
    jpeg_quality: int = 80


@dataclass
class PPEConfig:
    model_path: str = "ppe.pt"
    confidence: float = 0.35
    min_person_height_frac: float = 0.30  # bbox height / frame height
    entry_cooldown_sec: float = 4.0       # per-person cooldown between checks


@dataclass
class WorldConfig:
    """Multi-camera ground-plane fusion (wspólny układ z markerów ArUco)."""
    assoc_threshold_m: float = 0.7    # cluster obs from different cameras
    id_match_threshold_m: float = 0.9  # inherit fused_id from previous tick
    obs_ttl_sec: float = 1.5          # drop camera obs older than this


@dataclass
class VehicleMotionConfig:
    """Bramkowanie alarmu ruchem pojazdu.

    Alarm „osoba przy pojeździe" leci tylko dla pojazdu, który się PORUSZA
    (jedzie, obraca się, albo rusza osprzętem — np. koparka łyżką). Stan
    „niebezpieczny" utrzymuje się `hold_sec` po ostatnim ruchu, więc pojazd,
    który stanął na kilkanaście sekund, dalej jest niebezpieczny.
    """
    enabled: bool = True
    match_iou: float = 0.3          # dopasowanie pojazdu do toru między klatkami
    displacement_frac: float = 0.04  # przesunięcie środka bboxa / przekątna → jazda
    size_change_frac: float = 0.12   # zmiana rozmiaru bboxa → dojazd/obrót
    pixel_diff_thresh: int = 25      # próg różnicy piksela (0-255) w ROI pojazdu
    internal_motion_frac: float = 0.03  # % zmienionych pikseli ROI → ruch osprzętu
    hold_sec: float = 15.0           # jak długo „niebezpieczny" po ostatnim ruchu


@dataclass
class AppConfig:
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    danger: DangerConfig = field(default_factory=DangerConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    ppe: PPEConfig = field(default_factory=PPEConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    vehicle_motion: VehicleMotionConfig = field(default_factory=VehicleMotionConfig)
    flagged_frames_dir: str = "data/flagged_frames"
    host: str = "0.0.0.0"
    port: int = 8000


def _from_env() -> AppConfig:
    cfg = AppConfig()
    if v := os.getenv("YOLO_MODEL"):
        cfg.yolo.model_name = v
    if v := os.getenv("YOLO_CONFIDENCE"):
        cfg.yolo.confidence_threshold = float(v)
    if v := os.getenv("YOLO_DEVICE"):
        cfg.yolo.device = v
    if v := os.getenv("YOLO_IMG_SIZE"):
        cfg.yolo.img_size = int(v)
    if v := os.getenv("DANGER_PROXIMITY_PX"):
        cfg.danger.proximity_px = int(v)
    if v := os.getenv("DANGER_CONSECUTIVE_FRAMES"):
        cfg.danger.consecutive_frames_required = int(v)
    if v := os.getenv("DANGER_COOLDOWN_SEC"):
        cfg.danger.cooldown_seconds = float(v)
    if v := os.getenv("PPE_MODEL"):
        cfg.ppe.model_path = v
    if v := os.getenv("PPE_CONFIDENCE"):
        cfg.ppe.confidence = float(v)
    if v := os.getenv("PPE_MIN_PERSON_HEIGHT_FRAC"):
        cfg.ppe.min_person_height_frac = float(v)
    if v := os.getenv("PPE_COOLDOWN_SEC"):
        cfg.ppe.entry_cooldown_sec = float(v)
    if v := os.getenv("WORLD_ASSOC_THRESHOLD_M"):
        cfg.world.assoc_threshold_m = float(v)
    if v := os.getenv("WORLD_OBS_TTL_SEC"):
        cfg.world.obs_ttl_sec = float(v)
    if v := os.getenv("VEHICLE_MOTION_ENABLED"):
        cfg.vehicle_motion.enabled = v.lower() not in ("0", "false", "no")
    if v := os.getenv("VEHICLE_HOLD_SEC"):
        cfg.vehicle_motion.hold_sec = float(v)
    if v := os.getenv("VEHICLE_DISPLACEMENT_FRAC"):
        cfg.vehicle_motion.displacement_frac = float(v)
    if v := os.getenv("VEHICLE_INTERNAL_MOTION_FRAC"):
        cfg.vehicle_motion.internal_motion_frac = float(v)
    if v := os.getenv("SERVER_PORT"):
        cfg.port = int(v)
    return cfg


CONFIG = _from_env()
