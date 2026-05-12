import sys
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _base_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # JWT signing — change in production; required unless VIOSCAN_DEV_DEFAULT_SECRET=1 (dev only)
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    DATABASE_URL: str = "sqlite:///./vioscan.db"

    # Optional: Geoapify reverse geocoding (https://www.geoapify.com/). Leave empty for manual district only.
    GEOAPIFY_API_KEY: str = ""

    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 10
    AI_STREET_MODEL_PATH: str = "../best_floor.pt"
    # Inference recall is tuned with a low YOLO confidence; the counter then
    # discards anything under MIN_FLOOR_CONFIDENCE and clusters the survivors
    # by vertical position. IOU=0.45 matches Ultralytics' default — the older
    # 0.3 was eating real storeys whose boxes happened to overlap slightly.
    AI_STREET_MODEL_CONFIDENCE: float = 0.15
    AI_STREET_MODEL_IOU: float = 0.45
    AI_STREET_MODEL_MIN_FLOOR_CONFIDENCE: float = 0.25
    AI_STREET_MODEL_MIN_WIDTH_RATIO: float = 0.4
    AI_STREET_MODEL_FLOOR_GAP_RATIO: float = 0.6
    # 960 gives the model ~50% more pixels per storey than the YOLO default of
    # 640 (matters when citizens shoot tall buildings from across the street).
    AI_STREET_MODEL_IMGSZ: int = 960
    # Test-time augmentation: slower but recovers storeys hidden by sun glare /
    # power-line clutter. Disable in low-latency setups.
    AI_STREET_MODEL_AUGMENT: bool = True

    # Aerial encroachment building segmenter — train on the Roboflow dataset
    # described in backend/data/encroachment_dataset.yaml.
    AI_ENCROACHMENT_MODEL_PATH: str = "../best_encroachment.pt"
    AI_ENCROACHMENT_CONFIDENCE: float = 0.25
    AI_ENCROACHMENT_IOU: float = 0.45
    # Real-world span (m) of the longer side of an aerial submission; tune to
    # the zoom level citizens typically capture.
    AI_ENCROACHMENT_IMAGE_SPAN_M: float = 300.0
    # Half-width (m) applied to OSM road centerlines to approximate ROW.
    AI_ENCROACHMENT_ROAD_BUFFER_M: float = 4.0

    AI_DEVICE: str = "cpu"

    FRONTEND_URL: str = "http://localhost:5173"

    # Override path to built Vite output (default: repo ../frontend/dist; frozen: _MEIPASS/dist)
    FRONTEND_DIST: str = ""

    # When true, do not mount frontend/dist at / on the API server (use http://localhost:5173 for UI in dev).
    DISABLE_SPA_ON_API: bool = False

    # Printable notice (optional — blank means template shows fill-in lines)
    NOTICE_REPLY_DAYS: int = 7
    NOTICE_OFFICE_LINE1: str = ""
    NOTICE_OFFICE_LINE2: str = ""
    NOTICE_CONTACT_LINE: str = ""

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_or_dev(cls, v: str) -> str:
        if v:
            return v
        import os

        if os.environ.get("VIOSCAN_DEV_DEFAULT_SECRET") == "1" or getattr(sys, "frozen", False):
            return "vioscan-dev-insecure-change-me-use-env-secret-key"
        raise ValueError("SECRET_KEY is required. Set in .env or export VIOSCAN_DEV_DEFAULT_SECRET=1 for local dev only.")

    def resolved_frontend_dist(self) -> Path | None:
        if self.FRONTEND_DIST:
            p = Path(self.FRONTEND_DIST)
            if not p.is_absolute():
                p = _base_dir() / p
        elif getattr(sys, "frozen", False):
            # PyInstaller bundles SPA at _MEIPASS/dist
            p = _base_dir() / "dist"
        else:
            p = _base_dir().parent / "frontend" / "dist"
        if (p / "index.html").exists():
            return p.resolve()
        return None

    def resolved_ai_street_model_path(self) -> Path:
        p = Path(self.AI_STREET_MODEL_PATH)
        if not p.is_absolute():
            p = _base_dir() / p
        return p.resolve()

    def resolved_ai_encroachment_model_path(self) -> Path:
        p = Path(self.AI_ENCROACHMENT_MODEL_PATH)
        if not p.is_absolute():
            p = _base_dir() / p
        return p.resolve()

    def cors_origins(self) -> list[str]:
        u = self.FRONTEND_URL.rstrip("/")
        out = {
            u,
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
        }
        if self.resolved_frontend_dist():
            out.update({"http://127.0.0.1:8000", "http://localhost:8000"})
        return sorted(out)


settings = Settings()
