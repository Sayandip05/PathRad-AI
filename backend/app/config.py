from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application configuration settings"""

    # App
    APP_NAME: str = "PathRad AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./database/pathrad.db"

    # File Storage
    UPLOAD_DIR: str = "./uploads"
    REPORTS_DIR: str = "./reports"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

    # Hugging Face Token
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Model Paths (User will configure these)
    MEDGEMMA_MODEL_PATH: str = os.getenv("MEDGEMMA_MODEL_PATH", "./models/medgemma")
    CXR_MODEL_PATH: str = os.getenv("CXR_MODEL_PATH", "./models/cxr_foundation")
    PATH_MODEL_PATH: str = os.getenv("PATH_MODEL_PATH", "./models/path_foundation")

    # API Keys (Optional - for Groq fallback)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]

    # ADK Configuration
    ADK_PROJECT_ID: str = os.getenv("ADK_PROJECT_ID", "pathrad-ai")
    ADK_LOCATION: str = os.getenv("ADK_LOCATION", "us-central1")

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
