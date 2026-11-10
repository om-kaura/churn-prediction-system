"""
Centralized configuration, loaded from environment variables / .env (see
.env.example at the project root for every variable this reads and its default).

Using pydantic-settings instead of reading os.environ ad-hoc in main.py means
every config value is typed, validated, and documented in one place, and a
missing/malformed .env fails loudly at startup instead of silently at request time.
"""
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    app_name: str = "Customer Churn Prediction API"
    api_version: str = "3.0.0"
    port: int = 8000

    # CORS — comma-separated origins, or "*" for all (dev default)
    cors_origins: str = "*"

    # Rate limiting (slowapi / limits syntax, e.g. "30/minute", "1000/day")
    rate_limit_predict: str = "30/minute"
    rate_limit_default: str = "120/minute"

    # A customer is flagged "Yes" when their estimated churn probability is >= this.
    # Unset (recommended) = the "balanced" operating point read from
    # outputs/calibration.json (~0.265, i.e. the class base rate) — the point the
    # reported Accuracy/Precision/Recall were measured at. Callers can override
    # per request with ?threshold=
    default_threshold: Optional[float] = Field(default=None, ge=0.01, le=0.99)

    # If set, /predictions/history and /predictions/export.csv require this value
    # in an X-API-Key header (they expose logged customer inputs). Empty = open (dev).
    admin_api_key: str = ""

    # Paths (relative to the project root, i.e. api/main.py's parent directory)
    model_dir: str = "outputs"
    database_path: str = "outputs/predictions.db"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
