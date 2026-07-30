"""Centralized application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables with sensible defaults."""

    # File handling
    MAX_FILE_SIZE: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS: str = ".txt,.md,.csv,.pdf,.docx,.xlsx,.xls,.png,.jpg,.jpeg,.heic"

    # Rate limiting (requests per minute)
    RATE_LIMIT_QUERY: str = "100/minute"
    RATE_LIMIT_UPLOAD: str = "20/minute"
    RATE_LIMIT_READ: str = "200/minute"

    # CORS
    CORS_ORIGINS: str = "*"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Data directory
    DATA_DIR: str = "data"

    # Environment mode (set to "test" in tests to disable rate limiting)
    ENVIRONMENT: str = "production"

    model_config = {"env_prefix": "APP_", "case_sensitive": True}

    @property
    def allowed_extensions_set(self) -> set[str]:
        """Return ALLOWED_EXTENSIONS as a set."""
        return {ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")}

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS_ORIGINS as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


# Module-level singleton
settings = Settings()
