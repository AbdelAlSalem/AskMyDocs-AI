import os
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    demo_mode: bool = False
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout_seconds: float = Field(default=35, gt=0, le=120)

    chroma_dir: Path = PROJECT_DIR / "data" / "chroma"
    upload_dir: Path = PROJECT_DIR / "data" / "uploads"
    sqlite_path: Path = PROJECT_DIR / "data" / "documents.sqlite3"
    conversations_path: Path = Field(
        default_factory=lambda settings: settings["sqlite_path"].with_name("conversations.sqlite3")
    )
    max_upload_mb: int = Field(default=10, ge=1, le=10)

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_min_similarity: float = Field(default=0.30, ge=0, le=1)
    rag_debug_context: bool = False
    chunk_size: int = Field(default=800, ge=1, le=4000)
    chunk_overlap: int = Field(default=120, ge=0)

    @classmethod
    def load(cls) -> Self:
        env = {**dotenv_values(BACKEND_DIR / ".env"), **os.environ}
        settings = {}
        for name in cls.model_fields:
            value = env.get(name.upper())
            if value and value.strip():
                settings[name] = value.strip()
        return cls.model_validate(settings)

    @field_validator("chroma_dir", "upload_dir", "sqlite_path", "conversations_path")
    @classmethod
    def resolve_storage_path(cls, path: Path) -> Path:
        return (PROJECT_DIR / path.expanduser()).resolve()

    @model_validator(mode="after")
    def validate_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE.")
        return self

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.get_secret_value())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


class DemoSettings(Settings):
    demo_allowed_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    demo_max_body_bytes: int = Field(default=16384, ge=1024, le=65536)
    demo_max_question_chars: int = Field(default=600, ge=1, le=2000)
    demo_max_context_chars: int = Field(default=4500, ge=200, le=10000)
    demo_top_k: int = Field(default=4, ge=1, le=8)
    demo_max_history_messages: int = Field(default=4, ge=0, le=8)
    demo_max_history_chars: int = Field(default=1800, ge=0, le=6000)
    demo_max_completion_tokens: int = Field(default=1536, ge=128, le=2500)

    demo_ip_rate_limit: int = Field(default=5, ge=1, le=1000)
    demo_global_rate_limit: int = Field(default=15, ge=1, le=1000)
    demo_rate_window_seconds: int = Field(default=60, ge=1, le=3600)
    demo_daily_request_limit: int = Field(default=80, ge=1, le=10000)
    demo_max_concurrent_requests: int = Field(default=3, ge=1, le=20)
    demo_trust_render_client_ip: bool = False

    demo_turnstile_site_key: str = ""
    demo_turnstile_secret_key: SecretStr = SecretStr("")
    demo_turnstile_hostname: str = Field(default="", max_length=253, pattern=r"^(?:[A-Za-z0-9][A-Za-z0-9.-]*)?$")
    demo_turnstile_action: str = Field(default="demo_chat", pattern=r"^[A-Za-z0-9_-]{1,32}$")

    @field_validator("demo_allowed_origins", mode="before")
    @classmethod
    def split_demo_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(origin.strip() for origin in value.split(",") if origin.strip())
        return value

    @field_validator("demo_allowed_origins")
    @classmethod
    def validate_demo_origins(cls, origins: tuple[str, ...]) -> tuple[str, ...]:
        if not origins:
            raise ValueError("DEMO_ALLOWED_ORIGINS must contain at least one origin.")
        for origin in origins:
            url = urlsplit(origin)
            if (
                url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
                or url.path or url.query or url.fragment or "*" in origin or any(char.isspace() for char in origin)
                or (url.port is None and url.netloc.endswith(":"))
            ):
                raise ValueError("DEMO_ALLOWED_ORIGINS accepts only exact HTTP or HTTPS origins without paths.")
        return origins

    @model_validator(mode="after")
    def validate_turnstile(self) -> Self:
        if bool(self.demo_turnstile_site_key) != self.turnstile_required:
            raise ValueError("Configure DEMO_TURNSTILE_SITE_KEY and DEMO_TURNSTILE_SECRET_KEY together.")
        if self.turnstile_required and not self.demo_turnstile_hostname:
            raise ValueError("Set DEMO_TURNSTILE_HOSTNAME to the frontend hostname when Turnstile is enabled.")
        return self

    @property
    def turnstile_required(self) -> bool:
        return bool(self.demo_turnstile_secret_key.get_secret_value())
