"""Validated runtime configuration."""

from pydantic import Field, HttpUrl, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """Configuration shared by the orchestrator and vLLM client."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vllm_endpoint_url: str = Field(default="http://127.0.0.1:8000/v1", min_length=1)
    vllm_model_name: str = "google/gemma-4-12B-it"
    embedding_model_name: str = "nomic-embed-text:latest"
    memory_database_path: str = "data/atomic_memory.sqlite3"
    vllm_api_key: str | None = Field(default=None, repr=False)
    enable_thinking: bool = False
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, gt=0, le=30)
    max_prompt_bytes: int = Field(default=256_000, gt=0, le=4_000_000)
    max_response_bytes: int = Field(default=2_000_000, gt=0, le=16_000_000)

    @field_validator("vllm_endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, value: str) -> str:
        TypeAdapter(HttpUrl).validate_python(value)
        return value.rstrip("/")
