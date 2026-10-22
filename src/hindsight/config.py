from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HINDSIGHT_", env_file=".env", extra="ignore")

    service_token: SecretStr = Field(min_length=32)
    database_url: SecretStr
    chat_model: str = "openai:gpt-4.1-mini-2025-04-14"
    openai_api_key: SecretStr | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://us.cloud.langfuse.com"
    langfuse_environment: str = "development"

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  # filled from env
