from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HINDSIGHT_", env_file=".env", extra="ignore")

    service_token: SecretStr = Field(min_length=32)
    database_url: SecretStr


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  # filled from env
