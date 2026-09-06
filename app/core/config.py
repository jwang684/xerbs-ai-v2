from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Xerbs AI v2"
    app_version: str = "0.10.0"
    environment: str = "development"
    llm_provider: str = "mock"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    database_url: str = "sqlite:///./xerbs_ai_v2.db"
    clinical_reviewer_roles: str = "CLINICAL_REVIEWER,CLINICAL_ADMIN"
    port: int = 8080
    base44_contract_version: str = "base44-xerbs-ai-v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
