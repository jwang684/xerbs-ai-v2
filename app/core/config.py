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

    # Service-to-service authentication (X1D-E2E1).
    #
    # This service is called by xerbs-core, never by a browser, so the
    # credential is a single service secret from the environment. When it is
    # unset, protected endpoints refuse every request: an unconfigured
    # deployment must fail closed rather than stay open.
    service_auth_token: str | None = None
    service_caller_name: str = "xerbs-core"

    # Escape hatch for local development only. Anonymous access requires this
    # to be set deliberately AND service_auth_token to be empty, so merely
    # forgetting the secret never opens the service up. Health reports it.
    allow_insecure_local: bool = False

    # Governance mutation switch (X1D-E2E1 Step 5).
    #
    # reviewer_role arrives in the request body and is therefore self-asserted:
    # a caller could previously grant itself clinical review authority just by
    # naming the role. Until this service has real human reviewer identity,
    # external governance mutation is disabled by default and corpus content is
    # established by the controlled bootstrap instead.
    allow_external_governance_mutation: bool = False

    # X1D-AIV2-ATTEST1: the reverse call to xerbs-core, for verifying a human
    # clinical review attestation.
    #
    # A payload core pushes is just a string once it has left core, and whoever
    # holds the service credential can send any string. So ai-v2 asks core
    # whether the attestation actually exists and is still authoritative.
    # Minting one requires an authenticated admin session holding
    # clinical:review -- a path no service credential has.
    #
    # Both unset by default, so approval fails closed on a deployment that has
    # not been wired up. No secret value exists in this repository.
    core_base_url: str | None = None
    core_internal_service_token: str | None = None
    core_lookup_timeout_seconds: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
