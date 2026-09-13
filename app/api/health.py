from fastapi import APIRouter, Response
from app.core.config import get_settings
from app.services.llm.factory import configuration_error, provider_policy

router = APIRouter(tags=["health"])

# Configuration verdicts reported by health.
#
# These answer a different question from liveness, and the staging incident
# turned on conflating the two: the service was up and answering every request,
# and every answer came from the offline contract-test provider. "healthy" alone
# could not have revealed that.
CONFIG_OK = "OK"
CONFIG_NOT_PRODUCTION_READY = "NOT_PRODUCTION_READY"


def _configuration_status(problem: str | None) -> str:
    return CONFIG_OK if problem is None else CONFIG_NOT_PRODUCTION_READY


@router.get("/health")
async def health():
    """Liveness, plus an explicit verdict on clinical-inference configuration.

    This endpoint stays 200 whenever the process is answering, because it is
    what a platform health check watches: making it fail on a configuration
    problem would restart the container in a loop and destroy the very
    diagnostic the operator needs. The configuration verdict is carried in the
    body, and /health/clinical turns the same verdict into a status code.

    Only the provider name, environment and a validity flag are reported --
    never a base URL, key or model. Health is unauthenticated.
    """
    s = get_settings()
    policy = provider_policy()
    problem = configuration_error()

    return {
        "status": "healthy",
        "service": s.app_name,
        "version": s.app_version,
        "clinical_inference": {
            **policy,
            "usable": problem is None,
            "configuration_status": _configuration_status(problem),
            # Names which variables are wrong; never their values.
            "problem": problem,
        },
    }


@router.get("/health/clinical")
async def clinical_readiness(response: Response):
    """Readiness for clinical work, as a status code.

    503 when the deployment is not production-ready -- which includes the exact
    invariant this guard exists for:

        ENVIRONMENT=staging or production + LLM_PROVIDER=mock
        => not production-ready

    Kept separate from /health on purpose. Liveness must stay green so the
    container is not restarted; readiness must go red so nobody concludes from a
    200 that patient-facing inference is real.
    """
    problem = configuration_error()
    policy = provider_policy()

    if problem is not None:
        response.status_code = 503

    return {
        "ready_for_clinical_use": problem is None,
        "configuration_status": _configuration_status(problem),
        "llm_provider": policy["llm_provider"],
        "environment": policy["environment"],
        "performs_clinical_inference": policy["performs_clinical_inference"],
        "problem": problem,
    }
