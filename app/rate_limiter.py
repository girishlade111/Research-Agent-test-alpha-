"""Rate limiting configuration using slowapi."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings


def get_rate_limit_key(request: Request) -> str:
    """Extract rate limit key from x-user-id header, falling back to IP."""
    user_id = request.headers.get("x-user-id")
    if user_id:
        return user_id
    return get_remote_address(request)


def _is_test_mode() -> bool:
    """Check if running in test environment."""
    return settings.ENVIRONMENT == "test"


# Create limiter with custom key function
limiter = Limiter(
    key_func=get_rate_limit_key,
    enabled=not _is_test_mode(),
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom handler for rate limit exceeded errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    retry_after = exc.detail.split("per")[0].strip() if exc.detail else "60"

    response = JSONResponse(
        status_code=429,
        content={
            "request_id": request_id,
            "error": "RateLimitExceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "timestamp": "",
        },
    )
    response.headers["Retry-After"] = retry_after
    response.headers["X-Request-ID"] = request_id
    return response
