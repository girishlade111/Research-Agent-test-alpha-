"""Custom middleware for error handling, request IDs, and logging."""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("app.middleware")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generate a UUID request_id for each request and add to response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with method, path, status_code, duration_ms, user_id."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        user_id = request.headers.get("x-user-id", "anonymous")
        request_id = getattr(request.state, "request_id", "unknown")

        extra: dict[str, Any] = {
            "request_id": request_id,
            "user_id": user_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }

        log_record = logging.LogRecord(
            name="app.api",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"{request.method} {request.url.path} {response.status_code} {duration_ms}ms",
            args=None,
            exc_info=None,
        )
        for key, value in extra.items():
            setattr(log_record, key, value)

        logging.getLogger("app.api").handle(log_record)

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return structured JSON error responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            timestamp = datetime.now(UTC).isoformat() + "Z"

            # Log the full traceback
            logger.error(
                "Unhandled exception: %s",
                str(exc),
                exc_info=True,
                extra={"request_id": request_id},
            )

            error_response = {
                "request_id": request_id,
                "timestamp": timestamp,
                "error": type(exc).__name__,
                "detail": str(exc),
            }

            return JSONResponse(status_code=500, content=error_response)
