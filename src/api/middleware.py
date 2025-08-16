"""
Authentication & logging middleware
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Callable, Optional

import secure
from fastapi import HTTPException, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Context variables for request tracing
correlation_id_context: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context"""
    return correlation_id_context.get()


def get_request_id() -> Optional[str]:
    """Get the current request ID from context"""
    return request_id_context.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in context"""
    correlation_id_context.set(correlation_id)


def set_request_id(request_id: str) -> None:
    """Set the request ID in context"""
    request_id_context.set(request_id)


# Create rate limiter instance
limiter = Limiter(key_func=get_remote_address, enabled=settings.rate_limit_enabled)

# Security headers configuration
secure_headers = secure.Secure(
    csp=secure.ContentSecurityPolicy().default_src("'self'").script_src("'self'"),
    hsts=secure.StrictTransportSecurity().max_age(31536000).include_subdomains(),
    referrer=secure.ReferrerPolicy().no_referrer(),
    permissions=secure.PermissionsPolicy().geolocation("none").camera("none"),
    cache=secure.CacheControl().no_cache().no_store().must_revalidate(),
)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with headers, size limits, and validation"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size limits
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_request_size:
            client_host = request.client.host if request.client else "unknown"
            logger.warning(
                f"Request too large: {content_length} bytes from {client_host}"
            )
            raise HTTPException(
                status_code=413,
                detail=f"Request entity too large. Maximum size: {settings.max_request_size} bytes",
            )

        # Process request
        response = await call_next(request)

        # Add security headers
        if settings.is_production:
            # Apply security headers manually since secure.framework may not exist
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers[
                "Strict-Transport-Security"
            ] = "max-age=31536000; includeSubDomains"

        return response  # type: ignore[no-any-return]


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for Bearer token authentication"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth if no API key is configured - all endpoints are public
        if not settings.api_key:
            return await call_next(request)  # type: ignore[no-any-return]

        # Only the root endpoint is public when API_KEY is configured
        public_paths = ["/"]
        if request.url.path in public_paths:
            return await call_next(request)  # type: ignore[no-any-return]

        # Check Bearer token for protected endpoints
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            client_host = request.client.host if request.client else "unknown"
            logger.warning(f"Missing or invalid Bearer token from {client_host}")
            # Return JSON response directly from middleware
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authorization header missing or invalid",
                    "message": "Bearer token required",
                    "type": "authentication_error",
                    "timestamp": str(time.time()),
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Extract token from Bearer header
        token = auth_header[7:]  # Remove "Bearer " prefix

        if token != settings.api_key:
            client_host = request.client.host if request.client else "unknown"
            logger.warning(f"Invalid Bearer token attempt from {client_host}")
            # Return JSON response directly from middleware
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid Bearer token",
                    "message": "Access denied",
                    "type": "authentication_error",
                    "timestamp": str(time.time()),
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)  # type: ignore[no-any-return]


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracing with correlation IDs and performance monitoring"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Generate or extract correlation ID and request ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        # Set context variables for this request
        set_correlation_id(correlation_id)
        set_request_id(request_id)

        # Enhanced request logging with tracing context
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "correlation_id": correlation_id,
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "content_length": request.headers.get("Content-Length"),
                "event_type": "request_start",
            },
        )

        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time
            response_size = response.headers.get("Content-Length", "unknown")

            # Enhanced response logging with performance metrics
            logger.info(
                f"Request completed: {response.status_code} - {request.method} {request.url.path} - {process_time:.3f}s",
                extra={
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "response_size": response_size,
                    "client_ip": client_ip,
                    "event_type": "request_complete",
                },
            )

            # Add tracing headers to response
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"

            # Performance monitoring - log slow requests
            if process_time > 5.0:  # Log requests taking more than 5 seconds
                logger.warning(
                    f"Slow request detected: {request.method} {request.url.path} took {process_time:.3f}s",
                    extra={
                        "correlation_id": correlation_id,
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "process_time": process_time,
                        "event_type": "slow_request",
                    },
                )

            return response  # type: ignore[no-any-return]

        except Exception as e:
            process_time = time.time() - start_time

            # Enhanced error logging with full tracing context
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)} - {process_time:.3f}s",
                extra={
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": str(request.query_params),
                    "process_time": process_time,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "event_type": "request_error",
                },
                exc_info=True,
            )
            raise
