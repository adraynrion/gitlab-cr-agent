"""
GitLab AI Code Review Agent
Main FastAPI application with PydanticAI integration
"""

import asyncio
import logging
import signal
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agents.code_reviewer import initialize_review_agent
from src.api import health, webhooks
from src.api.middleware import (
    AuthenticationMiddleware,
    RequestTracingMiddleware,
    SecurityMiddleware,
)
from src.config.settings import settings
from src.exceptions import (
    AIProviderException,
    ConfigurationException,
    GitLabAPIException,
    GitLabReviewerException,
    RateLimitException,
    SecurityException,
    WebhookValidationException,
)
from src.utils.version import get_version

# Configure structured logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Encapsulated application state with thread safety"""

    review_agent: Optional[Any] = None
    initialized: bool = False
    shutdown_event: Optional[asyncio.Event] = None
    _lock: Optional[threading.Lock] = None

    def __post_init__(self):
        """Initialize thread-safe lock after dataclass creation"""
        if self._lock is None:
            object.__setattr__(self, "_lock", threading.Lock())
        if self.shutdown_event is None:
            object.__setattr__(self, "shutdown_event", asyncio.Event())

    def mark_initialized(self, agent: Any) -> None:
        """Thread-safe method to mark application as initialized"""
        if self._lock:
            with self._lock:
                self.review_agent = agent
                self.initialized = True

    def get_review_agent(self) -> Optional[Any]:
        """Thread-safe method to get review agent"""
        if self._lock:
            with self._lock:
                return self.review_agent
        return self.review_agent

    def is_initialized(self) -> bool:
        """Thread-safe method to check initialization status"""
        if self._lock:
            with self._lock:
                return self.initialized
        return self.initialized

    def clear(self) -> None:
        """Thread-safe method to clear application state"""
        if self._lock:
            with self._lock:
                self.review_agent = None
                self.initialized = False
        else:
            self.review_agent = None
            self.initialized = False


# Global application state instance
app_state = AppState()


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        if app_state.shutdown_event:
            app_state.shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


async def initialize_resources():
    """Initialize application resources"""
    logger.info("Initializing application resources")

    try:
        # Initialize review agent
        review_agent = await initialize_review_agent()
        app_state.mark_initialized(review_agent)
        logger.info(f"Review agent initialized with model: {settings.ai_model}")

    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        raise ConfigurationException(
            message="Application startup failed",
            details={"initialization_stage": "resources"},
            original_error=e,
        )


async def cleanup_resources():
    """Cleanup application resources"""
    logger.info("Cleaning up application resources")

    try:
        # Cleanup review agent if it exists
        review_agent = app_state.get_review_agent()
        if review_agent:
            # Check if review agent has cleanup methods and call them
            if hasattr(review_agent, "close") and callable(
                getattr(review_agent, "close")
            ):
                if asyncio.iscoroutinefunction(review_agent.close):
                    await review_agent.close()
                else:
                    review_agent.close()
            logger.info("Review agent cleanup complete")

        # Clear app state
        app_state.clear()

        # Clear secrets cache for security
        from src.utils.secrets import clear_secrets_cache

        clear_secrets_cache()
        logger.info("Secrets cache cleared")

        logger.info("Resource cleanup complete")

    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with graceful shutdown"""
    # Startup
    logger.info(f"Starting GitLab AI Reviewer in {settings.environment} mode")
    logger.info(f"GitLab URL: {settings.gitlab_url}")
    logger.info(
        f"Rate limiting: {'enabled' if settings.rate_limit_enabled else 'disabled'}"
    )

    await initialize_resources()

    yield

    # Shutdown
    logger.info("Shutting down GitLab AI Reviewer")
    if app_state.shutdown_event:
        app_state.shutdown_event.set()
    await cleanup_resources()
    logger.info("Graceful shutdown complete")


# Setup signal handlers
setup_signal_handlers()

# Create FastAPI application
app = FastAPI(
    title="GitLab AI Code Review Agent",
    version=get_version(),
    description="AI-powered code review automation for GitLab",
    lifespan=lifespan,
)


# Exception handlers for FastAPI HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTPException with enhanced logging"""
    # Determine log level based on status code
    if exc.status_code >= 500:
        log_method = logger.error
    elif exc.status_code >= 400:
        log_method = logger.warning
    else:
        log_method = logger.info

    log_method(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method,
            "client_host": request.client.host if request.client else None,
        },
    )

    # If detail is a dict (our enhanced format), use it as-is
    if isinstance(exc.detail, dict):
        content = exc.detail
        # Add timestamp if not present
        if "timestamp" not in content:
            content["timestamp"] = str(asyncio.get_event_loop().time())
    else:
        # Convert simple string detail to our format
        content = {
            "error": "HTTP Error",
            "message": str(exc.detail),
            "type": "http_error",
            "timestamp": str(asyncio.get_event_loop().time()),
        }

    return JSONResponse(status_code=exc.status_code, content=content)


# Exception handlers for custom exceptions
@app.exception_handler(SecurityException)
async def security_exception_handler(request: Request, exc: SecurityException):
    """Handle security-related exceptions"""
    logger.warning(
        f"Security exception: {exc.message}",
        extra={
            "exception_type": "SecurityException",
            "details": exc.details,
            "path": str(request.url.path),
            "client_host": request.client.host if request.client else None,
        },
    )
    return JSONResponse(
        status_code=401
        if "authentication" in exc.details.get("security_context", "")
        else 403,
        content={
            "error": "Security Error",
            "message": "Access denied",
            "type": "security_error",
            "timestamp": str(asyncio.get_event_loop().time()),
        },
    )


@app.exception_handler(RateLimitException)
async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
    """Handle rate limiting exceptions"""
    logger.warning(
        f"Rate limit exceeded: {exc.message}",
        extra={
            "exception_type": "RateLimitException",
            "details": exc.details,
            "path": str(request.url.path),
            "client_host": request.client.host if request.client else None,
        },
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate Limit Exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": exc.retry_after,
            "type": "rate_limit_error",
            "timestamp": str(asyncio.get_event_loop().time()),
        },
        headers={"Retry-After": str(exc.retry_after)} if exc.retry_after else {},
    )


@app.exception_handler(WebhookValidationException)
async def webhook_validation_exception_handler(
    request: Request, exc: WebhookValidationException
):
    """Handle webhook validation exceptions"""
    logger.error(
        f"Webhook validation failed: {exc.message}",
        extra={
            "exception_type": "WebhookValidationException",
            "details": exc.details,
            "path": str(request.url.path),
        },
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid Webhook",
            "message": "Webhook payload validation failed",
            "type": "validation_error",
            "timestamp": str(asyncio.get_event_loop().time()),
        },
    )


@app.exception_handler(GitLabAPIException)
async def gitlab_api_exception_handler(request: Request, exc: GitLabAPIException):
    """Handle GitLab API exceptions"""
    logger.error(
        f"GitLab API error: {exc.message}",
        extra={
            "exception_type": "GitLabAPIException",
            "details": exc.details,
            "status_code": exc.status_code,
            "path": str(request.url.path),
        },
    )
    return JSONResponse(
        status_code=502,
        content={
            "error": "External Service Error",
            "message": "GitLab API is currently unavailable",
            "type": "external_service_error",
            "timestamp": str(asyncio.get_event_loop().time()),
        },
    )


@app.exception_handler(AIProviderException)
async def ai_provider_exception_handler(request: Request, exc: AIProviderException):
    """Handle AI provider exceptions"""
    logger.error(
        f"AI provider error: {exc.message}",
        extra={
            "exception_type": "AIProviderException",
            "details": exc.details,
            "provider": exc.provider,
            "model": exc.model,
            "path": str(request.url.path),
        },
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": "AI Service Unavailable",
            "message": "AI review service is temporarily unavailable",
            "type": "service_unavailable",
            "timestamp": str(asyncio.get_event_loop().time()),
        },
    )


@app.exception_handler(ConfigurationException)
async def configuration_exception_handler(
    request: Request, exc: ConfigurationException
):
    """Handle configuration exceptions"""
    logger.error(
        f"Configuration error: {exc.message}",
        extra={
            "exception_type": "ConfigurationException",
            "details": exc.details,
            "path": str(request.url.path),
        },
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Configuration Error",
            "message": "Service is misconfigured. Please contact administrator.",
            "type": "configuration_error",
            "timestamp": str(asyncio.get_event_loop().time()),
        },
    )


@app.exception_handler(GitLabReviewerException)
async def gitlab_reviewer_exception_handler(
    request: Request, exc: GitLabReviewerException
):
    """Handle any other GitLabReviewerException (catch-all)"""
    logger.error(
        f"GitLab reviewer error: {exc.message}",
        extra={
            "exception_type": type(exc).__name__,
            "details": exc.details,
            "path": str(request.url.path),
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Service Error",
            "message": "An error occurred while processing your request",
            "type": "internal_error",
            "timestamp": str(asyncio.get_event_loop().time()),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unexpected exceptions"""
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={
            "exception_type": type(exc).__name__,
            "path": str(request.url.path),
            "client_host": request.client.host if request.client else None,
        },
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "type": "unexpected_error",
            "timestamp": str(asyncio.get_event_loop().time()),
        },
    )


# Add CORS middleware with secure defaults
if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],  # Restrict to required methods
        allow_headers=["Authorization", "X-Gitlab-Token", "Content-Type"],
    )
    logger.info(f"CORS enabled for origins: {settings.allowed_origins}")
else:
    logger.warning("No CORS origins configured - CORS middleware not added")

# Add custom middleware (order matters!)
app.add_middleware(SecurityMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(RequestTracingMiddleware)

# Include routers
app.include_router(webhooks.router, prefix="/webhook", tags=["webhooks"])
app.include_router(health.router, prefix="/health", tags=["health"])


@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "service": "GitLab AI Code Review Agent",
        "version": get_version(),
        "status": "running" if app_state.is_initialized() else "starting",
        "environment": settings.environment,
        "features": {
            "rate_limiting": settings.rate_limit_enabled,
            "cors": len(settings.allowed_origins) > 0,
            "api_authentication": settings.api_key is not None,
            "webhook_secret": settings.gitlab_webhook_secret is not None,
        },
    }


# Dependency injection functions
async def get_review_agent():
    """Dependency injection for review agent"""
    agent = app_state.get_review_agent()
    if not agent:
        raise ConfigurationException(
            message="Review agent not initialized", config_key="review_agent"
        )
    return agent


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
