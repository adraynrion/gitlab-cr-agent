"""
Authentication & logging middleware
"""

import logging
import time
from typing import Callable
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import secure

from src.config.settings import settings
from src.exceptions import SecurityException, RateLimitException

logger = logging.getLogger(__name__)

# Create rate limiter instance
limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled
)

# Security headers configuration
secure_headers = secure.Secure(
    csp=secure.ContentSecurityPolicy().default_src("'self'").script_src("'self'"),
    hsts=secure.StrictTransportSecurity().max_age(31536000).include_subdomains(),
    referrer=secure.ReferrerPolicy().no_referrer(),
    permissions=secure.PermissionsPolicy().geolocation("none").camera("none"),
    cache=secure.CacheControl().no_cache().no_store().must_revalidate()
)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with headers, size limits, and validation"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size limits
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > settings.max_request_size:
            logger.warning(f"Request too large: {content_length} bytes from {request.client.host}")
            raise HTTPException(
                status_code=413, 
                detail=f"Request entity too large. Maximum size: {settings.max_request_size} bytes"
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        if settings.is_production:
            secure_headers.framework.fastapi(response)
        
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for health checks, root endpoint, and webhook endpoints
        public_paths = ["/", "/health/live", "/health/ready", "/webhook/gitlab"]
        if request.url.path in public_paths:
            return await call_next(request)
        
        # Skip auth if no API key is configured
        if not settings.api_key:
            return await call_next(request)
        
        # Check API key for protected endpoints
        api_key = request.headers.get("X-API-Key")
        if api_key != settings.api_key:
            logger.warning(f"Invalid API key attempt from {request.client.host}")
            # Return JSON response directly from middleware
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid API key",
                    "message": "Access denied",
                    "type": "authentication_error",
                    "timestamp": str(time.time())
                }
            )
        
        return await call_next(request)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} - "
                f"{request.method} {request.url} - "
                f"Time: {process_time:.3f}s"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url} - "
                f"Error: {str(e)} - Time: {process_time:.3f}s",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "process_time": process_time,
                    "error_type": type(e).__name__,
                    "client_host": str(request.client.host) if request.client else None
                }
            )
            raise