"""
Authentication & logging middleware
"""

import logging
import time
from typing import Callable
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.config.settings import settings

logger = logging.getLogger(__name__)

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
            raise HTTPException(status_code=401, detail="Invalid API key")
        
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
                f"Error: {str(e)} - Time: {process_time:.3f}s"
            )
            raise