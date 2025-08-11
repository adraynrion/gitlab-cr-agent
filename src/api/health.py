"""
Health check endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import httpx
import logging
from datetime import datetime

from src.config.settings import settings
from src.exceptions import GitLabAPIException

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness probe - checks if the service is running
    Used by Kubernetes/Docker to determine if container should be restarted
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "GitLab AI Code Review Agent"
    }

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe - checks if the service is ready to accept traffic
    Used by Kubernetes/Docker to determine if container should receive requests
    """
    checks = {}
    all_healthy = True
    
    # Check GitLab connectivity
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.gitlab_url}/api/v4/version",
                headers={"PRIVATE-TOKEN": settings.gitlab_token}
            )
            if response.status_code == 200:
                checks["gitlab"] = {
                    "status": "healthy", 
                    "response_time": response.elapsed.total_seconds() if response.elapsed else 0
                }
            else:
                checks["gitlab"] = {
                    "status": "unhealthy", 
                    "error": f"HTTP {response.status_code}",
                    "details": response.text[:200] if hasattr(response, 'text') else None
                }
                all_healthy = False
    except httpx.RequestError as e:
        logger.warning(f"GitLab health check failed - network error: {e}")
        checks["gitlab"] = {"status": "unhealthy", "error": f"Network error: {str(e)}"}
        all_healthy = False
    except httpx.HTTPStatusError as e:
        logger.warning(f"GitLab health check failed - HTTP error: {e}")
        checks["gitlab"] = {"status": "unhealthy", "error": f"HTTP {e.response.status_code}: {e.response.text[:100]}"}
        all_healthy = False
    except Exception as e:
        logger.error(f"GitLab health check failed - unexpected error: {e}")
        checks["gitlab"] = {"status": "unhealthy", "error": f"Unexpected error: {str(e)}"}
        all_healthy = False
    
    # Check AI model configuration
    try:
        from src.agents.providers import get_llm_model
        model = get_llm_model(settings.ai_model)
        checks["ai_model"] = {
            "status": "healthy", 
            "model": settings.ai_model,
            "provider_type": type(model).__name__
        }
    except Exception as e:
        logger.warning(f"AI model configuration check failed: {e}")
        checks["ai_model"] = {
            "status": "unhealthy", 
            "error": str(e),
            "model": settings.ai_model
        }
        all_healthy = False
    
    status_code = 200 if all_healthy else 503
    
    result = {
        "status": "healthy" if all_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }
    
    if not all_healthy:
        raise HTTPException(status_code=status_code, detail=result)
    
    return result

@router.get("/status")
async def status() -> Dict[str, Any]:
    """
    Detailed status endpoint for monitoring and debugging
    """
    return {
        "service": "GitLab AI Code Review Agent",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment,
        "configuration": {
            "gitlab_url": settings.gitlab_url,
            "ai_model": settings.ai_model,
            "trigger_tag": settings.gitlab_trigger_tag,
            "debug_mode": settings.debug
        }
    }