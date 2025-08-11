"""
GitLab AI Code Review Agent
Main FastAPI application with PydanticAI integration
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn

from src.api import webhooks, health
from src.api.middleware import AuthenticationMiddleware, LoggingMiddleware
from src.config.settings import settings
from src.agents.code_reviewer import initialize_review_agent

# Configure structured logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent instance
review_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global review_agent
    
    # Startup
    logger.info(f"Starting GitLab AI Reviewer in {settings.environment} mode")
    review_agent = await initialize_review_agent()
    logger.info(f"Initialized AI agent with model: {settings.ai_model}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GitLab AI Reviewer")

# Create FastAPI application
app = FastAPI(
    title="GitLab AI Code Review Agent",
    version="1.0.0",
    description="AI-powered code review automation for GitLab",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(webhooks.router, prefix="/webhook", tags=["webhooks"])
app.include_router(health.router, prefix="/health", tags=["health"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "GitLab AI Code Review Agent",
        "version": "1.0.0",
        "status": "operational"
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )