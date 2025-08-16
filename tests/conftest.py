"""Pytest configuration and fixtures for the GitLab Code Review Agent tests."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response

# Add src to path for imports - must be before src imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Now import from src
from agents.code_reviewer import CodeReviewAgent
from config.settings import Settings
from models.gitlab_models import (
    GitLabProject,
    GitLabUser,
    MergeRequestAttributes,
    MergeRequestEvent,
)
from services.gitlab_service import GitLabService
from services.review_service import ReviewService

# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture(fixtures_dir: Path):
    """Factory to load fixture files."""

    def _load(filename: str) -> Any:
        filepath = fixtures_dir / filename
        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                return json.load(f)
        else:
            with open(filepath, "r") as f:
                return f.read()

    return _load


# ============================================================================
# Settings and Configuration Fixtures
# ============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    env_vars = {
        "GITLAB_TOKEN": "test-gitlab-token-1234567890",  # 20+ characters
        "GITLAB_URL": "https://gitlab.example.com",
        "GITLAB_WEBHOOK_SECRET": "test-webhook-secret-1234567890",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "AZURE_OPENAI_API_KEY": "test-azure-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "GEMINI_API_KEY": "test-gemini-key",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "REVIEW_PROVIDER": "openai",
        "REVIEW_MODEL": "gpt-4",
        "REVIEW_MAX_TOKENS": "2000",
        "REVIEW_TEMPERATURE": "0.3",
        "LOG_LEVEL": "INFO",
        "RATE_LIMIT_REQUESTS": "100",
        "RATE_LIMIT_PERIOD": "60",
        "CORS_ORIGINS": '["http://localhost:3000", "https://gitlab.example.com"]',
        "ALLOWED_USERS": '["user1", "user2"]',
        "ALLOWED_GROUPS": '["group1", "group2"]',
        "REVIEW_SKIP_DRAFT": "true",
        "REVIEW_SKIP_WIP": "true",
        "REVIEW_FILE_EXTENSIONS": '[".py", ".js", ".ts", ".go"]',
        "REVIEW_IGNORE_PATTERNS": '["**/test_*.py", "**/__pycache__/**"]',
        "REVIEW_MAX_FILES": "50",
        "REVIEW_MAX_LINES": "1000",
        "AUTH_ENABLED": "true",
        "AUTH_TOKEN": "test-auth-token",
        # New performance and security settings
        "MAX_DIFF_SIZE": "1048576",  # 1MB
        "REQUEST_TIMEOUT": "30.0",
        "MAX_CONNECTIONS": "100",
        "MAX_KEEPALIVE_CONNECTIONS": "20",
        "KEEPALIVE_EXPIRY": "30.0",
        "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "5",
        "CIRCUIT_BREAKER_TIMEOUT": "60",
        "CIRCUIT_BREAKER_EXPECTED_EXCEPTION": "httpx.HTTPStatusError,httpx.RequestError",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def settings(mock_env_vars) -> Settings:
    """Create a Settings instance with test configuration."""
    return Settings()


@pytest.fixture
def mock_settings(settings: Settings) -> Generator[Settings, None, None]:
    """Mock the settings module."""
    with patch("config.settings.settings", settings):
        yield settings


# ============================================================================
# Application Fixtures
# ============================================================================


@pytest.fixture
def app(mock_env_vars):
    """Create a FastAPI app instance for testing."""
    # Import here to avoid circular imports and ensure env vars are set
    from main import app

    return app


@pytest.fixture
def client(app) -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def sample_user() -> GitLabUser:
    """Create a sample GitLab user."""
    return GitLabUser(
        id=1,
        username="testuser",
        name="Test User",
        email="test@example.com",
        avatar_url="https://example.com/avatar.jpg",
    )


@pytest.fixture
def sample_project() -> GitLabProject:
    """Create a sample GitLab project."""
    return GitLabProject(
        id=100,
        name="test-project",
        description="Test project description",
        web_url="https://gitlab.example.com/test-group/test-project",
        avatar_url="https://example.com/avatar.jpg",
        git_ssh_url="git@gitlab.example.com:test-group/test-project.git",
        git_http_url="https://gitlab.example.com/test-group/test-project.git",
        namespace="test-group",
        visibility_level=0,
        path_with_namespace="test-group/test-project",
        default_branch="main",
    )


@pytest.fixture
def sample_repository() -> Dict[str, Any]:
    """Create a sample repository."""
    return {
        "name": "test-project",
        "url": "https://gitlab.example.com/test-group/test-project.git",
        "description": "Test project",
        "homepage": "https://gitlab.example.com/test-group/test-project",
    }


@pytest.fixture
def sample_object_attributes() -> MergeRequestAttributes:
    """Create sample object attributes."""
    return MergeRequestAttributes(
        id=1,
        iid=10,
        title="Add new feature",
        description="This MR adds a new feature",
        state="opened",
        created_at="2024-08-15 12:00:00 UTC",
        updated_at="2024-08-15 12:30:00 UTC",
        target_branch="main",
        source_branch="feature-branch",
        source_project_id=100,
        target_project_id=100,
        author_id=1,
        url="https://gitlab.example.com/test-group/test-project/-/merge_requests/10",
        source={"id": 100, "name": "test-project"},
        target={"id": 100, "name": "test-project"},
        last_commit={"id": "abc123", "message": "Add feature"},
        work_in_progress=False,
        action="open",
    )


@pytest.fixture
def merge_request_event(
    sample_user: GitLabUser,
    sample_project: GitLabProject,
    sample_repository: Dict[str, Any],
    sample_object_attributes: MergeRequestAttributes,
) -> MergeRequestEvent:
    """Create a complete merge request event."""
    return MergeRequestEvent(
        object_kind="merge_request",
        event_type="merge_request",
        user=sample_user,
        project=sample_project,
        repository=sample_repository,
        object_attributes=sample_object_attributes,
    )


# Removed problematic fixtures that depend on non-existent models


# ============================================================================
# Service Fixtures
# ============================================================================


@pytest.fixture
def mock_gitlab_service(settings: Settings) -> Mock:
    """Create a mock GitLab service."""
    mock_service = Mock(spec=GitLabService)
    mock_service.settings = settings
    mock_service.session = Mock()

    # Set up common return values for core functions only
    mock_service.get_merge_request = AsyncMock(
        return_value={"id": 1, "iid": 10, "title": "Test MR", "state": "opened"}
    )

    mock_service.get_merge_request_diff = AsyncMock(
        return_value=[
            {
                "old_path": "src/main.py",
                "new_path": "src/main.py",
                "diff": "@@ -1,5 +1,6 @@\n def hello():\n-    print('hello')\n+    print('hello world')",
            }
        ]
    )

    mock_service.post_merge_request_comment = AsyncMock(return_value={"id": 1})

    return mock_service


@pytest.fixture
def mock_review_service(settings: Settings, mock_gitlab_service: Mock) -> Mock:
    """Create a mock review service."""
    mock_service = Mock(spec=ReviewService)
    mock_service.settings = settings
    mock_service.gitlab_service = mock_gitlab_service
    mock_service.process_review = AsyncMock()
    return mock_service


@pytest.fixture
def mock_code_review_agent(settings: Settings) -> Mock:
    """Create a mock code review agent."""
    mock_agent = Mock(spec=CodeReviewAgent)
    mock_agent.settings = settings
    mock_agent.review_merge_request = AsyncMock()
    return mock_agent


# ============================================================================
# FastAPI Application Fixtures (Removed - not needed for ReviewService tests)
# ============================================================================


# ============================================================================
# HTTP Response Fixtures
# ============================================================================


@pytest.fixture
def mock_http_response() -> Mock:
    """Create a mock HTTP response."""
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={"status": "success"})
    mock_response.text = "Success"
    mock_response.headers = {"content-type": "application/json"}
    return mock_response


@pytest.fixture
def mock_gitlab_api_responses() -> Dict[str, Any]:
    """Create mock responses for GitLab API calls."""
    return {
        "merge_request": {
            "id": 1,
            "iid": 10,
            "title": "Test MR",
            "state": "opened",
            "web_url": "https://gitlab.example.com/test/merge_requests/10",
        },
        "changes": {
            "changes": [
                {
                    "old_path": "test.py",
                    "new_path": "test.py",
                    "diff": "@@ -1 +1 @@\n-old line\n+new line",
                }
            ]
        },
        "comment": {
            "id": 100,
            "body": "Test comment",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "user": {
            "id": 1,
            "username": "testuser",
            "email": "test@example.com",
            "is_admin": False,
        },
        "project": {
            "id": 100,
            "name": "test-project",
            "namespace": {"full_path": "test-group"},
        },
    }


# ============================================================================
# Async Helpers
# ============================================================================


@pytest.fixture
def async_return():
    """Helper to create async return values."""

    def _async_return(value):
        future = AsyncMock()
        future.return_value = value
        return future

    return _async_return


# ============================================================================
# Test Data Generators
# ============================================================================


@pytest.fixture
def generate_webhook_payload():
    """Generate webhook payload with custom attributes."""

    def _generate(**kwargs):
        base_payload = {
            "object_kind": "merge_request",
            "event_type": "merge_request",
            "user": {
                "id": 1,
                "username": "testuser",
                "name": "Test User",
                "email": "test@example.com",
            },
            "project": {
                "id": 100,
                "name": "test-project",
                "namespace": "test-group",
                "web_url": "https://gitlab.example.com/test-group/test-project",
                "git_ssh_url": "git@gitlab.example.com:test-group/test-project.git",
                "git_http_url": "https://gitlab.example.com/test-group/test-project.git",
                "visibility_level": 0,
                "path_with_namespace": "test-group/test-project",
                "default_branch": "main",
            },
            "repository": {
                "name": "test-project",
                "url": "https://gitlab.example.com/test-group/test-project.git",
            },
            "object_attributes": {
                "id": 1,
                "iid": 10,
                "title": "Test MR",
                "description": "Test description",
                "state": "opened",
                "action": "open",
                "source_branch": "feature",
                "target_branch": "main",
                "source_project_id": 100,
                "target_project_id": 100,
                "author_id": 1,
                "url": "https://gitlab.example.com/test-group/test-project/-/merge_requests/10",
                "created_at": "2024-08-15 12:00:00 UTC",
                "updated_at": "2024-08-15 12:00:00 UTC",
                "source": {"id": 100, "name": "test-project"},
                "target": {"id": 100, "name": "test-project"},
                "last_commit": {"id": "abc123", "message": "Test commit"},
                "work_in_progress": False,
                "draft": False,
            },
            "assignees": [],
        }

        # Deep merge kwargs into base_payload
        def deep_merge(base, update):
            for key, value in update.items():
                if (
                    isinstance(value, dict)
                    and key in base
                    and isinstance(base[key], dict)
                ):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(base_payload, kwargs)
        return base_payload

    return _generate


# ============================================================================
# New Feature Fixtures (Version System, Security, Performance)
# ============================================================================


@pytest.fixture
def mock_version_file(tmp_path):
    """Create a temporary version.txt file for testing."""

    def create_version_file(version: str = "2.1.0"):
        version_file = tmp_path / "version.txt"
        version_file.write_text(f"{version}\n")
        return version_file

    return create_version_file


@pytest.fixture
def mock_version_utility():
    """Mock the version utility functions."""
    with patch("src.utils.version.get_version") as mock_get_version, patch(
        "src.utils.version.get_version_info"
    ) as mock_get_version_info:
        mock_get_version.return_value = "2.1.0"
        mock_get_version_info.return_value = {
            "version": "2.1.0",
            "major": 2,
            "minor": 1,
            "patch": 0,
            "full": "v2.1.0",
        }

        yield {
            "get_version": mock_get_version,
            "get_version_info": mock_get_version_info,
        }


@pytest.fixture
def webhook_headers():
    """Generate basic webhook headers for testing."""
    return {
        "Content-Type": "application/json",
        "X-Gitlab-Event": "Merge Request Hook",
        "X-Gitlab-Token": "test-webhook-secret",
    }


@pytest.fixture
def webhook_headers_with_timestamp():
    """Generate webhook headers with timestamp for security testing."""
    import time

    def create_headers(
        secret: str = "test-webhook-secret",
        timestamp: float = None,
        include_timestamp: bool = True,
    ) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": secret,
        }

        if include_timestamp:
            actual_timestamp = timestamp or time.time()
            headers["X-Gitlab-Timestamp"] = str(actual_timestamp)

        return headers

    return create_headers


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing."""

    def create_mock_breaker(failure_threshold: int = 5, timeout: int = 60):
        breaker = Mock()
        breaker.failure_threshold = failure_threshold
        breaker.timeout = timeout
        breaker.failure_count = 0
        breaker.state = "closed"  # closed, open, half_open
        breaker.last_failure_time = None

        def call_with_breaker(func):
            """Simulate circuit breaker behavior."""
            if breaker.state == "open":
                # Check if timeout has passed
                import time

                if (
                    breaker.last_failure_time
                    and (time.time() - breaker.last_failure_time) > timeout
                ):
                    breaker.state = "half_open"
                else:
                    raise Exception("Circuit breaker is open")

            try:
                result = func()
                if breaker.state == "half_open":
                    breaker.state = "closed"
                    breaker.failure_count = 0
                return result
            except Exception as e:
                breaker.failure_count += 1
                if breaker.failure_count >= failure_threshold:
                    breaker.state = "open"
                    breaker.last_failure_time = time.time()
                raise e

        breaker.call = call_with_breaker
        return breaker

    return create_mock_breaker


@pytest.fixture
def security_test_data():
    """Provide test data for security feature testing."""
    import time

    return {
        "valid_timestamp": time.time(),
        "old_timestamp": time.time() - 400,  # 6+ minutes ago
        "future_timestamp": time.time() + 400,  # 6+ minutes in future
        "invalid_timestamps": [
            "not-a-number",
            "2024-01-01T12:00:00Z",
            "",
            "inf",
            "nan",
        ],
        "valid_webhook_secret": "secure-webhook-secret-12345",
        "invalid_webhook_secrets": ["", "weak", "123", None],
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_async_mocks():
    """Cleanup async mocks after each test."""
    yield
    # Clean up any lingering async tasks
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
    except RuntimeError:
        pass


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "security: mark test as security-related")
