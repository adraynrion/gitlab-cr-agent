"""Unit tests for health check endpoints."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.health import liveness_check, readiness_check
from src.api.health import router as health_router
from src.api.health import status


class TestHealthStatus:
    """Test the main health status endpoint."""

    @patch("src.api.health.get_settings")
    @patch("src.api.health.get_version")
    @pytest.mark.asyncio
    async def test_status_endpoint(self, mock_get_version, mock_get_settings):
        """Test status endpoint returns healthy."""
        mock_settings = Mock()
        mock_settings.environment = "development"
        mock_settings.gitlab_url = "https://gitlab.example.com"
        mock_settings.ai_model = "openai:gpt-4o"
        mock_settings.gitlab_trigger_tag = "review"
        mock_settings.debug = False
        mock_get_settings.return_value = mock_settings
        mock_get_version.return_value = "2.1.0"

        result = await status()

        assert result["service"] == "GitLab AI Code Review Agent"
        assert result["version"] == "2.1.0"
        assert result["environment"] == "development"
        assert "timestamp" in result
        assert "configuration" in result


class TestLivenessCheck:
    """Test the liveness probe endpoint."""

    @pytest.mark.asyncio
    async def test_liveness_check_success(self):
        """Test liveness check returns healthy status."""
        result = await liveness_check()

        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert result["service"] == "GitLab AI Code Review Agent"

    def test_liveness_check_function_direct(self):
        """Test liveness_check function directly (sync call)"""
        import asyncio

        result = asyncio.run(liveness_check())

        assert result["status"] == "healthy"
        assert result["service"] == "GitLab AI Code Review Agent"
        assert "timestamp" in result


class TestReadinessCheck:
    """Test the readiness probe endpoint."""

    @pytest.mark.asyncio
    @patch("src.api.health.get_settings")
    @patch("src.api.health.httpx.AsyncClient")
    async def test_readiness_check_healthy(self, mock_client, mock_settings):
        """Test readiness check with healthy dependencies."""
        # Mock settings
        mock_settings.return_value = Mock(
            gitlab_url="https://gitlab.example.com",
            gitlab_token="test-token",
            ai_model="openai:gpt-4",
        )

        # Mock successful GitLab response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.5

        mock_client_instance = Mock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance

        # Mock AI model check
        with patch("src.agents.providers.get_llm_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            result = await readiness_check()

            assert result["status"] == "healthy"
            assert "timestamp" in result
            assert "checks" in result
            assert result["checks"]["gitlab"]["status"] == "healthy"
            assert result["checks"]["ai_model"]["status"] == "healthy"

    @pytest.mark.asyncio
    @patch("src.api.health.get_settings")
    @patch("src.api.health.httpx.AsyncClient")
    async def test_readiness_check_gitlab_unhealthy(self, mock_client, mock_settings):
        """Test readiness check with unhealthy GitLab."""
        # Mock settings
        mock_settings.return_value = Mock(
            gitlab_url="https://gitlab.example.com",
            gitlab_token="test-token",
            ai_model="openai:gpt-4",
        )

        # Mock failed GitLab response
        mock_client_instance = Mock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.get = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.return_value = mock_client_instance

        # Mock AI model check
        with patch("src.agents.providers.get_llm_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            with pytest.raises(HTTPException) as exc_info:
                await readiness_check()

            assert exc_info.value.status_code == 503
            detail = exc_info.value.detail
            assert detail["status"] == "unhealthy"
            assert detail["checks"]["gitlab"]["status"] == "unhealthy"


class TestVersionIntegration:
    """Test version integration in health endpoints."""

    @patch("src.api.health.get_version")
    @pytest.mark.asyncio
    async def test_version_integration_in_status(self, mock_get_version):
        """Test that status endpoint correctly uses version utility."""
        mock_get_version.return_value = "2.1.0"

        with patch("src.api.health.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                environment="development",
                gitlab_url="https://gitlab.example.com",
                ai_model="openai:gpt-4o",
                gitlab_trigger_tag="review",
                debug=False,
            )

            result = await status()

            assert result["version"] == "2.1.0"
            mock_get_version.assert_called_once()

    @patch("src.api.health.get_version")
    @pytest.mark.asyncio
    async def test_version_error_handling(self, mock_get_version):
        """Test version error handling in health endpoint."""
        mock_get_version.side_effect = Exception("Version file not found")

        with patch("src.api.health.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                environment="development",
                gitlab_url="https://gitlab.example.com",
                ai_model="openai:gpt-4o",
                gitlab_trigger_tag="review",
                debug=False,
            )

            # Should handle version error gracefully
            with pytest.raises(Exception):
                await status()

    @patch("src.api.health.get_version")
    @patch("src.api.health.get_settings")
    @pytest.mark.asyncio
    async def test_status_comprehensive_config(self, mock_settings, mock_get_version):
        """Test status endpoint with comprehensive configuration."""
        mock_get_version.return_value = "2.5.0"
        mock_settings.return_value = Mock(
            environment="production",
            gitlab_url="https://gitlab.prod.com",
            ai_model="anthropic:claude-3-opus",
            gitlab_trigger_tag="ai-review",
            debug=True,
        )

        result = await status()

        assert result["version"] == "2.5.0"
        assert result["environment"] == "production"
        assert result["configuration"]["gitlab_url"] == "https://gitlab.prod.com"
        assert result["configuration"]["ai_model"] == "anthropic:claude-3-opus"
        assert result["configuration"]["trigger_tag"] == "ai-review"
        assert result["configuration"]["debug_mode"] is True


class TestHealthRouter:
    """Test health check router configuration."""

    def test_health_router_has_routes(self):
        """Test that health router has expected routes."""
        route_paths = [route.path for route in health_router.routes]

        expected_paths = ["/live", "/ready", "/status"]
        for path in expected_paths:
            assert path in route_paths

    def test_router_route_methods(self):
        """Test routes have correct HTTP methods."""
        for route in health_router.routes:
            if hasattr(route, "methods"):
                if route.path == "/live":
                    assert "GET" in route.methods
                elif route.path == "/status":
                    assert "GET" in route.methods
                elif route.path == "/ready":
                    assert "GET" in route.methods

    def test_health_router_configuration(self):
        """Test basic router configuration."""
        assert health_router is not None
        assert len(health_router.routes) >= 3


class TestHealthEndpointIntegration:
    """Test health endpoints with FastAPI test client."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app with health routes."""
        app = FastAPI()
        app.include_router(health_router, prefix="/health")
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    def test_liveness_endpoint_integration(self, client):
        """Test liveness endpoint returns success."""
        response = client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data

    @patch("src.api.health.get_version")
    def test_status_endpoint_integration(self, mock_get_version, client):
        """Test status endpoint integration."""
        mock_get_version.return_value = "2.1.0"

        with patch("src.api.health.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                environment="development",
                gitlab_url="https://gitlab.example.com",
                ai_model="openai:gpt-4o",
                gitlab_trigger_tag="review",
                debug=False,
            )

            response = client.get("/health/status")
            assert response.status_code == 200

            data = response.json()
            assert data["service"] == "GitLab AI Code Review Agent"
            assert data["version"] == "2.1.0"

    @patch("src.api.health.get_settings")
    @patch("src.api.health.httpx.AsyncClient")
    def test_readiness_endpoint_integration(self, mock_client, mock_settings, client):
        """Test readiness endpoint integration."""
        # Mock settings
        mock_settings.return_value = Mock(
            gitlab_url="https://gitlab.example.com",
            gitlab_token="test-token",
            ai_model="openai:gpt-4",
        )

        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.3

        mock_client_instance = Mock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance

        with patch("src.agents.providers.get_llm_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            response = client.get("/health/ready")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert "checks" in data
