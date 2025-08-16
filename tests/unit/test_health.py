"""Unit tests for health check endpoints."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.health import liveness_check
from src.api.health import router as health_router
from src.api.health import status


class TestHealthStatus:
    """Test the main health status endpoint."""

    @patch("src.api.health.settings")
    @patch("src.api.health.get_version")
    @pytest.mark.asyncio
    async def test_status_endpoint(self, mock_get_version, mock_settings):
        """Test status endpoint returns healthy."""
        mock_settings.environment = "development"
        mock_settings.gitlab_url = "https://gitlab.example.com"
        mock_settings.ai_model = "openai:gpt-4o"
        mock_settings.gitlab_trigger_tag = "review"
        mock_settings.debug = False
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


class TestVersionIntegration:
    """Test version integration in health endpoints."""

    @patch("src.api.health.get_version")
    @pytest.mark.asyncio
    async def test_version_integration_in_status(self, mock_get_version):
        """Test that status endpoint correctly uses version utility."""
        mock_get_version.return_value = "2.1.0"

        with patch("src.api.health.settings") as mock_settings:
            mock_settings.environment = "development"
            mock_settings.gitlab_url = "https://gitlab.example.com"
            mock_settings.ai_model = "openai:gpt-4o"
            mock_settings.gitlab_trigger_tag = "review"
            mock_settings.debug = False

            result = await status()

            assert result["version"] == "2.1.0"
            mock_get_version.assert_called_once()

    @patch("src.api.health.get_version")
    @pytest.mark.asyncio
    async def test_version_error_handling(self, mock_get_version):
        """Test version error handling in health endpoint."""
        mock_get_version.side_effect = Exception("Version file not found")

        with patch("src.api.health.settings") as mock_settings:
            mock_settings.environment = "development"
            mock_settings.gitlab_url = "https://gitlab.example.com"
            mock_settings.ai_model = "openai:gpt-4o"
            mock_settings.gitlab_trigger_tag = "review"
            mock_settings.debug = False

            # Should handle version error gracefully
            with pytest.raises(Exception):
                await status()


class TestHealthRouter:
    """Test health check router configuration."""

    def test_health_router_has_routes(self):
        """Test that health router has expected routes."""
        route_paths = [route.path for route in health_router.routes]

        expected_paths = ["/live", "/ready", "/status"]
        for path in expected_paths:
            assert path in route_paths

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

        with patch("src.api.health.settings") as mock_settings:
            mock_settings.environment = "development"
            mock_settings.gitlab_url = "https://gitlab.example.com"
            mock_settings.ai_model = "openai:gpt-4o"
            mock_settings.gitlab_trigger_tag = "review"
            mock_settings.debug = False

            response = client.get("/health/status")
            assert response.status_code == 200

            data = response.json()
            assert data["service"] == "GitLab AI Code Review Agent"
            assert data["version"] == "2.1.0"
