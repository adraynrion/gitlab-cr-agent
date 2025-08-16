"""
Simple middleware tests without complex dependencies
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api.middleware import AuthenticationMiddleware, SecurityMiddleware


class TestSecurityMiddleware:
    """Tests for SecurityMiddleware covering request size limits and security headers"""

    def test_middleware_initialization(self):
        """Test that SecurityMiddleware can be initialized"""
        middleware = SecurityMiddleware(Mock())
        assert middleware is not None

    def test_normal_request_passes(self):
        """Test that normal requests pass through security middleware"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"message": "success"}

    def test_request_size_limit_enforcement(self):
        """Test that requests exceeding size limits are rejected"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.post("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)

        # Mock settings to have a small max request size
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.max_request_size = 100  # 100 bytes

            # Send request with large content
            large_data = "x" * 200  # 200 bytes
            response = client.post(
                "/test", content=large_data, headers={"content-length": "200"}
            )

            assert response.status_code == 413
            assert "Request entity too large" in response.text

    def test_request_size_limit_no_content_length(self):
        """Test requests without content-length header pass through"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.post("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)
        response = client.post("/test", json={"data": "test"})
        # Should pass through (no content-length header to check)
        assert response.status_code in [200, 404, 405]  # Any non-413 response

    @pytest.mark.parametrize("is_production", [True, False])
    def test_security_headers_production_only(self, is_production):
        """Test security headers are only added in production"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.is_production = is_production

            client = TestClient(app)
            response = client.get("/test")

            if is_production:
                assert "X-Content-Type-Options" in response.headers
                assert "X-Frame-Options" in response.headers
                assert "X-XSS-Protection" in response.headers
                assert "Strict-Transport-Security" in response.headers

                assert response.headers["X-Content-Type-Options"] == "nosniff"
                assert response.headers["X-Frame-Options"] == "DENY"
                assert response.headers["X-XSS-Protection"] == "1; mode=block"
                assert (
                    "max-age=31536000" in response.headers["Strict-Transport-Security"]
                )
            else:
                assert "X-Content-Type-Options" not in response.headers
                assert "X-Frame-Options" not in response.headers
                assert "X-XSS-Protection" not in response.headers
                assert "Strict-Transport-Security" not in response.headers


class TestAuthenticationMiddleware:
    """Tests for AuthenticationMiddleware covering API key authentication"""

    def test_middleware_initialization(self):
        """Test that AuthenticationMiddleware can be initialized"""
        middleware = AuthenticationMiddleware(Mock())
        assert middleware is not None

    def test_public_paths_bypass_auth(self):
        """Test that public paths bypass authentication"""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware)

        @app.get("/")
        async def root():
            return {"message": "root"}

        @app.get("/health/live")
        async def health_live():
            return {"status": "ok"}

        @app.get("/health/ready")
        async def health_ready():
            return {"status": "ready"}

        @app.post("/webhook/gitlab")
        async def webhook():
            return {"status": "received"}

        client = TestClient(app)

        public_paths = [
            ("/", "GET"),
            ("/health/live", "GET"),
            ("/health/ready", "GET"),
            ("/webhook/gitlab", "POST"),
        ]

        for path, method in public_paths:
            if method == "POST":
                response = client.post(path)
            else:
                response = client.get(path)

            # Should not get 401 (may get 404/405 if endpoint doesn't exist)
            assert response.status_code != 401

    def test_protected_endpoint_requires_api_key(self):
        """Test that protected endpoints require API key when configured"""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "protected"}

        client = TestClient(app)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-api-key-123"

            # Request without API key
            response = client.get("/protected")
            assert response.status_code == 401

            response_data = response.json()
            assert response_data["error"] == "Invalid API key"
            assert response_data["message"] == "Access denied"
            assert response_data["type"] == "authentication_error"
            assert "timestamp" in response_data

    def test_valid_api_key_grants_access(self):
        """Test that valid API key grants access to protected endpoints"""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "protected"}

        client = TestClient(app)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-api-key-123"

            response = client.get(
                "/protected", headers={"X-API-Key": "test-api-key-123"}
            )
            assert response.status_code == 200
            assert response.json() == {"message": "protected"}

    def test_no_api_key_configured_allows_all(self):
        """Test that when no API key is configured, all requests are allowed"""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "protected"}

        client = TestClient(app)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = None

            response = client.get("/protected")
            assert response.status_code == 200


class TestMiddlewareIntegration:
    """Integration tests for multiple middleware working together"""

    def test_middleware_execution_order(self, caplog):
        """Test that middleware executes in the correct order"""
        app = FastAPI()

        # Add middleware in reverse order (FastAPI adds them in reverse)
        app.add_middleware(AuthenticationMiddleware)
        app.add_middleware(SecurityMiddleware)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "success"}

        client = TestClient(app)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.max_request_size = 1024 * 1024
            mock_settings.is_production = True

            # Request with valid API key
            response = client.get("/protected", headers={"X-API-Key": "test-key"})

            assert response.status_code == 200

            # Should have security headers (SecurityMiddleware)
            assert "X-Content-Type-Options" in response.headers

    def test_authentication_failure_still_logged(self, caplog):
        """Test that authentication failures are logged even with multiple middleware"""
        app = FastAPI()

        # Add middleware in reverse order
        app.add_middleware(AuthenticationMiddleware)
        app.add_middleware(SecurityMiddleware)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "success"}

        client = TestClient(app)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "correct-key"
            mock_settings.max_request_size = 1024 * 1024

            response = client.get("/protected", headers={"X-API-Key": "wrong-key"})

            assert response.status_code == 401

            # Should have authentication error log
            assert "Invalid API key attempt" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
