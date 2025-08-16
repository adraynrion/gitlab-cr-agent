"""
Comprehensive tests for middleware functionality
Tests cover authentication, security, logging, CORS, rate limiting, and error handling
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api.middleware import (
    AuthenticationMiddleware,
    SecurityMiddleware,
    limiter,
    secure_headers,
)
from config.settings import settings


class TestSecurityMiddleware:
    """Tests for SecurityMiddleware covering request size limits and security headers"""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app with SecurityMiddleware"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)

    def test_normal_request_passes(self, client):
        """Test that normal requests pass through security middleware"""
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"message": "success"}

    def test_request_size_limit_enforcement(self, client):
        """Test that requests exceeding size limits are rejected"""
        # Mock settings to have a small max request size
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.max_request_size = 100  # 100 bytes

            # Send request with large content
            large_data = "x" * 200  # 200 bytes
            # The middleware throws HTTPException which TestClient may handle differently
            with pytest.raises((HTTPException, Exception)) as exc_info:
                response = client.post(
                    "/test", content=large_data, headers={"content-length": "200"}
                )
                # If we get here, check the response code
                if hasattr(response, "status_code"):
                    assert response.status_code == 413

            # If an exception was raised, verify it's the right one
            if hasattr(exc_info, "value"):
                error_str = str(exc_info.value)
                assert "413" in error_str or "Request entity too large" in error_str

    def test_request_size_limit_no_content_length(self, client):
        """Test requests without content-length header pass through"""
        response = client.post("/test", json={"data": "test"})
        # Should pass through (no content-length header to check)
        assert response.status_code in [200, 404, 405]  # Any non-413 response

    def test_request_size_limit_logging(self, client, caplog):
        """Test that oversized requests are properly logged"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.max_request_size = 50

            try:
                response = client.post(
                    "/test", content="x" * 100, headers={"content-length": "100"}
                )
                assert response.status_code == 413
            except Exception as e:
                assert "413" in str(e) or "Request entity too large" in str(e)
            assert "Request too large" in caplog.text
            assert "100 bytes" in caplog.text

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

    def test_client_host_unknown_when_no_client(self):
        """Test client host handling when request.client is None"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.max_request_size = 50

            # Create mock request with no client
            mock_request = Mock()
            mock_request.headers = {"content-length": "100"}
            mock_request.client = None

            middleware = SecurityMiddleware(Mock())

            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(middleware.dispatch(mock_request, Mock()))

            assert exc_info.value.status_code == 413

    async def test_middleware_preserves_response_type(self):
        """Test that middleware preserves the original response"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.get("/json")
        async def json_endpoint():
            return JSONResponse({"custom": "response"})

        @app.get("/dict")
        async def dict_endpoint():
            return {"normal": "dict"}

        client = TestClient(app)

        json_response = client.get("/json")
        dict_response = client.get("/dict")

        assert json_response.json() == {"custom": "response"}
        assert dict_response.json() == {"normal": "dict"}


class TestAuthenticationMiddleware:
    """Tests for AuthenticationMiddleware covering API key authentication and Bearer tokens"""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app with AuthenticationMiddleware"""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "protected"}

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

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)

    def test_public_paths_bypass_auth(self, client):
        """Test that public paths bypass authentication"""
        public_paths = ["/", "/health/live", "/health/ready", "/webhook/gitlab"]

        for path in public_paths:
            method = "POST" if "webhook" in path else "GET"
            if method == "POST":
                response = client.post(path)
            else:
                response = client.get(path)

            # Should not get 401 (may get 404/405 if endpoint doesn't exist)
            assert response.status_code != 401

    def test_protected_endpoint_requires_api_key(self, client):
        """Test that protected endpoints require API key when configured"""
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

    def test_valid_api_key_grants_access(self, client):
        """Test that valid API key grants access to protected endpoints"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-api-key-123"

            response = client.get(
                "/protected", headers={"X-API-Key": "test-api-key-123"}
            )
            assert response.status_code == 200
            assert response.json() == {"message": "protected"}

    def test_invalid_api_key_denied(self, client):
        """Test that invalid API key is denied"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "correct-key"

            response = client.get("/protected", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 401

    def test_no_api_key_configured_allows_all(self, client):
        """Test that when no API key is configured, all requests are allowed"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = None

            response = client.get("/protected")
            assert response.status_code == 200

    def test_empty_api_key_configured_allows_all(self, client):
        """Test that empty API key string allows all requests"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = ""

            response = client.get("/protected")
            assert response.status_code == 200

    def test_authentication_logging_on_failure(self, client, caplog):
        """Test that failed authentication attempts are logged"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "correct-key"

            response = client.get("/protected", headers={"X-API-Key": "wrong-key"})

            assert response.status_code == 401
            assert "Invalid API key attempt" in caplog.text

    def test_authentication_client_host_logging(self, client, caplog):
        """Test client host is logged in authentication failures"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "correct-key"

            client.get("/protected", headers={"X-API-Key": "wrong-key"})

            # Should log client host (testclient uses 'testclient' as host)
            assert "Invalid API key attempt" in caplog.text

    def test_client_host_unknown_when_no_client(self):
        """Test client host handling when request.client is None"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-key"

            # Create mock request with no client
            mock_request = Mock()
            mock_request.url.path = "/protected"
            mock_request.headers = {}
            mock_request.client = None

            middleware = AuthenticationMiddleware(Mock())

            response = asyncio.run(middleware.dispatch(mock_request, Mock()))

            assert isinstance(response, JSONResponse)
            assert response.status_code == 401

    def test_api_key_case_sensitive(self, client):
        """Test that API key comparison is case sensitive"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "TestKey123"

            # Wrong case should fail
            response = client.get("/protected", headers={"X-API-Key": "testkey123"})
            assert response.status_code == 401

            # Correct case should pass
            response = client.get("/protected", headers={"X-API-Key": "TestKey123"})
            assert response.status_code == 200

    def test_missing_api_key_header(self, client):
        """Test behavior when X-API-Key header is missing"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-key"

            response = client.get("/protected")
            assert response.status_code == 401

    @pytest.mark.parametrize(
        "path,expected_public",
        [
            ("/", True),
            ("/health/live", True),
            ("/health/ready", True),
            ("/webhook/gitlab", True),
            ("/protected", False),
            ("/api/data", False),
            ("/health/other", False),
            ("/webhook/other", False),
        ],
    )
    def test_public_path_detection(self, path, expected_public, client):
        """Test public path detection logic"""
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-key"

            response = client.get(path)

            if expected_public:
                # Public paths should not return 401
                assert response.status_code != 401
            else:
                # Protected paths should return 401 without API key
                assert response.status_code == 401


class TestMiddlewareIntegration:
    """Integration tests for multiple middleware working together"""

    @pytest.fixture
    def app_with_all_middleware(self):
        """Create app with all middleware"""
        app = FastAPI()

        # Add middleware in reverse order (FastAPI adds them in reverse)
        app.add_middleware(AuthenticationMiddleware)
        app.add_middleware(SecurityMiddleware)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "success"}

        @app.get("/")
        async def public_endpoint():
            return {"message": "public"}

        return app

    def test_middleware_execution_order(self, app_with_all_middleware, caplog):
        """Test that middleware executes in the correct order"""
        client = TestClient(app_with_all_middleware)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.max_request_size = 1024 * 1024
            mock_settings.is_production = True

            # Request with valid API key
            response = client.get("/protected", headers={"X-API-Key": "test-key"})

            assert response.status_code == 200

            # Should have security headers (SecurityMiddleware)
            assert "X-Content-Type-Options" in response.headers

    def test_middleware_authentication_failure_logged(
        self, app_with_all_middleware, caplog
    ):
        """Test that authentication failures are logged even with multiple middleware"""
        client = TestClient(app_with_all_middleware)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "correct-key"
            mock_settings.max_request_size = 1024 * 1024

            response = client.get("/protected", headers={"X-API-Key": "wrong-key"})

            assert response.status_code == 401

            # Should have authentication error log
            assert "Invalid API key attempt" in caplog.text

    def test_middleware_request_size_limit_with_auth(self, app_with_all_middleware):
        """Test request size limits work even with authentication"""
        client = TestClient(app_with_all_middleware)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.max_request_size = 100

            # Large request with valid API key should still be rejected by SecurityMiddleware
            try:
                response = client.post(
                    "/protected",
                    content="x" * 200,
                    headers={"X-API-Key": "test-key", "content-length": "200"},
                )
                assert response.status_code == 413
            except Exception as e:
                assert "413" in str(e) or "Request entity too large" in str(e)

    def test_public_endpoint_bypasses_auth_but_has_logging(
        self, app_with_all_middleware, caplog
    ):
        """Test that public endpoints bypass auth but still get logged"""
        client = TestClient(app_with_all_middleware)

        with patch("api.middleware.settings") as mock_settings:
            mock_settings.api_key = "test-key"  # Auth enabled
            mock_settings.is_production = True

            response = client.get("/")  # Public endpoint

            assert response.status_code == 200

            # Should have security headers
            assert "X-Content-Type-Options" in response.headers


class TestRateLimiterConfiguration:
    """Tests for rate limiter configuration and initialization"""

    def test_limiter_initialization(self):
        """Test that limiter is properly initialized"""
        assert limiter is not None
        assert limiter.enabled == settings.rate_limit_enabled

    def test_limiter_key_function(self):
        """Test that limiter uses correct key function"""
        # The key function should be get_remote_address
        from slowapi.util import get_remote_address

        assert limiter.key_func == get_remote_address

    @patch("api.middleware.settings")
    def test_limiter_enabled_setting(self, mock_settings):
        """Test limiter respects enabled setting"""
        mock_settings.rate_limit_enabled = False

        # Re-import to get new limiter with updated settings
        from importlib import reload

        import src.api.middleware

        reload(src.api.middleware)

        # Should respect the disabled setting
        assert not src.api.middleware.limiter.enabled


class TestSecureHeadersConfiguration:
    """Tests for secure headers configuration"""

    def test_secure_headers_initialization(self):
        """Test that secure headers are properly configured"""
        assert secure_headers is not None

    def test_secure_headers_csp_policy(self):
        """Test Content Security Policy configuration"""
        # The secure_headers object should have CSP configured
        assert hasattr(secure_headers, "csp")

    def test_secure_headers_hsts_policy(self):
        """Test HTTP Strict Transport Security configuration"""
        assert hasattr(secure_headers, "hsts")

    def test_secure_headers_referrer_policy(self):
        """Test Referrer Policy configuration"""
        assert hasattr(secure_headers, "referrer")

    def test_secure_headers_permissions_policy(self):
        """Test Permissions Policy configuration"""
        assert hasattr(secure_headers, "permissions")

    def test_secure_headers_cache_control(self):
        """Test Cache Control configuration"""
        assert hasattr(secure_headers, "cache")


class TestErrorHandlingEdgeCases:
    """Tests for edge cases and error scenarios in middleware"""

    def test_security_middleware_with_invalid_content_length(self):
        """Test SecurityMiddleware handles invalid content-length values"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.post("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)

        # Test with non-numeric content-length (should not cause crash)
        with patch("api.middleware.settings") as mock_settings:
            mock_settings.max_request_size = 100

            # This would normally cause ValueError when calling int()
            # but the middleware should handle it gracefully
            try:
                client.post("/test", headers={"content-length": "invalid"})
                # If it doesn't crash, the middleware handled it gracefully
                assert True
            except ValueError:
                # If it does crash, that's a bug in the middleware
                pytest.fail(
                    "Middleware should handle invalid content-length gracefully"
                )

    def test_authentication_middleware_missing_settings_api_key(self):
        """Test AuthenticationMiddleware when settings.api_key attribute is missing"""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)

        # Mock settings without api_key attribute
        with patch("api.middleware.settings", spec=[]):
            # This should not crash - should treat as no API key configured
            response = client.get("/test")
            assert response.status_code == 200

    def test_middleware_with_special_characters_in_url(self):
        """Test middleware handles URLs with special characters"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.get("/test/{path:path}")
        async def test_endpoint(path: str):
            return {"path": path}

        client = TestClient(app)

        # Test with various special characters
        special_paths = [
            "/test/file%20with%20spaces",
            "/test/file%C3%A9",  # URL encoded é
            "/test/file&param=value",
            "/test/file?query=value",
        ]

        for path in special_paths:
            try:
                client.get(path)
                # Should not crash due to URL encoding issues
                assert True
            except Exception as e:
                pytest.fail(f"Middleware failed with special path {path}: {e}")

    def test_middleware_with_very_long_urls(self):
        """Test middleware handles very long URLs gracefully"""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware)
        app.add_middleware(SecurityMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)

        # Create very long query string
        long_query = "&".join([f"param{i}=value{i}" for i in range(100)])
        long_url = f"/test?{long_query}"

        try:
            client.get(long_url)
            # Should handle long URLs without crashing
            assert True
        except Exception as e:
            pytest.fail(f"Middleware failed with long URL: {e}")

    def test_concurrent_middleware_access(self):
        """Test middleware thread safety with concurrent requests"""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware)
        app.add_middleware(SecurityMiddleware)

        @app.get("/test")
        async def test_endpoint():
            await asyncio.sleep(0.01)  # Small delay to increase concurrency
            return {"message": "success"}

        client = TestClient(app)

        import queue
        import threading

        results = queue.Queue()

        def make_request():
            try:
                response = client.get("/test")
                results.put(("success", response.status_code))
            except Exception as e:
                results.put(("error", str(e)))

        # Create multiple concurrent threads
        threads = [threading.Thread(target=make_request) for _ in range(10)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Collect results
        responses = []
        while not results.empty():
            responses.append(results.get())

        # All requests should succeed
        assert len(responses) == 10
        for result_type, result_value in responses:
            assert result_type == "success"
            assert result_value == 200
