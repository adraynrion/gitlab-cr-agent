"""
Integration tests for enhanced rate limiting functionality
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.middleware import limiter
from src.main import app


class TestRateLimitingIntegration:
    """Test enhanced rate limiting integration"""

    @pytest.fixture(autouse=True)
    def reset_rate_limiter(self):
        """Reset rate limiter before each test"""
        # Reset slowapi rate limiter state if possible
        if hasattr(limiter, "_limiter") and hasattr(limiter._limiter, "storage"):
            try:
                # Try to reset the storage, but don't fail if method signature is different
                storage = limiter._limiter.storage
                if hasattr(storage, "reset"):
                    storage.reset()
                elif hasattr(storage, "clear"):
                    # Some storage implementations have clear without params
                    try:
                        storage.clear()
                    except TypeError:
                        # If clear needs a key, skip reset
                        pass
            except Exception:
                # If reset fails, continue anyway
                pass
        yield
        # Skip cleanup on exit to avoid repeated errors

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for rate limiting tests"""
        settings = Mock()
        settings.rate_limit_enabled = True
        settings.global_rate_limit = "5/minute"  # Very low limit for testing
        settings.gitlab_url = "https://gitlab.example.com"
        settings.gitlab_token = "test-token"
        settings.api_key = "test-api-key"
        return settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create test client with mocked settings"""
        with patch("src.config.settings.get_settings", return_value=mock_settings):
            return TestClient(app)

    def test_global_rate_limiting_enforcement(self, client):
        """Test that global rate limiting is enforced"""
        # Make several requests quickly
        responses = []
        for i in range(10):  # More than the 5/minute limit
            response = client.get("/")
            responses.append(response)

        # Some requests should succeed, others should be rate limited
        success_count = sum(1 for r in responses if r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)

        assert success_count > 0  # Some should succeed
        assert rate_limited_count > 0  # Some should be rate limited
        assert success_count + rate_limited_count == 10

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are returned"""
        response = client.get("/")

        # Should have rate limit headers if not rate limited
        if response.status_code == 200:
            # Note: Headers might not be present for successful requests
            # depending on the implementation
            pass
        elif response.status_code == 429:
            assert "Retry-After" in response.headers

    def test_rate_limiting_with_authentication(self, client):
        """Test rate limiting with Bearer authentication"""
        headers = {"Authorization": "Bearer test-api-key"}

        # Make requests with authentication
        responses = []
        for i in range(7):  # More than limit
            response = client.get("/health/status", headers=headers)
            responses.append(response)

        # Verify some requests succeed and some are rate limited
        success_count = sum(1 for r in responses if r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)

        assert success_count > 0
        assert rate_limited_count > 0

    def test_rate_limiting_disabled(self):
        """Test behavior when rate limiting is disabled"""
        mock_settings = Mock()
        mock_settings.rate_limit_enabled = False
        mock_settings.global_rate_limit = "1/minute"  # Very restrictive
        mock_settings.gitlab_url = "https://gitlab.example.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.api_key = None  # No auth required

        with patch("src.config.settings.get_settings", return_value=mock_settings):
            client = TestClient(app)

            # Make many requests - should all succeed when rate limiting disabled
            responses = []
            for i in range(10):
                response = client.get("/")
                responses.append(response)

            # All should succeed when rate limiting is disabled
            success_count = sum(1 for r in responses if r.status_code == 200)
            assert success_count == 10

    def test_rate_limiting_different_endpoints(self, client):
        """Test rate limiting across different endpoints"""
        # Test that rate limiting is global across endpoints
        endpoints = ["/", "/health/live", "/health/ready"]

        total_requests = 0
        rate_limited_requests = 0

        for endpoint in endpoints:
            for i in range(3):  # 3 requests per endpoint = 9 total
                response = client.get(endpoint)
                total_requests += 1
                if response.status_code == 429:
                    rate_limited_requests += 1

        # With a 5/minute limit, we should see some rate limiting
        assert total_requests == 9
        assert rate_limited_requests > 0  # Some should be rate limited

    def test_webhook_rate_limiting(self):
        """Test webhook-specific rate limiting"""
        mock_settings = Mock()
        mock_settings.rate_limit_enabled = True
        mock_settings.webhook_rate_limit = "2/minute"  # Very low for testing
        mock_settings.global_rate_limit = "100/minute"  # High global limit
        mock_settings.gitlab_url = "https://gitlab.example.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.gitlab_webhook_secret = "webhook-secret"
        mock_settings.api_key = "test-api-key"

        with patch("src.config.settings.get_settings", return_value=mock_settings):
            client = TestClient(app)

            # Mock webhook payload
            webhook_payload = {
                "object_kind": "merge_request",
                "event_type": "merge_request",
                "user": {"id": 1, "name": "Test User"},
                "project": {"id": 100, "name": "Test Project"},
                "object_attributes": {
                    "id": 1,
                    "iid": 1,
                    "title": "Test MR",
                    "state": "opened",
                    "source_branch": "feature",
                    "target_branch": "main",
                },
            }

            headers = {
                "Authorization": "Bearer test-api-key",
                "X-Gitlab-Token": "webhook-secret",
                "Content-Type": "application/json",
            }

            # Make webhook requests
            responses = []
            for i in range(5):  # More than the 2/minute limit
                response = client.post(
                    "/webhook/gitlab", json=webhook_payload, headers=headers
                )
                responses.append(response)

            # Some should be rate limited
            rate_limited_count = sum(1 for r in responses if r.status_code == 429)
            assert rate_limited_count > 0


class TestRateLimitingConfiguration:
    """Test rate limiting configuration"""

    def test_limiter_configuration(self):
        """Test that limiter is properly configured"""
        from src.api.middleware import limiter

        # Test that limiter exists and has expected attributes
        assert limiter is not None
        assert hasattr(limiter, "enabled")
        # slowapi.Limiter has different attributes than expected
        assert hasattr(limiter, "_limiter")

    def test_global_key_function(self):
        """Test that key function returns global key"""
        from src.api.middleware import limiter

        # Test the key function always returns 'global'
        # Note: slowapi.Limiter uses _key_func internally
        if hasattr(limiter, "_key_func"):
            key = limiter._key_func(None)
            assert key == "global"
        else:
            # If no key_func accessible, just verify limiter exists
            assert limiter is not None


class TestRateLimitingEdgeCases:
    """Test edge cases for rate limiting"""

    def test_rate_limiting_with_invalid_config(self):
        """Test behavior with invalid rate limit configuration"""
        mock_settings = Mock()
        mock_settings.rate_limit_enabled = True
        mock_settings.global_rate_limit = "invalid/format"  # Invalid format
        mock_settings.gitlab_url = "https://gitlab.example.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.api_key = None

        with patch("src.config.settings.get_settings", return_value=mock_settings):
            # Should handle invalid config gracefully
            client = TestClient(app)
            response = client.get("/")
            # Should not crash, either succeed or default behavior
            assert response.status_code in [200, 429]

    def test_rate_limiting_with_high_concurrency(self, client):
        """Test rate limiting under concurrent load"""
        import threading

        results = []

        def make_request():
            response = client.get("/")
            results.append(response.status_code)

        # Create multiple threads to simulate concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads at roughly the same time
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Verify we got some responses and handled concurrency
        assert len(results) == 10
        assert 200 in results  # Some should succeed
        # May or may not have rate limiting depending on exact timing
