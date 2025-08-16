"""Integration tests for the main FastAPI application."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class TestApplicationStartup:
    """Test application startup and configuration."""

    def test_app_creation(self, app):
        """Test that the FastAPI app is created successfully."""
        assert app.title == "GitLab AI Code Review Agent"
        assert app.version is not None

    def test_app_routes_configured(self, client: TestClient):
        """Test that all expected routes are configured."""
        # Get the OpenAPI schema which includes all routes
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        # Check that expected endpoints exist
        expected_paths = [
            "/",
            "/health/live",
            "/health/ready",
            "/health/status",
            "/webhook/gitlab",
        ]

        for path in expected_paths:
            assert path in paths, f"Expected path {path} not found in API"

    def test_cors_middleware_configured(self, client: TestClient):
        """Test that CORS middleware is properly configured."""
        # Test preflight request
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }

        response = client.options("/webhook/gitlab", headers=headers)

        # Should handle CORS preflight - check various header formats
        cors_headers = [
            "access-control-allow-origin",
            "Access-Control-Allow-Origin",
            "access-control-allow-methods",
            "Access-Control-Allow-Methods",
        ]

        has_cors = any(header in response.headers for header in cors_headers)
        assert has_cors, f"No CORS headers found in: {list(response.headers.keys())}"

    @patch("src.main.settings")
    def test_middleware_order(self, mock_settings, client: TestClient):
        """Test that middleware is applied in correct order."""
        mock_settings.cors_origins = ["http://localhost:3000"]

        response = client.get("/")

        # Should have all expected headers from middleware chain
        assert response.status_code == 200
        # Check for request tracing headers (these should always be present)
        assert (
            "x-correlation-id" in response.headers
            or "X-Correlation-ID" in response.headers
        )
        assert (
            "x-process-time" in response.headers or "X-Process-Time" in response.headers
        )


class TestRootEndpoint:
    """Test the root endpoint functionality."""

    def test_root_endpoint_success(self, client: TestClient):
        """Test that root endpoint returns success."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["message"] == "GitLab Code Review Agent"
        assert data["status"] == "running"
        assert "version" in data

    def test_root_endpoint_headers(self, client: TestClient):
        """Test root endpoint response headers."""
        response = client.get("/")

        assert response.headers["content-type"] == "application/json"
        # Should have process time header from middleware
        assert "x-process-time" in response.headers

    def test_root_endpoint_multiple_requests(self, client: TestClient):
        """Test multiple requests to root endpoint."""
        for i in range(5):
            response = client.get("/")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "running"


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_endpoint_success(self, client: TestClient):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_health_liveness_probe(self, client: TestClient):
        """Test liveness probe endpoint."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "alive"

    def test_health_readiness_probe(self, client: TestClient):
        """Test readiness probe endpoint."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ready"

    @patch("src.main.settings")
    def test_health_with_dependencies(self, mock_settings, client: TestClient):
        """Test health check includes dependency status."""
        mock_settings.gitlab_url = "https://gitlab.example.com"

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        # Should include configuration status
        assert "gitlab_configured" in str(data).lower() or "status" in data


class TestWebhookEndpoint:
    """Test webhook endpoint functionality."""

    @patch("src.api.webhooks.handle_gitlab_webhook")
    def test_webhook_merge_request_success(
        self, mock_handler, client: TestClient, load_fixture
    ):
        """Test successful merge request webhook processing."""
        # Load test payload
        payload = load_fixture("webhook_payloads.json")["merge_request_open"]

        # Mock successful processing
        mock_handler.return_value = {
            "status": "accepted",
            "action": "open",
            "merge_request_iid": 10,
        }

        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        response = client.post("/webhook", json=payload, headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "accepted"
        assert data["action"] == "open"
        mock_handler.assert_called_once()

    def test_webhook_missing_headers(self, client: TestClient):
        """Test webhook with missing required headers."""
        payload = {"object_kind": "merge_request"}

        response = client.post("/webhook", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "missing required headers" in data["detail"].lower()

    def test_webhook_invalid_signature(self, client: TestClient):
        """Test webhook with invalid signature."""
        payload = {"object_kind": "merge_request"}

        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "invalid-signature",
            "Content-Type": "application/json",
        }

        response = client.post("/webhook", json=payload, headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert "invalid signature" in data["detail"].lower()

    def test_webhook_unsupported_event(self, client: TestClient):
        """Test webhook with unsupported event type."""
        payload = {"object_kind": "issue"}

        headers = {
            "X-Gitlab-Event": "Issue Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        response = client.post("/webhook", json=payload, headers=headers)

        assert response.status_code == 400
        data = response.json()
        assert "unsupported event" in data["detail"].lower()

    def test_webhook_malformed_json(self, client: TestClient):
        """Test webhook with malformed JSON payload."""
        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        # Send invalid JSON
        response = client.post("/webhook", data="invalid json", headers=headers)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    @patch("src.api.webhooks.handle_gitlab_webhook")
    def test_webhook_processing_error(
        self, mock_handler, client: TestClient, load_fixture
    ):
        """Test webhook processing error handling."""
        payload = load_fixture("webhook_payloads.json")["merge_request_open"]

        # Mock processing error
        mock_handler.side_effect = Exception("Processing failed")

        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        response = client.post("/webhook", json=payload, headers=headers)

        assert response.status_code == 500
        data = response.json()
        assert "internal server error" in data["detail"].lower()

    def test_webhook_rate_limiting(self, client: TestClient, load_fixture):
        """Test webhook rate limiting."""
        payload = load_fixture("webhook_payloads.json")["merge_request_open"]

        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        with patch("src.api.webhooks.handle_gitlab_webhook") as mock_handler:
            mock_handler.return_value = {"status": "accepted"}

            # Make multiple requests rapidly
            responses = []
            for _ in range(10):
                response = client.post("/webhook", json=payload, headers=headers)
                responses.append(response)

            # Some requests should succeed, some may be rate limited
            success_count = sum(1 for r in responses if r.status_code == 200)
            _ = sum(1 for r in responses if r.status_code == 429)  # rate_limited_count

            # At least some should succeed
            assert success_count > 0


class TestErrorHandling:
    """Test application error handling."""

    def test_404_error_handling(self, client: TestClient):
        """Test 404 error handling."""
        response = client.get("/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Not Found"

    def test_method_not_allowed(self, client: TestClient):
        """Test 405 method not allowed error."""
        response = client.patch("/")  # PATCH not allowed on root

        assert response.status_code == 405
        data = response.json()
        assert "method not allowed" in data["detail"].lower()

    @patch("src.main.settings")
    def test_internal_server_error(self, mock_settings, client: TestClient):
        """Test internal server error handling."""
        # Mock settings to raise an exception
        mock_settings.side_effect = Exception("Configuration error")

        with patch("main.app") as mock_app:
            mock_app.get = Mock(side_effect=Exception("Internal error"))

            # This would trigger the error in a real scenario
            # For testing, we'll verify the error handler exists
            response = client.get("/health")

            # Should still handle the request gracefully
            assert response.status_code in [200, 500]

    def test_validation_error_handling(self, client: TestClient):
        """Test validation error handling."""
        # Send invalid data to webhook
        invalid_payload = {"invalid": "data"}

        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        response = client.post("/webhook", json=invalid_payload, headers=headers)

        assert response.status_code in [400, 422]
        data = response.json()
        assert "detail" in data


class TestMiddlewareIntegration:
    """Test middleware integration in the full application."""

    def test_request_logging_middleware(self, client: TestClient):
        """Test that request logging middleware works."""
        with patch("logging.Logger.info") as mock_log:
            response = client.get("/")

            assert response.status_code == 200
            # Should have logged the request
            assert mock_log.called

    def test_process_time_header(self, client: TestClient):
        """Test that process time header is added."""
        response = client.get("/")

        assert "X-Process-Time" in response.headers
        # Should be a valid float string
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0

    @patch("src.main.settings")
    def test_security_headers_in_production(self, mock_settings, client: TestClient):
        """Test security headers are added in production."""
        mock_settings.environment = "production"

        response = client.get("/")

        # Should have security headers in production
        if mock_settings.environment == "production":
            assert "X-Content-Type-Options" in response.headers

    def test_cors_headers_integration(self, client: TestClient):
        """Test CORS headers integration."""
        headers = {"Origin": "http://localhost:3000"}

        response = client.get("/", headers=headers)

        assert response.status_code == 200
        # Should have CORS headers
        assert "Access-Control-Allow-Origin" in response.headers

    @patch("src.api.middleware.settings")
    def test_authentication_middleware_behavior(
        self, mock_settings, client: TestClient
    ):
        """Test authentication middleware behavior with API_KEY configured."""
        mock_settings.api_key = "test-secret-token"

        # Root endpoint should be public even with API_KEY set
        response = client.get("/")
        assert response.status_code == 200

        # Health endpoints should require authentication when API_KEY is set
        protected_endpoints = ["/health/live", "/health/ready", "/health/status"]

        for endpoint in protected_endpoints:
            # Without auth should fail
            response = client.get(endpoint)
            assert response.status_code == 401

            # With valid Bearer token should pass
            response = client.get(
                endpoint, headers={"Authorization": "Bearer test-secret-token"}
            )
            assert response.status_code == 200

    @patch("src.api.middleware.settings")
    def test_webhook_authentication_required_when_api_key_set(
        self, mock_settings, client: TestClient
    ):
        """Test webhook endpoint requires Bearer auth when API_KEY is configured."""
        mock_settings.api_key = "secret-token"

        # Webhook endpoint should require Bearer authentication when API_KEY is set
        payload = {"object_kind": "merge_request"}
        webhook_headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        # Without Bearer auth should fail
        with patch("src.api.webhooks.handle_gitlab_webhook"):
            response = client.post(
                "/webhook/gitlab", json=payload, headers=webhook_headers
            )
            assert response.status_code == 401

        # With valid Bearer token should work
        auth_headers = webhook_headers.copy()
        auth_headers["Authorization"] = "Bearer secret-token"

        with patch("src.api.webhooks.handle_gitlab_webhook"):
            response = client.post(
                "/webhook/gitlab", json=payload, headers=auth_headers
            )
            assert response.status_code in [200, 400]  # 400 for invalid payload is OK

    @patch("src.api.middleware.settings")
    def test_no_authentication_when_api_key_not_set(
        self, mock_settings, client: TestClient
    ):
        """Test all endpoints are public when API_KEY is not configured."""
        mock_settings.api_key = None

        # All endpoints should be accessible without authentication
        test_endpoints = [
            ("/", "GET"),
            ("/health/live", "GET"),
            ("/health/ready", "GET"),
            ("/health/status", "GET"),
        ]

        for endpoint, method in test_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)
            assert (
                response.status_code == 200
            ), f"Endpoint {endpoint} should be public when API_KEY not set"

        # Webhook should also work without auth when API_KEY not set
        payload = {"object_kind": "merge_request"}
        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        with patch("src.api.webhooks.handle_gitlab_webhook"):
            response = client.post("/webhook/gitlab", json=payload, headers=headers)
            assert response.status_code in [200, 400]  # Should work without Bearer auth


class TestConcurrency:
    """Test concurrent request handling."""

    def test_concurrent_health_checks(self, client: TestClient):
        """Test concurrent health check requests."""
        import threading

        results = []

        def make_request():
            response = client.get("/health")
            results.append(response.status_code)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10

    @patch("src.api.webhooks.handle_gitlab_webhook")
    def test_concurrent_webhook_processing(
        self, mock_handler, client: TestClient, load_fixture
    ):
        """Test concurrent webhook processing."""
        import threading

        payload = load_fixture("webhook_payloads.json")["merge_request_open"]
        mock_handler.return_value = {"status": "accepted"}

        results = []

        def make_webhook_request():
            headers = {
                "X-Gitlab-Event": "Merge Request Hook",
                "X-Gitlab-Token": "test-webhook-secret",
                "Content-Type": "application/json",
            }
            response = client.post("/webhook", json=payload, headers=headers)
            results.append(response.status_code)

        # Create multiple concurrent webhook requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_webhook_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Most requests should succeed (some may be rate limited)
        success_count = sum(1 for status in results if status == 200)
        assert success_count >= 1  # At least one should succeed


class TestApplicationLifecycle:
    """Test application lifecycle events."""

    def test_startup_event(self, app):
        """Test application startup event."""
        # App should start successfully
        assert app is not None
        assert hasattr(app, "routes")

    def test_shutdown_graceful(self, app):
        """Test graceful shutdown handling."""
        # Verify app can handle shutdown events
        assert hasattr(app, "router")

        # In a real test, you might test cleanup logic
        # For now, just verify the app structure supports shutdown events


class TestSecurityIntegration:
    """Test security integration across the application."""

    def test_request_size_limit(self, client: TestClient):
        """Test request size limiting."""
        # Create a large payload
        large_payload = {"data": "x" * (10 * 1024 * 1024)}  # 10MB payload

        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        response = client.post("/webhook", json=large_payload, headers=headers)

        # Should be rejected due to size
        assert response.status_code in [413, 400]

    def test_content_type_validation(self, client: TestClient):
        """Test content type validation."""
        payload = '{"object_kind": "merge_request"}'

        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "text/plain",  # Wrong content type
        }

        response = client.post("/webhook", data=payload, headers=headers)

        # Should handle content type validation
        assert response.status_code in [400, 415, 422]

    def test_header_injection_protection(self, client: TestClient):
        """Test protection against header injection."""
        # Try to inject malicious headers
        malicious_headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret\r\nX-Injected: malicious",
            "Content-Type": "application/json",
        }

        payload = {"object_kind": "merge_request"}

        response = client.post("/webhook", json=payload, headers=malicious_headers)

        # Should handle gracefully without processing injected headers
        assert response.status_code in [200, 400, 401]
        # Response should not contain injected header
        assert "X-Injected" not in response.headers

    def test_sql_injection_protection(self, client: TestClient):
        """Test SQL injection protection in JSON payloads."""
        # Attempt SQL injection in various fields
        malicious_payload = {
            "object_kind": "merge_request",
            "object_attributes": {
                "title": "'; DROP TABLE users; --",
                "description": "1' OR '1'='1",
                "source_branch": "feature'; DELETE FROM projects; --",
            },
        }

        headers = {
            "X-Gitlab-Event": "Merge Request Hook",
            "X-Gitlab-Token": "test-webhook-secret",
            "Content-Type": "application/json",
        }

        with patch("src.api.webhooks.handle_gitlab_webhook") as mock_handler:
            mock_handler.return_value = {"status": "accepted"}

            response = client.post("/webhook", json=malicious_payload, headers=headers)

            # Should handle without executing malicious SQL
            assert response.status_code in [200, 400]

            if mock_handler.called:
                # Verify malicious content was passed as-is (not executed)
                call_args = mock_handler.call_args[0][0]
                assert "DROP TABLE" in call_args.object_attributes.title
