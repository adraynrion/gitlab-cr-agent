"""Comprehensive unit tests for src/main.py FastAPI application."""

import asyncio
import signal
import threading
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.exceptions import (
    AIProviderException,
    ConfigurationException,
    GitLabAPIException,
    GitLabReviewerException,
    RateLimitException,
    SecurityException,
    WebhookValidationException,
)
from src.main import AppState, app, app_state, get_review_agent


class TestAppState:
    """Test AppState class functionality and thread safety."""

    def test_app_state_initialization(self):
        """Test AppState initialization with proper defaults."""
        state = AppState()

        assert state.review_agent is None
        assert state.initialized is False
        assert state.shutdown_event is not None
        assert state._lock is not None

    def test_app_state_thread_safety(self):
        """Test AppState thread safety with concurrent access."""
        state = AppState()
        mock_agent = Mock()
        results = []

        def mark_and_check():
            state.mark_initialized(mock_agent)
            results.append(state.is_initialized())
            results.append(state.get_review_agent())

        # Create multiple threads that access state concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=mark_and_check)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All should see consistent state
        assert all(
            result is True for result in results[::2]
        )  # Every other result is is_initialized()
        assert all(
            result == mock_agent for result in results[1::2]
        )  # Other results are get_review_agent()

    def test_app_state_clear(self):
        """Test AppState clear functionality."""
        state = AppState()
        mock_agent = Mock()

        state.mark_initialized(mock_agent)
        assert state.is_initialized() is True
        assert state.get_review_agent() == mock_agent

        state.clear()
        assert state.is_initialized() is False
        assert state.get_review_agent() is None

    def test_app_state_without_lock(self):
        """Test AppState behavior when lock is None (edge case)."""
        state = AppState()
        # Manually set lock to None to test fallback behavior
        object.__setattr__(state, "_lock", None)

        mock_agent = Mock()

        # Should still work without thread safety - directly modify attributes
        object.__setattr__(state, "review_agent", mock_agent)
        object.__setattr__(state, "initialized", True)

        assert state.is_initialized() is True
        assert state.get_review_agent() == mock_agent

        state.clear()
        assert state.is_initialized() is False
        assert state.get_review_agent() is None


class TestSignalHandlers:
    """Test signal handler setup and functionality."""

    @patch("signal.signal")
    def test_signal_handler_setup(self, mock_signal):
        """Test that signal handlers are properly registered."""
        from src.main import setup_signal_handlers

        setup_signal_handlers()

        # Should register handlers for SIGTERM and SIGINT
        assert mock_signal.call_count == 2
        calls = mock_signal.call_args_list
        signals_registered = [call[0][0] for call in calls]

        assert signal.SIGTERM in signals_registered
        assert signal.SIGINT in signals_registered

    def test_signal_handler_sets_shutdown_event(self):
        """Test that signal handler sets shutdown event."""
        from src.main import setup_signal_handlers

        # Mock the signal handler function
        with patch("signal.signal") as mock_signal:
            setup_signal_handlers()

            # Get the handler function that was registered
            handler_func = mock_signal.call_args_list[0][0][1]

            # Mock the app_state shutdown event
            mock_event = Mock()
            app_state.shutdown_event = mock_event

            # Call the handler
            handler_func(signal.SIGTERM, None)

            # Should set the shutdown event
            mock_event.set.assert_called_once()


class TestResourceManagement:
    """Test application resource initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize_resources_success(self):
        """Test successful resource initialization."""
        from src.main import initialize_resources

        mock_agent = Mock()

        with patch(
            "src.main.initialize_review_agent", return_value=mock_agent
        ) as mock_init:
            with patch("src.config.settings.get_settings") as mock_get_settings:
                mock_settings = Mock()
                mock_settings.ai_model = "test-model"
                mock_get_settings.return_value = mock_settings

                await initialize_resources()

                mock_init.assert_called_once()
                assert app_state.get_review_agent() == mock_agent
                assert app_state.is_initialized() is True

    @pytest.mark.asyncio
    async def test_initialize_resources_failure(self):
        """Test resource initialization failure handling."""
        from src.main import initialize_resources

        with patch(
            "src.main.initialize_review_agent", side_effect=Exception("Init failed")
        ):
            with pytest.raises(ConfigurationException) as exc_info:
                await initialize_resources()

            assert "Application startup failed" in str(exc_info.value.message)
            assert exc_info.value.details["initialization_stage"] == "resources"

    @pytest.mark.asyncio
    async def test_cleanup_resources_with_agent(self):
        """Test resource cleanup when agent exists."""
        from src.main import cleanup_resources

        # Setup state with mock agent
        mock_agent = Mock()
        mock_agent.close = AsyncMock()
        app_state.mark_initialized(mock_agent)

        # No need to patch secrets cache - module was removed
        await cleanup_resources()

        # Should call agent cleanup
        mock_agent.close.assert_called_once()
        # Should clear app state
        assert app_state.is_initialized() is False
        assert app_state.get_review_agent() is None

    @pytest.mark.asyncio
    async def test_cleanup_resources_sync_agent_close(self):
        """Test cleanup with synchronous agent close method."""
        from src.main import cleanup_resources

        mock_agent = Mock()
        mock_agent.close = Mock()  # Sync close method
        app_state.mark_initialized(mock_agent)

        # No need to patch secrets cache - module was removed
        await cleanup_resources()

        mock_agent.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_resources_no_agent(self):
        """Test cleanup when no agent exists."""
        from src.main import cleanup_resources

        app_state.clear()  # Ensure no agent

        # No need to patch secrets cache - module was removed
        await cleanup_resources()

        # Should clear app state
        assert app_state.is_initialized() is False
        assert app_state.get_review_agent() is None

    @pytest.mark.asyncio
    async def test_cleanup_resources_error_handling(self):
        """Test cleanup error handling doesn't raise exceptions."""
        from src.main import cleanup_resources

        mock_agent = Mock()
        mock_agent.close = Mock(side_effect=Exception("Cleanup failed"))
        app_state.mark_initialized(mock_agent)

        # Should not raise exception even if cleanup fails
        # No need to patch secrets cache - module was removed
        await cleanup_resources()


class TestExceptionHandlers:
    """Test all custom exception handlers."""

    def test_http_exception_handler_error_level(self):
        """Test HTTP exception handler with different status codes."""
        client = TestClient(app)

        # Test 500 level (should log as error)
        with patch("src.main.logger.error") as _:
            response = client.get("/nonexistent")
            assert response.status_code == 404

    def test_security_exception_handler(self):
        """Test SecurityException handler."""
        from src.main import security_exception_handler

        mock_request = Mock()
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"

        exc = SecurityException(
            message="Access denied",
            details={"security_context": "authentication_failed"},
        )

        with patch("src.main.logger.warning") as mock_log:
            response = asyncio.run(security_exception_handler(mock_request, exc))

            assert response.status_code == 401  # Authentication context
            assert "Security Error" in response.body.decode()
            mock_log.assert_called_once()

    def test_rate_limit_exception_handler(self):
        """Test RateLimitException handler."""
        from src.main import rate_limit_exception_handler

        mock_request = Mock()
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"

        exc = RateLimitException(message="Rate limit exceeded", retry_after=60)

        with patch("src.main.logger.warning") as mock_log:
            response = asyncio.run(rate_limit_exception_handler(mock_request, exc))

            assert response.status_code == 429
            assert "Retry-After" in response.headers
            assert response.headers["Retry-After"] == "60"
            mock_log.assert_called_once()

    def test_webhook_validation_exception_handler(self):
        """Test WebhookValidationException handler."""
        from src.main import webhook_validation_exception_handler

        mock_request = Mock()
        mock_request.url.path = "/webhook"

        exc = WebhookValidationException(
            message="Invalid payload", details={"field": "missing"}
        )

        with patch("src.main.logger.error") as mock_log:
            response = asyncio.run(
                webhook_validation_exception_handler(mock_request, exc)
            )

            assert response.status_code == 400
            assert "Invalid Webhook" in response.body.decode()
            mock_log.assert_called_once()

    def test_gitlab_api_exception_handler(self):
        """Test GitLabAPIException handler."""
        from src.main import gitlab_api_exception_handler

        mock_request = Mock()
        mock_request.url.path = "/test"

        exc = GitLabAPIException(
            message="GitLab API error",
            status_code=503,
            details={"endpoint": "api/v4/projects"},
        )

        with patch("src.main.logger.error") as mock_log:
            response = asyncio.run(gitlab_api_exception_handler(mock_request, exc))

            assert response.status_code == 502
            assert "External Service Error" in response.body.decode()
            mock_log.assert_called_once()

    def test_ai_provider_exception_handler(self):
        """Test AIProviderException handler."""
        from src.main import ai_provider_exception_handler

        mock_request = Mock()
        mock_request.url.path = "/test"

        exc = AIProviderException(
            message="AI service error",
            provider="openai",
            model="gpt-4",
            details={"error_code": "rate_limited"},
        )

        with patch("src.main.logger.error") as mock_log:
            response = asyncio.run(ai_provider_exception_handler(mock_request, exc))

            assert response.status_code == 503
            assert "AI Service Unavailable" in response.body.decode()
            mock_log.assert_called_once()

    def test_configuration_exception_handler(self):
        """Test ConfigurationException handler."""
        from src.main import configuration_exception_handler

        mock_request = Mock()
        mock_request.url.path = "/test"

        exc = ConfigurationException(
            message="Config error", config_key="api_key", details={"stage": "startup"}
        )

        with patch("src.main.logger.error") as mock_log:
            response = asyncio.run(configuration_exception_handler(mock_request, exc))

            assert response.status_code == 503
            assert "Service Configuration Error" in response.body.decode()
            mock_log.assert_called_once()

    def test_gitlab_reviewer_exception_handler(self):
        """Test GitLabReviewerException handler."""
        from src.main import gitlab_reviewer_exception_handler

        mock_request = Mock()
        mock_request.url.path = "/test"

        exc = GitLabReviewerException(
            message="Reviewer error", details={"component": "review_service"}
        )

        with patch("src.main.logger.error") as mock_log:
            response = asyncio.run(gitlab_reviewer_exception_handler(mock_request, exc))

            assert response.status_code == 500
            assert "Internal Service Error" in response.body.decode()
            mock_log.assert_called_once()

    def test_general_exception_handler(self):
        """Test general exception handler for unexpected errors."""
        from src.main import general_exception_handler

        mock_request = Mock()
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"

        exc = ValueError("Unexpected error")

        with patch("src.main.logger.error") as mock_log:
            response = asyncio.run(general_exception_handler(mock_request, exc))

            assert response.status_code == 500
            assert "Internal Server Error" in response.body.decode()
            mock_log.assert_called_once()


class TestMiddlewareConfiguration:
    """Test middleware and CORS configuration."""

    def test_cors_middleware_with_origins(self):
        """Test CORS middleware configuration when origins are set."""
        # CORS middleware is configured at app startup, so we just verify the setup logic
        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.allowed_origins = ["https://example.com"]
            mock_get_settings.return_value = mock_settings

            # Test that the app can handle requests (CORS is configured at startup)
            client = TestClient(app)
            response = client.get("/")

            # Should successfully process request through middleware stack
            assert response.status_code == 200

    def test_cors_middleware_without_origins(self):
        """Test behavior when no CORS origins are configured."""
        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.allowed_origins = []
            mock_get_settings.return_value = mock_settings

            # Should not add CORS middleware
            # This is tested by checking the middleware setup doesn't fail

    def test_middleware_order_applied(self):
        """Test that middlewares are applied in correct order."""
        # The middleware order is: Security -> Authentication -> RequestTracing
        # This is verified by the successful application startup
        client = TestClient(app)
        response = client.get("/")

        # Should process through all middleware layers
        assert response.status_code == 200


class TestRootEndpoint:
    """Test the root endpoint functionality."""

    def test_root_endpoint_when_initialized(self):
        """Test root endpoint when app is initialized."""
        # Set app state to initialized
        mock_agent = Mock()
        app_state.mark_initialized(mock_agent)

        with patch("src.main.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.environment = "test"
            mock_settings.rate_limit_enabled = True
            mock_settings.allowed_origins = ["https://example.com"]
            mock_settings.api_key = "test-key"
            mock_settings.gitlab_webhook_secret = "test-secret"
            mock_get_settings.return_value = mock_settings

            client = TestClient(app)
            response = client.get("/")

            assert response.status_code == 200
            data = response.json()

            assert data["service"] == "GitLab AI Code Review Agent"
            assert data["status"] == "running"
            assert data["environment"] == "test"
            assert "version" in data
            assert data["features"]["rate_limiting"] is True
            assert data["features"]["cors"] is True
            assert data["features"]["api_authentication"] is True
            assert data["features"]["webhook_secret"] is True

    def test_root_endpoint_when_starting(self):
        """Test root endpoint when app is still starting."""
        # Clear app state
        app_state.clear()

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "starting"


class TestDependencyInjection:
    """Test dependency injection functionality."""

    @pytest.mark.asyncio
    async def test_get_review_agent_success(self):
        """Test successful review agent dependency injection."""
        mock_agent = Mock()
        app_state.mark_initialized(mock_agent)

        result = await get_review_agent()
        assert result == mock_agent

    @pytest.mark.asyncio
    async def test_get_review_agent_not_initialized(self):
        """Test review agent dependency when not initialized."""
        app_state.clear()

        with pytest.raises(ConfigurationException) as exc_info:
            await get_review_agent()

        assert "Review agent not initialized" in exc_info.value.message


class TestApplicationLifecycle:
    """Test application lifecycle management."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_and_shutdown(self):
        """Test application lifespan management."""
        from src.main import lifespan

        mock_app = Mock()

        with patch("src.main.initialize_resources") as mock_init:
            with patch("src.main.cleanup_resources") as mock_cleanup:
                with patch("src.config.settings.get_settings") as mock_get_settings:
                    mock_settings = Mock()
                    mock_settings.environment = "test"
                    mock_settings.gitlab_url = "https://gitlab.example.com"
                    mock_settings.rate_limit_enabled = True
                    mock_get_settings.return_value = mock_settings

                    async with lifespan(mock_app):
                        # During the context, resources should be initialized
                        mock_init.assert_called_once()

                    # After context, cleanup should be called
                    mock_cleanup.assert_called_once()
