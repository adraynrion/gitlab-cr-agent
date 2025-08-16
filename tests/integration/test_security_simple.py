"""Simplified integration tests for security features."""

import time
from unittest.mock import Mock, patch

import pytest

from src.config.settings import Settings


class TestSecurityIntegration:
    """Simplified security integration tests."""

    @pytest.mark.security
    async def test_webhook_timestamp_validation(self):
        """Test webhook timestamp validation."""
        from src.api.webhooks import verify_gitlab_token

        # Create test settings
        test_settings = Mock(spec=Settings)
        test_settings.gitlab_webhook_secret = "test-secret"

        # Create mock request
        request = Mock()
        request.headers = {
            "X-Gitlab-Token": "test-secret",
            "X-Gitlab-Timestamp": str(time.time()),
        }

        with patch("src.api.webhooks.settings", test_settings):
            # Should not raise for valid timestamp
            verify_gitlab_token(request)

            # Should raise for old timestamp
            request.headers["X-Gitlab-Timestamp"] = str(time.time() - 400)
            with pytest.raises(Exception):
                verify_gitlab_token(request)

    @pytest.mark.integration
    async def test_version_system(self):
        """Test version system integration."""
        from src.utils.version import get_version

        version = get_version()
        assert version == "2.1.0"
        assert isinstance(version, str)

        # Test caching
        version2 = get_version()
        assert version2 == version

    @pytest.mark.integration
    async def test_settings_loading(self):
        """Test settings load with new fields."""
        with patch.dict(
            "os.environ",
            {
                "GITLAB_TOKEN": "glpat-test-token-123456789",
                "GITLAB_URL": "https://gitlab.example.com",
                "GITLAB_WEBHOOK_SECRET": "secret",
                "AI_MODEL": "openai:gpt-4o",
                "OPENAI_API_KEY": "sk-test",
            },
        ):
            settings = Settings()

            # New fields should have defaults
            assert settings.max_diff_size == 1 * 1024 * 1024
            assert settings.request_timeout == 30.0
            assert settings.max_connections == 100
            assert settings.circuit_breaker_failure_threshold == 5
            assert settings.circuit_breaker_timeout == 60
