"""Unit tests for custom exception handling."""

import pytest

from src.exceptions import (
    AIProviderException,
    ConfigurationException,
    GitLabAPIException,
    GitLabReviewerException,
    RateLimitException,
    ReviewProcessException,
    SecurityException,
    WebhookValidationException,
)


class TestGitLabReviewerException:
    """Test the base GitLabReviewerException exception."""

    def test_basic_error_creation(self):
        """Test creating basic error with message."""
        error = GitLabReviewerException("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"

    def test_error_with_details(self):
        """Test creating error with details."""
        details = {"key": "value", "number": 42}
        error = GitLabReviewerException("Test error", details=details)

        assert error.message == "Test error"
        assert error.details == details
        assert "Details: " in str(error)

    def test_error_with_original_error(self):
        """Test creating error with original error."""
        original = ValueError("Original error")
        error = GitLabReviewerException("Wrapped error", original_error=original)

        assert error.message == "Wrapped error"
        assert error.original_error is original

    def test_error_inheritance(self):
        """Test that GitLabReviewerException inherits from Exception."""
        error = GitLabReviewerException("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, GitLabReviewerException)

    def test_error_with_unicode_message(self):
        """Test error with unicode characters."""
        unicode_message = "Errör with unicödé chars: 🚨"
        error = GitLabReviewerException(unicode_message)

        assert str(error) == unicode_message

    def test_error_repr(self):
        """Test error string representation."""
        error = GitLabReviewerException("Test error")

        repr_str = repr(error)
        assert "GitLabReviewerException" in repr_str
        assert "Test error" in repr_str

    def test_error_with_empty_details(self):
        """Test error with empty details dict."""
        error = GitLabReviewerException("Test", details={})

        assert error.details == {}
        assert str(error) == "Test"  # Should not show empty details

    def test_error_with_none_details(self):
        """Test error with None details."""
        error = GitLabReviewerException("Test", details=None)

        assert error.details == {}
        assert str(error) == "Test"


class TestGitLabAPIException:
    """Test GitLabAPIException exception."""

    def test_gitlab_api_error_creation(self):
        """Test creating GitLabAPIException."""
        error = GitLabAPIException("API request failed")

        assert str(error) == "API request failed"
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, GitLabAPIException)

    def test_gitlab_api_error_with_status_code(self):
        """Test GitLabAPIException with HTTP status code."""
        error = GitLabAPIException(
            "Request failed", status_code=404, response_body='{"message": "Not Found"}'
        )

        assert error.status_code == 404
        assert "404" in str(error)
        assert "Not Found" in str(error)

    def test_gitlab_api_error_inheritance_chain(self):
        """Test GitLabAPIException inheritance chain."""
        error = GitLabAPIException("API Error")

        assert isinstance(error, Exception)
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, GitLabAPIException)

    def test_gitlab_api_error_with_details(self):
        """Test GitLabAPIException with custom details."""
        details = {"endpoint": "/api/v4/projects", "method": "GET"}
        error = GitLabAPIException("Request failed", status_code=403, details=details)

        assert error.status_code == 403
        assert error.details["endpoint"] == "/api/v4/projects"
        assert "endpoint" in str(error)

    def test_gitlab_api_error_catching(self):
        """Test catching GitLabAPIException."""
        with pytest.raises(GitLabAPIException) as exc_info:
            raise GitLabAPIException("Test API error")

        assert "Test API error" in str(exc_info.value)

        # Should also be catchable as base class
        with pytest.raises(GitLabReviewerException):
            raise GitLabAPIException("Test API error")


class TestAIProviderException:
    """Test AIProviderException exception."""

    def test_ai_provider_error_creation(self):
        """Test creating AIProviderException."""
        error = AIProviderException("AI provider failed")

        assert str(error) == "AI provider failed"
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, AIProviderException)

    def test_ai_provider_error_with_provider_info(self):
        """Test AIProviderException with provider and model info."""
        error = AIProviderException(
            "Rate limit exceeded", provider="openai", model="gpt-4"
        )

        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert "openai" in str(error)
        assert "gpt-4" in str(error)

    def test_ai_provider_error_with_details(self):
        """Test AIProviderException with additional details."""
        details = {"request_id": "req_123", "tokens_used": 1500}
        error = AIProviderException(
            "Processing failed", provider="anthropic", details=details
        )

        assert error.provider == "anthropic"
        assert error.details["request_id"] == "req_123"
        assert "request_id" in str(error)

    def test_ai_provider_error_inheritance(self):
        """Test AIProviderException inheritance."""
        error = AIProviderException("Provider error")

        assert isinstance(error, Exception)
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, AIProviderException)

    def test_ai_provider_error_with_original_error(self):
        """Test AIProviderException wrapping original error."""
        original = ConnectionError("Connection timeout")
        error = AIProviderException(
            "Provider unavailable", provider="gemini", original_error=original
        )

        assert error.provider == "gemini"
        assert error.original_error is original


class TestWebhookValidationException:
    """Test WebhookValidationException exception."""

    def test_webhook_validation_error_creation(self):
        """Test creating WebhookValidationException."""
        error = WebhookValidationException("Invalid webhook payload")

        assert str(error) == "Invalid webhook payload"
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, WebhookValidationException)

    def test_webhook_validation_error_with_payload_section(self):
        """Test WebhookValidationException with payload section."""
        error = WebhookValidationException(
            "Missing required field", payload_section="object_attributes"
        )

        assert "object_attributes" in str(error)

    def test_webhook_validation_error_with_details(self):
        """Test WebhookValidationException with validation details."""
        details = {"field": "merge_request_iid", "expected_type": "int"}
        error = WebhookValidationException(
            "Type validation failed",
            payload_section="object_attributes",
            details=details,
        )

        assert error.details["field"] == "merge_request_iid"
        assert "field" in str(error)

    def test_webhook_validation_error_inheritance(self):
        """Test WebhookValidationException inheritance."""
        error = WebhookValidationException("Validation error")

        assert isinstance(error, Exception)
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, WebhookValidationException)

    def test_webhook_validation_error_signature_context(self):
        """Test WebhookValidationException with signature context."""
        error = WebhookValidationException(
            "Invalid signature",
            payload_section="headers",
            details={"expected_header": "X-Gitlab-Token"},
        )

        assert "headers" in str(error)
        assert "X-Gitlab-Token" in str(error)


class TestConfigurationException:
    """Test ConfigurationException exception."""

    def test_configuration_error_creation(self):
        """Test creating ConfigurationException."""
        error = ConfigurationException("Missing API key configuration")

        assert str(error) == "Missing API key configuration"
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, ConfigurationException)

    def test_configuration_error_with_config_key(self):
        """Test ConfigurationException with config key."""
        error = ConfigurationException("Invalid value", config_key="GITLAB_TOKEN")

        assert "GITLAB_TOKEN" in str(error)

    def test_configuration_error_with_validation_details(self):
        """Test ConfigurationException with validation details."""
        details = {"min_length": 20, "actual_length": 5}
        error = ConfigurationException(
            "Token too short", config_key="GITLAB_WEBHOOK_SECRET", details=details
        )

        assert error.details["min_length"] == 20
        assert "min_length" in str(error)

    def test_configuration_error_inheritance(self):
        """Test ConfigurationException inheritance."""
        error = ConfigurationException("Config error")

        assert isinstance(error, Exception)
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, ConfigurationException)

    def test_configuration_error_startup_scenario(self):
        """Test ConfigurationException in startup scenario."""
        details = {"missing_keys": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]}
        error = ConfigurationException(
            "Multiple provider keys missing",
            config_key="provider_config",
            details=details,
        )

        assert "OPENAI_API_KEY" in str(error)
        assert "ANTHROPIC_API_KEY" in str(error)


class TestReviewProcessException:
    """Test ReviewProcessException exception."""

    def test_review_process_error_creation(self):
        """Test creating ReviewProcessException."""
        error = ReviewProcessException("Review processing failed")

        assert str(error) == "Review processing failed"
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, ReviewProcessException)

    def test_review_process_error_with_mr_context(self):
        """Test ReviewProcessException with merge request context."""
        error = ReviewProcessException(
            "Processing failed", merge_request_iid=123, project_id=456
        )

        assert error.merge_request_iid == 123
        assert error.project_id == 456
        assert "123" in str(error)
        assert "456" in str(error)

    def test_review_process_error_with_details(self):
        """Test ReviewProcessException with processing details."""
        details = {"stage": "diff_analysis", "files_processed": 3}
        error = ReviewProcessException(
            "Analysis failed", merge_request_iid=10, details=details
        )

        assert error.details["stage"] == "diff_analysis"
        assert "diff_analysis" in str(error)

    def test_review_process_error_inheritance(self):
        """Test ReviewProcessException inheritance."""
        error = ReviewProcessException("Process error")

        assert isinstance(error, Exception)
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, ReviewProcessException)

    def test_review_process_error_wrapping(self):
        """Test ReviewProcessException wrapping other errors."""
        original = ValueError("Invalid diff format")
        error = ReviewProcessException(
            "Failed to parse changes", merge_request_iid=10, original_error=original
        )

        assert error.original_error is original
        assert error.merge_request_iid == 10


class TestSecurityException:
    """Test SecurityException exception."""

    def test_security_error_creation(self):
        """Test creating SecurityException."""
        error = SecurityException("Security violation detected")

        assert str(error) == "Security violation detected"
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, SecurityException)

    def test_security_error_with_context(self):
        """Test SecurityException with security context."""
        error = SecurityException(
            "Unauthorized access attempt", security_context="webhook_validation"
        )

        assert "webhook_validation" in str(error)

    def test_security_error_with_details(self):
        """Test SecurityException with security details."""
        details = {
            "ip_address": "192.168.1.1",
            "attempted_action": "bypass_auth",
            "user_agent": "Malicious Bot",
        }
        error = SecurityException(
            "Security breach attempted",
            security_context="authentication",
            details=details,
        )

        assert error.details["ip_address"] == "192.168.1.1"
        assert "bypass_auth" in str(error)

    def test_security_error_inheritance(self):
        """Test SecurityException inheritance."""
        error = SecurityException("Security error")

        assert isinstance(error, Exception)
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, SecurityException)

    def test_security_error_token_context(self):
        """Test SecurityException with token security context."""
        details = {"token_type": "webhook", "validation_failed": True}
        error = SecurityException(
            "Invalid token signature",
            security_context="token_validation",
            details=details,
        )

        assert "token_validation" in str(error)
        assert "webhook" in str(error)


class TestRateLimitException:
    """Test RateLimitException exception."""

    def test_rate_limit_error_creation(self):
        """Test creating RateLimitException."""
        error = RateLimitException("Rate limit exceeded")

        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, RateLimitException)

    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitException with retry after information."""
        error = RateLimitException("Rate limit exceeded", retry_after=60)

        assert error.retry_after == 60
        assert "60" in str(error)

    def test_rate_limit_error_with_details(self):
        """Test RateLimitException with rate limiting details."""
        details = {
            "limit": 100,
            "remaining": 0,
            "reset_time": "2024-01-15T13:00:00Z",
            "provider": "openai",
        }
        error = RateLimitException(
            "API rate limit exceeded", retry_after=120, details=details
        )

        assert error.retry_after == 120
        assert error.details["limit"] == 100
        assert "openai" in str(error)

    def test_rate_limit_error_inheritance(self):
        """Test RateLimitException inheritance."""
        error = RateLimitException("Rate limit error")

        assert isinstance(error, Exception)
        assert isinstance(error, GitLabReviewerException)
        assert isinstance(error, RateLimitException)

    def test_rate_limit_error_provider_context(self):
        """Test RateLimitException with provider context."""
        details = {"provider": "anthropic", "request_count": 1000}
        error = RateLimitException(
            "Daily limit reached", retry_after=3600, details=details
        )

        assert "anthropic" in str(error)
        assert "1000" in str(error)


class TestExceptionHierarchy:
    """Test exception hierarchy and relationships."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from GitLabReviewerException."""
        exception_classes = [
            GitLabAPIException,
            AIProviderException,
            WebhookValidationException,
            ConfigurationException,
            ReviewProcessException,
            SecurityException,
            RateLimitException,
        ]

        for exc_class in exception_classes:
            error = exc_class("Test message")
            assert isinstance(error, GitLabReviewerException)
            assert isinstance(error, Exception)

    def test_exception_catching_hierarchy(self):
        """Test catching exceptions at different hierarchy levels."""
        # All custom exceptions should be catchable as GitLabReviewerException
        custom_exceptions = [
            GitLabAPIException("API error"),
            AIProviderException("Provider error"),
            WebhookValidationException("Validation error"),
            ConfigurationException("Config error"),
            ReviewProcessException("Process error"),
            SecurityException("Security error"),
            RateLimitException("Rate limit error"),
        ]

        for exception in custom_exceptions:
            with pytest.raises(GitLabReviewerException):
                raise exception

    def test_specific_exception_catching(self):
        """Test catching specific exception types."""
        # Should be able to catch specific exception types
        with pytest.raises(GitLabAPIException):
            raise GitLabAPIException("Specific API error")

        with pytest.raises(AIProviderException):
            raise AIProviderException("Specific provider error")

        with pytest.raises(SecurityException):
            raise SecurityException("Specific security error")

    def test_exception_inheritance_order(self):
        """Test method resolution order for exception inheritance."""
        error = GitLabAPIException("Test")

        # Check MRO (Method Resolution Order)
        mro = type(error).__mro__

        assert GitLabAPIException in mro
        assert GitLabReviewerException in mro
        assert Exception in mro

        # GitLabAPIException should come before GitLabReviewerException in MRO
        api_index = mro.index(GitLabAPIException)
        base_index = mro.index(GitLabReviewerException)
        assert api_index < base_index

    def test_exception_type_checking(self):
        """Test exception type checking with isinstance."""
        api_error = GitLabAPIException("API error")
        provider_error = AIProviderException("Provider error")
        base_error = GitLabReviewerException("Base error")

        # Type checking should work correctly
        assert isinstance(api_error, GitLabAPIException)
        assert isinstance(api_error, GitLabReviewerException)
        assert not isinstance(api_error, AIProviderException)

        assert isinstance(provider_error, AIProviderException)
        assert isinstance(provider_error, GitLabReviewerException)
        assert not isinstance(provider_error, GitLabAPIException)

        assert isinstance(base_error, GitLabReviewerException)
        assert not isinstance(base_error, GitLabAPIException)
        assert not isinstance(base_error, AIProviderException)


class TestExceptionContext:
    """Test exceptions with context and additional data."""

    def test_exception_with_context_dict(self):
        """Test creating exceptions with context information."""
        context = {
            "merge_request_id": 123,
            "project_id": 456,
            "user": "testuser",
            "action": "review",
        }

        error = ReviewProcessException(
            "Review failed",
            merge_request_iid=context["merge_request_id"],
            project_id=context["project_id"],
            details={"user": context["user"], "action": context["action"]},
        )

        assert error.merge_request_iid == 123
        assert error.project_id == 456
        assert "testuser" in str(error)
        assert "review" in str(error)

    def test_exception_chaining(self):
        """Test exception chaining with 'from' clause."""
        original_error = ValueError("Invalid data format")

        try:
            try:
                raise original_error
            except ValueError as e:
                raise ReviewProcessException("Review processing failed") from e
        except ReviewProcessException as review_error:
            # Should maintain chain
            assert review_error.__cause__ is original_error
            assert str(review_error) == "Review processing failed"

    def test_exception_with_detailed_context(self):
        """Test exception with detailed context information."""
        details = {
            "endpoint": "/api/v4/projects/123/merge_requests/456/notes",
            "method": "POST",
            "status_code": 403,
            "response_time": 1.5,
        }

        error = GitLabAPIException(
            "Comment posting failed", status_code=403, details=details
        )

        assert error.status_code == 403
        assert "merge_requests" in str(error)
        assert "POST" in str(error)

    def test_exception_with_nested_details(self):
        """Test exception with nested details structure."""
        nested_details = {
            "request": {
                "url": "https://api.openai.com/v1/chat/completions",
                "headers": {"Authorization": "Bearer ***"},
                "payload_size": 2048,
            },
            "response": {"status": 429, "headers": {"Retry-After": "60"}},
        }

        error = AIProviderException(
            "API request failed",
            provider="openai",
            model="gpt-4",
            details=nested_details,
        )

        assert "openai" in str(error)
        assert "gpt-4" in str(error)
        assert "429" in str(error)


class TestExceptionEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_exception_with_none_message(self):
        """Test exception with None as message."""
        # None gets converted to string by Python's Exception class
        error = GitLabReviewerException(None)
        assert str(error) == "Unknown error"
        assert error.message is None

    def test_exception_with_empty_message(self):
        """Test exception with empty message."""
        error = GitLabReviewerException("")

        assert str(error) == ""
        assert error.message == ""

    def test_exception_with_numeric_details(self):
        """Test exception with numeric values in details."""
        details = {"count": 42, "rate": 3.14, "timestamp": 1642261234}
        error = GitLabReviewerException("Test error", details=details)

        assert error.details["count"] == 42
        assert error.details["rate"] == 3.14
        assert "42" in str(error)
        assert "3.14" in str(error)

    def test_exception_with_boolean_details(self):
        """Test exception with boolean values in details."""
        details = {"success": False, "retry_allowed": True}
        error = GitLabReviewerException("Operation failed", details=details)

        assert error.details["success"] is False
        assert error.details["retry_allowed"] is True
        assert "False" in str(error)
        assert "True" in str(error)

    def test_exception_with_list_details(self):
        """Test exception with list values in details."""
        details = {
            "errors": ["Error 1", "Error 2"],
            "tried_providers": ["openai", "anthropic"],
        }
        error = GitLabReviewerException("Multiple failures", details=details)

        assert "Error 1" in str(error)
        assert "openai" in str(error)

    def test_exception_with_very_long_message(self):
        """Test exception with very long message."""
        long_message = "Error: " + "x" * 1000  # Very long error message
        error = GitLabReviewerException(long_message)

        assert str(error) == long_message
        assert len(str(error)) >= 1000

    def test_exception_with_special_characters(self):
        """Test exception with special characters."""
        special_message = "Error with special chars: !@#$%^&*()[]{}|\\:;\"'<>?,./"
        error = GitLabReviewerException(special_message)

        assert str(error) == special_message
        assert "!@#$%^&*()" in str(error)

    def test_exception_details_immutability(self):
        """Test that exception details don't affect original dict."""
        original_details = {"key": "value"}
        error = GitLabReviewerException("Test", details=original_details)

        # Modify the original dict
        original_details["new_key"] = "new_value"

        # Exception's details should be independent
        assert "new_key" not in error.details

    def test_exception_string_representation_consistency(self):
        """Test that string representation is consistent."""
        details = {"key": "value"}
        error = GitLabReviewerException("Test message", details=details)

        # Multiple calls to str() should return same result
        str1 = str(error)
        str2 = str(error)

        assert str1 == str2
        assert "Test message" in str1
        assert "key" in str1
