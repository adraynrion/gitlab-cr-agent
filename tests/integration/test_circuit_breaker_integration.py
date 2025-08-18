"""
Integration tests for circuit breaker functionality
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.exceptions import AIProviderException
from src.services.review_service import ReviewService
from src.utils.circuit_breaker import get_ai_circuit_breaker, reset_circuit_breaker


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration in review workflow"""

    @pytest.fixture(autouse=True)
    def reset_circuit_breaker_state(self):
        """Reset circuit breaker before each test"""
        reset_circuit_breaker()
        yield
        reset_circuit_breaker()

    @pytest.fixture
    def mock_review_agent(self):
        """Mock review agent for testing"""
        agent = Mock()
        agent.review_merge_request = AsyncMock()
        return agent

    @pytest.fixture
    def review_service(self, mock_review_agent):
        """Create review service with mocked agent"""
        return ReviewService(review_agent=mock_review_agent)

    @pytest.fixture
    def sample_mr_event(self, merge_request_event):
        """Use the merge request event fixture"""
        return merge_request_event

    @pytest.fixture
    def sample_mr_details(self):
        """Sample MR details"""
        return {
            "id": 123,
            "iid": 10,
            "title": "Test MR",
            "description": "Test description",
            "state": "opened",
            "source_branch": "feature",
            "target_branch": "main",
        }

    @pytest.fixture
    def sample_mr_diff(self):
        """Sample MR diff"""
        return [
            {
                "old_path": "test.py",
                "new_path": "test.py",
                "new_file": False,
                "renamed_file": False,
                "deleted_file": False,
                "diff": "@@ -1,3 +1,4 @@\n def test():\n-    pass\n+    print('hello')\n+    return True",
            }
        ]

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(
        self,
        review_service,
        sample_mr_event,
        sample_mr_details,
        sample_mr_diff,
    ):
        """Test that circuit breaker opens after multiple AI failures"""
        # Configure mock to always fail
        review_service.review_agent.review_merge_request.side_effect = (
            AIProviderException("AI service unavailable", "openai")
        )

        # Circuit breaker should open after multiple failures
        failure_count = 0
        while failure_count < 10:  # Safety limit
            try:
                await review_service.review_merge_request(
                    mr_details=sample_mr_details,
                    mr_diff=sample_mr_diff,
                    mr_event=sample_mr_event,
                )
                break  # If we succeed, break out
            except AIProviderException as e:
                failure_count += 1
                if "circuit breaker is open" in str(e).lower():
                    # Circuit breaker is now open
                    break

        # Verify circuit breaker opened
        cb = get_ai_circuit_breaker()
        assert cb.state == "open"
        assert failure_count >= 3  # Should have failed at least threshold times

    @pytest.mark.asyncio
    async def test_circuit_breaker_protects_subsequent_calls(
        self,
        review_service,
        sample_mr_event,
        sample_mr_details,
        sample_mr_diff,
    ):
        """Test that circuit breaker protects subsequent calls when open"""
        # First, trigger circuit breaker to open
        review_service.review_agent.review_merge_request.side_effect = (
            AIProviderException("AI service down", "openai")
        )

        # Make enough calls to open the circuit
        for _ in range(5):
            try:
                await review_service.review_merge_request(
                    mr_details=sample_mr_details,
                    mr_diff=sample_mr_diff,
                    mr_event=sample_mr_event,
                )
            except AIProviderException:
                pass

        # Verify circuit is open
        cb = get_ai_circuit_breaker()
        assert cb.state == "open"

        # Now subsequent calls should fail immediately with circuit breaker message
        with pytest.raises(AIProviderException) as exc_info:
            await review_service.review_merge_request(
                mr_details=sample_mr_details,
                mr_diff=sample_mr_diff,
                mr_event=sample_mr_event,
            )

        assert "circuit breaker is open" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_successful_calls(
        self,
        review_service,
        sample_mr_event,
        sample_mr_details,
        sample_mr_diff,
        sample_review_result,
    ):
        """Test that circuit breaker allows successful calls through"""
        # Configure mock to succeed
        review_service.review_agent.review_merge_request.return_value = (
            sample_review_result
        )

        # Multiple successful calls should work fine
        for _ in range(5):
            result = await review_service.review_merge_request(
                mr_details=sample_mr_details,
                mr_diff=sample_mr_diff,
                mr_event=sample_mr_event,
            )
            assert result == sample_review_result

        # Circuit should remain closed
        cb = get_ai_circuit_breaker()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_mixed_success_failure(
        self,
        review_service,
        sample_mr_event,
        sample_mr_details,
        sample_mr_diff,
        sample_review_result,
    ):
        """Test circuit breaker with mixed success and failure patterns"""
        call_count = 0

        def mock_review_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail every 3rd call
            if call_count % 3 == 0:
                raise AIProviderException("Intermittent failure", "openai")
            return sample_review_result

        review_service.review_agent.review_merge_request.side_effect = (
            mock_review_side_effect
        )

        # Make several calls with mixed results
        success_count = 0
        failure_count = 0

        for _ in range(10):
            try:
                await review_service.review_merge_request(
                    mr_details=sample_mr_details,
                    mr_diff=sample_mr_diff,
                    mr_event=sample_mr_event,
                )
                success_count += 1
            except AIProviderException as e:
                if "circuit breaker is open" in str(e).lower():
                    # Circuit opened, stop testing
                    break
                failure_count += 1

        # Should have some successes and some failures
        assert success_count > 0
        assert failure_count > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_respects_different_exception_types(
        self,
        review_service,
        sample_mr_event,
        sample_mr_details,
        sample_mr_diff,
    ):
        """Test that circuit breaker only counts expected exception types"""
        call_count = 0

        def mock_review_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                # Non-AI exceptions shouldn't trigger circuit breaker
                raise ValueError("Not an AI provider error")
            else:
                # AI exceptions should trigger circuit breaker
                raise AIProviderException("AI service down", "openai")

        review_service.review_agent.review_merge_request.side_effect = (
            mock_review_side_effect
        )

        # First few calls with ValueError shouldn't affect circuit breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                await review_service.review_merge_request(
                    mr_details=sample_mr_details,
                    mr_diff=sample_mr_diff,
                    mr_event=sample_mr_event,
                )

        # Circuit should still be closed
        cb = get_ai_circuit_breaker()
        assert cb.state == "closed"
        assert cb.failure_count == 0

        # Now AI exceptions should start affecting the circuit breaker
        with pytest.raises(AIProviderException):
            await review_service.review_merge_request(
                mr_details=sample_mr_details,
                mr_diff=sample_mr_diff,
                mr_event=sample_mr_event,
            )

        # Circuit breaker should now have recorded the AI failure
        assert cb.failure_count > 0


class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration and settings"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_uses_settings(self):
        """Test that circuit breaker respects configuration settings"""
        mock_settings = Mock()
        mock_settings.circuit_breaker_failure_threshold = 2  # Lower threshold
        mock_settings.circuit_breaker_timeout = 10  # Shorter timeout

        with patch(
            "src.utils.circuit_breaker.get_settings", return_value=mock_settings
        ):
            from src.utils.circuit_breaker import AIProviderCircuitBreaker

            cb = AIProviderCircuitBreaker()
            assert cb.failure_threshold == 2
            assert cb.recovery_timeout == 10

    @pytest.mark.asyncio
    async def test_circuit_breaker_global_instance(self):
        """Test that global circuit breaker is properly managed"""
        cb1 = get_ai_circuit_breaker()
        cb2 = get_ai_circuit_breaker()

        # Should be the same instance
        assert cb1 is cb2

        # Reset should affect the global instance
        reset_circuit_breaker()
        assert cb1.failure_count == 0
