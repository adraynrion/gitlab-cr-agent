"""
Integration tests for service layer interactions and workflows
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.review_models import ReviewResult
from src.services.gitlab_service import GitLabService
from src.services.review_service import ReviewService


class TestServiceIntegration:
    """Test integration between services"""

    def test_gitlab_service_initialization(self):
        """Test GitLab service can be initialized with settings"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_connections = 100
        mock_settings.max_keepalive_connections = 20
        mock_settings.keepalive_expiry = 30.0

        with patch(
            "src.services.gitlab_service.get_settings", return_value=mock_settings
        ):
            service = GitLabService()
            assert service is not None

    def test_review_service_initialization(self):
        """Test review service can be initialized"""
        mock_agent = Mock()
        mock_agent.review_merge_request = AsyncMock()

        service = ReviewService(review_agent=mock_agent)
        assert service.review_agent is mock_agent

    def test_review_service_without_agent(self):
        """Test review service creates default agent when none provided"""
        with patch("src.services.review_service.CodeReviewAgent") as mock_agent_class:
            mock_instance = Mock()
            mock_agent_class.return_value = mock_instance

            service = ReviewService()
            mock_agent_class.assert_called_once()
            assert service.review_agent is mock_instance

    @pytest.mark.asyncio
    async def test_review_service_comment_formatting_integration(self):
        """Test review service comment formatting with realistic data"""
        mock_agent = Mock()
        mock_agent.review_merge_request = AsyncMock()

        service = ReviewService(review_agent=mock_agent)

        # Create a realistic review result
        result = ReviewResult(
            overall_assessment="approve_with_changes",
            risk_level="medium",
            summary="Code looks good with some improvements needed",
            issues=[],
            positive_feedback=["Good error handling", "Clear variable names"],
            metrics={"files_changed": 3, "lines_added": 45, "lines_removed": 12},
        )

        comment = service.format_review_comment(result)

        # Verify comment structure
        assert "⚠️ AI Code Review" in comment
        assert "**Overall Assessment:** Approve With Changes" in comment
        assert "**Risk Level:** Medium" in comment
        assert "Code looks good with some improvements needed" in comment
        assert "### ✨ Positive Feedback" in comment
        assert "Good error handling" in comment
        assert "### 📊 Review Metrics" in comment
        assert "**Files Changed:** 3" in comment

    def test_settings_integration_across_services(self):
        """Test that settings are properly shared across services"""
        from src.config.settings import get_settings

        settings = get_settings()

        # Verify settings have required fields for both services
        assert hasattr(settings, "gitlab_url")
        assert hasattr(settings, "gitlab_token")
        assert hasattr(settings, "request_timeout")
        assert hasattr(settings, "ai_model")
        assert hasattr(settings, "max_diff_size")

    def test_exception_handling_integration(self):
        """Test custom exceptions integrate properly"""
        from src.exceptions import (
            AIProviderException,
            GitLabAPIException,
            ReviewProcessException,
        )

        # Test exception hierarchy
        gitlab_exc = GitLabAPIException("API Error", 500, "Internal Error")
        assert isinstance(gitlab_exc, Exception)
        assert gitlab_exc.status_code == 500

        ai_exc = AIProviderException("Model Error", "openai", "gpt-4")
        assert isinstance(ai_exc, Exception)
        assert ai_exc.provider == "openai"

        review_exc = ReviewProcessException("Process Error", 123, 456)
        assert isinstance(review_exc, Exception)
        assert review_exc.merge_request_iid == 123


class TestWorkflowIntegration:
    """Test complete workflow integrations"""

    @pytest.mark.asyncio
    async def test_merge_request_review_workflow_components(self, merge_request_event):
        """Test components work together in MR review workflow"""
        # Mock dependencies
        mock_agent = Mock()
        mock_agent.review_merge_request = AsyncMock()

        mock_result = ReviewResult(
            overall_assessment="approve", risk_level="low", summary="LGTM", issues=[]
        )
        mock_agent.review_merge_request.return_value = mock_result

        # Create service
        review_service = ReviewService(review_agent=mock_agent)

        # Test data
        mr_details = {
            "id": 123,
            "iid": 10,
            "title": "Test MR",
            "description": "Test description",
            "state": "opened",
        }

        mr_diff = [
            {"old_path": "test.py", "new_path": "test.py", "diff": "+ print('hello')"}
        ]

        # Execute workflow
        result = await review_service.review_merge_request(
            mr_details=mr_details, mr_diff=mr_diff, mr_event=merge_request_event
        )

        # Verify workflow execution
        assert result is mock_result
        assert result.metrics["files_changed"] == 1
        assert result.metrics["mr_iid"] == merge_request_event.object_attributes.iid
        assert "review_timestamp" in result.metrics

        # Verify agent was called with proper context
        mock_agent.review_merge_request.assert_called_once()
        call_args = mock_agent.review_merge_request.call_args

        context = call_args[1]["context"]
        assert context.repository_url == merge_request_event.project.web_url
        assert context.merge_request_iid == merge_request_event.object_attributes.iid

    def test_model_validation_integration(self):
        """Test that Pydantic models work correctly across services"""
        from src.models.gitlab_models import GitLabProject, GitLabUser
        from src.models.review_models import CodeIssue, ReviewContext

        # Test GitLab models
        user = GitLabUser(
            id=1, username="test", name="Test User", email="test@example.com"
        )
        assert user.username == "test"

        project = GitLabProject(
            id=100,
            name="test-project",
            namespace="test-namespace",
            web_url="https://gitlab.com/test/project",
            git_ssh_url="git@gitlab.com:test/project.git",
            git_http_url="https://gitlab.com/test/project.git",
            visibility_level=20,
            path_with_namespace="test/project",
            default_branch="main",
        )
        assert project.web_url == "https://gitlab.com/test/project"

        # Test review models
        context = ReviewContext(
            repository_url="https://gitlab.com/test/repo",
            merge_request_iid=123,
            source_branch="feature",
            target_branch="main",
            trigger_tag="ai-review",
            file_changes=[],
        )
        assert context.merge_request_iid == 123

        issue = CodeIssue(
            file_path="test.py",
            line_number=10,
            severity="medium",
            category="style",
            description="Style issue",
            suggestion="Fix style",
        )
        assert issue.severity == "medium"

    def test_configuration_integration(self):
        """Test configuration integration across the application"""
        import os

        from src.config.settings import Settings

        # Test that settings can be created with environment variables
        with patch.dict(
            os.environ,
            {
                "GITLAB_TOKEN": "test-token-123456789012345678901",
                "GITLAB_URL": "https://gitlab.example.com",
                "AI_MODEL": "openai:gpt-4o",
            },
            clear=True,
        ):
            settings = Settings()

            assert settings.gitlab_token == "test-token-123456789012345678901"
            assert settings.gitlab_url == "https://gitlab.example.com"
            assert settings.ai_model == "openai:gpt-4o"

            # Test derived settings
            assert settings.openai_model_name == "gpt-4o"


class TestHealthCheckIntegration:
    """Test health check integration"""

    def test_health_endpoints_components(self):
        """Test health check components are available"""
        from src.api import health

        # Health module should be importable
        assert health is not None

    def test_version_integration(self):
        """Test version system integration"""
        from src.utils.version import get_version

        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0

        # Test version caching
        version2 = get_version()
        assert version == version2


class TestMiddlewareIntegration:
    """Test middleware integration"""

    def test_middleware_components_available(self):
        """Test middleware components can be imported and initialized"""
        from src.api.middleware import RequestTracingMiddleware, SecurityMiddleware

        # Test middleware can be instantiated
        app_mock = Mock()
        security_mw = SecurityMiddleware(app_mock)
        tracing_mw = RequestTracingMiddleware(app_mock)

        assert hasattr(security_mw, "dispatch")
        assert hasattr(tracing_mw, "dispatch")

    def test_app_integration_components(self):
        """Test FastAPI app has required components"""
        from src.main import app

        # App should be created
        assert app is not None

        # Should have middleware stack attribute
        assert hasattr(app, "middleware_stack")

        # Should have routes
        assert hasattr(app, "routes")
        assert len(app.routes) > 0
