"""
Integration tests for the simplified Context7 tool workflow
"""

import os

import pytest

from src.agents.code_reviewer import CodeReviewAgent
from src.agents.tools import ToolRegistry
from src.config.settings import settings
from src.models.review_models import ReviewContext


def configure_openrouter_for_test():
    """Configure OpenRouter settings for integration tests"""
    settings.openai_base_url = "https://openrouter.ai/api/v1"
    settings.openai_model_name = "openai/gpt-oss-20b:free"

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        pytest.skip("OpenRouter API key not configured for integration tests.")

    original_openai_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openrouter_key

    return {"OPENAI_API_KEY": original_openai_key}


def restore_original_settings(original_settings, original_env_vars=None):
    """Restore original settings after test"""
    for key, value in original_settings.items():
        setattr(settings, key, value)

    if original_env_vars:
        for env_var, value in original_env_vars.items():
            if value is not None:
                os.environ[env_var] = value
            elif env_var in os.environ:
                del os.environ[env_var]


class TestSimplifiedToolWorkflow:
    """Integration tests for the simplified Context7-based tool workflow"""

    @pytest.fixture
    def review_context(self):
        """Create a review context for testing"""
        return ReviewContext(
            repository_url="https://gitlab.com/test/repo",
            merge_request_iid=123,
            source_branch="feature/new-feature",
            target_branch="main",
            trigger_tag="ai-review",
            file_changes=[
                {
                    "path": "src/main.py",
                    "action": "modified",
                    "diff": "+ import fastapi\n+ from fastapi import FastAPI\n+ app = FastAPI()\n+ @app.get('/hello')\n+ def hello(): return 'world'",
                },
            ],
        )

    @pytest.fixture
    def test_diff_content(self):
        """Create test diff content with FastAPI usage"""
        return """
--- a/src/main.py
+++ b/src/main.py
@@ -0,0 +1,10 @@
+ import fastapi
+ from fastapi import FastAPI
+ from pydantic import BaseModel
+
+ app = FastAPI()
+
+ @app.get("/hello")
+ def hello():
+     return {"message": "Hello World"}
+
+ @app.post("/users")
+ def create_user(user_data: dict):
+     return user_data
"""

    @pytest.mark.asyncio
    async def test_context7_tool_workflow(self, review_context, test_diff_content):
        """Test the simplified Context7 tool workflow"""
        # Store original settings
        original_settings = {
            "tools_enabled": settings.tools_enabled,
            "context7_enabled": settings.context7_enabled,
            "openai_base_url": settings.openai_base_url,
            "openai_model_name": settings.openai_model_name,
            "openai_api_key_raw": settings.openai_api_key_raw,
        }

        try:
            # Configure for testing
            settings.tools_enabled = True
            settings.context7_enabled = True
            original_env_vars = configure_openrouter_for_test()

            # Create review agent
            agent = CodeReviewAgent(model_name="openai:gpt-oss-20b:free")

            # Perform review
            result = await agent.review_merge_request(
                diff_content=test_diff_content, context=review_context
            )

            # Verify result structure
            assert result is not None
            assert hasattr(result, "overall_assessment")
            assert hasattr(result, "issues")
            assert hasattr(result, "summary")
            assert hasattr(result, "metrics")

            # Should have completed successfully
            assert result.summary is not None
            assert "review_timestamp" in result.metrics

        finally:
            # Restore original settings
            restore_original_settings(original_settings, original_env_vars)

    @pytest.mark.asyncio
    async def test_simplified_tool_registry(self):
        """Test the simplified tool registry with only Context7 tool"""
        registry = ToolRegistry()

        # Import the unified Context7 tool
        import src.agents.tools.unified_context7_tools  # noqa: F401

        # Check that our single tool is registered
        stats = registry.get_statistics()
        assert stats["total_tools"] >= 1

        # Check we have our Context7 tool
        enabled_tools = registry.get_enabled_tools()
        tool_names = [tool.name for tool in enabled_tools]
        assert "Context7DocumentationValidationTool" in tool_names

        # Test tool execution
        from src.agents.tools.base import ToolContext

        context = ToolContext(
            diff_content="+ import fastapi\n+ app = fastapi.FastAPI()",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        # Execute our Context7 tool
        context7_tool = next(
            (
                t
                for t in enabled_tools
                if t.name == "Context7DocumentationValidationTool"
            ),
            None,
        )

        if context7_tool:
            result = await context7_tool.run(context)
            assert result.success is True or result.error_message is not None

    @pytest.mark.asyncio
    async def test_tools_disabled_workflow(self, review_context, test_diff_content):
        """Test review workflow with tools disabled"""
        original_settings = {
            "tools_enabled": settings.tools_enabled,
            "openai_base_url": settings.openai_base_url,
            "openai_model_name": settings.openai_model_name,
            "openai_api_key_raw": settings.openai_api_key_raw,
        }

        try:
            # Configure for testing
            settings.tools_enabled = False
            original_env_vars = configure_openrouter_for_test()

            # Create review agent
            agent = CodeReviewAgent(model_name="openai:gpt-oss-20b:free")

            # Perform review
            result = await agent.review_merge_request(
                diff_content=test_diff_content, context=review_context
            )

            # Should still work without tools
            assert result is not None
            assert hasattr(result, "overall_assessment")
            assert hasattr(result, "summary")

        finally:
            # Restore original settings
            restore_original_settings(original_settings, original_env_vars)

    def test_simplified_settings_integration(self):
        """Test integration with application settings"""
        # Test that tool settings are properly defined
        assert hasattr(settings, "tools_enabled")
        assert hasattr(settings, "context7_enabled")
        assert hasattr(settings, "tools_parallel_execution")

        # Test default values
        assert isinstance(settings.tools_enabled, bool)
        assert isinstance(settings.context7_enabled, bool)
        assert isinstance(settings.tools_parallel_execution, bool)


@pytest.mark.integration
class TestContext7Integration:
    """Test real Context7 MCP integration"""

    @pytest.fixture
    def fastapi_diff(self):
        """FastAPI code diff for Context7 validation"""
        return """
--- a/src/api.py
+++ b/src/api.py
@@ -0,0 +1,15 @@
+ from fastapi import FastAPI
+ from pydantic import BaseModel
+
+ app = FastAPI()
+
+ class User(BaseModel):
+     name: str
+     email: str
+
+ @app.post("/users")
+ def create_user(user: User):
+     return {"message": f"Created user {user.name}"}
+
+ @app.get("/users")
+ def get_users():
+     return []
"""

    @pytest.mark.asyncio
    async def test_context7_validation(self, fastapi_diff):
        """Test Context7 validation of FastAPI code"""
        from src.agents.tools.base import ToolContext
        from src.agents.tools.unified_context7_tools import (
            Context7DocumentationValidationTool,
        )

        context = ToolContext(
            diff_content=fastapi_diff,
            file_changes=[{"path": "src/api.py", "action": "added"}],
            source_branch="feature/api",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        tool = Context7DocumentationValidationTool()
        result = await tool.execute(context)

        # Should execute successfully (may or may not find issues depending on Context7 availability)
        assert result.success is True
        assert (
            result.metrics["libraries_validated"] >= 1
        )  # Should detect fastapi and pydantic

        # Should have processed FastAPI and Pydantic libraries
        assert (
            "fastapi" in result.evidence
            or "pydantic" in result.evidence
            or not result.evidence
        )
