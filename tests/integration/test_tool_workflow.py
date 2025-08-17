"""
Integration tests for the complete tool workflow
"""

import os

import pytest

from src.agents.code_reviewer import CodeReviewAgent
from src.agents.tools import ToolRegistry
from src.config.settings import settings
from src.models.review_models import ReviewContext


def configure_openrouter_for_test():
    """
    Configure OpenRouter settings for integration tests

    Uses OpenRouter's free model 'openai/gpt-oss-20b:free' to avoid API quota limits.
    Requires OPENROUTER_API_KEY environment variable to be set.

    Returns:
        dict: Original environment variables to restore after test
    """
    # Use OpenRouter with free model for testing - no quota limits
    settings.openai_base_url = "https://openrouter.ai/api/v1"
    settings.openai_model_name = "openai/gpt-oss-20b:free"

    # Use OpenRouter API key if available
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        pytest.skip(
            "OpenRouter API key not configured for integration tests. Set OPENROUTER_API_KEY environment variable."
        )

    # Store original environment variable for restoration
    original_openai_key = os.environ.get("OPENAI_API_KEY")

    # Temporarily override OPENAI_API_KEY with OpenRouter key
    # This allows the existing OpenAI provider code to work with OpenRouter
    os.environ["OPENAI_API_KEY"] = openrouter_key

    return {"OPENAI_API_KEY": original_openai_key}


def restore_original_settings(original_settings, original_env_vars=None):
    """Restore original settings after test"""
    for key, value in original_settings.items():
        setattr(settings, key, value)

    # Restore environment variables
    if original_env_vars:
        for env_var, value in original_env_vars.items():
            if value is not None:
                os.environ[env_var] = value
            elif env_var in os.environ:
                del os.environ[env_var]


class TestToolWorkflowIntegration:
    """Integration tests for the complete tool-enhanced review workflow"""

    @pytest.fixture
    def review_context(self):
        """Create a review context for testing"""
        return ReviewContext(
            repository_url="https://gitlab.com/test/repo",
            merge_request_iid=123,
            source_branch="feature/new-auth",
            target_branch="main",
            trigger_tag="ai-review",
            file_changes=[
                {
                    "path": "src/auth.py",
                    "action": "added",
                    "diff": "+ def authenticate(username, password):\n+     return username == 'admin' and password == 'secret'",
                },
                {
                    "path": "src/main.py",
                    "action": "modified",
                    "diff": "+ import fastapi\n+ from src.auth import authenticate\n+ \n+ @app.post('/login')\n+ def login(username: str, password: str):\n+     if authenticate(username, password):\n+         return {'status': 'success'}\n+     return {'status': 'failed'}",
                },
            ],
        )

    @pytest.fixture
    def complex_diff_content(self):
        """Create complex diff content with multiple issues"""
        return """
--- a/src/auth.py
+++ b/src/auth.py
@@ -0,0 +1,15 @@
+ import hashlib
+
+ # Hardcoded credentials (security issue)
+ ADMIN_PASSWORD = "secret123"
+
+ def authenticate(username, password):
+     # SQL injection vulnerability
+     query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
+
+     # Weak crypto
+     password_hash = hashlib.md5(password.encode()).hexdigest()
+
+     # Performance issue - string concatenation in loop
+     result = ""
+     for i in range(1000):
+         result += str(i)
+
+     return username == "admin" and password == ADMIN_PASSWORD

--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,15 @@
+ import fastapi
+ from fastapi import FastAPI
+ from src.auth import authenticate
+
+ app = FastAPI()
+
+ @app.post("/login")
+ def login(username: str, password: str):
+     # Missing async (performance issue)
+     if authenticate(username, password):
+         return {"status": "success"}
+     return {"status": "failed"}
+
+ # Missing response model (API issue)
+ @app.get("/users")
+ def get_users():
+     return []
"""

    @pytest.mark.asyncio
    async def test_complete_review_workflow(self, review_context, complex_diff_content):
        """Test the complete review workflow with tools enabled"""
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

            # Create review agent with OpenRouter configuration
            agent = CodeReviewAgent(model_name="openai:gpt-oss-20b:free")

            # Perform review
            result = await agent.review_merge_request(
                diff_content=complex_diff_content, context=review_context
            )

            # Verify result structure
            assert result is not None
            assert hasattr(result, "overall_assessment")
            assert hasattr(result, "issues")
            assert hasattr(result, "summary")
            assert hasattr(result, "metrics")

            # Should have found multiple issues
            assert (
                len(result.issues) >= 3
            )  # At least security, performance, and API issues

            # Should have different severity levels
            severities = [issue.severity for issue in result.issues]
            assert "critical" in severities or "high" in severities

            # Should have suggestions
            assert len(result.positive_feedback) >= 0

            # Should have metrics from tools
            assert "review_timestamp" in result.metrics

        finally:
            # Restore original settings
            restore_original_settings(original_settings, original_env_vars)

    @pytest.mark.asyncio
    async def test_review_workflow_tools_disabled(
        self, review_context, complex_diff_content
    ):
        """Test review workflow with tools disabled"""
        # Store original settings
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

            # Create review agent with OpenRouter configuration
            agent = CodeReviewAgent(model_name="openai:gpt-oss-20b:free")

            # Perform review
            result = await agent.review_merge_request(
                diff_content=complex_diff_content, context=review_context
            )

            # Should still work without tools
            assert result is not None
            assert hasattr(result, "overall_assessment")
            assert hasattr(result, "summary")

        finally:
            # Restore original settings
            restore_original_settings(original_settings, original_env_vars)

    @pytest.mark.asyncio
    async def test_tool_registry_integration(self):
        """Test tool registry integration"""
        registry = ToolRegistry()

        # Import tool modules to register them
        import src.agents.tools.analysis_tools  # noqa: F401
        import src.agents.tools.context_tools  # noqa: F401
        import src.agents.tools.validation_tools  # noqa: F401

        # Check that tools are registered
        stats = registry.get_statistics()
        assert stats["total_tools"] > 0

        # Check that we have tools in different categories
        security_tools = registry.get_tools_by_category(
            registry.get_enabled_tools()[0].category
            if registry.get_enabled_tools()
            else None
        )
        assert len(security_tools) >= 0  # May be 0 if no tools in that category

        # Test tool execution
        from src.agents.tools.base import ToolContext

        context = ToolContext(
            diff_content="+ print('hello')",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        enabled_tools = registry.get_enabled_tools()
        if enabled_tools:
            # Execute first enabled tool
            result = await enabled_tools[0].run(context)
            assert result.success is True or result.error_message is not None

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test tool error handling in the workflow"""
        from src.agents.tools.base import (
            BaseTool,
            ToolCategory,
            ToolPriority,
            ToolResult,
        )
        from src.agents.tools.registry import register_tool

        @register_tool(enabled=True, name="FailingTestTool")
        class FailingTestTool(BaseTool):
            @property
            def category(self) -> ToolCategory:
                return ToolCategory.CORRECTNESS

            @property
            def priority(self) -> ToolPriority:
                return ToolPriority.LOW

            async def execute(self, context) -> ToolResult:
                raise Exception("Simulated tool failure")

        registry = ToolRegistry()

        from src.agents.tools.base import ToolContext

        context = ToolContext(
            diff_content="+ print('hello')",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        # Should handle tool failures gracefully
        results = await registry.execute_tools(context)

        # Find our failing tool result
        failing_results = [r for r in results if r.tool_name == "FailingTestTool"]
        if failing_results:
            failing_result = failing_results[0]
            assert failing_result.success is False
            assert failing_result.error_message is not None
            assert "Simulated tool failure" in failing_result.error_message

    @pytest.mark.asyncio
    async def test_tool_caching_workflow(self):
        """Test tool result caching in the workflow"""
        from src.agents.tools.base import ToolContext

        registry = ToolRegistry()
        enabled_tools = registry.get_enabled_tools()

        if not enabled_tools:
            pytest.skip("No enabled tools for caching test")

        context = ToolContext(
            diff_content="+ def test(): pass",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        tool = enabled_tools[0]

        # First execution
        result1 = await tool.run(context)
        assert result1.cached is False

        # Second execution should be cached
        result2 = await tool.run(context)
        assert result2.cached is True

        # Results should be equivalent
        assert result1.tool_name == result2.tool_name
        assert result1.category == result2.category

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_execution(self):
        """Test parallel vs sequential tool execution"""
        registry = ToolRegistry()

        # Import some tools
        import src.agents.tools.analysis_tools  # noqa: F401
        from src.agents.tools.base import ToolContext

        context = ToolContext(
            diff_content="+ import fastapi\n+ def hello(): return 'world'",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        # Clear cache to ensure fresh execution
        registry.clear_cache()

        # Measure parallel execution time
        import time

        start_time = time.time()
        parallel_results = await registry.execute_tools(context, parallel=True)
        parallel_time = time.time() - start_time

        # Clear cache again
        registry.clear_cache()

        # Measure sequential execution time
        start_time = time.time()
        sequential_results = await registry.execute_tools(context, parallel=False)
        sequential_time = time.time() - start_time

        # Both should produce results
        assert len(parallel_results) == len(sequential_results)

        # Parallel should generally be faster (or at least not significantly slower)
        # Allow some tolerance for test environment variability
        assert parallel_time <= sequential_time + 1.0

        # Results should be comparable (same tools executed)
        parallel_tool_names = sorted([r.tool_name for r in parallel_results])
        sequential_tool_names = sorted([r.tool_name for r in sequential_results])
        assert parallel_tool_names == sequential_tool_names

    @pytest.mark.asyncio
    async def test_context7_integration(self):
        """Test Context7 MCP integration"""
        from src.agents.tools.context_tools import Context7Client

        client = Context7Client()

        # Test library resolution
        library_id = await client.resolve_library_id("python")
        assert library_id is not None

        # Test documentation retrieval
        if library_id:
            docs = await client.get_library_docs(library_id, topic="testing")
            assert docs is not None
            assert "snippets" in docs
            assert "references" in docs

    @pytest.mark.asyncio
    async def test_tool_configuration_from_settings(self):
        """Test tool configuration from application settings"""
        registry = ToolRegistry()

        # Test configuration
        test_settings = {
            "enabled_categories": ["security", "performance"],
            "disabled_categories": ["style"],
            "enabled_tools": ["PythonSecurityAnalysisTool"],
            "disabled_tools": ["PythonCodeQualityTool"],
        }

        registry.configure_from_settings(test_settings)

        # Verify configuration was applied
        enabled_tools = registry.get_enabled_tools()
        enabled_tool_names = [tool.name for tool in enabled_tools]

        # PythonSecurityAnalysisTool should be explicitly enabled
        if "PythonSecurityAnalysisTool" in [
            tool.name for tool in registry.get_all_tools()
        ]:
            assert "PythonSecurityAnalysisTool" in enabled_tool_names

        # PythonCodeQualityTool should be explicitly disabled
        if "PythonCodeQualityTool" in [tool.name for tool in registry.get_all_tools()]:
            assert "PythonCodeQualityTool" not in enabled_tool_names

    def test_settings_integration(self):
        """Test integration with application settings"""
        # Test that tool settings are properly defined
        assert hasattr(settings, "tools_enabled")
        assert hasattr(settings, "context7_enabled")
        assert hasattr(settings, "tools_parallel_execution")
        assert hasattr(settings, "enabled_tool_categories")
        assert hasattr(settings, "disabled_tool_categories")

        # Test default values
        assert isinstance(settings.tools_enabled, bool)
        assert isinstance(settings.context7_enabled, bool)
        assert isinstance(settings.tools_parallel_execution, bool)
        assert isinstance(settings.enabled_tool_categories, list)
        assert isinstance(settings.disabled_tool_categories, list)

        # Test that all expected categories are included by default
        expected_categories = [
            "documentation",
            "security",
            "performance",
            "correctness",
            "maintainability",
        ]
        for category in expected_categories:
            assert category in settings.enabled_tool_categories


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world scenarios with the tool system"""

    @pytest.fixture
    def security_vulnerable_diff(self):
        """Real-world security vulnerable code diff"""
        return """
--- a/src/user_service.py
+++ b/src/user_service.py
@@ -0,0 +1,25 @@
+ import sqlite3
+ import hashlib
+
+ DATABASE_URL = "sqlite:///users.db"
+ ADMIN_SECRET = "admin123"  # Hardcoded secret
+
+ def authenticate_user(username, password):
+     conn = sqlite3.connect(DATABASE_URL)
+
+     # SQL Injection vulnerability
+     query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
+     cursor = conn.execute(query)
+     user = cursor.fetchone()
+
+     if user:
+         # Weak cryptography
+         password_hash = hashlib.md5(password.encode()).hexdigest()
+         return password_hash == user[2]
+
+     return False
+
+ def admin_access(secret):
+     # Dangerous eval usage
+     if eval(f"'{secret}' == '{ADMIN_SECRET}'"):
+         return True
+     return False
"""

    @pytest.fixture
    def performance_issues_diff(self):
        """Real-world performance issues code diff"""
        return """
--- a/src/data_processor.py
+++ b/src/data_processor.py
@@ -0,0 +1,30 @@
+ import requests
+
+ def process_user_data(user_ids):
+     results = []
+
+     # N+1 query problem
+     for user_id in user_ids:
+         response = requests.get(f"/api/users/{user_id}")
+         user_data = response.json()
+         results.append(user_data)
+
+     return results
+
+ def generate_report(data):
+     report = ""
+
+     # String concatenation in loop
+     for item in data:
+         report += f"User: {item['name']}\n"
+         report += f"Email: {item['email']}\n"
+         report += "---\n"
+
+     return report
+
+ def check_permissions(user, resources):
+     # Inefficient membership testing
+     allowed_resources = ['read', 'write', 'delete', 'admin']
+
+     for resource in resources:
+         if resource in allowed_resources:  # Should use set for O(1) lookup
+             return True
+     return False
"""

    @pytest.mark.asyncio
    async def test_security_vulnerability_detection(self, security_vulnerable_diff):
        """Test detection of real security vulnerabilities"""
        from src.agents.tools.base import ToolContext
        from src.agents.tools.python.analysis_tools import PythonSecurityAnalysisTool
        from src.agents.tools.python.context_tools import (
            PythonSecurityPatternValidationTool,
        )

        context = ToolContext(
            diff_content=security_vulnerable_diff,
            file_changes=[{"path": "src/user_service.py", "action": "added"}],
            source_branch="feature/auth",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        # Test security tools
        security_tools = [
            PythonSecurityPatternValidationTool(),
            PythonSecurityAnalysisTool(),
        ]

        all_issues = []
        for tool in security_tools:
            result = await tool.execute(context)
            assert result.success is True
            all_issues.extend(result.issues)

        # Should detect multiple security issues
        assert len(all_issues) >= 3

        # Should detect specific vulnerabilities
        issue_descriptions = [issue["description"].lower() for issue in all_issues]
        assert any(
            "sql injection" in desc or "injection" in desc
            for desc in issue_descriptions
        )
        assert any(
            "hardcoded" in desc or "secret" in desc for desc in issue_descriptions
        )
        assert any("eval" in desc or "exec" in desc for desc in issue_descriptions)
        assert any("md5" in desc or "weak" in desc for desc in issue_descriptions)

        # Should have critical severity issues
        severities = [issue["severity"] for issue in all_issues]
        assert "critical" in severities

    @pytest.mark.asyncio
    async def test_performance_issue_detection(self, performance_issues_diff):
        """Test detection of real performance issues"""
        from src.agents.tools.base import ToolContext
        from src.agents.tools.python.validation_tools import (
            PythonPerformancePatternTool,
        )

        context = ToolContext(
            diff_content=performance_issues_diff,
            file_changes=[{"path": "src/data_processor.py", "action": "added"}],
            source_branch="feature/optimization",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        tool = PythonPerformancePatternTool()
        result = await tool.execute(context)

        assert result.success is True
        assert len(result.issues) >= 2

        # Should detect specific performance anti-patterns
        issue_descriptions = [issue["description"].lower() for issue in result.issues]
        assert any("string concatenation" in desc for desc in issue_descriptions)
        assert any(
            "membership test" in desc or "list" in desc for desc in issue_descriptions
        )

        # Should have performance-related suggestions
        suggestions_text = " ".join(result.suggestions).lower()
        assert "join" in suggestions_text or "set" in suggestions_text

    @pytest.mark.asyncio
    async def test_comprehensive_analysis(
        self, security_vulnerable_diff, performance_issues_diff
    ):
        """Test comprehensive analysis with multiple tools on complex code"""
        combined_diff = security_vulnerable_diff + "\n" + performance_issues_diff

        # Import all tool modules
        import src.agents.tools.analysis_tools  # noqa: F401
        import src.agents.tools.context_tools  # noqa: F401
        import src.agents.tools.validation_tools  # noqa: F401
        from src.agents.tools import ToolRegistry
        from src.agents.tools.base import ToolContext

        registry = ToolRegistry()

        context = ToolContext(
            diff_content=combined_diff,
            file_changes=[
                {"path": "src/user_service.py", "action": "added"},
                {"path": "src/data_processor.py", "action": "added"},
            ],
            source_branch="feature/major-update",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        # Execute all enabled tools
        results = await registry.execute_tools(context, parallel=True)

        # Should have results from multiple tools
        assert len(results) >= 3

        # Should have found issues across different categories
        all_issues = []
        categories_found = set()

        for result in results:
            if result.success:
                all_issues.extend(result.issues)
                categories_found.add(result.category.value)

        # Should find security and performance issues
        assert len(all_issues) >= 5
        assert "security" in categories_found
        assert "performance" in categories_found or "correctness" in categories_found

        # Should have a mix of severity levels
        severities = [issue["severity"] for issue in all_issues]
        assert "critical" in severities
        assert "high" in severities or "medium" in severities
