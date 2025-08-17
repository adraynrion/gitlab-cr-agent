"""
Simple tests for unified Context7 tool
"""

import pytest

from src.agents.tools.base import ToolCategory, ToolContext, ToolPriority
from src.agents.tools.unified_context7_tools import Context7DocumentationValidationTool


class TestContext7DocumentationValidationTool:
    """Test the unified Context7 validation tool"""

    @pytest.fixture
    def tool(self):
        """Create tool instance"""
        return Context7DocumentationValidationTool()

    @pytest.fixture
    def tool_context(self):
        """Create tool context with sample data"""
        return ToolContext(
            diff_content="""
+from fastapi import FastAPI
+from django.http import HttpResponse
+import requests
+
+app = FastAPI()
            """,
            file_changes=[
                {"path": "main.py", "new_file": False},
                {"path": "views.py", "new_file": False},
            ],
            source_branch="feature/test",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    def test_tool_properties(self, tool):
        """Test tool properties"""
        assert tool.category == ToolCategory.CORRECTNESS
        assert tool.priority == ToolPriority.HIGH
        assert tool.requires_network is True
        assert tool.name == "Context7DocumentationValidationTool"

    def test_extract_libraries(self, tool):
        """Test library extraction from diff content"""
        diff_content = """
+from fastapi import FastAPI
+import django
+from requests import Session
+import os
+import sys
        """

        libraries = tool._extract_libraries(diff_content)

        # Should extract external libraries but not standard library
        assert "fastapi" in libraries
        assert "django" in libraries
        assert "requests" in libraries
        assert "os" not in libraries
        assert "sys" not in libraries

    def test_extract_libraries_empty(self, tool):
        """Test library extraction with no imports"""
        diff_content = "# Just a comment\nprint('hello')"

        libraries = tool._extract_libraries(diff_content)

        assert len(libraries) == 0

    @pytest.mark.asyncio
    async def test_execute_no_libraries(self, tool, tool_context):
        """Test execution when no libraries are found"""
        tool_context.diff_content = "# No imports here\nprint('hello')"

        result = await tool.execute(tool_context)

        assert result.success is True
        assert result.tool_name == "Context7DocumentationValidationTool"
        assert result.category == ToolCategory.CORRECTNESS
        assert len(result.positive_findings) == 1
        assert "No external libraries" in result.positive_findings[0]
        assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_execute_with_context7_default(self, tool, tool_context):
        """Test execution with default Context7 behavior (unavailable)"""
        result = await tool.execute(tool_context)

        # Should complete successfully even when Context7 is unavailable
        assert result.success is True
        assert result.metrics["context7_availability_rate"] == 0.0
        assert result.confidence_score == 0.3  # Low confidence without Context7

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, tool, tool_context):
        """Test tool failure handling"""
        # Force an error by setting invalid diff content
        tool_context.diff_content = None

        result = await tool.execute(tool_context)

        assert result.success is False
        assert result.error_message is not None
        assert "Context7 validation tool failed" in result.error_message
