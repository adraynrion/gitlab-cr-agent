"""
Minimal tests for Context7 MCP integration
"""

import pytest

from src.agents.tools.mcp_context7 import (
    CodeSnippet,
    LibraryDocumentation,
    LibraryResolutionResult,
    QuestionAnswer,
)


class TestContext7Models:
    """Test Context7 MCP data models"""

    def test_library_resolution_result_creation(self):
        """Test LibraryResolutionResult model creation"""
        result = LibraryResolutionResult(
            library_id="/fastapi/fastapi",
            name="FastAPI",
            description="Modern web framework for Python",
            trust_score=9.0,
            versions=["0.104.0", "0.103.0"],
            context7_available=True,
            unavailability_reason=None,
        )

        assert result.name == "FastAPI"
        assert result.library_id == "/fastapi/fastapi"
        assert result.trust_score == 9.0
        assert result.context7_available is True

    def test_code_snippet_creation(self):
        """Test CodeSnippet model creation"""
        snippet = CodeSnippet(
            title="Basic FastAPI App",
            description="Create a basic FastAPI application",
            code="from fastapi import FastAPI\napp = FastAPI()",
            source="https://fastapi.tiangolo.com/tutorial/first-steps/",
            language="python",
        )

        assert snippet.title == "Basic FastAPI App"
        assert snippet.language == "python"
        assert "FastAPI" in snippet.code

    def test_question_answer_creation(self):
        """Test QuestionAnswer model creation"""
        qa = QuestionAnswer(
            topic="Getting Started",
            question="How to create a FastAPI app?",
            answer="Use the FastAPI() constructor",
            source="https://fastapi.tiangolo.com/tutorial/",
        )

        assert qa.topic == "Getting Started"
        assert "FastAPI" in qa.question
        assert "constructor" in qa.answer

    def test_library_documentation_creation(self):
        """Test LibraryDocumentation model creation"""
        doc = LibraryDocumentation(
            library_id="/fastapi/fastapi",
            topic="routing",
            snippets=[],
            questions_answers=[],
            references=[],
            context7_available=True,
            unavailability_reason=None,
        )

        assert doc.library_id == "/fastapi/fastapi"
        assert doc.topic == "routing"
        assert doc.context7_available is True
        assert isinstance(doc.snippets, list)


class TestContext7Integration:
    """Test Context7 MCP integration functionality"""

    @pytest.mark.asyncio
    async def test_context7_availability_check(self):
        """Test Context7 availability checking"""
        from src.agents.tools.mcp_context7 import _check_context7_available

        # In test environment, Context7 should not be available
        result = await _check_context7_available()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_resolve_library_fallback(self):
        """Test library resolution fallback when Context7 unavailable"""
        from src.agents.tools.mcp_context7 import resolve_library_id

        result = await resolve_library_id("fastapi")

        assert isinstance(result, LibraryResolutionResult)
        assert result.name == "fastapi"
        # Should have fallback behavior when Context7 unavailable
        assert result.context7_available is False

    @pytest.mark.asyncio
    async def test_get_docs_fallback(self):
        """Test documentation retrieval fallback"""
        from src.agents.tools.mcp_context7 import get_library_docs

        result = await get_library_docs("/fastapi/fastapi", "routing", 1000)

        assert isinstance(result, LibraryDocumentation)
        assert result.library_id == "/fastapi/fastapi"
        # Should have fallback behavior when Context7 unavailable

    @pytest.mark.asyncio
    async def test_search_docs_fallback(self):
        """Test documentation search fallback"""
        from src.agents.tools.mcp_context7 import search_documentation

        result = await search_documentation("FastAPI routing", ["fastapi"], 5)

        assert isinstance(result, list)
        # Should return empty list or fallback results when Context7 unavailable

    @pytest.mark.asyncio
    async def test_validate_api_fallback(self):
        """Test API validation fallback"""
        from src.agents.tools.mcp_context7 import validate_api_usage

        code = "from fastapi import FastAPI\napp = FastAPI()"
        result = await validate_api_usage("fastapi", code, "Basic app setup")

        assert isinstance(result, dict)
        assert "library_name" in result
        # Should have fallback behavior when Context7 unavailable


class TestContext7Configuration:
    """Test Context7 MCP configuration and settings"""

    def test_context7_settings_integration(self):
        """Test that Context7 settings are properly integrated"""
        from src.config.settings import get_settings

        settings = get_settings()

        # Check that Context7 settings exist
        assert hasattr(settings, "context7_enabled")
        assert hasattr(settings, "context7_api_url")
        assert hasattr(settings, "context7_max_tokens")
        assert hasattr(settings, "context7_cache_ttl")

    def test_context7_basic_functionality(self):
        """Test basic Context7 functionality exists"""
        # Import test to ensure no syntax errors
        from src.agents.tools import mcp_context7

        assert hasattr(mcp_context7, "resolve_library_id")
        assert hasattr(mcp_context7, "get_library_docs")
        assert hasattr(mcp_context7, "search_documentation")
        assert hasattr(mcp_context7, "validate_api_usage")
