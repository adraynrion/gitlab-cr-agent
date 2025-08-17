"""
Simple tests for Context7 MCP integration
"""

from unittest.mock import Mock, patch

from src.agents.tools.mcp_context7 import (
    CodeSnippet,
    LibraryDocumentation,
    LibraryResolutionResult,
    QuestionAnswer,
    _check_mcp_available,
    get_library_docs,
    resolve_library_id,
    search_documentation,
    validate_api_usage,
)


class TestContext7MCPAvailability:
    """Test Context7 MCP availability checks"""

    def test_check_mcp_available_false(self):
        """Test MCP availability check returns False when not available"""
        result = _check_mcp_available()
        # Should return False in test environment (no MCP available)
        assert result is False

    def test_check_mcp_not_available(self):
        """Test MCP availability when not available"""
        with patch("builtins.hasattr", return_value=False):
            result = _check_mcp_available()
            assert result is False

    @patch("builtins.hasattr")
    def test_check_mcp_available(self, mock_hasattr):
        """Test MCP availability when available"""
        mock_hasattr.return_value = True
        result = _check_mcp_available()
        assert result is True


class TestLibraryResolution:
    """Test library resolution functions"""

    def test_resolve_library_id_unavailable(self):
        """Test library resolution when Context7 is unavailable"""
        result = resolve_library_id("fastapi")

        assert isinstance(result, LibraryResolutionResult)
        assert result.name == "fastapi"
        assert result.library_id is None
        assert result.context7_available is False
        assert result.unavailability_reason is not None
        assert "Context7 MCP" in result.unavailability_reason

    @patch("src.agents.tools.mcp_context7._check_mcp_available", return_value=True)
    @patch("builtins.hasattr", return_value=True)
    def test_resolve_library_id_with_mcp(self, mock_hasattr, mock_check):
        """Test library resolution when MCP is available"""
        mock_resolve = Mock(
            return_value={
                "library_id": "/tiangolo/fastapi",
                "name": "fastapi",
                "description": "FastAPI framework",
                "trust_score": 9.5,
                "versions": ["0.100.0", "0.99.0"],
            }
        )

        with patch(
            "builtins.globals",
            return_value={"mcp__context7__resolve_library_id": mock_resolve},
        ):
            result = resolve_library_id("fastapi")

            assert result.library_id == "/tiangolo/fastapi"
            assert result.name == "fastapi"
            assert result.trust_score == 9.5
            assert result.context7_available is True

    def test_resolve_library_id_without_mcp(self):
        """Test library resolution when MCP is not available"""
        with patch(
            "src.agents.tools.mcp_context7._check_mcp_available", return_value=False
        ):
            result = resolve_library_id("django")

            assert result.name == "django"
            assert result.library_id is None
            assert result.context7_available is False
            assert "not available" in result.unavailability_reason

    @patch("src.agents.tools.mcp_context7._check_mcp_available", return_value=True)
    @patch("builtins.hasattr", return_value=True)
    def test_resolve_library_id_with_error(self, mock_hasattr, mock_check):
        """Test library resolution with error"""
        mock_resolve = Mock(side_effect=Exception("API error"))

        with patch(
            "builtins.globals",
            return_value={"mcp__context7__resolve_library_id": mock_resolve},
        ):
            result = resolve_library_id("requests")

            assert result.context7_available is False
            assert "Error during Context7 resolution" in result.unavailability_reason

    def test_get_library_docs_unavailable(self):
        """Test documentation retrieval when Context7 is unavailable"""
        result = get_library_docs("/tiangolo/fastapi", topic="authentication")

        assert isinstance(result, LibraryDocumentation)
        assert result.library_id == "/tiangolo/fastapi"
        assert result.topic == "authentication"
        assert len(result.snippets) == 0
        assert len(result.questions_answers) == 0
        assert result.context7_available is False
        assert result.unavailability_reason is not None

    @patch("src.agents.tools.mcp_context7._check_mcp_available", return_value=True)
    @patch("builtins.hasattr", return_value=True)
    def test_get_library_docs_with_mcp(self, mock_hasattr, mock_check):
        """Test documentation retrieval when MCP is available"""
        mock_get_docs = Mock(
            return_value={
                "documentation": "FastAPI documentation content",
                "code_snippets": [{"title": "Basic app", "code": "app = FastAPI()"}],
                "questions_answers": [
                    {"question": "How to create app?", "answer": "Use FastAPI()"}
                ],
            }
        )

        with patch(
            "builtins.globals",
            return_value={"mcp__context7__get_library_docs": mock_get_docs},
        ):
            result = get_library_docs(
                "/tiangolo/fastapi", topic="routing", max_tokens=1000
            )

            assert result.library_id == "/tiangolo/fastapi"
            assert result.topic == "routing"
            assert result.context7_available is True
            assert len(result.snippets) > 0
            assert len(result.questions_answers) > 0

    def test_get_library_docs_without_mcp(self):
        """Test documentation retrieval when MCP is not available"""
        with patch(
            "src.agents.tools.mcp_context7._check_mcp_available", return_value=False
        ):
            result = get_library_docs("/django/django", topic="models")

            assert result.library_id == "/django/django"
            assert result.topic == "models"
            assert result.context7_available is False
            assert len(result.snippets) == 0


class TestSearchDocumentation:
    """Test documentation search functionality"""

    def test_search_documentation_no_libraries(self):
        """Test search with no libraries specified"""
        result = search_documentation("authentication")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_search_documentation_with_libraries(self):
        """Test search with libraries when Context7 unavailable"""
        result = search_documentation("authentication", libraries=["fastapi", "django"])

        assert isinstance(result, list)
        assert len(result) == 2

        # Check that unavailability is properly reported
        for item in result:
            assert "library" in item
            assert "context7_available" in item
            assert item["context7_available"] is False
            assert "unavailability_reason" in item

    @patch("src.agents.tools.mcp_context7.resolve_library_id")
    @patch("src.agents.tools.mcp_context7.get_library_docs")
    def test_search_documentation_with_mcp_available(self, mock_get_docs, mock_resolve):
        """Test search with multiple libraries when MCP is available"""
        mock_resolve.side_effect = [
            LibraryResolutionResult(
                library_id="/tiangolo/fastapi", name="fastapi", context7_available=True
            ),
            LibraryResolutionResult(
                library_id="/django/django", name="django", context7_available=True
            ),
        ]

        mock_get_docs.side_effect = [
            LibraryDocumentation(
                library_id="/tiangolo/fastapi",
                topic="authentication",
                documentation="FastAPI auth docs",
                context7_available=True,
            ),
            LibraryDocumentation(
                library_id="/django/django",
                topic="authentication",
                documentation="Django auth docs",
                context7_available=True,
            ),
        ]

        result = search_documentation("authentication", libraries=["fastapi", "django"])

        assert len(result) == 2
        assert result[0]["library"] == "fastapi"
        assert result[1]["library"] == "django"

    @patch("src.agents.tools.mcp_context7.resolve_library_id")
    def test_search_documentation_with_unavailable_library(self, mock_resolve):
        """Test search when library resolution fails"""
        mock_resolve.return_value = LibraryResolutionResult(
            name="unknown",
            context7_available=False,
            unavailability_reason="Library not found",
        )

        result = search_documentation("testing", libraries=["unknown"])

        assert len(result) == 1
        assert result[0]["context7_available"] is False


class TestAPIValidation:
    """Test API usage validation"""

    def test_validate_api_usage_unavailable(self):
        """Test API validation when Context7 is unavailable"""
        result = validate_api_usage(
            library_name="fastapi",
            code_snippet="from fastapi import FastAPI\napp = FastAPI()",
            context="web framework",
        )

        assert isinstance(result, dict)
        assert result["library_name"] == "fastapi"
        assert result["valid"] is None  # Cannot determine without Context7
        assert result["context7_available"] is False
        assert result["unavailability_reason"] is not None
        assert len(result["issues"]) == 0
        assert len(result["suggestions"]) == 0

    @patch("src.agents.tools.mcp_context7.resolve_library_id")
    @patch("src.agents.tools.mcp_context7._check_mcp_available", return_value=True)
    @patch("builtins.hasattr", return_value=True)
    def test_validate_api_usage_success(self, mock_hasattr, mock_check, mock_resolve):
        """Test successful API usage validation"""
        mock_resolve.return_value = LibraryResolutionResult(
            library_id="/tiangolo/fastapi", name="fastapi", context7_available=True
        )

        mock_validate = Mock(
            return_value={
                "valid": True,
                "issues": [],
                "suggestions": ["Consider using dependency injection"],
                "references": ["https://fastapi.tiangolo.com"],
            }
        )

        with patch(
            "builtins.globals",
            return_value={"mcp__context7__validate_code_against_docs": mock_validate},
        ):
            result = validate_api_usage(
                "fastapi", "from fastapi import FastAPI\napp = FastAPI()", "web api"
            )

            assert result["valid"] is True
            assert len(result["issues"]) == 0
            assert len(result["suggestions"]) > 0

    @patch("src.agents.tools.mcp_context7.resolve_library_id")
    def test_validate_api_usage_library_not_found(self, mock_resolve):
        """Test validation when library is not found"""
        mock_resolve.return_value = LibraryResolutionResult(
            name="unknown", context7_available=False, unavailability_reason="Not found"
        )

        result = validate_api_usage("unknown", "import unknown", "test")

        assert result["valid"] is None
        assert result["context7_available"] is False

    @patch("src.agents.tools.mcp_context7.resolve_library_id")
    def test_validate_api_usage_with_error(self, mock_resolve):
        """Test validation with error handling"""
        mock_resolve.side_effect = Exception("Network error")

        result = validate_api_usage("requests", "import requests", "http")

        assert result["context7_available"] is False
        assert "Error during validation" in result["unavailability_reason"]


class TestErrorHandling:
    """Test error handling scenarios"""

    @patch("src.agents.tools.mcp_context7._check_mcp_available")
    def test_resolve_library_id_with_mcp_available(self, mock_check):
        """Test library resolution when MCP is available but function missing"""
        mock_check.return_value = True

        # Mock builtins to not have the function
        with patch("builtins.hasattr", return_value=False):
            result = resolve_library_id("fastapi")

            assert result.context7_available is False
            assert "not accessible" in result.unavailability_reason

    def test_error_handling_in_resolve(self):
        """Test error handling in resolve_library_id"""
        with patch(
            "src.agents.tools.mcp_context7._check_mcp_available",
            side_effect=Exception("Test error"),
        ):
            result = resolve_library_id("fastapi")

            assert result.context7_available is False
            assert "Error during Context7 resolution" in result.unavailability_reason

    def test_error_handling_in_validation(self):
        """Test error handling in validate_api_usage"""
        with patch(
            "src.agents.tools.mcp_context7.resolve_library_id",
            side_effect=Exception("Test error"),
        ):
            result = validate_api_usage("fastapi", "code", "context")

            assert result["context7_available"] is False
            assert "Error during validation" in result["unavailability_reason"]
            assert len(result["issues"]) == 1
            assert "Error during validation" in result["issues"][0]


class TestLibraryModels:
    """Test Pydantic models for Context7 data"""

    def test_library_resolution_result_creation(self):
        """Test LibraryResolutionResult model creation"""
        result = LibraryResolutionResult(
            library_id="/test/library",
            name="test-library",
            description="Test library",
            trust_score=9.5,
            versions=["1.0.0", "2.0.0"],
            context7_available=True,
        )

        assert result.library_id == "/test/library"
        assert result.name == "test-library"
        assert result.trust_score == 9.5
        assert len(result.versions) == 2
        assert result.context7_available is True

    def test_library_documentation_creation(self):
        """Test LibraryDocumentation model creation"""
        doc = LibraryDocumentation(
            library_id="/test/library", topic="authentication", context7_available=True
        )

        assert doc.library_id == "/test/library"
        assert doc.topic == "authentication"
        assert len(doc.snippets) == 0
        assert len(doc.questions_answers) == 0
        assert doc.context7_available is True

    def test_models_with_unavailability(self):
        """Test models with unavailability information"""
        result = LibraryResolutionResult(
            name="test", context7_available=False, unavailability_reason="Test reason"
        )

        assert result.context7_available is False
        assert result.unavailability_reason == "Test reason"
        assert result.library_id is None

    def test_code_snippet_model(self):
        """Test CodeSnippet model"""
        snippet = CodeSnippet(
            title="Example",
            code="print('test')",
            description="Test snippet",
            language="python",
        )

        assert snippet.title == "Example"
        assert snippet.language == "python"

    def test_question_answer_model(self):
        """Test QuestionAnswer model"""
        qa = QuestionAnswer(
            question="What is Python?", answer="A programming language", confidence=0.95
        )

        assert qa.confidence == 0.95

    def test_library_resolution_comprehensive(self):
        """Test comprehensive LibraryResolutionResult model"""
        result = LibraryResolutionResult(
            library_id="/test/library",
            name="test-library",
            description="Test library",
            trust_score=9.5,
            versions=["1.0.0", "2.0.0"],
            context7_available=True,
        )

        assert result.library_id == "/test/library"
        assert result.name == "test-library"
        assert result.trust_score == 9.5
        assert len(result.versions) == 2
        assert result.context7_available is True

    def test_library_documentation_comprehensive(self):
        """Test comprehensive LibraryDocumentation model"""
        doc = LibraryDocumentation(
            library_id="/test/library", topic="authentication", context7_available=True
        )

        assert doc.library_id == "/test/library"
        assert doc.topic == "authentication"
        assert len(doc.snippets) == 0
        assert len(doc.questions_answers) == 0
        assert doc.context7_available is True
