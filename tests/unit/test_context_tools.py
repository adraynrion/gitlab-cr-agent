"""
Unit tests for Context7 integration tools
"""


import pytest

from src.agents.tools.base import ToolCategory, ToolContext, ToolPriority
from src.agents.tools.python.context_tools import (
    Context7Client,
    PythonAPIUsageValidationTool,
    PythonDocumentationLookupTool,
    PythonSecurityPatternValidationTool,
)


class TestContext7Client:
    """Test Context7Client wrapper"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return Context7Client(settings={"cache_ttl": 60})

    @pytest.mark.asyncio
    async def test_resolve_library_id(self, client):
        """Test library ID resolution"""
        # Test exact match
        library_id = await client.resolve_library_id("python")
        assert library_id == "/websites/python-3"

        # Test partial match
        library_id = await client.resolve_library_id("fast")
        assert library_id == "/tiangolo/fastapi"

        # Test unknown library
        library_id = await client.resolve_library_id("unknown_library")
        assert library_id is None

    @pytest.mark.asyncio
    async def test_get_library_docs(self, client):
        """Test documentation retrieval"""
        docs = await client.get_library_docs("/websites/python-3", topic="testing")

        assert docs is not None
        assert docs["library_id"] == "/websites/python-3"
        assert docs["topic"] == "testing"
        assert "snippets" in docs
        assert "references" in docs
        assert len(docs["snippets"]) > 0
        assert len(docs["references"]) > 0

    @pytest.mark.asyncio
    async def test_docs_caching(self, client):
        """Test documentation caching"""
        # First call
        docs1 = await client.get_library_docs("/websites/python-3", topic="async")

        # Second call should be cached
        docs2 = await client.get_library_docs("/websites/python-3", topic="async")

        # Should be the same object (from cache)
        assert docs1 == docs2


class TestPythonDocumentationLookupTool:
    """Test PythonDocumentationLookupTool"""

    @pytest.fixture
    def tool(self):
        """Create a documentation lookup tool"""
        return PythonDocumentationLookupTool()

    @pytest.fixture
    def context_with_imports(self):
        """Create context with import statements"""
        diff_content = """
        + import fastapi
        + from pydantic import BaseModel
        + import unknown_lib
        + from typing import List
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    @pytest.fixture
    def context_no_imports(self):
        """Create context without imports"""
        diff_content = """
        + def hello():
        +     print("Hello World")
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    def test_tool_properties(self, tool):
        """Test tool properties"""
        assert tool.category == ToolCategory.DOCUMENTATION
        assert tool.priority == ToolPriority.HIGH
        assert tool.requires_network is True

    @pytest.mark.asyncio
    async def test_execute_with_imports(self, tool, context_with_imports):
        """Test execution with import statements"""
        result = await tool.execute(context_with_imports)

        assert result.success is True
        assert result.tool_name == "PythonDocumentationLookupTool"
        assert result.category == ToolCategory.DOCUMENTATION
        assert result.confidence_score > 0.5

        # Should have found fastapi and pydantic imports
        assert len(result.evidence) >= 1
        assert len(result.references) >= 1

    @pytest.mark.asyncio
    async def test_execute_no_imports(self, tool, context_no_imports):
        """Test execution without import statements"""
        result = await tool.execute(context_no_imports)

        assert result.success is True
        assert len(result.positive_findings) == 1
        assert "No new imports to validate" in result.positive_findings[0]

    def test_extract_imports(self, tool):
        """Test import extraction"""
        diff_content = """
        + import os
        + import fastapi
        + from pydantic import BaseModel
        + from typing import List, Dict
        + from src.models import User
        """

        imports = tool._extract_imports(diff_content)

        # Should exclude standard library imports (os, typing)
        import_names = [imp["library"] for imp in imports]
        assert "fastapi" in import_names
        assert "pydantic" in import_names
        assert "src" in import_names
        assert "os" not in import_names  # Standard library
        assert "typing" not in import_names  # Standard library

    def test_analyze_usage(self, tool):
        """Test usage analysis"""
        import_info = {"library": "fastapi", "module": None}
        docs = {
            "snippets": [
                {
                    "description": "This feature is deprecated in v2.0",
                    "code": "example code",
                }
            ],
            "references": ["https://fastapi.tiangolo.com"],
        }

        analysis = tool._analyze_usage(import_info, docs)

        assert "issues" in analysis
        assert "suggestions" in analysis
        assert len(analysis["issues"]) == 1
        assert "deprecated" in analysis["issues"][0]["description"]
        assert len(analysis["suggestions"]) == 1


class TestPythonAPIUsageValidationTool:
    """Test PythonAPIUsageValidationTool"""

    @pytest.fixture
    def tool(self):
        """Create an API validation tool"""
        return PythonAPIUsageValidationTool()

    @pytest.fixture
    def context_with_api_calls(self):
        """Create context with API calls"""
        diff_content = """
        + user = User.get(user_id)
        + await session.commit()
        + result = client.post("/api/users")
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    def test_tool_properties(self, tool):
        """Test tool properties"""
        assert tool.category == ToolCategory.CORRECTNESS
        assert tool.priority == ToolPriority.HIGH
        assert tool.requires_network is True

    @pytest.mark.asyncio
    async def test_execute_with_api_calls(self, tool, context_with_api_calls):
        """Test execution with API calls"""
        result = await tool.execute(context_with_api_calls)

        assert result.success is True
        assert result.tool_name == "PythonAPIUsageValidationTool"
        assert result.category == ToolCategory.CORRECTNESS

    def test_extract_api_calls(self, tool):
        """Test API call extraction"""
        diff_content = """
        + user = User.get(123)
        + await client.post("/api/test")
        + result = session.query()
        """

        api_calls = tool._extract_api_calls(diff_content)

        assert len(api_calls) >= 2

        # Check extracted calls
        call_methods = [(call["library"], call["method"]) for call in api_calls]
        assert ("User", "get") in call_methods
        assert ("client", "post") in call_methods or (
            "session",
            "query",
        ) in call_methods

    def test_validate_api_call(self, tool):
        """Test API call validation"""
        api_call = {"library": "User", "method": "get", "async": False}
        docs = {"snippets": [{"code": "User.get(id)"}, {"code": "User.create(data)"}]}

        validation = tool._validate_api_call(api_call, docs)

        assert "issues" in validation
        assert "suggestions" in validation
        # Method should be found in documentation
        assert len(validation["issues"]) == 0


class TestPythonSecurityPatternValidationTool:
    """Test PythonSecurityPatternValidationTool"""

    @pytest.fixture
    def tool(self):
        """Create a security validation tool"""
        return PythonSecurityPatternValidationTool()

    @pytest.fixture
    def context_with_security_issues(self):
        """Create context with security issues"""
        diff_content = """
        + password = "hardcoded_secret"
        + query = f"SELECT * FROM users WHERE id = {user_id}"
        + eval(user_input)
        + allow_origins = ["*"]
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    @pytest.fixture
    def context_secure(self):
        """Create context with secure code"""
        diff_content = """
        + password = os.getenv("PASSWORD")
        + query = "SELECT * FROM users WHERE id = ?"
        + result = json.loads(user_input)
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    def test_tool_properties(self, tool):
        """Test tool properties"""
        assert tool.category == ToolCategory.SECURITY
        assert tool.priority == ToolPriority.CRITICAL
        assert tool.requires_network is True

    @pytest.mark.asyncio
    async def test_execute_with_security_issues(
        self, tool, context_with_security_issues
    ):
        """Test execution with security issues"""
        result = await tool.execute(context_with_security_issues)

        assert result.success is True
        assert result.tool_name == "PythonSecurityPatternValidationTool"
        assert result.category == ToolCategory.SECURITY
        assert len(result.issues) >= 1
        assert result.confidence_score > 0.9

        # Check that OWASP reference is included
        assert any("owasp.org" in ref for ref in result.references)

        # Check severity levels
        severities = [issue["severity"] for issue in result.issues]
        assert "critical" in severities or "high" in severities

    @pytest.mark.asyncio
    async def test_execute_secure_code(self, tool, context_secure):
        """Test execution with secure code"""
        result = await tool.execute(context_secure)

        assert result.success is True
        # Should have fewer or no security issues
        assert len(result.issues) == 0 or all(
            issue["severity"] in ["low", "medium"] for issue in result.issues
        )

    def test_detect_security_patterns(self, tool):
        """Test security pattern detection"""
        diff_content = """
        + password = "secret123"
        + execute(f"DELETE FROM users WHERE id = {user_id}")
        + eval(user_code)
        + hashlib.md5(password.encode())
        + open(f"../../../etc/passwd")
        + allow_origins = ["*"]
        """

        patterns = tool._detect_security_patterns(diff_content)

        pattern_types = [p["type"] for p in patterns]
        assert "hardcoded_secret" in pattern_types
        assert "sql_injection" in pattern_types
        assert "eval_usage" in pattern_types
        assert "weak_crypto" in pattern_types
        assert "file_traversal" in pattern_types
        assert "cors_wildcard" in pattern_types

    def test_get_severity(self, tool):
        """Test severity level assignment"""
        assert tool._get_severity("sql_injection") == "critical"
        assert tool._get_severity("hardcoded_secret") == "critical"
        assert tool._get_severity("weak_crypto") == "high"
        assert tool._get_severity("eval_usage") == "high"
        assert tool._get_severity("cors_wildcard") == "medium"
        assert tool._get_severity("unknown_pattern") == "medium"

    def test_validate_security_pattern(self, tool):
        """Test security pattern validation"""
        pattern = {
            "type": "sql_injection",
            "match": "query = f'SELECT * FROM users WHERE id = {user_id}'",
            "severity": "critical",
        }
        docs = {"snippets": []}

        validation = tool._validate_security_pattern(pattern, docs)

        assert "issues" in validation
        assert "suggestions" in validation
        assert len(validation["issues"]) == 1
        assert validation["issues"][0]["severity"] == "critical"
        assert "SQL injection" in validation["issues"][0]["description"]
        assert len(validation["suggestions"]) == 1
        assert "parameterized queries" in validation["suggestions"][0]


class TestToolIntegration:
    """Integration tests for multiple tools working together"""

    @pytest.fixture
    def context_complex(self):
        """Create context with complex code changes"""
        diff_content = """
        + import fastapi
        + from pydantic import BaseModel
        +
        + password = "hardcoded_password"
        +
        + async def get_user(user_id: int):
        +     query = f"SELECT * FROM users WHERE id = {user_id}"
        +     result = await database.execute(query)
        +     return result
        +
        + @app.get("/users/{user_id}")
        + def get_user_endpoint(user_id: int):
        +     return get_user(user_id)
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[{"path": "main.py", "action": "modified"}],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
            settings={
                "context7": {"enabled": True, "max_tokens": 2000, "cache_ttl": 3600}
            },
        )

    @pytest.mark.asyncio
    async def test_multiple_tools_execution(self, context_complex):
        """Test executing multiple tools on the same context"""
        tools = [
            PythonDocumentationLookupTool(),
            PythonAPIUsageValidationTool(),
            PythonSecurityPatternValidationTool(),
        ]

        results = []
        for tool in tools:
            result = await tool.execute(context_complex)
            results.append(result)

        # All tools should succeed
        assert all(result.success for result in results)

        # Should have different categories
        categories = [result.category for result in results]
        assert ToolCategory.DOCUMENTATION in categories
        assert ToolCategory.CORRECTNESS in categories
        assert ToolCategory.SECURITY in categories

        # Security tool should find critical issues
        security_result = next(
            r for r in results if r.category == ToolCategory.SECURITY
        )
        assert len(security_result.issues) >= 1
        assert any(issue["severity"] == "critical" for issue in security_result.issues)

        # Documentation tool should find imports
        doc_result = next(
            r for r in results if r.category == ToolCategory.DOCUMENTATION
        )
        assert len(doc_result.evidence) >= 1 or len(doc_result.references) >= 1
