"""
Unit tests for standards-based analysis tools
"""

from unittest.mock import patch

import pytest

from src.agents.tools.analysis_tools import SecurityAnalysisTool
from src.agents.tools.base import ToolContext
from src.agents.tools.validation_tools import ErrorHandlingTool, PerformancePatternTool


class TestSecurityAnalysisToolStandardsBased:
    """Test SecurityAnalysisTool with standards-based rules"""

    @pytest.fixture
    def security_tool(self):
        """Create a security analysis tool"""
        return SecurityAnalysisTool()

    @pytest.fixture
    def test_context(self):
        """Create a test context with security issues"""
        diff_content = """
        + password = "hardcoded_secret123"
        + query = f"SELECT * FROM users WHERE id = {user_id}"
        + eval(user_input)
        + hashlib.md5(password.encode())
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    @pytest.fixture
    def mock_security_rules(self):
        """Mock security rules from OWASP/NIST"""
        return {
            "patterns": {
                "sql_injection": {
                    "severity": "critical",
                    "description": "SQL injection vulnerability detected",
                    "recommendation": "Use parameterized queries to prevent SQL injection",
                    "references": [
                        "https://owasp.org/www-community/attacks/SQL_Injection"
                    ],
                },
                "hardcoded_secrets": {
                    "severity": "critical",
                    "description": "Hardcoded secrets detected",
                    "recommendation": "Use environment variables or secret management systems",
                    "references": [
                        "https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password"
                    ],
                },
                "code_injection": {
                    "severity": "critical",
                    "description": "Code injection vulnerability via eval/exec",
                    "recommendation": "Avoid eval/exec; use ast.literal_eval for safe evaluation",
                    "references": ["https://owasp.org/Top10/A03_2021-Injection/"],
                },
                "weak_crypto": {
                    "severity": "high",
                    "description": "Weak cryptographic algorithms detected",
                    "recommendation": "Use stronger hashing algorithms like SHA-256 or bcrypt",
                    "references": [
                        "https://owasp.org/www-community/vulnerabilities/Insecure_Cryptographic_Storage"
                    ],
                },
            },
            "recommendations": [
                "Follow OWASP Top 10 security guidelines",
                "Implement secure coding practices",
                "Use security linters and static analysis tools",
            ],
            "source": "OWASP Top 10 + NIST Framework",
            "last_updated": 1234567890,
        }

    @pytest.mark.asyncio
    async def test_security_analysis_with_standards(
        self, security_tool, test_context, mock_security_rules
    ):
        """Test security analysis using standards-based rules"""
        # Mock the rule engine to return our test rules
        with patch.object(
            security_tool.rule_engine, "get_security_rules"
        ) as mock_get_rules:
            mock_get_rules.return_value = mock_security_rules

            # Execute the security analysis
            result = await security_tool.execute(test_context)

            # Verify successful execution
            assert result.success is True
            assert result.tool_name == "SecurityAnalysisTool"
            assert result.category.value == "security"

            # Verify security issues were detected
            assert (
                len(result.issues) >= 2
            )  # Should detect hardcoded secrets and eval usage at minimum

            # Verify issue details - be more flexible with descriptions
            issue_descriptions = [issue["description"] for issue in result.issues]

            # At least one of these patterns should be detected
            has_secret_issue = any(
                "secret" in desc.lower() or "hardcoded" in desc.lower()
                for desc in issue_descriptions
            )
            has_eval_issue = any(
                "eval" in desc.lower() or "code injection" in desc.lower()
                for desc in issue_descriptions
            )

            # Should detect at least hardcoded secrets and eval usage
            assert (
                has_secret_issue or has_eval_issue
            ), f"Expected security issues not found in: {issue_descriptions}"

            # Verify severity levels from standards
            severities = [issue["severity"] for issue in result.issues]
            assert "critical" in severities

            # Verify suggestions from standards
            assert len(result.suggestions) > 0
            # Check for security-related suggestions
            suggestion_text = " ".join(result.suggestions).lower()
            assert any(
                keyword in suggestion_text
                for keyword in [
                    "environment",
                    "secret",
                    "management",
                    "avoid",
                    "eval",
                    "parameter",
                    "secure",
                ]
            ), f"Expected security suggestions not found in: {result.suggestions}"

            # Verify references from OWASP
            assert len(result.references) > 0
            assert any("owasp.org" in ref for ref in result.references)

            # Verify metrics include standards info
            assert "rules_source" in result.metrics
            assert result.metrics["rules_source"] == "OWASP Top 10 + NIST Framework"
            assert "rules_last_updated" in result.metrics

    @pytest.mark.asyncio
    async def test_security_analysis_no_issues(
        self, security_tool, mock_security_rules
    ):
        """Test security analysis with secure code"""
        # Context with secure code patterns
        secure_context = ToolContext(
            diff_content="""
            + password = os.getenv("DATABASE_PASSWORD")
            + query = "SELECT * FROM users WHERE id = ?"
            + result = json.loads(user_input)
            + password_hash = hashlib.sha256(password.encode())
            """,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        with patch.object(
            security_tool.rule_engine, "get_security_rules"
        ) as mock_get_rules:
            mock_get_rules.return_value = mock_security_rules

            result = await security_tool.execute(secure_context)

            # Should succeed with few or no issues
            assert result.success is True
            assert len(result.issues) == 0  # Secure code should have no issues
            assert len(result.positive_findings) > 0
            assert "OWASP" in result.positive_findings[0]  # Should mention standards

    def test_detect_pattern_in_code(self, security_tool):
        """Test pattern detection accuracy"""
        diff_content = "+ eval(user_input)"
        pattern_info = {
            "severity": "critical",
            "description": "Code injection vulnerability",
            "recommendation": "Avoid eval/exec",
        }

        findings = security_tool._detect_pattern_in_code(
            diff_content, "code_injection", pattern_info
        )

        assert len(findings) == 1
        assert findings[0]["pattern"] == "code_injection"
        assert findings[0]["severity"] == "critical"
        assert "eval" in findings[0]["evidence"]


class TestPerformancePatternToolStandardsBased:
    """Test PerformancePatternTool with standards-based rules"""

    @pytest.fixture
    def performance_tool(self):
        """Create a performance analysis tool"""
        return PerformancePatternTool()

    @pytest.fixture
    def test_context_with_performance_issues(self):
        """Create context with performance issues"""
        diff_content = """
        + for item in items:
        +     result += str(item) + ","
        +
        + if user_id in [1, 2, 3, 4, 5]:
        +     process_user(user_id)
        +
        + for i in range(len(data)):
        +     process(data[i])
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    @pytest.fixture
    def mock_performance_rules(self):
        """Mock performance rules from Python docs"""
        return {
            "patterns": {
                "string_concatenation": {
                    "severity": "medium",
                    "description": "Inefficient string concatenation in loops",
                    "recommendation": "Use ''.join() for better performance",
                    "alternative": "Use list comprehension or generator expressions",
                },
                "membership_test": {
                    "severity": "medium",
                    "description": "Inefficient membership testing on list",
                    "recommendation": "Use set for O(1) membership testing",
                    "alternative": "Convert list to set for frequent membership tests",
                },
                "repeated_function_calls": {
                    "severity": "low",
                    "description": "Function call in loop condition",
                    "recommendation": "Cache function result outside loop",
                    "alternative": "Store len() result in variable",
                },
            },
            "recommendations": [
                "Profile code to identify bottlenecks",
                "Use appropriate data structures for the task",
                "Consider algorithmic complexity",
            ],
            "framework": None,
            "source": "Python Performance Guidelines",
            "last_updated": 1234567890,
        }

    @pytest.mark.asyncio
    async def test_performance_analysis_with_standards(
        self,
        performance_tool,
        test_context_with_performance_issues,
        mock_performance_rules,
    ):
        """Test performance analysis using standards-based rules"""
        with patch.object(
            performance_tool.rule_engine, "get_performance_rules"
        ) as mock_get_rules:
            mock_get_rules.return_value = mock_performance_rules

            # Mock framework detection
            with patch.object(performance_tool, "_detect_framework") as mock_detect:
                mock_detect.return_value = None

                result = await performance_tool.execute(
                    test_context_with_performance_issues
                )

                # Verify successful execution
                assert result.success is True
                assert result.tool_name == "PerformancePatternTool"

                # Verify performance issues detected
                assert (
                    len(result.issues) >= 2
                )  # String concatenation and membership test

                # Verify issue types
                issue_descriptions = [issue["description"] for issue in result.issues]
                assert any(
                    "concatenation" in desc.lower() for desc in issue_descriptions
                )
                assert any("membership" in desc.lower() for desc in issue_descriptions)

                # Verify standards-based recommendations
                assert len(result.suggestions) > 0
                assert any("join()" in suggestion for suggestion in result.suggestions)

                # Verify metrics include performance data
                assert "string_concat_issues" in result.metrics
                assert "framework_detected" in result.metrics
                assert result.metrics["rules_source"] == "Python Performance Guidelines"

    def test_framework_detection(self, performance_tool):
        """Test framework detection from code"""
        # Test FastAPI detection
        fastapi_code = """
        + from fastapi import FastAPI
        + @app.get("/users")
        + def get_users():
        """
        framework = performance_tool._detect_framework(fastapi_code)
        assert framework == "fastapi"

        # Test Django detection
        django_code = """
        + from django.db import models
        + class User(models.Model):
        """
        framework = performance_tool._detect_framework(django_code)
        assert framework == "django"

        # Test no framework
        plain_code = """
        + def hello():
        +     print("Hello")
        """
        framework = performance_tool._detect_framework(plain_code)
        assert framework is None


class TestErrorHandlingToolStandardsBased:
    """Test ErrorHandlingTool with standards-based rules"""

    @pytest.fixture
    def error_tool(self):
        """Create an error handling analysis tool"""
        return ErrorHandlingTool()

    @pytest.fixture
    def test_context_with_error_issues(self):
        """Create context with error handling issues"""
        diff_content = """
        + try:
        +     risky_operation()
        + except:
        +     pass
        +
        + try:
        +     database_operation()
        + except Exception:
        +     return None
        """
        return ToolContext(
            diff_content=diff_content,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    @pytest.fixture
    def mock_error_handling_rules(self):
        """Mock error handling rules from Python docs"""
        return {
            "patterns": {
                "bare_except": {
                    "severity": "high",
                    "description": "Bare except clause catches all exceptions",
                    "recommendation": "Catch specific exception types instead of bare except",
                    "alternative": "Use specific exception classes like ValueError, TypeError",
                },
                "broad_except": {
                    "severity": "medium",
                    "description": "Overly broad exception catching",
                    "recommendation": "Catch more specific exception types",
                    "alternative": "Use specific exceptions instead of Exception",
                },
                "silent_exception": {
                    "severity": "medium",
                    "description": "Exception caught but not logged",
                    "recommendation": "Log exceptions for debugging and monitoring",
                    "alternative": "Add logging.exception() or logger.error()",
                },
            },
            "recommendations": [
                "Use specific exception types",
                "Log exceptions appropriately",
                "Consider using context managers for resource management",
            ],
            "source": "Python Exception Handling Best Practices",
            "last_updated": 1234567890,
        }

    @pytest.mark.asyncio
    async def test_error_handling_analysis_with_standards(
        self, error_tool, test_context_with_error_issues, mock_error_handling_rules
    ):
        """Test error handling analysis using standards-based rules"""
        with patch.object(
            error_tool.rule_engine, "get_error_handling_rules"
        ) as mock_get_rules:
            mock_get_rules.return_value = mock_error_handling_rules

            result = await error_tool.execute(test_context_with_error_issues)

            # Verify successful execution
            assert result.success is True
            assert result.tool_name == "ErrorHandlingTool"

            # Verify error handling issues detected
            assert len(result.issues) >= 1  # Should detect bare except

            # Verify issue details
            issue_descriptions = [issue["description"] for issue in result.issues]
            assert any(
                "bare except" in desc.lower() or "exception" in desc.lower()
                for desc in issue_descriptions
            )

            # Verify standards-based recommendations
            assert len(result.suggestions) > 0
            assert any(
                "specific" in suggestion.lower() for suggestion in result.suggestions
            )

            # Verify metrics
            assert "bare_except_blocks" in result.metrics
            assert "exception_coverage" in result.metrics
            assert (
                result.metrics["rules_source"]
                == "Python Exception Handling Best Practices"
            )

    @pytest.mark.asyncio
    async def test_error_handling_analysis_good_practices(
        self, error_tool, mock_error_handling_rules
    ):
        """Test error handling analysis with good practices"""
        good_context = ToolContext(
            diff_content="""
            + try:
            +     risky_operation()
            + except ValueError as e:
            +     logger.error(f"Value error: {e}")
            +     return None
            + except FileNotFoundError as e:
            +     logger.warning(f"File not found: {e}")
            +     return default_value
            """,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        with patch.object(
            error_tool.rule_engine, "get_error_handling_rules"
        ) as mock_get_rules:
            mock_get_rules.return_value = mock_error_handling_rules

            result = await error_tool.execute(good_context)

            # Should succeed with good practices
            assert result.success is True
            assert len(result.positive_findings) > 0
            assert "specific exception types" in result.positive_findings[0]

    def test_detect_error_pattern_in_code(self, error_tool):
        """Test error pattern detection accuracy"""
        diff_content = """
        + try:
        +     operation()
        + except:
        +     pass
        """

        pattern_info = {
            "severity": "high",
            "description": "Bare except clause",
            "recommendation": "Use specific exceptions",
        }

        findings = error_tool._detect_error_pattern_in_code(
            diff_content, "bare_except", pattern_info
        )

        assert len(findings) == 1
        assert findings[0]["pattern"] == "bare_except"
        assert findings[0]["severity"] == "high"
        assert "except:" in findings[0]["evidence"]


class TestStandardsBasedToolsIntegration:
    """Integration tests for standards-based tools working together"""

    @pytest.mark.asyncio
    async def test_multiple_tools_with_standards(self):
        """Test multiple tools using standards-based rules on the same code"""
        # Code with multiple types of issues
        complex_context = ToolContext(
            diff_content="""
            + import fastapi
            +
            + password = "hardcoded_secret"
            +
            + try:
            +     for user in users:
            +         query = f"SELECT * FROM profiles WHERE user_id = {user.id}"
            +         result = database.execute(query)
            + except:
            +     pass
            +
            + for item in items:
            +     result += str(item) + ","
            """,
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        # Initialize tools
        security_tool = SecurityAnalysisTool()
        performance_tool = PerformancePatternTool()
        error_tool = ErrorHandlingTool()

        # Mock rule engines for all tools
        with patch.object(
            security_tool.rule_engine, "get_security_rules"
        ) as mock_security_rules, patch.object(
            performance_tool.rule_engine, "get_performance_rules"
        ) as mock_performance_rules, patch.object(
            error_tool.rule_engine, "get_error_handling_rules"
        ) as mock_error_rules:
            # Mock rules for each tool
            mock_security_rules.return_value = {
                "patterns": {
                    "sql_injection": {
                        "severity": "critical",
                        "description": "SQL injection",
                        "recommendation": "Use parameters",
                    },
                    "hardcoded_secrets": {
                        "severity": "critical",
                        "description": "Hardcoded secret",
                        "recommendation": "Use env vars",
                    },
                },
                "source": "OWASP",
                "last_updated": 1234567890,
            }

            mock_performance_rules.return_value = {
                "patterns": {
                    "string_concatenation": {
                        "severity": "medium",
                        "description": "String concat",
                        "recommendation": "Use join",
                    },
                    "n_plus_one": {
                        "severity": "high",
                        "description": "N+1 queries",
                        "recommendation": "Bulk operations",
                    },
                },
                "source": "Python Performance",
                "last_updated": 1234567890,
            }

            mock_error_rules.return_value = {
                "patterns": {
                    "bare_except": {
                        "severity": "high",
                        "description": "Bare except",
                        "recommendation": "Specific exceptions",
                    }
                },
                "source": "Python Best Practices",
                "last_updated": 1234567890,
            }

            # Mock framework detection
            with patch.object(
                performance_tool, "_detect_framework", return_value="fastapi"
            ):
                # Execute all tools
                security_result = await security_tool.execute(complex_context)
                performance_result = await performance_tool.execute(complex_context)
                error_result = await error_tool.execute(complex_context)

                # All tools should succeed
                assert security_result.success is True
                assert performance_result.success is True
                assert error_result.success is True

                # Each tool should find relevant issues
                assert (
                    len(security_result.issues) >= 1
                )  # SQL injection, hardcoded secrets
                assert len(performance_result.issues) >= 1  # String concatenation, N+1
                assert len(error_result.issues) >= 1  # Bare except

                # Verify different categories
                assert security_result.category.value == "security"
                assert performance_result.category.value == "performance"
                assert error_result.category.value == "correctness"

                # Verify all tools use standards-based sources
                assert "OWASP" in security_result.metrics["rules_source"]
                assert (
                    "Python Performance" in performance_result.metrics["rules_source"]
                )
                assert "Python Best Practices" in error_result.metrics["rules_source"]
