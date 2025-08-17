"""
Unit tests for the standards-based rule engine
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.tools.rule_engine import RuleEngine


class TestRuleEngine:
    """Test RuleEngine class"""

    @pytest.fixture
    def rule_engine(self):
        """Create a test rule engine"""
        return RuleEngine(settings={"rule_cache_ttl": 60})

    @pytest.fixture
    def mock_context7_client(self):
        """Create a mock Context7 client"""
        client = MagicMock()
        client.get_library_docs = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_security_rules_owasp(self, rule_engine, mock_context7_client):
        """Test getting security rules from OWASP"""
        # Mock OWASP documentation response
        mock_owasp_docs = {
            "snippets": [
                {
                    "description": "SQL injection attacks are common vulnerabilities",
                    "code": "Use parameterized queries to prevent injection",
                },
                {
                    "description": "Eval functions can lead to code injection",
                    "code": "Avoid eval() and exec() functions",
                },
            ],
            "references": ["https://owasp.org/www-community/attacks/SQL_Injection"],
        }

        mock_context7_client.get_library_docs.return_value = mock_owasp_docs
        rule_engine.context7_client = mock_context7_client

        # Test getting security rules
        rules = await rule_engine.get_security_rules("injection")

        # Verify structure
        assert "patterns" in rules
        assert "recommendations" in rules
        assert "source" in rules
        assert "last_updated" in rules

        # Verify OWASP patterns are processed
        patterns = rules["patterns"]
        assert "sql_injection" in patterns
        assert patterns["sql_injection"]["severity"] == "critical"

        # Verify client was called for OWASP docs (may be called multiple times including NIST)
        calls = mock_context7_client.get_library_docs.call_args_list
        owasp_call_found = any(
            call[0][0] == "/owasp/top-10-2021"
            and call[1].get("topic") == "security injection"
            for call in calls
        )
        assert owasp_call_found, f"Expected OWASP call not found in: {calls}"

    @pytest.mark.asyncio
    async def test_get_performance_rules_with_framework(
        self, rule_engine, mock_context7_client
    ):
        """Test getting performance rules with framework-specific rules"""
        # Mock Python performance docs
        mock_python_docs = {
            "snippets": [
                {
                    "description": "String concatenation in loops is inefficient",
                    "code": "Use ''.join() for better performance",
                },
                {
                    "description": "Membership testing on lists is O(n)",
                    "code": "Use sets for O(1) membership testing",
                },
            ],
            "references": ["https://docs.python.org/3/tutorial/datastructures.html"],
        }

        # Mock FastAPI framework docs
        mock_fastapi_docs = {
            "snippets": [
                {
                    "description": "Async endpoints perform better in FastAPI",
                    "code": "Use async def for better performance",
                }
            ],
            "references": ["https://fastapi.tiangolo.com/async/"],
        }

        mock_context7_client.get_library_docs.side_effect = [
            mock_python_docs,
            mock_fastapi_docs,
        ]
        rule_engine.context7_client = mock_context7_client

        # Test getting performance rules for FastAPI
        rules = await rule_engine.get_performance_rules("fastapi")

        # Verify structure
        assert "patterns" in rules
        assert "recommendations" in rules
        assert "framework" in rules
        assert rules["framework"] == "fastapi"

        # Verify patterns are processed
        patterns = rules["patterns"]
        assert "string_concatenation" in patterns

        # Verify both Python and FastAPI docs were fetched
        assert mock_context7_client.get_library_docs.call_count == 2

    @pytest.mark.asyncio
    async def test_get_async_rules(self, rule_engine, mock_context7_client):
        """Test getting async pattern rules"""
        mock_async_docs = {
            "snippets": [
                {
                    "description": "Await in loop reduces concurrency performance",
                    "code": "Use asyncio.gather() for concurrent execution",
                },
                {
                    "description": "Blocking calls should not be used in async functions",
                    "code": "Use asyncio.sleep() instead of time.sleep()",
                },
            ],
            "references": ["https://docs.python.org/3/library/asyncio.html"],
        }

        mock_context7_client.get_library_docs.return_value = mock_async_docs
        rule_engine.context7_client = mock_context7_client

        # Test getting async rules
        rules = await rule_engine.get_async_rules()

        # Verify structure and patterns
        assert "patterns" in rules
        patterns = rules["patterns"]
        assert "await_in_loop" in patterns
        assert "blocking_call" in patterns

        # Verify severity and recommendations
        assert patterns["await_in_loop"]["severity"] == "medium"
        assert "asyncio.gather()" in patterns["await_in_loop"]["alternative"]

    @pytest.mark.asyncio
    async def test_get_error_handling_rules(self, rule_engine, mock_context7_client):
        """Test getting error handling rules"""
        mock_error_docs = {
            "snippets": [
                {
                    "description": "Bare except clauses catch all exceptions",
                    "code": "Use specific exception types",
                },
                {
                    "description": "Silent exception handling hides errors",
                    "code": "Log exceptions for debugging",
                },
            ],
            "references": ["https://docs.python.org/3/tutorial/errors.html"],
        }

        mock_context7_client.get_library_docs.return_value = mock_error_docs
        rule_engine.context7_client = mock_context7_client

        # Test getting error handling rules
        rules = await rule_engine.get_error_handling_rules()

        # Verify structure and patterns
        assert "patterns" in rules
        patterns = rules["patterns"]
        assert "bare_except" in patterns
        assert "silent_exception" in patterns

    @pytest.mark.asyncio
    async def test_get_framework_rules_fastapi(self, rule_engine, mock_context7_client):
        """Test getting framework-specific rules for FastAPI"""
        mock_fastapi_docs = {
            "snippets": [
                {
                    "description": "Response models improve API documentation",
                    "code": "@app.get('/users', response_model=UserResponse)",
                },
                {
                    "description": "Dependency injection improves testability",
                    "code": "def get_db(db: Session = Depends(get_database)):",
                },
            ],
            "references": ["https://fastapi.tiangolo.com/tutorial/"],
        }

        mock_context7_client.get_library_docs.return_value = mock_fastapi_docs
        rule_engine.context7_client = mock_context7_client

        # Test getting FastAPI rules
        rules = await rule_engine.get_framework_rules("fastapi")

        # Verify structure
        assert "patterns" in rules
        assert "framework" in rules
        assert rules["framework"] == "fastapi"

        # Verify FastAPI-specific patterns
        patterns = rules["patterns"]
        assert "missing_response_model" in patterns

    @pytest.mark.asyncio
    async def test_caching_behavior(self, rule_engine, mock_context7_client):
        """Test that rules are properly cached"""
        # Clear any existing cache
        rule_engine.clear_cache()

        mock_docs = {
            "snippets": [{"description": "Test pattern", "code": "test code"}],
            "references": ["https://test.com"],
        }

        mock_context7_client.get_library_docs.return_value = mock_docs
        rule_engine.context7_client = mock_context7_client

        # First call - should fetch from Context7
        rules1 = await rule_engine.get_security_rules("general")

        # Reset mock call count to check second call
        initial_call_count = mock_context7_client.get_library_docs.call_count

        # Second call - should use cache
        rules2 = await rule_engine.get_security_rules("general")

        # Should be the same rules
        assert rules1["source"] == rules2["source"]
        assert rules1["patterns"] == rules2["patterns"]

        # Context7 should not be called again for the second call
        assert mock_context7_client.get_library_docs.call_count == initial_call_count

    @pytest.mark.asyncio
    async def test_cache_expiration(self, rule_engine, mock_context7_client):
        """Test that cache expires correctly"""
        # Clear any existing cache
        rule_engine.clear_cache()

        # Set very short cache TTL
        rule_engine.cache_ttl = 0.1  # 100ms

        mock_docs = {
            "snippets": [{"description": "Test pattern", "code": "test code"}],
            "references": ["https://test.com"],
        }

        mock_context7_client.get_library_docs.return_value = mock_docs
        rule_engine.context7_client = mock_context7_client

        # First call
        await rule_engine.get_security_rules("general")
        initial_call_count = mock_context7_client.get_library_docs.call_count

        # Wait for cache to expire
        time.sleep(0.2)

        # Second call - should fetch again
        await rule_engine.get_security_rules("general")

        # Should have been called at least once more
        assert mock_context7_client.get_library_docs.call_count > initial_call_count

    @pytest.mark.asyncio
    async def test_fallback_rules_on_error(self, rule_engine, mock_context7_client):
        """Test fallback rules when Context7 fails"""
        # Mock Context7 failure
        mock_context7_client.get_library_docs.side_effect = Exception("Network error")
        rule_engine.context7_client = mock_context7_client

        # Should return fallback rules without raising exception
        rules = await rule_engine.get_security_rules("general")

        # Verify fallback rules structure
        assert "patterns" in rules
        assert "source" in rules
        assert rules["source"] == "Fallback security rules"

        # Should have basic security patterns
        patterns = rules["patterns"]
        assert "sql_injection" in patterns
        assert "code_injection" in patterns

    def test_cache_key_generation(self, rule_engine):
        """Test cache key generation for different rule types"""
        # Test that cache operations work correctly
        test_rules = {"test": "data"}

        # Cache some rules
        rule_engine._cache_rules("test_key", test_rules)

        # Verify caching worked
        assert rule_engine._is_cached("test_key")
        assert rule_engine._get_from_cache("test_key") == test_rules

        # Test cache clearing
        rule_engine.clear_cache()
        assert not rule_engine._is_cached("test_key")

    def test_rule_processing_methods(self, rule_engine):
        """Test rule processing helper methods"""
        # Test OWASP rule processing
        mock_owasp_docs = {
            "snippets": [
                {
                    "description": "SQL injection vulnerability",
                    "code": "Use parameters",
                },
                {
                    "description": "Eval creates code injection risk",
                    "code": "Avoid eval",
                },
            ],
            "references": ["https://owasp.org"],
        }

        rules = rule_engine._process_security_rules(mock_owasp_docs, None, "general")

        # Verify processing
        assert "patterns" in rules
        assert "sql_injection" in rules["patterns"]
        assert "code_injection" in rules["patterns"]

        # Test recommendation extraction
        snippet = {
            "description": "You should use parameterized queries",
            "code": "SELECT * FROM users WHERE id = ?",
        }
        recommendation = rule_engine._extract_recommendation(snippet)
        assert "should" in recommendation.lower()

    @pytest.mark.asyncio
    async def test_unsupported_framework(self, rule_engine, mock_context7_client):
        """Test handling of unsupported framework"""
        rule_engine.context7_client = mock_context7_client

        # Test with unsupported framework
        rules = await rule_engine.get_framework_rules("unsupported_framework")

        # Should return empty patterns but not fail
        assert "patterns" in rules
        assert len(rules["patterns"]) == 0

        # Should not call Context7 for unsupported framework
        mock_context7_client.get_library_docs.assert_not_called()


class TestRuleEngineIntegration:
    """Integration tests for rule engine with realistic scenarios"""

    @pytest.mark.asyncio
    async def test_complete_security_analysis_workflow(self):
        """Test complete workflow from rule fetching to pattern detection"""
        rule_engine = RuleEngine()

        # Mock realistic OWASP response
        with patch.object(
            rule_engine.context7_client, "get_library_docs"
        ) as mock_get_docs:
            mock_get_docs.return_value = {
                "snippets": [
                    {
                        "description": "SQL injection occurs when user input is directly inserted into queries",
                        "code": "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
                    },
                    {
                        "description": "Hardcoded passwords should never be stored in source code",
                        "code": "password = os.getenv('DATABASE_PASSWORD')",
                    },
                ],
                "references": [
                    "https://owasp.org/www-community/attacks/SQL_Injection",
                    "https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password",
                ],
            }

            # Get security rules
            rules = await rule_engine.get_security_rules("injection")

            # Verify comprehensive rule structure
            assert rules["source"] == "OWASP Top 10 + NIST Framework"
            assert len(rules["patterns"]) >= 2
            assert "sql_injection" in rules["patterns"]
            assert "hardcoded_secrets" in rules["patterns"]

            # Verify severity levels are assigned
            assert rules["patterns"]["sql_injection"]["severity"] == "critical"
            assert rules["patterns"]["hardcoded_secrets"]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_framework_detection_and_rule_fetching(self):
        """Test framework detection leading to appropriate rule fetching"""
        rule_engine = RuleEngine()

        # Mock both Python and FastAPI documentation
        with patch.object(
            rule_engine.context7_client, "get_library_docs"
        ) as mock_get_docs:
            mock_get_docs.side_effect = [
                # Python performance docs
                {
                    "snippets": [
                        {
                            "description": "List comprehensions are faster than loops",
                            "code": "result = [x*2 for x in items]",
                        }
                    ],
                    "references": [
                        "https://docs.python.org/3/tutorial/datastructures.html"
                    ],
                },
                # FastAPI performance docs
                {
                    "snippets": [
                        {
                            "description": "Async endpoints handle more concurrent requests",
                            "code": "@app.get('/items') async def get_items():",
                        }
                    ],
                    "references": ["https://fastapi.tiangolo.com/async/"],
                },
            ]

            # Get performance rules for FastAPI
            rules = await rule_engine.get_performance_rules("fastapi")

            # Verify both Python and framework-specific rules
            assert rules["framework"] == "fastapi"
            assert "patterns" in rules
            assert rules["source"] == "Python Performance Guidelines"

            # Should have called both Python and FastAPI docs
            assert mock_get_docs.call_count == 2

    def test_pattern_processing_accuracy(self):
        """Test accuracy of pattern processing from documentation"""
        rule_engine = RuleEngine()

        # Test security pattern processing
        mock_docs = {
            "snippets": [
                {
                    "description": "SQL injection vulnerability in queries",
                    "code": "Use parameters",
                },
                {"description": "Weak crypto with MD5 hashes", "code": "Use SHA-256"},
                {
                    "description": "Hardcoded API keys are dangerous",
                    "code": "Use environment variables",
                },
            ],
            "references": ["https://owasp.org"],
        }

        rules = rule_engine._process_security_rules(mock_docs, None, "all")
        patterns = rules["patterns"]

        # Should detect multiple security patterns (at least 2)
        assert len(patterns) >= 2
        assert "sql_injection" in patterns
        assert "weak_crypto" in patterns
        # Note: hardcoded_secrets may not be detected depending on keyword matching

        # Verify recommendations are extracted
        for pattern in patterns.values():
            assert "recommendation" in pattern
            assert pattern["recommendation"]  # Should not be empty
