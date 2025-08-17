"""
Standards-based rule engine for dynamic code review rules

This module provides a central system for fetching and caching code review rules
from authoritative sources like OWASP, NIST, Python PEPs, and framework documentation
using Context7 MCP integration.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.agents.tools.python.context_tools import Context7Client

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Language-aware rule engine for fetching standards-based code review rules

    Integrates with Context7 to dynamically load current standards from:
    - OWASP Top 10 security guidelines
    - NIST Cybersecurity Framework
    - Python Enhancement Proposals (PEPs)
    - Framework-specific best practices (FastAPI, Django, Flask)
    - Language-specific patterns and recommendations
    """

    def __init__(
        self, settings: Optional[Dict[str, Any]] = None, language: str = "python"
    ):
        """
        Initialize the rule engine with configuration and target language

        Args:
            settings: Optional configuration settings
            language: Target programming language for rules (default: python)
        """
        self.settings = settings or {}
        self.language = language.lower()
        self.context7_client = Context7Client(settings)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = self.settings.get("rule_cache_ttl", 3600)  # 1 hour default

        # Language-aware rule sources mapping
        self.rule_sources = self._get_language_rule_sources(self.language)

    def _get_language_rule_sources(self, language: str) -> Dict[str, Dict[str, str]]:
        """Get rule sources based on target language"""
        if language == "python":
            return {
                "security": {
                    "owasp": "/owasp/top-10-2021",
                    "nist": "/nist/cybersecurity-framework",
                    "python_security": "/websites/python-3",
                },
                "performance": {
                    "python_performance": "/websites/python-3",
                    "best_practices": "/python/python-patterns",
                },
                "correctness": {
                    "python_peps": "/websites/python-3",
                    "async_patterns": "/websites/python-3",
                },
                "style": {"pep8": "/python/pep8", "black": "/psf/black"},
                "framework": {
                    "fastapi": "/tiangolo/fastapi",
                    "django": "/django/django",
                    "flask": "/pallets/flask",
                    "pytest": "/pytest-dev/pytest",
                    "pydantic": "/pydantic/pydantic",
                },
            }
        elif language == "javascript":
            return {
                "security": {
                    "owasp": "/owasp/top-10-2021",
                    "nist": "/nist/cybersecurity-framework",
                    "js_security": "/nodejs/node",
                },
                "performance": {
                    "js_performance": "/nodejs/node",
                    "best_practices": "/airbnb/javascript",
                },
                "correctness": {
                    "eslint": "/eslint/eslint",
                    "typescript": "/microsoft/typescript",
                },
                "style": {"prettier": "/prettier/prettier", "eslint": "/eslint/eslint"},
                "framework": {
                    "react": "/facebook/react",
                    "nextjs": "/vercel/next.js",
                    "express": "/expressjs/express",
                    "jest": "/facebook/jest",
                },
            }
        elif language == "go":
            return {
                "security": {
                    "owasp": "/owasp/top-10-2021",
                    "nist": "/nist/cybersecurity-framework",
                    "go_security": "/golang/go",
                },
                "performance": {
                    "go_performance": "/golang/go",
                    "best_practices": "/golang/go",
                },
                "correctness": {
                    "effective_go": "/golang/go",
                    "go_patterns": "/golang/go",
                },
                "style": {"gofmt": "/golang/go", "golint": "/golang/go"},
                "framework": {
                    "gin": "/gin-gonic/gin",
                    "gorilla": "/gorilla/mux",
                    "testify": "/stretchr/testify",
                },
            }
        else:
            # Default to universal security rules only
            return {
                "security": {
                    "owasp": "/owasp/top-10-2021",
                    "nist": "/nist/cybersecurity-framework",
                },
                "performance": {},
                "correctness": {},
                "style": {},
                "framework": {},
            }

    async def get_security_rules(self, rule_type: str = "general") -> Dict[str, Any]:
        """
        Get current security rules from OWASP and NIST standards for the target language

        Args:
            rule_type: Type of security rules ('injection', 'authentication', 'crypto', etc.)

        Returns:
            Dictionary containing security patterns, severity levels, and recommendations
        """
        cache_key = f"security:{self.language}:{rule_type}"

        # Check cache first
        if self._is_cached(cache_key):
            return self._get_from_cache(cache_key)

        try:
            # Fetch OWASP Top 10 guidelines
            owasp_docs = await self.context7_client.get_library_docs(
                self.rule_sources["security"]["owasp"],
                topic=f"security {rule_type}",
                max_tokens=2000,
            )

            # Fetch NIST framework if available
            nist_docs = None
            try:
                nist_docs = await self.context7_client.get_library_docs(
                    self.rule_sources["security"]["nist"],
                    topic=f"cybersecurity {rule_type}",
                    max_tokens=1000,
                )
            except Exception as e:
                logger.warning(f"NIST docs unavailable: {e}")

            # Process and structure the rules
            rules = self._process_security_rules(owasp_docs, nist_docs, rule_type)

            # Cache the results
            self._cache_rules(cache_key, rules)

            return rules

        except Exception as e:
            logger.error(f"Failed to fetch security rules for {rule_type}: {e}")
            return self._get_fallback_security_rules(rule_type)

    async def get_performance_rules(
        self, framework: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance optimization rules from language documentation and framework guides

        Args:
            framework: Optional framework name for framework-specific rules

        Returns:
            Dictionary containing performance patterns and optimization guidelines
        """
        cache_key = f"performance:{self.language}:{framework or 'general'}"

        if self._is_cached(cache_key):
            return self._get_from_cache(cache_key)

        try:
            # Fetch Python performance guidelines
            python_docs = await self.context7_client.get_library_docs(
                self.rule_sources["performance"]["python_performance"],
                topic="performance optimization patterns",
                max_tokens=2000,
            )

            # Fetch framework-specific rules if specified
            framework_docs = None
            if framework and framework in self.rule_sources["framework"]:
                framework_docs = await self.context7_client.get_library_docs(
                    self.rule_sources["framework"][framework],
                    topic="performance best practices",
                    max_tokens=1500,
                )

            rules = self._process_performance_rules(
                python_docs, framework_docs, framework
            )
            self._cache_rules(cache_key, rules)

            return rules

        except Exception as e:
            logger.error(f"Failed to fetch performance rules: {e}")
            return self._get_fallback_performance_rules()

    async def get_async_rules(self) -> Dict[str, Any]:
        """
        Get async/await pattern rules from Python documentation

        Returns:
            Dictionary containing async patterns, anti-patterns, and best practices
        """
        cache_key = f"async:{self.language}:patterns"

        if self._is_cached(cache_key):
            return self._get_from_cache(cache_key)

        try:
            docs = await self.context7_client.get_library_docs(
                self.rule_sources["correctness"]["async_patterns"],
                topic="asyncio async await patterns concurrency",
                max_tokens=2000,
            )

            rules = self._process_async_rules(docs)
            self._cache_rules(cache_key, rules)

            return rules

        except Exception as e:
            logger.error(f"Failed to fetch async rules: {e}")
            return self._get_fallback_async_rules()

    async def get_error_handling_rules(self) -> Dict[str, Any]:
        """
        Get error handling best practices from Python PEPs and documentation

        Returns:
            Dictionary containing exception handling patterns and guidelines
        """
        cache_key = f"error_handling:{self.language}:patterns"

        if self._is_cached(cache_key):
            return self._get_from_cache(cache_key)

        try:
            docs = await self.context7_client.get_library_docs(
                self.rule_sources["correctness"]["python_peps"],
                topic="exception handling error patterns try except",
                max_tokens=1500,
            )

            rules = self._process_error_handling_rules(docs)
            self._cache_rules(cache_key, rules)

            return rules

        except Exception as e:
            logger.error(f"Failed to fetch error handling rules: {e}")
            return self._get_fallback_error_handling_rules()

    async def get_framework_rules(self, framework: str) -> Dict[str, Any]:
        """
        Get framework-specific best practices and patterns

        Args:
            framework: Framework name (fastapi, django, flask, etc.)

        Returns:
            Dictionary containing framework-specific rules and patterns
        """
        cache_key = f"framework:{self.language}:{framework}"

        if self._is_cached(cache_key):
            return self._get_from_cache(cache_key)

        if framework not in self.rule_sources["framework"]:
            logger.warning(f"Framework {framework} not supported")
            return {"patterns": {}, "recommendations": []}

        try:
            docs = await self.context7_client.get_library_docs(
                self.rule_sources["framework"][framework],
                topic=f"{framework} best practices patterns",
                max_tokens=2000,
            )

            rules = self._process_framework_rules(docs, framework)
            self._cache_rules(cache_key, rules)

            return rules

        except Exception as e:
            logger.error(f"Failed to fetch {framework} rules: {e}")
            return self._get_fallback_framework_rules(framework)

    def _process_security_rules(
        self, owasp_docs: Optional[Dict], nist_docs: Optional[Dict], rule_type: str
    ) -> Dict[str, Any]:
        """Process OWASP and NIST documentation into structured security rules"""
        patterns = {}
        recommendations: List[str] = []

        if owasp_docs and "snippets" in owasp_docs:
            for snippet in owasp_docs["snippets"]:
                description = snippet.get("description", "").lower()

                # Extract security patterns based on OWASP guidelines
                if "injection" in description or "sql" in description:
                    patterns["sql_injection"] = {
                        "severity": "critical",
                        "description": "SQL injection vulnerability detected",
                        "recommendation": self._extract_recommendation(snippet),
                        "references": owasp_docs.get("references", []),
                    }

                if "eval" in description or "code injection" in description:
                    patterns["code_injection"] = {
                        "severity": "critical",
                        "description": "Code injection vulnerability via eval/exec",
                        "recommendation": self._extract_recommendation(snippet),
                        "references": owasp_docs.get("references", []),
                    }

                if "password" in description or "secret" in description:
                    patterns["hardcoded_secrets"] = {
                        "severity": "critical",
                        "description": "Hardcoded secrets or credentials detected",
                        "recommendation": self._extract_recommendation(snippet),
                        "references": owasp_docs.get("references", []),
                    }

                if "crypto" in description or "hash" in description:
                    patterns["weak_crypto"] = {
                        "severity": "high",
                        "description": "Weak cryptographic algorithms detected",
                        "recommendation": self._extract_recommendation(snippet),
                        "references": owasp_docs.get("references", []),
                    }

        # Add NIST recommendations if available
        if nist_docs and "snippets" in nist_docs:
            for snippet in nist_docs["snippets"]:
                recommendations.append(self._extract_recommendation(snippet))

        return {
            "patterns": patterns,
            "recommendations": recommendations,
            "source": "OWASP Top 10 + NIST Framework",
            "last_updated": time.time(),
        }

    def _process_performance_rules(
        self,
        python_docs: Optional[Dict],
        framework_docs: Optional[Dict],
        framework: Optional[str],
    ) -> Dict[str, Any]:
        """Process Python and framework documentation into performance rules"""
        patterns = {}
        recommendations: List[str] = []

        if python_docs and "snippets" in python_docs:
            for snippet in python_docs["snippets"]:
                description = snippet.get("description", "").lower()
                code = snippet.get("code", "").lower()

                # Extract performance patterns from documentation
                if "loop" in description or "iteration" in description:
                    if "join" in code or "comprehension" in description:
                        patterns["string_concatenation"] = {
                            "severity": "medium",
                            "description": "Inefficient string concatenation in loops",
                            "recommendation": self._extract_recommendation(snippet),
                            "alternative": "Use ''.join() or list comprehension",
                        }

                if "membership" in description or "in operator" in description:
                    patterns["membership_test"] = {
                        "severity": "medium",
                        "description": "Inefficient membership testing",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Use set for O(1) membership testing",
                    }

                if "memory" in description or "allocation" in description:
                    patterns["memory_usage"] = {
                        "severity": "low",
                        "description": "Potential memory inefficiency",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Consider generator expressions or memory-efficient alternatives",
                    }

        # Add framework-specific performance patterns
        if framework_docs and "snippets" in framework_docs:
            for snippet in framework_docs["snippets"]:
                description = snippet.get("description", "").lower()

                if framework == "fastapi" and "async" in description:
                    patterns["sync_in_async"] = {
                        "severity": "high",
                        "description": "Synchronous operations in async FastAPI endpoints",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Use async variants of operations",
                    }

                if framework and "query" in description:
                    patterns["n_plus_one"] = {
                        "severity": "high",
                        "description": "Potential N+1 query pattern",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Use bulk operations or select_related/prefetch_related",
                    }

        return {
            "patterns": patterns,
            "recommendations": recommendations,
            "framework": framework,
            "source": "Python Performance Guidelines",
            "last_updated": time.time(),
        }

    def _process_async_rules(self, docs: Optional[Dict]) -> Dict[str, Any]:
        """Process Python async documentation into async pattern rules"""
        patterns = {}
        recommendations: List[str] = []

        if docs and "snippets" in docs:
            for snippet in docs["snippets"]:
                description = snippet.get("description", "").lower()
                snippet.get("code", "").lower()

                if "await" in description and "loop" in description:
                    patterns["await_in_loop"] = {
                        "severity": "medium",
                        "description": "Sequential await calls in loop reduce concurrency",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Use asyncio.gather() or asyncio.as_completed()",
                    }

                if "sleep" in description or "blocking" in description:
                    patterns["blocking_call"] = {
                        "severity": "high",
                        "description": "Blocking call in async function",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Use async equivalents (asyncio.sleep, aiofiles, etc.)",
                    }

                if "context manager" in description or "async with" in description:
                    patterns["async_context"] = {
                        "severity": "medium",
                        "description": "Missing async context manager usage",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Use 'async with' for async context managers",
                    }

        return {
            "patterns": patterns,
            "recommendations": recommendations,
            "source": "Python asyncio Documentation",
            "last_updated": time.time(),
        }

    def _process_error_handling_rules(self, docs: Optional[Dict]) -> Dict[str, Any]:
        """Process Python documentation into error handling rules"""
        patterns = {}
        recommendations: List[str] = []

        if docs and "snippets" in docs:
            for snippet in docs["snippets"]:
                description = snippet.get("description", "").lower()
                snippet.get("code", "").lower()

                if "except" in description and (
                    "bare" in description or "catch all" in description
                ):
                    patterns["bare_except"] = {
                        "severity": "high",
                        "description": "Bare except clause catches all exceptions",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Catch specific exception types",
                    }

                if "exception" in description and (
                    "broad" in description or "generic" in description
                ):
                    patterns["broad_except"] = {
                        "severity": "medium",
                        "description": "Overly broad exception catching",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Catch specific exception types",
                    }

                if "logging" in description or "silent" in description:
                    patterns["silent_exception"] = {
                        "severity": "medium",
                        "description": "Exception caught but not logged",
                        "recommendation": self._extract_recommendation(snippet),
                        "alternative": "Log exceptions for debugging and monitoring",
                    }

        return {
            "patterns": patterns,
            "recommendations": recommendations,
            "source": "Python Exception Handling Best Practices",
            "last_updated": time.time(),
        }

    def _process_framework_rules(
        self, docs: Optional[Dict], framework: str
    ) -> Dict[str, Any]:
        """Process framework documentation into framework-specific rules"""
        patterns = {}
        recommendations: List[str] = []

        if docs and "snippets" in docs:
            for snippet in docs["snippets"]:
                description = snippet.get("description", "").lower()
                snippet.get("code", "").lower()

                if framework == "fastapi":
                    if (
                        "response_model" in description
                        or "response model" in description
                    ):
                        patterns["missing_response_model"] = {
                            "severity": "medium",
                            "description": "FastAPI endpoint missing response_model",
                            "recommendation": self._extract_recommendation(snippet),
                            "alternative": "Add response_model for better API documentation",
                        }

                    if "dependency" in description and "injection" in description:
                        patterns["dependency_injection"] = {
                            "severity": "low",
                            "description": "Consider using FastAPI dependency injection",
                            "recommendation": self._extract_recommendation(snippet),
                            "alternative": "Use Depends() for better testability",
                        }

                elif framework == "pytest":
                    if "assert" in description or "assertion" in description:
                        patterns["missing_assertion"] = {
                            "severity": "medium",
                            "description": "Test function appears to lack assertions",
                            "recommendation": self._extract_recommendation(snippet),
                            "alternative": "Add assertions to validate test expectations",
                        }

                elif framework == "pydantic":
                    if "field" in description and "description" in description:
                        patterns["missing_field_description"] = {
                            "severity": "low",
                            "description": "Pydantic field lacks description",
                            "recommendation": self._extract_recommendation(snippet),
                            "alternative": "Add description to Field() for better API docs",
                        }

        return {
            "patterns": patterns,
            "recommendations": recommendations,
            "framework": framework,
            "source": f"{framework.title()} Best Practices",
            "last_updated": time.time(),
        }

    def _extract_recommendation(self, snippet: Dict[str, Any]) -> str:
        """Extract actionable recommendation from documentation snippet"""
        description: str = snippet.get("description", "")
        code: str = snippet.get("code", "")

        # Simple heuristic to extract recommendations
        if "should" in description.lower():
            return description
        elif "use" in description.lower():
            return description
        elif code:
            return f"Follow this pattern: {code[:100]}..."
        else:
            return "Follow documented best practices"

    def _is_cached(self, cache_key: str) -> bool:
        """Check if rule set is cached and not expired"""
        if cache_key not in self._cache:
            return False

        timestamp: float = self._cache_timestamps.get(cache_key, 0.0)
        return bool((time.time() - timestamp) < self.cache_ttl)

    def _get_from_cache(self, cache_key: str) -> Dict[str, Any]:
        """Get rules from cache"""
        return self._cache.get(cache_key, {})

    def _cache_rules(self, cache_key: str, rules: Dict[str, Any]) -> None:
        """Cache rules with timestamp"""
        self._cache[cache_key] = rules
        self._cache_timestamps[cache_key] = time.time()

    def clear_cache(self) -> None:
        """Clear all cached rules"""
        self._cache.clear()
        self._cache_timestamps.clear()

    # Fallback methods for when Context7 is unavailable
    def _get_fallback_security_rules(self, rule_type: str) -> Dict[str, Any]:
        """Fallback security rules when external sources are unavailable"""
        return {
            "patterns": {
                "sql_injection": {
                    "severity": "critical",
                    "description": "Potential SQL injection vulnerability",
                    "recommendation": "Use parameterized queries",
                    "references": [
                        "https://owasp.org/www-community/attacks/SQL_Injection"
                    ],
                },
                "code_injection": {
                    "severity": "critical",
                    "description": "Code injection via eval/exec",
                    "recommendation": "Avoid eval/exec; use ast.literal_eval for safe evaluation",
                    "references": ["https://owasp.org/Top10/A03_2021-Injection/"],
                },
                "hardcoded_secrets": {
                    "severity": "critical",
                    "description": "Hardcoded secrets detected",
                    "recommendation": "Use environment variables or secret management",
                    "references": [
                        "https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password"
                    ],
                },
            },
            "recommendations": [
                "Follow OWASP Top 10 guidelines",
                "Implement secure coding practices",
            ],
            "source": "Fallback security rules",
            "last_updated": time.time(),
        }

    def _get_fallback_performance_rules(self) -> Dict[str, Any]:
        """Fallback performance rules"""
        return {
            "patterns": {
                "string_concatenation": {
                    "severity": "medium",
                    "description": "String concatenation in loop",
                    "recommendation": "Use ''.join() for better performance",
                    "alternative": "Use list comprehension or generator",
                },
                "membership_test": {
                    "severity": "medium",
                    "description": "Inefficient membership testing on list",
                    "recommendation": "Use set for O(1) membership testing",
                    "alternative": "Convert list to set for frequent membership tests",
                },
            },
            "recommendations": [
                "Profile code for bottlenecks",
                "Use appropriate data structures",
            ],
            "source": "Fallback performance rules",
            "last_updated": time.time(),
        }

    def _get_fallback_async_rules(self) -> Dict[str, Any]:
        """Fallback async rules"""
        return {
            "patterns": {
                "await_in_loop": {
                    "severity": "medium",
                    "description": "Sequential await in loop",
                    "recommendation": "Use asyncio.gather() for concurrent execution",
                    "alternative": "Consider asyncio.as_completed() for processing results as they arrive",
                },
                "blocking_call": {
                    "severity": "high",
                    "description": "Blocking call in async function",
                    "recommendation": "Use async equivalents",
                    "alternative": "Use asyncio.sleep() instead of time.sleep()",
                },
            },
            "recommendations": [
                "Use async/await properly",
                "Avoid blocking calls in async functions",
            ],
            "source": "Fallback async rules",
            "last_updated": time.time(),
        }

    def _get_fallback_error_handling_rules(self) -> Dict[str, Any]:
        """Fallback error handling rules"""
        return {
            "patterns": {
                "bare_except": {
                    "severity": "high",
                    "description": "Bare except clause",
                    "recommendation": "Catch specific exception types",
                    "alternative": "Use specific exception classes",
                },
                "silent_exception": {
                    "severity": "medium",
                    "description": "Exception not logged",
                    "recommendation": "Log exceptions for debugging",
                    "alternative": "Add logging.exception() or logger.error()",
                },
            },
            "recommendations": [
                "Use specific exception types",
                "Log exceptions appropriately",
            ],
            "source": "Fallback error handling rules",
            "last_updated": time.time(),
        }

    def _get_fallback_framework_rules(self, framework: str) -> Dict[str, Any]:
        """Fallback framework rules"""
        return {
            "patterns": {},
            "recommendations": [f"Follow {framework} best practices"],
            "framework": framework,
            "source": f"Fallback {framework} rules",
            "last_updated": time.time(),
        }
