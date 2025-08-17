"""
Enhanced validation tools for comprehensive code review
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.agents.tools.base import (
    BaseTool,
    ToolCategory,
    ToolContext,
    ToolPriority,
    ToolResult,
)
from src.agents.tools.registry import register_tool
from src.agents.tools.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


@register_tool(enabled=True, name="PythonPerformancePatternTool")
class PythonPerformancePatternTool(BaseTool):
    """Detect Python performance anti-patterns using standards-based rules"""

    def __init__(self, name: str = "PythonPerformancePatternTool"):
        super().__init__(name)
        self.rule_engine = RuleEngine(language="python")

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.PERFORMANCE

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.HIGH

    @property
    def requires_network(self) -> bool:
        return True  # Needs network access for Context7 rule fetching

    async def execute(self, context: ToolContext) -> ToolResult:
        """Analyze code for performance issues using current standards"""
        try:
            issues = []
            suggestions = []
            metrics = {}
            references: List[str] = []

            # Detect framework being used
            framework = self._detect_framework(context.diff_content)

            # Get current performance rules from Python docs and framework guides
            performance_rules = await self.rule_engine.get_performance_rules(framework)

            # Performance pattern checks
            perf_analysis = await self._analyze_performance_patterns(
                context.diff_content, performance_rules
            )

            for finding in perf_analysis:
                issues.append(
                    {
                        "severity": finding["severity"],
                        "category": "performance",
                        "description": finding["description"],
                        "file_path": finding.get("file_path", "unknown"),
                        "line_number": finding.get("line_number", 0),
                        "evidence": finding.get("evidence", ""),
                    }
                )

                if finding.get("recommendation"):
                    suggestions.append(finding["recommendation"])

                if finding.get("alternative"):
                    suggestions.append(f"Alternative: {finding['alternative']}")

            # Calculate performance metrics
            metrics.update(
                {
                    "performance_issues": len(issues),
                    "n_plus_one_queries": len(
                        [i for i in issues if "N+1" in i["description"]]
                    ),
                    "inefficient_loops": len(
                        [i for i in issues if "loop" in i["description"]]
                    ),
                    "memory_issues": len(
                        [i for i in issues if "memory" in i["description"]]
                    ),
                    "string_concat_issues": len(
                        [i for i in issues if "concatenation" in i["description"]]
                    ),
                    "framework_detected": framework or "none",
                    "rules_source": performance_rules.get("source", "Unknown"),
                    "rules_last_updated": performance_rules.get("last_updated", 0),
                }
            )

            positive_findings = []
            if not issues:
                positive_findings.append(
                    f"No performance issues detected using {performance_rules.get('source', 'current standards')}"
                )

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                positive_findings=positive_findings,
                metrics=metrics,
                references=references,
                confidence_score=0.85,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _detect_framework(self, diff_content: str) -> Optional[str]:
        """Detect which framework is being used in the code using fast string operations"""
        # Convert to lowercase once for case-insensitive matching
        content_lower = diff_content.lower()

        # Fast string containment checks (replace 4 regex patterns)
        framework_indicators = {
            "fastapi": [
                "from fastapi",
                "import fastapi",
                "@app.get",
                "@app.post",
                "@app.put",
                "@app.delete",
            ],
            "django": [
                "from django",
                "import django",
                "(model)",
                "class meta:",
                "django.db",
            ],
            "flask": ["from flask", "import flask", "@app.route", "flask."],
            "sqlalchemy": ["from sqlalchemy", "import sqlalchemy", "sqlalchemy."],
        }

        for framework, indicators in framework_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                return framework

        return None

    async def _analyze_performance_patterns(
        self, diff_content: str, performance_rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze code for performance anti-patterns using standards-based rules"""
        findings = []

        # Get patterns from current performance standards
        patterns = performance_rules.get("patterns", {})

        # Analyze each pattern type from standards
        for pattern_name, pattern_info in patterns.items():
            pattern_findings = self._detect_performance_pattern_in_code(
                diff_content, pattern_name, pattern_info
            )
            findings.extend(pattern_findings)

        # Add database-specific performance checks
        db_findings = self._check_database_patterns(diff_content, performance_rules)
        findings.extend(db_findings)

        return findings

    def _detect_performance_pattern_in_code(
        self, diff_content: str, pattern_name: str, pattern_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect performance patterns using optimized AST + text analysis"""
        from .code_parser import get_parser

        findings = []
        parser = get_parser()

        # Use AST-based performance pattern detection
        ast_patterns = parser.extract_performance_patterns(diff_content)

        # Filter patterns by type
        for pattern in ast_patterns:
            if pattern["type"] == pattern_name:
                findings.append(
                    {
                        "pattern": pattern_name,
                        "severity": pattern_info.get("severity", pattern["severity"]),
                        "description": pattern_info.get(
                            "description", pattern["description"]
                        ),
                        "recommendation": pattern_info.get(
                            "recommendation",
                            pattern.get("suggestion", "Optimize performance"),
                        ),
                        "evidence": pattern["evidence"],
                        "line": pattern.get("line", 0),
                    }
                )

        # Additional text-based detection for complex patterns
        text_patterns = self._detect_text_performance_patterns(
            diff_content, pattern_name, pattern_info
        )
        findings.extend(text_patterns)

        return findings

    def _detect_text_performance_patterns(
        self, diff_content: str, pattern_name: str, pattern_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Text-based performance pattern detection for complex cases"""
        findings = []
        added_lines = [
            line.strip()
            for line in diff_content.split("\n")
            if line.strip().startswith("+") and not line.strip().startswith("+++")
        ]

        for i, line in enumerate(added_lines):
            code_line = line[1:].strip()

            # N+1 query detection (requires context analysis)
            if pattern_name == "n_plus_one":
                if "for " in code_line and i + 1 < len(added_lines):
                    next_line = added_lines[i + 1][1:].strip()
                    if any(
                        db_method in next_line
                        for db_method in [".get(", ".filter(", ".query("]
                    ):
                        findings.append(
                            {
                                "pattern": pattern_name,
                                "severity": pattern_info.get("severity", "high"),
                                "description": pattern_info.get(
                                    "description", "Potential N+1 query pattern"
                                ),
                                "recommendation": pattern_info.get(
                                    "recommendation", "Use bulk operations or joins"
                                ),
                                "evidence": f"{code_line} -> {next_line}",
                                "line": i + 1,
                            }
                        )

            # Inefficient dict access
            elif pattern_name == "inefficient_dict_access":
                if ".keys()" in code_line and (
                    " in " in code_line or "for " in code_line
                ):
                    findings.append(
                        {
                            "pattern": pattern_name,
                            "severity": pattern_info.get("severity", "medium"),
                            "description": pattern_info.get(
                                "description", "Inefficient dictionary key access"
                            ),
                            "recommendation": pattern_info.get(
                                "recommendation", "Use direct key access or .items()"
                            ),
                            "evidence": code_line,
                            "line": i + 1,
                        }
                    )

            # Sync operations in async context
            elif pattern_name == "sync_in_async":
                if any(
                    sync_op in code_line
                    for sync_op in ["time.sleep(", "requests.", "urllib."]
                ):
                    # Check if we're in an async function (simplified check)
                    for j in range(max(0, i - 10), i):
                        if "async def" in added_lines[j]:
                            findings.append(
                                {
                                    "pattern": pattern_name,
                                    "severity": pattern_info.get("severity", "high"),
                                    "description": pattern_info.get(
                                        "description",
                                        "Synchronous operation in async function",
                                    ),
                                    "recommendation": pattern_info.get(
                                        "recommendation", "Use async alternatives"
                                    ),
                                    "evidence": code_line,
                                    "line": i + 1,
                                }
                            )
                            break

        return findings

    def _check_database_patterns(
        self, diff_content: str, performance_rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for database-related performance issues using standards-based rules"""
        findings = []

        # Get database patterns from performance rules
        patterns = performance_rules.get("patterns", {})

        # N+1 query pattern detection using cached patterns
        from .code_parser import get_compiled_patterns

        patterns_cache = get_compiled_patterns()

        if patterns_cache.search(
            r"\+\s*for\s+.*:\s*\n\s*\+\s*.*\.get\(", diff_content, re.MULTILINE
        ):
            n_plus_one_info = patterns.get(
                "n_plus_one",
                {
                    "severity": "high",
                    "description": "Potential N+1 query pattern detected",
                    "recommendation": "Use select_related() or prefetch_related() to optimize queries",
                    "alternative": "Bulk fetch related objects to avoid repeated queries",
                },
            )

            findings.append(
                {
                    "pattern": "n_plus_one_query",
                    "severity": n_plus_one_info.get("severity", "high"),
                    "description": n_plus_one_info.get(
                        "description", "N+1 query pattern detected"
                    ),
                    "recommendation": n_plus_one_info.get(
                        "recommendation", "Optimize database queries"
                    ),
                    "alternative": n_plus_one_info.get("alternative", ""),
                    "evidence": "Loop with individual database queries",
                    "line_number": 0,
                }
            )

        # Missing database indexes
        if patterns_cache.search(r"\+\s*.*\.filter\(.*=.*\)", diff_content):
            index_info = patterns.get(
                "missing_index",
                {
                    "severity": "medium",
                    "description": "Database filter without obvious index",
                    "recommendation": "Ensure database indexes exist for filtered fields",
                    "alternative": "Add database indexes for frequently queried fields",
                },
            )

            findings.append(
                {
                    "pattern": "potential_missing_index",
                    "severity": index_info.get("severity", "medium"),
                    "description": index_info.get(
                        "description", "Missing database index"
                    ),
                    "recommendation": index_info.get(
                        "recommendation", "Add appropriate database indexes"
                    ),
                    "alternative": index_info.get("alternative", ""),
                    "evidence": "Database query with filter",
                    "line_number": 0,
                }
            )

        return findings


@register_tool(enabled=True, name="PythonAsyncPatternValidationTool")
class PythonAsyncPatternValidationTool(BaseTool):
    """Validate Python async/await patterns and concurrency usage"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CORRECTNESS

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.HIGH

    async def execute(self, context: ToolContext) -> ToolResult:
        """Validate async patterns in the code"""
        try:
            issues = []
            suggestions = []
            metrics = {}

            # Analyze async patterns
            async_analysis = self._analyze_async_patterns(context.diff_content)

            for finding in async_analysis:
                issues.append(
                    {
                        "severity": finding["severity"],
                        "category": "correctness",
                        "description": finding["description"],
                        "file_path": finding.get("file_path", "unknown"),
                        "line_number": finding.get("line_number", 0),
                        "evidence": finding.get("evidence", ""),
                    }
                )

                if finding.get("suggestion"):
                    suggestions.append(finding["suggestion"])

            # Calculate async metrics using fast line counting
            async_funcs = 0
            await_calls = 0

            for line in context.diff_content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("+"):
                    code_line = stripped[1:].strip()
                    if "async def" in code_line:
                        async_funcs += 1
                    if "await " in code_line:
                        await_calls += 1

            metrics.update(
                {
                    "async_functions": async_funcs,
                    "await_calls": await_calls,
                    "async_issues": len(issues),
                    "potential_deadlocks": len(
                        [i for i in issues if "deadlock" in i["description"]]
                    ),
                }
            )

            positive_findings = []
            if async_funcs > 0 and not issues:
                positive_findings.append("Async patterns appear to be used correctly")

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                positive_findings=positive_findings,
                metrics=metrics,
                confidence_score=0.85,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Async pattern analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _analyze_async_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Analyze async/await patterns"""
        findings = []

        # Common async anti-patterns
        patterns = {
            "missing_await": {
                "regex": r"\+\s*(?!await\s+).*async_function\(",
                "severity": "high",
                "description": "Async function called without await",
                "suggestion": "Add await keyword before async function calls",
            },
            "await_in_loop": {
                "regex": r"\+\s*for\s+.*:\s*\n\s*\+\s*.*await\s+",
                "severity": "medium",
                "description": "Await in loop - consider using asyncio.gather()",
                "suggestion": "Use asyncio.gather() or asyncio.as_completed() for concurrent execution",
            },
            "sync_call_in_async": {
                "regex": r"\+\s*async\s+def\s+.*:\s*.*\n\s*\+\s*.*time\.sleep\(",
                "severity": "high",
                "description": "Blocking call in async function",
                "suggestion": "Use asyncio.sleep() instead of time.sleep() in async functions",
            },
            "missing_async_context": {
                "regex": r"\+\s*with\s+.*\(",
                "severity": "medium",
                "description": "Potential async context manager used with regular 'with'",
                "suggestion": "Use 'async with' for async context managers",
            },
        }

        for pattern_name, pattern_info in patterns.items():
            matches = re.finditer(pattern_info["regex"], diff_content, re.MULTILINE)
            for match in matches:
                findings.append(
                    {
                        "pattern": pattern_name,
                        "severity": pattern_info["severity"],
                        "description": pattern_info["description"],
                        "suggestion": pattern_info["suggestion"],
                        "evidence": match.group(0).strip(),
                        "line_number": diff_content[: match.start()].count("\n") + 1,
                    }
                )

        return findings


@register_tool(enabled=True, name="PythonErrorHandlingTool")
class PythonErrorHandlingTool(BaseTool):
    """Validate Python error handling patterns using standards-based rules"""

    def __init__(self, name: str = "PythonErrorHandlingTool"):
        super().__init__(name)
        self.rule_engine = RuleEngine(language="python")

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CORRECTNESS

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.HIGH

    @property
    def requires_network(self) -> bool:
        return True  # Needs network access for Context7 rule fetching

    async def execute(self, context: ToolContext) -> ToolResult:
        """Analyze error handling patterns using current standards"""
        try:
            issues = []
            suggestions = []
            metrics = {}
            references: List[str] = []

            # Get current error handling rules from Python documentation
            error_rules = await self.rule_engine.get_error_handling_rules()

            # Analyze error handling
            error_analysis = await self._analyze_error_handling(
                context.diff_content, error_rules
            )

            for finding in error_analysis:
                issues.append(
                    {
                        "severity": finding["severity"],
                        "category": "correctness",
                        "description": finding["description"],
                        "file_path": finding.get("file_path", "unknown"),
                        "line_number": finding.get("line_number", 0),
                        "evidence": finding.get("evidence", ""),
                    }
                )

                if finding.get("recommendation"):
                    suggestions.append(finding["recommendation"])

                if finding.get("alternative"):
                    suggestions.append(f"Alternative: {finding['alternative']}")

            # Calculate error handling metrics using fast line counting
            try_blocks = 0
            except_blocks = 0
            bare_excepts = 0

            # Process only added lines for counting
            for line in context.diff_content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("+"):
                    code_line = stripped[1:].strip()
                    if code_line.startswith("try:"):
                        try_blocks += 1
                    elif code_line.startswith("except"):
                        except_blocks += 1
                        if code_line == "except:":  # Bare except
                            bare_excepts += 1

            metrics.update(
                {
                    "try_blocks": try_blocks,
                    "except_blocks": except_blocks,
                    "bare_except_blocks": bare_excepts,
                    "error_handling_issues": len(issues),
                    "exception_coverage": (except_blocks - bare_excepts) / except_blocks
                    if except_blocks > 0
                    else 1.0,
                    "rules_source": error_rules.get("source", "Unknown"),
                    "rules_last_updated": error_rules.get("last_updated", 0),
                }
            )

            positive_findings = []
            if try_blocks > 0 and bare_excepts == 0:
                positive_findings.append(
                    f"Good exception handling with specific exception types using {error_rules.get('source', 'current standards')}"
                )
            elif try_blocks == 0:
                positive_findings.append(
                    "No exception handling patterns detected in changes"
                )

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                positive_findings=positive_findings,
                metrics=metrics,
                references=references,
                confidence_score=0.9,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Error handling analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    async def _analyze_error_handling(
        self, diff_content: str, error_rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze error handling patterns using standards-based rules"""
        findings = []

        # Get patterns from current error handling standards
        patterns = error_rules.get("patterns", {})

        # Analyze each pattern type from standards
        for pattern_name, pattern_info in patterns.items():
            pattern_findings = self._detect_error_pattern_in_code(
                diff_content, pattern_name, pattern_info
            )
            findings.extend(pattern_findings)

        return findings

    def _detect_error_pattern_in_code(
        self, diff_content: str, pattern_name: str, pattern_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect specific error handling pattern in code based on pattern type"""
        findings = []

        # Define detection regex patterns based on pattern type
        detection_patterns = {
            "bare_except": [
                r"\+\s*except\s*:",
            ],
            "broad_except": [
                r"\+\s*except\s+Exception\s*:",
                r"\+\s*except\s+BaseException\s*:",
            ],
            "silent_exception": [
                r"\+\s*except.*:\s*\n\s*\+\s*pass",
                r"\+\s*except.*:\s*\n\s*\+\s*(?!.*log).*return",
                r"\+\s*except.*:\s*\n\s*\+\s*continue",
            ],
            "missing_finally": [
                r"\+\s*try:.*\n\s*\+\s*.*\n\s*\+\s*except.*:(?!.*finally)",
            ],
            "exception_swallowing": [
                r"\+\s*except.*:\s*\n\s*\+\s*pass\s*$",
            ],
            "logging_exception": [
                r"\+\s*except.*:\s*\n\s*\+\s*(?!.*log).*",
            ],
            "resource_leak": [
                r"\+\s*try:\s*\n\s*\+\s*.*open\(",  # File not closed in finally
                r"\+\s*try:\s*\n\s*\+\s*.*connect\(",  # Connection not closed
            ],
        }

        # Get regex patterns for this pattern type
        regexes = detection_patterns.get(pattern_name, [])

        for regex in regexes:
            matches = re.finditer(
                regex, diff_content, re.IGNORECASE | re.MULTILINE | re.DOTALL
            )
            for match in matches:
                findings.append(
                    {
                        "pattern": pattern_name,
                        "severity": pattern_info.get("severity", "medium"),
                        "description": pattern_info.get(
                            "description", f"Error handling issue: {pattern_name}"
                        ),
                        "recommendation": pattern_info.get(
                            "recommendation", "Follow error handling best practices"
                        ),
                        "alternative": pattern_info.get("alternative", ""),
                        "evidence": match.group(0).strip(),
                        "line_number": diff_content[: match.start()].count("\n") + 1,
                    }
                )

        return findings


@register_tool(enabled=True, name="PythonTypeHintValidationTool")
class PythonTypeHintValidationTool(BaseTool):
    """Validate Python type hints and type safety"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.MAINTAINABILITY

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.MEDIUM

    async def execute(self, context: ToolContext) -> ToolResult:
        """Analyze type hint usage"""
        try:
            issues = []
            suggestions = []
            metrics = {}

            # Analyze type hints
            type_analysis = self._analyze_type_hints(context.diff_content)

            for finding in type_analysis:
                issues.append(
                    {
                        "severity": finding["severity"],
                        "category": "maintainability",
                        "description": finding["description"],
                        "file_path": finding.get("file_path", "unknown"),
                        "line_number": finding.get("line_number", 0),
                        "evidence": finding.get("evidence", ""),
                    }
                )

                if finding.get("suggestion"):
                    suggestions.append(finding["suggestion"])

            # Calculate type hint metrics
            functions = re.findall(
                r"\+\s*def\s+(\w+)\s*\(([^)]*)\)", context.diff_content
            )
            typed_functions = 0
            untyped_functions = 0

            for func_name, params in functions:
                if "->" in params or ":" in params:
                    typed_functions += 1
                else:
                    untyped_functions += 1

            metrics.update(
                {
                    "total_functions": len(functions),
                    "typed_functions": typed_functions,
                    "untyped_functions": untyped_functions,
                    "type_hint_coverage": typed_functions / len(functions)
                    if functions
                    else 0,
                    "type_issues": len(issues),
                }
            )

            positive_findings = []
            if typed_functions > untyped_functions:
                positive_findings.append(
                    "Good type hint coverage improves code maintainability"
                )

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                positive_findings=positive_findings,
                metrics=metrics,
                confidence_score=0.8,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Type hint analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _analyze_type_hints(self, diff_content: str) -> List[Dict[str, Any]]:
        """Analyze type hint patterns"""
        findings = []

        # Find function definitions without type hints
        func_pattern = r"\+\s*def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*)?:"
        functions = re.finditer(func_pattern, diff_content)

        for func_match in functions:
            func_name = func_match.group(1)
            params = func_match.group(2)
            full_match = func_match.group(0)

            # Skip dunder methods and simple functions
            if func_name.startswith("__") and func_name.endswith("__"):
                continue

            # Check for type hints in parameters
            has_param_hints = ":" in params and "->" in full_match
            has_return_hint = "->" in full_match

            if not has_param_hints and params.strip() and params.strip() != "self":
                findings.append(
                    {
                        "pattern": "missing_parameter_types",
                        "severity": "low",
                        "description": f"Function '{func_name}' parameters lack type hints",
                        "suggestion": "Add type hints to function parameters for better code clarity",
                        "evidence": f"def {func_name}({params})",
                        "line_number": diff_content[: func_match.start()].count("\n")
                        + 1,
                    }
                )

            if not has_return_hint:
                findings.append(
                    {
                        "pattern": "missing_return_type",
                        "severity": "low",
                        "description": f"Function '{func_name}' lacks return type hint",
                        "suggestion": "Add return type hint to clarify function output",
                        "evidence": f"def {func_name}(...)",
                        "line_number": diff_content[: func_match.start()].count("\n")
                        + 1,
                    }
                )

        # Check for improper type hint usage
        type_patterns = {
            "any_type_usage": {
                "regex": r"\+\s*.*:\s*Any\b",
                "severity": "medium",
                "description": "Usage of 'Any' type reduces type safety benefits",
                "suggestion": "Use more specific type hints instead of 'Any' when possible",
            },
            "missing_optional": {
                "regex": r"\+\s*.*:\s*.*=\s*None",
                "severity": "low",
                "description": "Parameter with None default should use Optional type",
                "suggestion": "Use Optional[Type] or Type | None for parameters with None default",
            },
        }

        for pattern_name, pattern_info in type_patterns.items():
            matches = re.finditer(pattern_info["regex"], diff_content)
            for match in matches:
                findings.append(
                    {
                        "pattern": pattern_name,
                        "severity": pattern_info["severity"],
                        "description": pattern_info["description"],
                        "suggestion": pattern_info["suggestion"],
                        "evidence": match.group(0).strip(),
                        "line_number": diff_content[: match.start()].count("\n") + 1,
                    }
                )

        return findings


@register_tool(enabled=True, name="PythonFrameworkSpecificTool")
class PythonFrameworkSpecificTool(BaseTool):
    """Validate Python framework-specific patterns and best practices"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CORRECTNESS

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.MEDIUM

    async def execute(self, context: ToolContext) -> ToolResult:
        """Analyze framework-specific patterns"""
        try:
            issues = []
            suggestions = []
            metrics = {}

            # Detect frameworks in use
            frameworks = self._detect_frameworks(context.diff_content)

            # Analyze each detected framework
            for framework in frameworks:
                framework_analysis = self._analyze_framework_patterns(
                    framework, context.diff_content
                )

                for finding in framework_analysis:
                    issues.append(
                        {
                            "severity": finding["severity"],
                            "category": "correctness",
                            "description": f"[{framework}] {finding['description']}",
                            "file_path": finding.get("file_path", "unknown"),
                            "line_number": finding.get("line_number", 0),
                            "evidence": finding.get("evidence", ""),
                        }
                    )

                    if finding.get("suggestion"):
                        suggestions.append(f"[{framework}] {finding['suggestion']}")

            metrics.update(
                {"detected_frameworks": frameworks, "framework_issues": len(issues)}
            )

            positive_findings = []
            if frameworks and not issues:
                positive_findings.append(
                    f"Framework patterns for {', '.join(frameworks)} look good"
                )

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                positive_findings=positive_findings,
                metrics=metrics,
                confidence_score=0.75,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Framework analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _detect_frameworks(self, diff_content: str) -> List[str]:
        """Detect frameworks used in the code"""
        frameworks = []

        framework_patterns = {
            "fastapi": r"from\s+fastapi|import\s+fastapi",
            "django": r"from\s+django|import\s+django",
            "flask": r"from\s+flask|import\s+flask",
            "pytest": r"import\s+pytest|from\s+pytest",
            "pydantic": r"from\s+pydantic|import\s+pydantic",
            "sqlalchemy": r"from\s+sqlalchemy|import\s+sqlalchemy",
        }

        for framework, pattern in framework_patterns.items():
            if re.search(pattern, diff_content, re.IGNORECASE):
                frameworks.append(framework)

        return frameworks

    def _analyze_framework_patterns(
        self, framework: str, diff_content: str
    ) -> List[Dict[str, Any]]:
        """Analyze patterns specific to a framework"""
        findings = []

        if framework == "fastapi":
            findings.extend(self._analyze_fastapi_patterns(diff_content))
        elif framework == "pytest":
            findings.extend(self._analyze_pytest_patterns(diff_content))
        elif framework == "pydantic":
            findings.extend(self._analyze_pydantic_patterns(diff_content))

        return findings

    def _analyze_fastapi_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Analyze FastAPI-specific patterns"""
        findings = []

        # FastAPI anti-patterns
        patterns = {
            "missing_response_model": {
                "regex": r"\+\s*@app\.(get|post|put|delete)\([^)]*\)\s*\n\s*\+\s*(?:async\s+)?def\s+\w+\(",
                "severity": "medium",
                "description": "FastAPI endpoint without response_model",
                "suggestion": "Add response_model to FastAPI decorators for better API documentation",
            },
            "sync_endpoint": {
                "regex": r"\+\s*@app\.(get|post|put|delete).*\n\s*\+\s*def\s+(?!async)",
                "severity": "low",
                "description": "Synchronous FastAPI endpoint",
                "suggestion": "Consider using async def for better performance in FastAPI",
            },
        }

        for pattern_name, pattern_info in patterns.items():
            matches = re.finditer(pattern_info["regex"], diff_content, re.MULTILINE)
            for match in matches:
                findings.append(
                    {
                        "pattern": pattern_name,
                        "severity": pattern_info["severity"],
                        "description": pattern_info["description"],
                        "suggestion": pattern_info["suggestion"],
                        "evidence": match.group(0).strip(),
                        "line_number": diff_content[: match.start()].count("\n") + 1,
                    }
                )

        return findings

    def _analyze_pytest_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Analyze pytest-specific patterns"""
        findings = []

        # Pytest patterns
        if re.search(r"\+\s*def\s+test_.*\(.*\):", diff_content):
            # Check for missing assertions
            test_functions = re.finditer(
                r"\+\s*def\s+(test_\w+)\([^)]*\):", diff_content
            )
            for test_match in test_functions:
                test_name = test_match.group(1)

                # Look for assertions in the test function (simplified)
                test_body_start = test_match.end()
                test_lines = diff_content[test_body_start:].split("\n")[
                    :20
                ]  # Look at next 20 lines

                has_assertion = any(
                    "assert" in line for line in test_lines if line.startswith("+")
                )

                if not has_assertion:
                    findings.append(
                        {
                            "pattern": "test_without_assertion",
                            "severity": "medium",
                            "description": f"Test function '{test_name}' appears to lack assertions",
                            "suggestion": "Add assertions to validate test expectations",
                            "evidence": f"def {test_name}(...)",
                            "line_number": diff_content[: test_match.start()].count(
                                "\n"
                            )
                            + 1,
                        }
                    )

        return findings

    def _analyze_pydantic_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Analyze Pydantic-specific patterns"""
        findings = []

        # Pydantic patterns
        if re.search(r"\+\s*class\s+\w+\(.*BaseModel.*\):", diff_content):
            # Check for missing Field descriptions
            field_pattern = r"\+\s*(\w+):\s*[^=]*=\s*(?!Field\(.*description)"
            fields_without_desc = re.finditer(field_pattern, diff_content)

            for field_match in fields_without_desc:
                field_name = field_match.group(1)
                findings.append(
                    {
                        "pattern": "missing_field_description",
                        "severity": "low",
                        "description": f"Pydantic field '{field_name}' lacks description",
                        "suggestion": "Add description to Field() for better API documentation",
                        "evidence": field_match.group(0).strip(),
                        "line_number": diff_content[: field_match.start()].count("\n")
                        + 1,
                    }
                )

        return findings
