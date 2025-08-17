"""
Enhanced validation tools for comprehensive code review
"""

import logging
import re
from typing import Any, Dict, List

from src.agents.tools.base import (
    BaseTool,
    ToolCategory,
    ToolContext,
    ToolPriority,
    ToolResult,
)
from src.agents.tools.registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(enabled=True, name="PerformancePatternTool")
class PerformancePatternTool(BaseTool):
    """Detect performance anti-patterns and optimization opportunities"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.PERFORMANCE

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.HIGH

    async def execute(self, context: ToolContext) -> ToolResult:
        """Analyze code for performance issues"""
        try:
            issues = []
            suggestions = []
            metrics = {}

            # Performance pattern checks
            perf_analysis = self._analyze_performance_patterns(context.diff_content)

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

                if finding.get("suggestion"):
                    suggestions.append(finding["suggestion"])

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
                }
            )

            positive_findings = []
            if not issues:
                positive_findings.append("No obvious performance issues detected")

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
            logger.error(f"Performance analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _analyze_performance_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Analyze code for performance anti-patterns"""
        findings = []

        # Performance anti-patterns
        patterns = {
            "string_concatenation_loop": {
                "regex": r"\+\s*for\s+.*:\s*\n\s*\+\s*.*\+=.*\+",
                "severity": "medium",
                "description": "String concatenation in loop - consider using join()",
                "suggestion": "Use ''.join() or a list comprehension for better performance",
            },
            "repeated_function_calls": {
                "regex": r"\+\s*for\s+.*:\s*.*len\(",
                "severity": "low",
                "description": "Function call in loop condition - cache the result",
                "suggestion": "Cache len() result outside the loop",
            },
            "inefficient_membership_test": {
                "regex": r"\+\s*.*in\s+\[.*\]",
                "severity": "medium",
                "description": "Membership test on list - consider using set",
                "suggestion": "Use set for O(1) membership testing instead of list O(n)",
            },
            "unnecessary_list_comprehension": {
                "regex": r"\+\s*list\(\[.*for.*\]\)",
                "severity": "low",
                "description": "Unnecessary list() around list comprehension",
                "suggestion": "Remove redundant list() call around list comprehension",
            },
            "inefficient_dict_access": {
                "regex": r"\+\s*if\s+.*in\s+.*\.keys\(\)",
                "severity": "low",
                "description": "Inefficient dictionary key check using .keys()",
                "suggestion": "Use 'if key in dict' instead of 'if key in dict.keys()'",
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

        # Check for database query patterns
        db_patterns = self._check_database_patterns(diff_content)
        findings.extend(db_patterns)

        return findings

    def _check_database_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Check for database-related performance issues"""
        findings = []

        # N+1 query pattern detection
        if re.search(
            r"\+\s*for\s+.*:\s*\n\s*\+\s*.*\.get\(", diff_content, re.MULTILINE
        ):
            findings.append(
                {
                    "pattern": "n_plus_one_query",
                    "severity": "high",
                    "description": "Potential N+1 query pattern detected",
                    "suggestion": "Use select_related() or prefetch_related() to optimize queries",
                    "evidence": "Loop with individual database queries",
                    "line_number": 0,
                }
            )

        # Missing database indexes
        if re.search(r"\+\s*.*\.filter\(.*=.*\)", diff_content):
            findings.append(
                {
                    "pattern": "potential_missing_index",
                    "severity": "medium",
                    "description": "Database filter without obvious index",
                    "suggestion": "Ensure database indexes exist for filtered fields",
                    "evidence": "Database query with filter",
                    "line_number": 0,
                }
            )

        return findings


@register_tool(enabled=True, name="AsyncPatternValidationTool")
class AsyncPatternValidationTool(BaseTool):
    """Validate async/await patterns and concurrency usage"""

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

            # Calculate async metrics
            async_funcs = len(re.findall(r"\+\s*async\s+def", context.diff_content))
            await_calls = len(re.findall(r"\+\s*.*await\s+", context.diff_content))

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


@register_tool(enabled=True, name="ErrorHandlingTool")
class ErrorHandlingTool(BaseTool):
    """Validate error handling patterns and exception safety"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CORRECTNESS

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.HIGH

    async def execute(self, context: ToolContext) -> ToolResult:
        """Analyze error handling patterns"""
        try:
            issues = []
            suggestions = []
            metrics = {}

            # Analyze error handling
            error_analysis = self._analyze_error_handling(context.diff_content)

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

                if finding.get("suggestion"):
                    suggestions.append(finding["suggestion"])

            # Calculate error handling metrics
            try_blocks = len(re.findall(r"\+\s*try:", context.diff_content))
            except_blocks = len(re.findall(r"\+\s*except", context.diff_content))
            bare_excepts = len(re.findall(r"\+\s*except:", context.diff_content))

            metrics.update(
                {
                    "try_blocks": try_blocks,
                    "except_blocks": except_blocks,
                    "bare_except_blocks": bare_excepts,
                    "error_handling_issues": len(issues),
                }
            )

            positive_findings = []
            if try_blocks > 0 and bare_excepts == 0:
                positive_findings.append(
                    "Good exception handling with specific exception types"
                )

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                positive_findings=positive_findings,
                metrics=metrics,
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

    def _analyze_error_handling(self, diff_content: str) -> List[Dict[str, Any]]:
        """Analyze error handling patterns"""
        findings = []

        # Error handling anti-patterns
        patterns = {
            "bare_except": {
                "regex": r"\+\s*except\s*:",
                "severity": "high",
                "description": "Bare except clause catches all exceptions",
                "suggestion": "Catch specific exception types instead of using bare except",
            },
            "empty_except": {
                "regex": r"\+\s*except.*:\s*\n\s*\+\s*pass",
                "severity": "high",
                "description": "Empty except block silently ignores errors",
                "suggestion": "Handle exceptions appropriately or log the error",
            },
            "too_broad_except": {
                "regex": r"\+\s*except\s+Exception\s*:",
                "severity": "medium",
                "description": "Catching Exception is too broad",
                "suggestion": "Catch more specific exception types",
            },
            "missing_finally": {
                "regex": r"\+\s*try:.*\n\s*\+\s*.*\n\s*\+\s*except.*:(?!.*finally)",
                "severity": "low",
                "description": "Try/except without finally block for cleanup",
                "suggestion": "Consider using finally block or context managers for cleanup",
            },
            "exception_without_logging": {
                "regex": r"\+\s*except.*:\s*\n\s*\+\s*(?!.*log).*return",
                "severity": "medium",
                "description": "Exception caught but not logged",
                "suggestion": "Log exceptions for debugging and monitoring",
            },
        }

        for pattern_name, pattern_info in patterns.items():
            matches = re.finditer(
                pattern_info["regex"], diff_content, re.MULTILINE | re.DOTALL
            )
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


@register_tool(enabled=True, name="TypeHintValidationTool")
class TypeHintValidationTool(BaseTool):
    """Validate type hints and type safety"""

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


@register_tool(enabled=True, name="FrameworkSpecificTool")
class FrameworkSpecificTool(BaseTool):
    """Validate framework-specific patterns and best practices"""

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
