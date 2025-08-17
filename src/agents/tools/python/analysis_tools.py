"""
Code analysis tools migrated from the original code reviewer
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
from src.agents.tools.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


@register_tool(enabled=True, name="PythonSecurityAnalysisTool")
class PythonSecurityAnalysisTool(BaseTool):
    """Analyze Python code for common security vulnerabilities using standards-based rules"""

    def __init__(self, name: str = "PythonSecurityAnalysisTool"):
        super().__init__(name)
        self.rule_engine = RuleEngine(language="python")

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SECURITY

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.CRITICAL

    @property
    def requires_network(self) -> bool:
        return True  # Needs network access for Context7 rule fetching

    async def execute(self, context: ToolContext) -> ToolResult:
        """Analyze code for security vulnerabilities using current standards"""
        try:
            issues = []
            suggestions = []
            metrics = {}
            references = []

            # Get current security rules from OWASP and NIST
            security_rules = await self.rule_engine.get_security_rules()

            # Analyze the diff content for security issues
            security_findings = await self._analyze_security_patterns(
                context.diff_content, security_rules
            )

            # Convert findings to structured issues
            for finding in security_findings:
                issues.append(
                    {
                        "severity": finding["severity"],
                        "category": "security",
                        "description": finding["description"],
                        "file_path": finding.get("file_path", "unknown"),
                        "line_number": finding.get("line_number", 0),
                        "evidence": finding.get("evidence", ""),
                    }
                )

                if finding.get("recommendation"):
                    suggestions.append(finding["recommendation"])

                # Collect references from standards
                if finding.get("references"):
                    references.extend(finding["references"])

            # Calculate security metrics
            metrics.update(
                {
                    "total_security_issues": len(issues),
                    "critical_issues": len(
                        [i for i in issues if i["severity"] == "critical"]
                    ),
                    "high_issues": len([i for i in issues if i["severity"] == "high"]),
                    "medium_issues": len(
                        [i for i in issues if i["severity"] == "medium"]
                    ),
                    "rules_source": security_rules.get("source", "Unknown"),
                    "rules_last_updated": security_rules.get("last_updated", 0),
                }
            )

            # Higher confidence when using current standards
            confidence = 0.95 if issues else 0.9

            positive_findings = []
            if not issues:
                positive_findings.append(
                    f"No security vulnerabilities detected using {security_rules.get('source', 'current standards')}"
                )

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                positive_findings=positive_findings,
                metrics=metrics,
                references=list(set(references)),  # Remove duplicates
                confidence_score=confidence,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    async def _analyze_security_patterns(
        self, diff_content: str, security_rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze code for security patterns using standards-based rules"""
        findings = []

        # Get patterns from current security standards
        patterns = security_rules.get("patterns", {})

        # Analyze each pattern type from standards
        for pattern_name, pattern_info in patterns.items():
            pattern_findings = self._detect_pattern_in_code(
                diff_content, pattern_name, pattern_info
            )
            findings.extend(pattern_findings)

        # Add recommendations from standards
        recommendations = security_rules.get("recommendations", [])
        for finding in findings:
            if not finding.get("recommendation") and recommendations:
                finding["recommendation"] = recommendations[
                    0
                ]  # Use first relevant recommendation

        return findings

    def _detect_pattern_in_code(
        self, diff_content: str, pattern_name: str, pattern_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect specific security pattern in code using AST analysis + text fallback"""
        from .code_parser import get_parser

        findings = []
        parser = get_parser()

        # Use AST-based security pattern detection first
        ast_patterns = parser.extract_security_patterns(diff_content)

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
                            "recommendation", "Review and fix security issue"
                        ),
                        "evidence": pattern["evidence"],
                        "line": pattern.get("line", 0),
                    }
                )

        # Fallback to text-based detection for patterns AST might miss
        fallback_patterns = self._detect_text_patterns(diff_content, pattern_name)
        findings.extend(fallback_patterns)

        return findings

    def _detect_text_patterns(
        self, diff_content: str, pattern_name: str
    ) -> List[Dict[str, Any]]:
        """Text-based pattern detection for patterns AST might miss"""
        findings = []
        added_lines = [
            line.strip()
            for line in diff_content.split("\n")
            if line.strip().startswith("+") and not line.strip().startswith("+++")
        ]

        for i, line in enumerate(added_lines):
            code_line = line[1:].strip().lower()

            # CORS wildcard detection (config-based, hard to detect with AST)
            if pattern_name == "cors_wildcard":
                if "allow_origins" in code_line and "*" in code_line:
                    findings.append(
                        {
                            "pattern": pattern_name,
                            "severity": "high",
                            "description": "CORS wildcard configuration detected",
                            "recommendation": "Specify explicit origins instead of wildcard",
                            "evidence": line.strip(),
                            "line": i + 1,
                        }
                    )

            # Insecure random detection
            elif pattern_name == "insecure_random":
                if "random.random(" in code_line or "random.choice(" in code_line:
                    findings.append(
                        {
                            "pattern": pattern_name,
                            "severity": "medium",
                            "description": "Insecure random number generation",
                            "recommendation": "Use secrets module for cryptographic purposes",
                            "evidence": line.strip(),
                            "line": i + 1,
                        }
                    )

            # File traversal detection
            elif pattern_name == "file_traversal":
                if "../" in code_line and (
                    "open(" in code_line or "path(" in code_line
                ):
                    findings.append(
                        {
                            "pattern": pattern_name,
                            "severity": "high",
                            "description": "Potential path traversal vulnerability",
                            "recommendation": "Validate and sanitize file paths",
                            "evidence": line.strip(),
                            "line": i + 1,
                        }
                    )

        return findings


@register_tool(enabled=True, name="PythonComplexityAnalysisTool")
class PythonComplexityAnalysisTool(BaseTool):
    """Calculate cyclomatic complexity and other Python code metrics"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.MAINTAINABILITY

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.MEDIUM

    async def execute(self, context: ToolContext) -> ToolResult:
        """Calculate code complexity metrics"""
        try:
            issues = []
            suggestions = []
            metrics = {}

            # Analyze complexity for each changed file
            file_complexities = self._analyze_complexity(context.diff_content)

            total_complexity = 0
            function_count = 0

            for file_analysis in file_complexities:
                for func_analysis in file_analysis.get("functions", []):
                    complexity = func_analysis["complexity"]
                    total_complexity += complexity
                    function_count += 1

                    # Flag high complexity functions
                    if complexity > 10:
                        issues.append(
                            {
                                "severity": "medium" if complexity <= 15 else "high",
                                "category": "maintainability",
                                "description": f"High cyclomatic complexity ({complexity}) in function",
                                "file_path": file_analysis.get("file_path", "unknown"),
                                "line_number": func_analysis.get("line_number", 0),
                                "evidence": f"Complexity score: {complexity}",
                            }
                        )

                        suggestions.append(
                            f"Consider refactoring function to reduce complexity (current: {complexity})"
                        )

            # Calculate average complexity
            avg_complexity = (
                total_complexity / function_count if function_count > 0 else 0
            )

            metrics.update(
                {
                    "total_functions": function_count,
                    "average_complexity": round(avg_complexity, 2),
                    "max_complexity": max(
                        [
                            f["complexity"]
                            for file in file_complexities
                            for f in file.get("functions", [])
                        ],
                        default=0,
                    ),
                    "high_complexity_functions": len(
                        [i for i in issues if "complexity" in i["description"]]
                    ),
                }
            )

            positive_findings = []
            if avg_complexity <= 5:
                positive_findings.append(
                    "Code has low complexity and good maintainability"
                )

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
            logger.error(f"Complexity analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _analyze_complexity(self, diff_content: str) -> List[Dict[str, Any]]:
        """Analyze cyclomatic complexity of functions using AST parsing"""
        from .code_parser import get_parser

        file_analyses = []
        parser = get_parser()

        # Extract functions using AST parsing
        functions = parser.extract_functions(diff_content)

        for func_info in functions:
            func_name = func_info["name"]
            complexity_data = func_info["complexity_indicators"]

            # Calculate overall complexity score
            complexity = (
                complexity_data.get("branches", 0)
                + complexity_data.get("loops", 0)
                + complexity_data.get("nested_functions", 0)
                + complexity_data.get("try_blocks", 0)
                + 1  # Base complexity
            )

            file_analyses.append(
                {
                    "file_path": "modified_file",
                    "functions": [
                        {
                            "name": func_name,
                            "complexity": complexity,
                            "line_number": func_info["line_number"],
                            "lines_of_code": complexity_data.get("lines", 0),
                            "is_async": func_info["is_async"],
                            "has_docstring": func_info["has_docstring"],
                            "decorators": func_info["decorators"],
                            "complexity_breakdown": complexity_data,
                        }
                    ],
                }
            )

        return file_analyses

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1  # Base complexity

        # Count decision points
        decision_keywords = ["if", "elif", "for", "while", "except", "and", "or"]

        for keyword in decision_keywords:
            # Use word boundaries to avoid false positives
            pattern = r"\b" + keyword + r"\b"
            matches = re.findall(pattern, code, re.IGNORECASE)
            complexity += len(matches)

        return complexity


@register_tool(enabled=True, name="PythonCodeQualityTool")
class PythonCodeQualityTool(BaseTool):
    """Analyze Python code quality issues"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.STYLE

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.MEDIUM

    async def execute(self, context: ToolContext) -> ToolResult:
        """Analyze code quality issues"""
        try:
            issues = []
            suggestions = []
            metrics = {}

            # Analyze various quality aspects
            quality_checks = [
                self._check_naming_conventions,
                self._check_code_duplication,
                self._check_function_length,
                self._check_import_organization,
                self._check_documentation,
            ]

            for check in quality_checks:
                check_results = check(context.diff_content)
                issues.extend(check_results.get("issues", []))
                suggestions.extend(check_results.get("suggestions", []))
                metrics.update(check_results.get("metrics", {}))

            positive_findings = []
            if not issues:
                positive_findings.append("Code follows good quality practices")

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
            logger.error(f"Code quality analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _check_naming_conventions(self, diff_content: str) -> Dict[str, Any]:
        """Check naming conventions"""
        issues = []
        suggestions = []

        # Check function naming (should be snake_case) using cached patterns
        from .code_parser import get_compiled_patterns

        patterns_cache = get_compiled_patterns()

        func_pattern = r"\+\s*def\s+([A-Z][a-zA-Z0-9_]*)\s*\("
        bad_func_names = patterns_cache.findall(func_pattern, diff_content)

        for func_name in bad_func_names:
            issues.append(
                {
                    "severity": "low",
                    "category": "style",
                    "description": f"Function name '{func_name}' should use snake_case",
                    "file_path": "unknown",
                    "line_number": 0,
                }
            )

        if bad_func_names:
            suggestions.append("Use snake_case for function names (e.g., my_function)")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "metrics": {"naming_violations": len(bad_func_names)},
        }

    def _check_code_duplication(self, diff_content: str) -> Dict[str, Any]:
        """Check for potential code duplication"""
        issues = []
        suggestions = []

        # Simple duplication check (look for similar lines)
        added_lines = [
            line[1:].strip()
            for line in diff_content.split("\n")
            if line.startswith("+") and line[1:].strip()
        ]

        duplicates = []
        for i, line in enumerate(added_lines):
            if len(line) > 20:  # Only check substantial lines
                for j, other_line in enumerate(added_lines[i + 1 :], i + 1):
                    if line == other_line:
                        duplicates.append((i, j, line))

        if duplicates:
            issues.append(
                {
                    "severity": "medium",
                    "category": "maintainability",
                    "description": f"Potential code duplication detected ({len(duplicates)} instances)",
                    "file_path": "unknown",
                    "line_number": 0,
                }
            )
            suggestions.append(
                "Consider extracting common code into reusable functions"
            )

        return {
            "issues": issues,
            "suggestions": suggestions,
            "metrics": {"duplicate_lines": len(duplicates)},
        }

    def _check_function_length(self, diff_content: str) -> Dict[str, Any]:
        """Check function length"""
        issues = []
        suggestions = []

        # Count lines in functions (simplified)
        func_pattern = r"\+\s*def\s+\w+\s*\([^)]*\):"
        functions = list(re.finditer(func_pattern, diff_content))

        long_functions = 0
        for i, func_match in enumerate(functions):
            # Count lines until next function or end
            start_pos = func_match.end()
            if i + 1 < len(functions):
                end_pos = functions[i + 1].start()
            else:
                end_pos = len(diff_content)

            func_lines = diff_content[start_pos:end_pos].count("\n+")

            if func_lines > 50:  # Arbitrary threshold
                long_functions += 1
                issues.append(
                    {
                        "severity": "medium",
                        "category": "maintainability",
                        "description": f"Function is very long ({func_lines} lines)",
                        "file_path": "unknown",
                        "line_number": diff_content[: func_match.start()].count("\n")
                        + 1,
                    }
                )

        if long_functions > 0:
            suggestions.append(
                "Consider breaking down long functions into smaller, focused functions"
            )

        return {
            "issues": issues,
            "suggestions": suggestions,
            "metrics": {"long_functions": long_functions},
        }

    def _check_import_organization(self, diff_content: str) -> Dict[str, Any]:
        """Check import organization"""
        issues = []
        suggestions = []

        # Check for imports in the middle of the file
        import_pattern = r"\+\s*(?:import|from)\s+"
        imports = list(re.finditer(import_pattern, diff_content))

        # Check if imports appear after code lines
        code_pattern = r"\+\s*(?:def|class|if|for|while|with|try)\s+"
        code_lines = list(re.finditer(code_pattern, diff_content))

        misplaced_imports = 0
        for imp in imports:
            for code in code_lines:
                if imp.start() > code.start():
                    misplaced_imports += 1
                    break

        if misplaced_imports > 0:
            issues.append(
                {
                    "severity": "low",
                    "category": "style",
                    "description": "Imports should be at the top of the file",
                    "file_path": "unknown",
                    "line_number": 0,
                }
            )
            suggestions.append("Move all imports to the top of the file")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "metrics": {"misplaced_imports": misplaced_imports},
        }

    def _check_documentation(self, diff_content: str) -> Dict[str, Any]:
        """Check for documentation"""
        issues = []
        suggestions = []

        # Check for functions without docstrings
        func_pattern = r"\+\s*def\s+(\w+)\s*\([^)]*\):"
        functions = re.findall(func_pattern, diff_content)

        # Simple check: look for docstrings after function definitions
        functions_without_docs = 0
        for func_name in functions:
            # Look for docstring pattern after function definition
            func_def_pattern = rf"\+\s*def\s+{re.escape(func_name)}\s*\([^)]*\):"
            match = re.search(func_def_pattern, diff_content)
            if match:
                # Check next few lines for docstring
                after_def = diff_content[match.end() :]
                lines = after_def.split("\n")[:5]  # Check first 5 lines

                has_docstring = any(
                    '"""' in line or "'''" in line
                    for line in lines
                    if line.startswith("+")
                )

                if not has_docstring:
                    functions_without_docs += 1

        if functions_without_docs > 0:
            issues.append(
                {
                    "severity": "low",
                    "category": "maintainability",
                    "description": f"{functions_without_docs} functions lack documentation",
                    "file_path": "unknown",
                    "line_number": 0,
                }
            )
            suggestions.append(
                "Add docstrings to document function purpose and parameters"
            )

        return {
            "issues": issues,
            "suggestions": suggestions,
            "metrics": {"undocumented_functions": functions_without_docs},
        }
