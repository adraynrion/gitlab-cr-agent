"""
Simple Context7 MCP-based validation tool

This single tool replaces ALL complex hard-coded analysis with Context7's
validate_code_against_docs function. No more patterns, no more complexity.

Just extract libraries from imports and validate against documentation.
Clean, simple, and always up-to-date.
"""

import logging
import re
from typing import List

from src.agents.tools.base import (
    BaseTool,
    ToolCategory,
    ToolContext,
    ToolPriority,
    ToolResult,
)
from src.agents.tools.mcp_context7 import validate_api_usage
from src.agents.tools.registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(enabled=True, name="Context7DocumentationValidationTool")
class Context7DocumentationValidationTool(BaseTool):
    """
    Single tool that uses Context7's validate_code_against_docs for everything.

    No hard-coded patterns. No complex analysis. No maintenance burden.
    Just validate code against current documentation.
    """

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CORRECTNESS

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.HIGH

    @property
    def requires_network(self) -> bool:
        return True

    async def execute(self, context: ToolContext) -> ToolResult:
        """Validate code using Context7's validate_code_against_docs"""
        try:
            issues = []
            suggestions = []
            evidence = {}
            references = []
            context7_unavailable_libraries = []
            context7_available_libraries = []

            # Extract libraries from imports
            libraries = self._extract_libraries(context.diff_content)

            if not libraries:
                return ToolResult(
                    tool_name=self.name,
                    category=self.category,
                    success=True,
                    positive_findings=["No external libraries to validate"],
                    error_message=None,
                    execution_time_ms=0,
                )

            # For each library, use Context7's validate_code_against_docs
            for library in libraries:
                try:
                    validation = await validate_api_usage(
                        library_name=library,
                        code_snippet=context.diff_content,
                        context="security performance best_practices",
                    )

                    # Track Context7 availability
                    is_context7_available = validation.get("context7_available", False)
                    unavailability_reason = validation.get("unavailability_reason")

                    if not is_context7_available:
                        context7_unavailable_libraries.append(
                            {
                                "library": library,
                                "reason": unavailability_reason or "Unknown reason",
                            }
                        )
                        logger.warning(
                            f"Context7 unavailable for {library}: {unavailability_reason}"
                        )
                        continue
                    else:
                        context7_available_libraries.append(library)

                    # Add issues
                    for issue in validation.get("issues", []):
                        issues.append(
                            {
                                "severity": "medium",
                                "category": "documentation",
                                "description": f"{library}: {issue}",
                                "file_path": "unknown",
                                "line_number": 0,
                                "evidence": "Context7 documentation validation",
                            }
                        )

                    # Add suggestions
                    for suggestion in validation.get("suggestions", []):
                        suggestions.append(f"{library}: {suggestion}")

                    # Collect references
                    references.extend(validation.get("references", []))

                    # Store evidence
                    if validation.get("issues") or validation.get("suggestions"):
                        evidence[library] = validation

                except Exception as e:
                    logger.warning(f"Validation failed for {library}: {e}")
                    context7_unavailable_libraries.append(
                        {"library": library, "reason": f"Validation error: {str(e)}"}
                    )

            # Enhanced metrics with Context7 availability info
            metrics = {
                "libraries_detected": len(libraries),
                "libraries_validated_with_context7": len(context7_available_libraries),
                "libraries_unavailable_context7": len(context7_unavailable_libraries),
                "total_issues": len(issues),
                "total_suggestions": len(suggestions),
                "context7_availability_rate": len(context7_available_libraries)
                / len(libraries)
                if libraries
                else 0,
            }

            positive_findings = []

            # Add Context7 availability information
            if context7_available_libraries:
                positive_findings.append(
                    f"Context7 validation available for: {', '.join(context7_available_libraries)}"
                )

            if not issues and context7_available_libraries:
                positive_findings.append(
                    "No documentation issues found for validated libraries"
                )

            # Create warnings for unavailable libraries
            context7_warnings = []
            if context7_unavailable_libraries:
                context7_warnings.append(
                    "Context7 MCP documentation validation was not available for some libraries"
                )
                for lib_info in context7_unavailable_libraries:
                    context7_warnings.append(
                        f"- {lib_info['library']}: {lib_info['reason']}"
                    )

            # Determine success - successful if no errors, even if Context7 is unavailable
            success = True
            error_message = None

            # Store Context7 availability info in evidence
            if context7_unavailable_libraries:
                evidence["context7_unavailability"] = {
                    "unavailable_libraries": context7_unavailable_libraries,
                    "impact": "Documentation validation was limited for some libraries",
                    "recommendation": "Code review will proceed with agent knowledge only for affected libraries",
                }

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=success,
                issues=issues,
                suggestions=suggestions + context7_warnings,
                positive_findings=positive_findings,
                evidence=evidence,
                references=list(set(references)),
                metrics=metrics,
                confidence_score=0.9 if context7_available_libraries else 0.3,
                error_message=error_message,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Context7 validation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=f"Context7 validation tool failed: {str(e)}",
                execution_time_ms=0,
            )

    def _extract_libraries(self, diff_content: str) -> List[str]:
        """Extract library names from import statements"""
        libraries = set()

        # Simple import extraction
        patterns = [
            r"\+.*from\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\+.*import\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, diff_content)
            libraries.update(matches)

        # Filter out standard library
        return [
            lib
            for lib in libraries
            if lib
            not in {"os", "sys", "re", "json", "time", "datetime", "typing", "logging"}
        ]
