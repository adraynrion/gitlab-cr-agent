"""
PydanticAI-based code review agent with multi-LLM support and tool system
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, RunContext
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.providers import get_llm_model
from src.agents.tools import ToolContext, ToolRegistry
from src.agents.tools.mcp_context7 import (
    LibraryDocumentation,
    LibraryResolutionResult,
    get_library_docs,
    resolve_library_id,
    search_documentation,
    validate_api_usage,
)
from src.config.settings import get_settings
from src.exceptions import AIProviderException, ReviewProcessException
from src.models.review_models import ReviewContext, ReviewResult

logger = logging.getLogger(__name__)

# System prompt for code review with tool integration
CODE_REVIEW_SYSTEM_PROMPT = """
You are an expert software engineer conducting thorough, evidence-based code reviews.
Your role is to analyze code changes using specialized tools and provide constructive, actionable feedback.

REVIEW FRAMEWORK:
1. **Correctness**: Identify logic errors, edge cases, and algorithm issues
2. **Security**: Detect vulnerabilities, input validation issues, authentication flaws
3. **Performance**: Find bottlenecks, inefficient algorithms, resource usage problems
4. **Maintainability**: Assess code clarity, structure, documentation quality
5. **Best Practices**: Check language conventions, design patterns, testing

SIMPLIFIED TOOL ARCHITECTURE:
You have access to a simplified, Context7-based tool system that provides evidence-based insights:
- Context7 Documentation Validation Tool: Validates code against current official documentation
- Library import detection and analysis
- Real-time documentation lookup when Context7 MCP is available
- Graceful degradation when Context7 is unavailable

CONTEXT7 MCP INTEGRATION:
You have direct access to Context7's comprehensive documentation database through these tools:
- resolve_library_documentation(library_name): Get Context7 library ID and metadata
- get_documentation(library_id, topic, max_tokens): Fetch current docs and code examples
- search_library_docs(query, libraries, max_results): Search across multiple libraries
- validate_code_against_docs(library_name, code_snippet, context): Validate API usage

IMPORTANT: Context7 MCP may not always be available. When Context7 is unavailable:
- Clearly inform the user that Context7 documentation validation is not available
- Explain that the review is based on your knowledge only for affected libraries
- Still provide a thorough review using your training knowledge
- Be transparent about limitations without Context7 verification

Use these Context7 tools to:
- Verify API usage patterns against official documentation
- Find relevant code examples and best practices
- Check for deprecated methods or patterns
- Get authoritative references for your recommendations

Use tool results to support your analysis with concrete evidence and references.

OUTPUT REQUIREMENTS:
- Integrate tool findings into your review
- Provide specific file paths and line numbers for each issue
- Categorize issues by severity (critical, high, medium, low)
- Include concrete suggestions backed by tool evidence
- Reference official documentation when available
- Show code examples when helpful
- Balance criticism with positive observations
- Consider the broader codebase context

EVIDENCE-BASED APPROACH:
- Prioritize issues backed by tool evidence
- Include references to official documentation
- Cite specific security guidelines (OWASP, etc.)
- Use quantitative metrics when available
- Distinguish between tool-detected issues and manual observations

TONE:
- Be constructive and educational
- Focus on substantial issues over style preferences
- Acknowledge good practices when observed
- Provide clear rationale for suggestions with supporting evidence
"""


@dataclass
class ReviewDependencies:
    """Dependencies for code review operations"""

    repository_url: str
    branch: str
    merge_request_iid: int
    gitlab_token: str
    diff_content: str
    file_changes: List[Dict[str, Any]]
    review_trigger_tag: str
    tool_results: Optional[List[Dict[str, Any]]] = None


class CodeReviewAgent:
    """Main code review agent using PydanticAI with tool system integration"""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the review agent with specified model"""
        settings = get_settings()
        self.model_name = model_name or settings.ai_model
        self.model = get_llm_model(self.model_name)

        # Create PydanticAI agent
        self.agent = Agent(
            model=self.model,
            output_type=ReviewResult,
            deps_type=ReviewDependencies,
            system_prompt=CODE_REVIEW_SYSTEM_PROMPT,
            retries=settings.ai_retries,
        )

        # Initialize tool system
        self.tool_registry = ToolRegistry()
        self._initialize_tools()

        # Register simplified tools for PydanticAI (tool results will be provided via context)
        self._register_pydantic_tools()

        logger.info(f"Initialized CodeReviewAgent with model: {self.model_name}")

    def _initialize_tools(self):
        """Initialize the tool system with settings"""
        settings = get_settings()
        if settings.tools_enabled:
            # Import tool modules to trigger registration
            try:
                # Import new unified Context7-based tools
                import src.agents.tools.unified_context7_tools  # noqa: F401

                # Configure registry from settings
                tool_settings = {
                    "enabled_categories": settings.enabled_tool_categories,
                    "disabled_categories": settings.disabled_tool_categories,
                    "enabled_tools": settings.enabled_tools,
                    "disabled_tools": settings.disabled_tools,
                    "context7": {
                        "enabled": settings.context7_enabled,
                        "max_tokens": settings.context7_max_tokens,
                        "cache_ttl": settings.context7_cache_ttl,
                    },
                }

                self.tool_registry.configure_from_settings(tool_settings)
                logger.info(
                    f"Tool system initialized: {self.tool_registry.get_statistics()}"
                )

            except ImportError as e:
                logger.warning(f"Failed to import tool modules: {e}")
                settings.tools_enabled = False
        else:
            logger.info("Tool system disabled in settings")

    def _register_pydantic_tools(self):
        """Register tools for PydanticAI that provide complete, unlimited access to tool results"""

        @self.agent.tool
        async def get_tool_insights(
            ctx: RunContext[ReviewDependencies], aspect: str
        ) -> str:
            """Get complete insights from tool results for a specific aspect"""
            if not ctx.deps.tool_results:
                return f"No tool insights available for {aspect}"

            insights = []
            for tool_result in ctx.deps.tool_results:
                if aspect.lower() in tool_result.get("category", "").lower():
                    # Extract ALL findings without limitation
                    issues = tool_result.get("issues", [])
                    suggestions = tool_result.get("suggestions", [])
                    positive_findings = tool_result.get("positive_findings", [])

                    tool_name = tool_result.get("tool_name", "Unknown")
                    insights.append(f"\n=== {tool_name} Results ===")

                    if issues:
                        insights.append(f"Issues found: {len(issues)}")
                        for issue in issues:  # Include ALL issues
                            severity = issue.get("severity", "unknown")
                            description = issue.get("description", "Unknown issue")
                            evidence = issue.get("evidence", "")
                            file_path = issue.get("file_path", "")
                            line_number = issue.get("line_number", 0)

                            issue_details = f"[{severity.upper()}] {description}"
                            if file_path and file_path != "unknown":
                                issue_details += f" (at {file_path}:{line_number})"
                            if evidence:
                                issue_details += f" | Evidence: {evidence}"
                            insights.append(f"- {issue_details}")

                    if suggestions:
                        insights.append("Suggestions:")
                        for suggestion in suggestions:  # Include ALL suggestions
                            insights.append(f"- {suggestion}")

                    if positive_findings:
                        insights.append("Positive findings:")
                        for (
                            finding
                        ) in positive_findings:  # Include ALL positive findings
                            insights.append(f"+ {finding}")

            return (
                "\n".join(insights)
                if insights
                else f"No specific insights for {aspect}"
            )

        @self.agent.tool
        async def get_evidence_references(
            ctx: RunContext[ReviewDependencies], topic: str
        ) -> str:
            """Get complete evidence and references from tool results"""
            if not ctx.deps.tool_results:
                return f"No evidence available for {topic}"

            all_references = []
            all_evidence = []

            for tool_result in ctx.deps.tool_results:
                tool_name = tool_result.get("tool_name", "Unknown")

                # Collect ALL references without limitation
                tool_refs = tool_result.get("references", [])
                if tool_refs:
                    all_references.append(f"\n=== {tool_name} References ===")
                    all_references.extend([f"- {ref}" for ref in tool_refs])

                # Collect ALL evidence without limitation
                tool_evidence = tool_result.get("evidence", {})
                if isinstance(tool_evidence, dict) and tool_evidence:
                    matching_evidence = []
                    for key, value in tool_evidence.items():
                        if (
                            topic.lower() in key.lower()
                            or not topic
                            or topic.lower() == "all"
                        ):
                            if isinstance(value, list):
                                for item in value:
                                    matching_evidence.append(f"{key}: {item}")
                            else:
                                matching_evidence.append(f"{key}: {value}")

                    if matching_evidence:
                        all_evidence.append(f"\n=== {tool_name} Evidence ===")
                        all_evidence.extend(
                            [f"- {evidence}" for evidence in matching_evidence]
                        )

            result_parts = []
            if all_evidence:
                result_parts.append("EVIDENCE:")
                result_parts.extend(all_evidence)

            if all_references:
                result_parts.append("\nREFERENCES:")
                result_parts.extend(all_references)

            return (
                "\n".join(result_parts)
                if result_parts
                else f"No evidence found for {topic}"
            )

        # Context7 MCP Tools for direct documentation access
        @self.agent.tool
        async def resolve_library_documentation(
            ctx: RunContext[ReviewDependencies], library_name: str
        ) -> LibraryResolutionResult:
            """Resolve a library name to get its Context7 library ID and metadata"""
            return resolve_library_id(library_name)

        @self.agent.tool
        async def get_documentation(
            ctx: RunContext[ReviewDependencies],
            context7_library_id: str,
            topic: Optional[str] = None,
            max_tokens: int = 2000,
        ) -> LibraryDocumentation:
            """Get up-to-date documentation for a library from Context7"""
            return get_library_docs(context7_library_id, topic, max_tokens)

        @self.agent.tool
        async def search_library_docs(
            ctx: RunContext[ReviewDependencies],
            query: str,
            libraries: Optional[List[str]] = None,
            max_results: int = 5,
        ) -> List[Dict[str, Any]]:
            """Search for documentation across multiple libraries"""
            return search_documentation(query, libraries, max_results)

        @self.agent.tool
        async def validate_code_against_docs(
            ctx: RunContext[ReviewDependencies],
            library_name: str,
            code_snippet: str,
            context: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Validate API usage against official documentation"""
            return validate_api_usage(library_name, code_snippet, context)

        @self.agent.tool
        async def get_metrics_summary(
            ctx: RunContext[ReviewDependencies],
        ) -> Dict[str, Any]:
            """Get complete summary of metrics from all tools"""
            if not ctx.deps.tool_results:
                return {"message": "No metrics available"}

            complete_metrics = {}
            total_issues = 0
            total_suggestions = 0
            confidence_scores = []

            for tool_result in ctx.deps.tool_results:
                tool_name = tool_result.get("tool_name", "unknown")

                # Collect ALL metrics without limitation
                issues = tool_result.get("issues", [])
                suggestions = tool_result.get("suggestions", [])
                positive_findings = tool_result.get("positive_findings", [])
                metrics = tool_result.get("metrics", {})
                confidence = tool_result.get("confidence_score", 0)
                execution_time = tool_result.get("execution_time_ms", 0)
                cached = tool_result.get("cached", False)

                complete_metrics[tool_name] = {
                    "issues_found": len(issues),
                    "suggestions_count": len(suggestions),
                    "positive_findings_count": len(positive_findings),
                    "confidence_score": confidence,
                    "execution_time_ms": execution_time,
                    "was_cached": cached,
                    "detailed_metrics": metrics,  # Include ALL detailed metrics
                    "all_issues": issues,  # Include complete issue details
                    "all_suggestions": suggestions,  # Include complete suggestions
                    "all_positive_findings": positive_findings,  # Include all positive findings
                }

                total_issues += len(issues)
                total_suggestions += len(suggestions)
                if confidence > 0:
                    confidence_scores.append(confidence)

            # Calculate overall statistics
            complete_metrics["summary"] = {
                "total_issues": total_issues,
                "total_suggestions": total_suggestions,
                "tools_executed": len(ctx.deps.tool_results),
                "average_confidence": sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0,
                "min_confidence": min(confidence_scores) if confidence_scores else 0,
                "max_confidence": max(confidence_scores) if confidence_scores else 0,
            }

            return complete_metrics

        @self.agent.tool
        async def get_all_tool_results(
            ctx: RunContext[ReviewDependencies],
        ) -> Dict[str, Any]:
            """Get complete, unfiltered tool results"""
            if not ctx.deps.tool_results:
                return {"message": "No tool results available"}

            return {
                "complete_tool_results": ctx.deps.tool_results,
                "total_tools": len(ctx.deps.tool_results),
                "execution_summary": {
                    tool_result.get("tool_name", f"tool_{i}"): {
                        "success": tool_result.get("success", False),
                        "category": tool_result.get("category", "unknown"),
                        "execution_time_ms": tool_result.get("execution_time_ms", 0),
                        "cached": tool_result.get("cached", False),
                        "error_message": tool_result.get("error_message"),
                    }
                    for i, tool_result in enumerate(ctx.deps.tool_results)
                },
            }

        @self.agent.tool
        async def check_context7_availability_status(
            ctx: RunContext[ReviewDependencies],
        ) -> Dict[str, Any]:
            """Check Context7 MCP availability status and report limitations to user"""
            if not ctx.deps.tool_results:
                return {"message": "No tool results available to check Context7 status"}

            context7_status: Dict[str, Any] = {
                "overall_available": True,
                "unavailable_libraries": [],
                "available_libraries": [],
                "limitations": [],
                "user_message": "",
            }

            # Check for Context7 availability information in tool results
            for tool_result in ctx.deps.tool_results:
                evidence = tool_result.get("evidence", {})

                # Check for Context7 unavailability evidence
                if "context7_unavailability" in evidence:
                    unavailability_info = evidence["context7_unavailability"]
                    unavailable_libs = unavailability_info.get(
                        "unavailable_libraries", []
                    )

                    for lib_info in unavailable_libs:
                        context7_status["unavailable_libraries"].append(
                            f"{lib_info['library']}: {lib_info['reason']}"
                        )

                    context7_status["limitations"].append(
                        unavailability_info.get("impact", "")
                    )
                    context7_status["overall_available"] = False

                # Check metrics for Context7 availability
                metrics = tool_result.get("metrics", {})
                if "context7_availability_rate" in metrics:
                    availability_rate = metrics["context7_availability_rate"]
                    validated_libs = metrics.get("libraries_validated_with_context7", 0)
                    unavailable_libs_count = metrics.get(
                        "libraries_unavailable_context7", 0
                    )

                    if availability_rate < 1.0:
                        context7_status["overall_available"] = False
                        context7_status["limitations"].append(
                            f"Context7 availability: {availability_rate:.0%} ({validated_libs} available, {unavailable_libs_count} unavailable)"
                        )

            # Generate user-friendly message
            if context7_status["overall_available"]:
                context7_status[
                    "user_message"
                ] = "✅ Context7 MCP documentation validation is fully available for this review."
            else:
                limitations_text = "\n".join(
                    [f"- {limitation}" for limitation in context7_status["limitations"]]
                )
                unavailable_text = "\n".join(
                    [f"- {lib}" for lib in context7_status["unavailable_libraries"]]
                )

                context7_status[
                    "user_message"
                ] = f"""⚠️ Context7 MCP Documentation Validation Status:

Context7 documentation validation was not fully available for this review.

Limitations:
{limitations_text}

Libraries affected:
{unavailable_text}

Impact: The code review for affected libraries is based on the AI agent's training knowledge only, without real-time documentation verification. The review quality remains high, but lacks the most current documentation validation for these specific libraries."""

            return context7_status

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((Exception,)),  # Retry on most exceptions
    )
    async def review_merge_request(
        self, diff_content: str, context: ReviewContext
    ) -> ReviewResult:
        """Perform comprehensive, tool-enhanced code review on merge request with retry logic"""

        settings = get_settings()
        logger.info(f"Starting tool-enhanced review for MR {context.merge_request_iid}")

        # Execute tools first to gather evidence-based insights
        tool_results: List[Dict[str, Any]] = []
        if settings.tools_enabled:
            logger.info("Executing analysis tools...")

            # Create tool context
            tool_context = ToolContext(
                diff_content=diff_content,
                file_changes=context.file_changes,
                source_branch=context.source_branch,
                target_branch=context.target_branch,
                repository_url=context.repository_url,
                project_id=getattr(context, "project_id", None),
                merge_request_iid=context.merge_request_iid,
                settings={
                    "context7": {
                        "enabled": settings.context7_enabled,
                        "max_tokens": settings.context7_max_tokens,
                        "cache_ttl": settings.context7_cache_ttl,
                    },
                    "tools": {
                        "timeout": settings.tools_timeout,
                        "cache_enabled": settings.tools_cache_enabled,
                        "cache_ttl": settings.tools_cache_ttl,
                    },
                },
            )

            try:
                # Execute all enabled tools
                raw_tool_results = await self.tool_registry.execute_tools(
                    context=tool_context, parallel=settings.tools_parallel_execution
                )

                # Convert tool results to serializable format
                tool_results = []
                for tool_result in raw_tool_results:
                    tool_data = {
                        "tool_name": tool_result.tool_name,
                        "category": tool_result.category.value,
                        "success": tool_result.success,
                        "issues": tool_result.issues,
                        "suggestions": tool_result.suggestions,
                        "positive_findings": tool_result.positive_findings,
                        "evidence": tool_result.evidence,
                        "references": tool_result.references,
                        "metrics": tool_result.metrics,
                        "confidence_score": tool_result.confidence_score,
                        "execution_time_ms": tool_result.execution_time_ms,
                        "cached": tool_result.cached,
                        "error_message": tool_result.error_message,
                        "partial_result": tool_result.partial_result,
                    }
                    tool_results.append(tool_data)

                logger.info(f"Tools executed: {len(tool_results)} tools completed")

                # Log tool execution summary
                successful_tools = [r for r in tool_results if r["success"]]
                failed_tools = [r for r in tool_results if not r["success"]]
                total_issues = sum(
                    len(r["issues"]) if isinstance(r["issues"], list) else 0
                    for r in successful_tools
                )

                logger.info(
                    f"Tool execution summary: {len(successful_tools)} successful, "
                    f"{len(failed_tools)} failed, {total_issues} total issues found"
                )

            except Exception as e:
                logger.warning(f"Tool execution failed: {e}")
                # Continue with review even if tools fail
                tool_results = []
        else:
            logger.info("Tool system disabled, proceeding with basic review")

        # Prepare dependencies with tool results
        deps = ReviewDependencies(
            repository_url=context.repository_url,
            branch=context.target_branch,
            merge_request_iid=context.merge_request_iid,
            gitlab_token=settings.gitlab_token,
            diff_content=diff_content,
            file_changes=context.file_changes,
            review_trigger_tag=context.trigger_tag,
            tool_results=tool_results,
        )

        # Construct enhanced review prompt with tool integration instructions
        review_prompt = f"""
        Please review the following code changes from a GitLab merge request using both the provided tool analysis and your expertise.

        Repository: {context.repository_url}
        Target Branch: {context.target_branch}
        Source Branch: {context.source_branch}

        DIFF CONTENT:
        {diff_content}

        TOOL ANALYSIS AVAILABLE:
        You have access to Context7-based analysis tools through get_tool_insights, get_evidence_references,
        get_metrics_summary, check_context7_availability_status, and get_all_tool_results functions.

        Tool execution completed: {len(tool_results)} tools ran
        Successful analyses: {len([r for r in tool_results if r["success"]])} tools completed successfully
        Tool categories executed: {", ".join(list(set(str(r["category"]) for r in tool_results if r["success"]))) if [r for r in tool_results if r["success"]] else "No successful tool executions"}

        Context7 Status: {'Available' if any('context7_availability_rate' in r.get('metrics', {}) and r['metrics']['context7_availability_rate'] > 0 for r in tool_results) else 'Check required - use check_context7_availability_status() first'}

        INSTRUCTIONS:
        1. FIRST: Use check_context7_availability_status() to check Context7 MCP status and inform the user of any limitations
        2. Use get_tool_insights() for documentation analysis (the primary tool category in this simplified architecture)
        3. Use get_evidence_references() to find Context7 documentation and evidence
        4. Use get_metrics_summary() to get Context7 availability metrics and library validation status
        5. If Context7 is available, leverage real-time documentation validation results
        6. If Context7 is unavailable, clearly explain limitations and rely on your training knowledge
        7. Integrate Context7 findings (when available) with your own analysis
        8. Prioritize issues backed by Context7 documentation evidence when available

        Provide a comprehensive review focusing on:
        1. Context7 availability status and any limitations (always start with this)
        2. Critical issues that must be fixed (prioritize Context7-validated issues when available)
        3. Library usage validation against official documentation (when Context7 is available)
        4. Security vulnerabilities and best practices (leverage your knowledge + Context7 evidence)
        5. Performance concerns and optimization opportunities
        6. Code quality and maintainability improvements
        7. Positive aspects worth highlighting

        IMPORTANT:
        - Always be transparent about Context7 availability status
        - Include ALL Context7 findings, evidence, and references when available
        - When Context7 is unavailable, clearly state review limitations
        - Provide thorough analysis using your training knowledge regardless of Context7 status
        - Do not summarize or truncate any available Context7 results
        """

        try:
            # Run the enhanced review agent
            result = await self.agent.run(review_prompt, deps=deps)

            # Log token usage for monitoring
            if hasattr(result, "usage"):
                usage = result.usage()
                logger.info(
                    f"Review completed. Tokens used: {usage.total_tokens if hasattr(usage, 'total_tokens') else 'unknown'}"
                )

            # Validate result has expected output
            if not hasattr(result, "output"):
                raise AIProviderException(
                    message="AI provider returned unexpected result structure",
                    provider=self.model_name,
                    details={"result_type": type(result).__name__},
                )

            # The output should be a ReviewResult from PydanticAI
            return result.output  # type: ignore[return-value]

        except AIProviderException:
            # Don't wrap AI provider exceptions
            raise
        except Exception as e:
            logger.error(f"Review failed for MR {context.merge_request_iid}: {e}")
            raise ReviewProcessException(
                message=f"Review process failed for MR {context.merge_request_iid}",
                merge_request_iid=context.merge_request_iid,
                details={"model_name": self.model_name},
                original_error=e,
            )


async def initialize_review_agent() -> CodeReviewAgent:
    """Factory function to initialize the review agent"""
    try:
        settings = get_settings()
        agent = CodeReviewAgent(model_name=settings.ai_model)
        logger.info(
            f"Review agent initialized successfully with model: {settings.ai_model}"
        )
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize review agent: {e}")
        settings = get_settings()
        raise ReviewProcessException(
            message="Failed to initialize AI review agent",
            details={"model_name": settings.ai_model},
            original_error=e,
        )
