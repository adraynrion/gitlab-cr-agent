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

TOOL-ENHANCED ANALYSIS:
You have access to specialized tools that provide evidence-based insights:
- Documentation lookup tools that verify API usage against official documentation
- Security pattern validators that check against known vulnerabilities
- Performance analyzers that detect common anti-patterns
- Code quality tools that assess maintainability metrics
- Framework-specific validators for best practices

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
                import src.agents.tools.analysis_tools  # noqa: F401
                import src.agents.tools.context_tools  # noqa: F401
                import src.agents.tools.validation_tools  # noqa: F401

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
        You have access to specialized analysis tools through the get_tool_insights, get_evidence_references,
        get_metrics_summary, and get_all_tool_results functions. These tools have already analyzed the code
        and found {len([r for r in tool_results if r["success"]])} successful analyses covering:
        {", ".join(list(set(str(r["category"]) for r in tool_results if r["success"])))}

        INSTRUCTIONS:
        1. Use get_tool_insights() for each relevant category (security, performance, correctness, maintainability, etc.)
        2. Use get_evidence_references() to find supporting documentation and evidence
        3. Use get_metrics_summary() to get quantitative analysis
        4. Integrate tool findings with your own analysis
        5. Prioritize issues backed by tool evidence
        6. Include all references and evidence in your review

        Provide a comprehensive review focusing on:
        1. Critical issues that must be fixed (prioritize tool-detected issues)
        2. Security vulnerabilities (use security tool insights)
        3. Performance concerns (use performance tool insights)
        4. Code quality and maintainability (use quality tool insights)
        5. Best practices and documentation compliance (use evidence and references)
        6. Positive aspects worth highlighting

        IMPORTANT: Include ALL tool findings, evidence, and references without limitation.
        Do not summarize or truncate tool results - provide complete information.
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
