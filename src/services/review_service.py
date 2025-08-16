"""
Review service for orchestrating code reviews
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Circuit breaker import temporarily disabled due to compatibility issues
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.code_reviewer import CodeReviewAgent
from src.api.middleware import get_correlation_id, get_request_id
from src.config.settings import settings
from src.exceptions import (
    AIProviderException,
    GitLabAPIException,
    RateLimitException,
    ReviewProcessException,
)
from src.models.gitlab_models import MergeRequestEvent
from src.models.review_models import CodeIssue, ReviewContext, ReviewResult

logger = logging.getLogger(__name__)


class ReviewService:
    """Service for orchestrating AI code reviews with enhanced retry and rate limiting"""

    def __init__(self, review_agent: Optional[CodeReviewAgent] = None):
        self.review_agent = review_agent or CodeReviewAgent()

        # Circuit breaker for AI provider calls temporarily disabled due to compatibility issues

        # Rate limiter state
        self._last_ai_call = 0.0
        self._min_ai_call_interval = 2.0  # Minimum 2 seconds between AI calls

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between AI provider calls"""
        import time

        current_time = time.time()
        time_since_last_call = current_time - self._last_ai_call

        if time_since_last_call < self._min_ai_call_interval:
            sleep_time = self._min_ai_call_interval - time_since_last_call
            logger.info(
                f"Rate limiting: waiting {sleep_time:.2f}s before next AI call",
                extra={
                    "correlation_id": get_correlation_id(),
                    "request_id": get_request_id(),
                    "operation": "rate_limit_wait",
                    "sleep_time": sleep_time,
                },
            )
            await asyncio.sleep(sleep_time)

        self._last_ai_call = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((AIProviderException, ConnectionError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _call_ai_with_retry(
        self, diff_content: str, review_context: ReviewContext
    ) -> ReviewResult:
        """Call AI provider with enhanced retry logic and circuit breaker protection"""
        try:
            # Enforce rate limiting
            await self._enforce_rate_limit()

            # Direct AI call (circuit breaker temporarily disabled)
            result: ReviewResult = await self.review_agent.review_merge_request(
                diff_content=diff_content, context=review_context
            )

            logger.info(
                "AI review completed successfully",
                extra={
                    "correlation_id": get_correlation_id(),
                    "request_id": get_request_id(),
                    "operation": "ai_review_success",
                    "issues_found": len(result.issues)
                    if hasattr(result, "issues")
                    else 0,
                },
            )

            return result

        # Circuit breaker exception handling temporarily disabled
        except Exception as e:
            logger.error(
                f"AI provider call failed: {e}",
                extra={
                    "correlation_id": get_correlation_id(),
                    "request_id": get_request_id(),
                    "operation": "ai_call_failed",
                    "error_type": type(e).__name__,
                },
            )
            raise

    async def review_merge_request(
        self,
        mr_details: Dict[str, Any],
        mr_diff: List[Dict[str, Any]],
        mr_event: MergeRequestEvent,
    ) -> ReviewResult:
        """
        Orchestrate complete merge request review process

        Args:
            mr_details: GitLab MR details from API
            mr_diff: GitLab MR diff from API
            mr_event: Webhook event data

        Returns:
            ReviewResult with comprehensive analysis
        """
        logger.info(
            f"Starting review orchestration for MR {mr_event.object_attributes.iid}",
            extra={
                "correlation_id": get_correlation_id(),
                "request_id": get_request_id(),
                "project_id": mr_event.project.id,
                "mr_iid": mr_event.object_attributes.iid,
                "operation": "review_orchestration_start",
            },
        )

        try:
            # Convert diff to readable format
            diff_content = self._format_diff_content(mr_diff)

            # Create review context
            # Extract trigger tag from labels (find the first matching trigger tag)
            trigger_tag = "ai-review"  # default
            if mr_event.object_attributes.labels:
                # Use the first label's title as trigger tag, or find the specific trigger tag
                label_titles = [
                    label.title for label in mr_event.object_attributes.labels
                ]

                if settings.gitlab_trigger_tag in label_titles:
                    trigger_tag = settings.gitlab_trigger_tag
                else:
                    trigger_tag = label_titles[0]  # fallback to first label

            context = ReviewContext(
                repository_url=mr_event.project.web_url,
                merge_request_iid=mr_event.object_attributes.iid,
                source_branch=mr_event.object_attributes.source_branch,
                target_branch=mr_event.object_attributes.target_branch,
                trigger_tag=trigger_tag,
                file_changes=mr_diff,
            )

            # Execute AI review with enhanced retry and rate limiting
            review_result = await self._call_ai_with_retry(diff_content, context)

            # Add metadata
            review_result.metrics.update(
                {
                    "files_changed": len(mr_diff),
                    "review_timestamp": datetime.utcnow().isoformat(),
                    "mr_iid": mr_event.object_attributes.iid,
                    "project_id": mr_event.project.id,
                }
            )

            logger.info(
                f"Review completed for MR {mr_event.object_attributes.iid}",
                extra={
                    "correlation_id": get_correlation_id(),
                    "request_id": get_request_id(),
                    "project_id": mr_event.project.id,
                    "mr_iid": mr_event.object_attributes.iid,
                    "operation": "review_orchestration_success",
                    "issues_found": len(review_result.issues)
                    if hasattr(review_result, "issues")
                    else 0,
                },
            )
            return review_result

        except (GitLabAPIException, AIProviderException) as e:
            logger.error(
                f"Review orchestration failed for MR {mr_event.object_attributes.iid}: {e.message}",
                extra={
                    "correlation_id": get_correlation_id(),
                    "request_id": get_request_id(),
                    "project_id": mr_event.project.id,
                    "mr_iid": mr_event.object_attributes.iid,
                    "operation": "review_orchestration_failed",
                    "error_type": type(e).__name__,
                },
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in review orchestration for MR {mr_event.object_attributes.iid}: {e}",
                extra={
                    "correlation_id": get_correlation_id(),
                    "request_id": get_request_id(),
                    "project_id": mr_event.project.id,
                    "mr_iid": mr_event.object_attributes.iid,
                    "operation": "review_orchestration_error",
                    "error_type": type(e).__name__,
                },
            )
            raise ReviewProcessException(
                message=f"Review orchestration failed for MR {mr_event.object_attributes.iid}",
                merge_request_iid=mr_event.object_attributes.iid,
                project_id=mr_event.project.id,
                original_error=e,
            )

    def _format_diff_content(self, mr_diff: List[Dict[str, Any]]) -> str:
        """
        Format GitLab diff data into readable text format with size validation

        Args:
            mr_diff: Raw GitLab diff data

        Returns:
            Formatted diff string

        Raises:
            ReviewProcessException: If diff content exceeds maximum allowed size
        """
        # Calculate total size before processing to prevent memory issues
        total_size = 0
        for diff_item in mr_diff:
            if diff_item.get("diff"):
                total_size += len(str(diff_item["diff"]))
                total_size += len(str(diff_item.get("old_path", "")))
                total_size += len(str(diff_item.get("new_path", "")))

        # Check against configurable maximum diff size
        if total_size > settings.max_diff_size:
            logger.warning(
                f"Diff size ({total_size} bytes) exceeds maximum allowed size "
                f"({settings.max_diff_size} bytes)"
            )
            raise ReviewProcessException(
                message=f"Diff too large for processing: {total_size} bytes "
                f"(maximum: {settings.max_diff_size} bytes)",
                details={
                    "diff_size_bytes": total_size,
                    "max_allowed_bytes": settings.max_diff_size,
                    "files_count": len(mr_diff),
                },
            )

        formatted_diff = []
        for diff_item in mr_diff:
            if diff_item.get("diff"):
                formatted_diff.append(f"--- {diff_item.get('old_path', 'unknown')}")
                formatted_diff.append(f"+++ {diff_item.get('new_path', 'unknown')}")
                formatted_diff.append(diff_item["diff"])
                formatted_diff.append("")  # Add spacing between files

        result = "\n".join(formatted_diff)
        logger.debug(
            f"Formatted diff content: {len(result)} bytes from {len(mr_diff)} files"
        )
        return result

    def format_review_comment(self, review_result: ReviewResult) -> str:
        """
        Format review result as a GitLab comment

        Args:
            review_result: AI review result

        Returns:
            Formatted markdown comment
        """
        # Determine emoji and color based on assessment
        assessment_config = {
            "approve": {"emoji": "✅", "color": "green"},
            "approve_with_changes": {"emoji": "⚠️", "color": "orange"},
            "needs_work": {"emoji": "❌", "color": "red"},
            "reject": {"emoji": "🚫", "color": "red"},
        }

        config = assessment_config.get(
            review_result.overall_assessment, {"emoji": "🔍", "color": "blue"}
        )

        # Build comment header
        comment_lines = [
            f"## {config['emoji']} AI Code Review",
            "",
            f"**Overall Assessment:** {review_result.overall_assessment.replace('_', ' ').title()}",
            f"**Risk Level:** {review_result.risk_level.title()}",
            "",
            "### Summary",
            review_result.summary,
            "",
        ]

        # Add issues section if any
        if review_result.issues:
            comment_lines.extend(
                [f"### Issues Found ({len(review_result.issues)})", ""]
            )

            # Group issues by severity
            issues_by_severity: Dict[str, List[CodeIssue]] = {}
            for issue in review_result.issues:
                if issue.severity not in issues_by_severity:
                    issues_by_severity[issue.severity] = []
                issues_by_severity[issue.severity].append(issue)

            # Add issues in order of severity
            for severity in ["critical", "high", "medium", "low"]:
                if severity in issues_by_severity:
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟡",
                        "medium": "🟠",
                        "low": "🔵",
                    }

                    comment_lines.append(
                        f"#### {severity_emoji[severity]} {severity.title()} Issues"
                    )
                    comment_lines.append("")

                    for issue in issues_by_severity[severity]:
                        comment_lines.extend(
                            [
                                f"**{issue.file_path}:{issue.line_number}** - {issue.category.title()}",
                                issue.description,
                                "",
                                f"💡 **Suggestion:** {issue.suggestion}",
                                "",
                            ]
                        )

                        if issue.code_example:
                            comment_lines.extend(
                                ["```python", issue.code_example, "```", ""]
                            )

        # Add positive feedback section
        if review_result.positive_feedback:
            comment_lines.extend(["### ✨ Positive Feedback", ""])

            for feedback in review_result.positive_feedback:
                comment_lines.append(f"- {feedback}")

            comment_lines.append("")

        # Add metrics footer
        if review_result.metrics:
            comment_lines.extend(["### 📊 Review Metrics", ""])

            for key, value in review_result.metrics.items():
                if key != "review_timestamp":  # Skip internal timestamp
                    formatted_key = key.replace("_", " ").title()
                    comment_lines.append(f"- **{formatted_key}:** {value}")

        # Add footer
        comment_lines.extend(
            ["", "---", "🤖 *Generated by GitLab AI Code Review Agent*"]
        )

        return "\n".join(comment_lines)
