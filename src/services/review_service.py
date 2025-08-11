"""
Review service for orchestrating code reviews
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from src.agents.code_reviewer import CodeReviewAgent
from src.models.review_models import ReviewResult, ReviewContext, CodeIssue
from src.models.gitlab_models import MergeRequestEvent
from src.services.gitlab_service import GitLabService
from src.exceptions import ReviewProcessException, GitLabAPIException, AIProviderException

logger = logging.getLogger(__name__)

class ReviewService:
    """Service for orchestrating AI code reviews"""
    
    def __init__(self, review_agent: CodeReviewAgent = None):
        self.review_agent = review_agent or CodeReviewAgent()
    
    async def review_merge_request(
        self,
        mr_details: Dict[str, Any],
        mr_diff: List[Dict[str, Any]], 
        mr_event: MergeRequestEvent
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
        logger.info(f"Starting review orchestration for MR {mr_event.object_attributes.iid}")
        
        try:
            # Convert diff to readable format
            diff_content = self._format_diff_content(mr_diff)
            
            # Create review context
            # Extract trigger tag from labels (find the first matching trigger tag)
            trigger_tag = "ai-review"  # default
            if mr_event.object_attributes.labels:
                # Use the first label's title as trigger tag, or find the specific trigger tag
                label_titles = [label.title for label in mr_event.object_attributes.labels]
                from src.config.settings import settings
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
                file_changes=mr_diff
            )
            
            # Execute AI review
            review_result = await self.review_agent.review_merge_request(
                diff_content=diff_content,
                context=context
            )
            
            # Add metadata
            review_result.metrics.update({
                "files_changed": len(mr_diff),
                "review_timestamp": datetime.utcnow().isoformat(),
                "mr_iid": mr_event.object_attributes.iid,
                "project_id": mr_event.project.id
            })
            
            logger.info(f"Review completed for MR {mr_event.object_attributes.iid}")
            return review_result
            
        except (GitLabAPIException, AIProviderException) as e:
            logger.error(f"Review orchestration failed for MR {mr_event.object_attributes.iid}: {e.message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in review orchestration for MR {mr_event.object_attributes.iid}: {e}")
            raise ReviewProcessException(
                message=f"Review orchestration failed for MR {mr_event.object_attributes.iid}",
                merge_request_iid=mr_event.object_attributes.iid,
                project_id=mr_event.project.id,
                original_error=e
            )
    
    def _format_diff_content(self, mr_diff: List[Dict[str, Any]]) -> str:
        """
        Format GitLab diff data into readable text format
        
        Args:
            mr_diff: Raw GitLab diff data
            
        Returns:
            Formatted diff string
        """
        formatted_diff = []
        
        for diff_item in mr_diff:
            if diff_item.get("diff"):
                formatted_diff.append(f"--- {diff_item.get('old_path', 'unknown')}")
                formatted_diff.append(f"+++ {diff_item.get('new_path', 'unknown')}")
                formatted_diff.append(diff_item["diff"])
                formatted_diff.append("")  # Add spacing between files
        
        return "\n".join(formatted_diff)
    
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
            "reject": {"emoji": "🚫", "color": "red"}
        }
        
        config = assessment_config.get(review_result.overall_assessment, {"emoji": "🔍", "color": "blue"})
        
        # Build comment header
        comment_lines = [
            f"## {config['emoji']} AI Code Review",
            f"",
            f"**Overall Assessment:** {review_result.overall_assessment.replace('_', ' ').title()}",
            f"**Risk Level:** {review_result.risk_level.title()}",
            f"",
            f"### Summary",
            f"{review_result.summary}",
            f""
        ]
        
        # Add issues section if any
        if review_result.issues:
            comment_lines.extend([
                f"### Issues Found ({len(review_result.issues)})",
                f""
            ])
            
            # Group issues by severity
            issues_by_severity = {}
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
                        "low": "🔵"
                    }
                    
                    comment_lines.append(f"#### {severity_emoji[severity]} {severity.title()} Issues")
                    comment_lines.append("")
                    
                    for issue in issues_by_severity[severity]:
                        comment_lines.extend([
                            f"**{issue.file_path}:{issue.line_number}** - {issue.category.title()}",
                            f"{issue.description}",
                            f"",
                            f"💡 **Suggestion:** {issue.suggestion}",
                            f""
                        ])
                        
                        if issue.code_example:
                            comment_lines.extend([
                                f"```python",
                                issue.code_example,
                                f"```",
                                f""
                            ])
        
        # Add positive feedback section
        if review_result.positive_feedback:
            comment_lines.extend([
                f"### ✨ Positive Feedback",
                f""
            ])
            
            for feedback in review_result.positive_feedback:
                comment_lines.append(f"- {feedback}")
            
            comment_lines.append("")
        
        # Add metrics footer
        if review_result.metrics:
            comment_lines.extend([
                f"### 📊 Review Metrics",
                f""
            ])
            
            for key, value in review_result.metrics.items():
                if key != "review_timestamp":  # Skip internal timestamp
                    formatted_key = key.replace("_", " ").title()
                    comment_lines.append(f"- **{formatted_key}:** {value}")
        
        # Add footer
        comment_lines.extend([
            f"",
            f"---",
            f"🤖 *Generated by GitLab AI Code Review Agent*"
        ])
        
        return "\n".join(comment_lines)