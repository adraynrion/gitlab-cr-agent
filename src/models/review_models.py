"""
Data models for code review operations
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class CodeIssue(BaseModel):
    """Individual code issue found during review"""

    file_path: str = Field(..., description="Path to the file containing the issue")
    line_number: int = Field(..., description="Line number where issue occurs")
    severity: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Severity level of the issue"
    )
    category: Literal[
        "security", "performance", "correctness", "style", "maintainability"
    ] = Field(..., description="Category of the issue")
    description: str = Field(..., description="Detailed description of the issue")
    suggestion: str = Field(..., description="Suggested fix or improvement")
    code_example: Optional[str] = Field(None, description="Example of corrected code")


class ReviewResult(BaseModel):
    """Complete code review result"""

    overall_assessment: Literal[
        "approve", "approve_with_changes", "needs_work", "reject"
    ] = Field(..., description="Overall recommendation for the merge request")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Overall risk assessment"
    )
    summary: str = Field(..., description="Executive summary of the review")
    issues: List[CodeIssue] = Field(
        default_factory=list, description="List of identified issues"
    )
    positive_feedback: List[str] = Field(
        default_factory=list, description="Positive aspects worth highlighting"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Review metrics and statistics"
    )


class ReviewContext(BaseModel):
    """Context information for code review"""

    repository_url: str
    merge_request_iid: int
    source_branch: str
    target_branch: str
    trigger_tag: str
    file_changes: List[Dict[str, Any]]
