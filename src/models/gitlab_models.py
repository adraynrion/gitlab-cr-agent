"""
GitLab webhook payload models
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class GitLabUser(BaseModel):
    """GitLab user model"""

    id: int
    name: str
    username: str
    email: str
    avatar_url: Optional[str] = None


class GitLabLabel(BaseModel):
    """GitLab label model"""

    id: int
    title: str
    color: str
    project_id: Optional[int] = None
    group_id: Optional[int] = None
    description: Optional[str] = None


class GitLabProject(BaseModel):
    """GitLab project model"""

    id: int
    name: str
    description: Optional[str] = None
    web_url: str
    avatar_url: Optional[str] = None
    git_ssh_url: str
    git_http_url: str
    namespace: str
    visibility_level: int
    path_with_namespace: str
    default_branch: str


class MergeRequestAttributes(BaseModel):
    """Merge request attributes from webhook"""

    id: int
    iid: int
    title: str
    description: Optional[str] = None
    state: str
    created_at: str  # GitLab sends as string, we'll parse it
    updated_at: str  # GitLab sends as string, we'll parse it
    target_branch: str
    source_branch: str
    source_project_id: int
    target_project_id: int
    author_id: int
    assignee_id: Optional[int] = None
    url: str
    source: Dict[str, Any]
    target: Dict[str, Any]
    last_commit: Dict[str, Any]
    work_in_progress: bool = False
    assignee: Optional[GitLabUser] = None
    labels: List[GitLabLabel] = Field(
        default_factory=list
    )  # GitLab sends label objects
    action: str

    @field_validator("created_at", "updated_at")
    @classmethod
    def parse_gitlab_datetime(cls, v: str) -> datetime:
        """Parse GitLab datetime format: '2025-08-11 16:33:01 UTC'"""
        if isinstance(v, str):
            # Remove 'UTC' and parse
            clean_dt = v.replace(" UTC", "")
            return datetime.fromisoformat(clean_dt.replace(" ", "T"))
        return v


class GitLabWebhookPayload(BaseModel):
    """Base GitLab webhook payload"""

    object_kind: str
    event_type: Optional[str] = None
    user: GitLabUser
    project: GitLabProject
    repository: Dict[str, Any]


class MergeRequestEvent(GitLabWebhookPayload):
    """Merge request webhook event"""

    object_attributes: MergeRequestAttributes
    labels: List[Dict[str, Any]] = Field(default_factory=list)
    changes: Optional[Dict[str, Any]] = None
