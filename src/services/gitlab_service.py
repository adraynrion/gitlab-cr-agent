"""
GitLab API integration service
"""

import httpx
from typing import Dict, Any, List, Optional
import logging

from src.config.settings import settings

logger = logging.getLogger(__name__)

class GitLabService:
    """Service for interacting with GitLab API"""
    
    def __init__(self):
        self.base_url = f"{settings.gitlab_url}/api/v4"
        self.headers = {
            "PRIVATE-TOKEN": settings.gitlab_token,
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=30.0
        )
    
    async def get_merge_request(
        self,
        project_id: int,
        mr_iid: int
    ) -> Dict[str, Any]:
        """Fetch merge request details"""
        try:
            response = await self.client.get(
                f"/projects/{project_id}/merge_requests/{mr_iid}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch MR {mr_iid}: {e}")
            raise
    
    async def get_merge_request_diff(
        self,
        project_id: int,
        mr_iid: int
    ) -> List[Dict[str, Any]]:
        """Fetch merge request diff"""
        try:
            response = await self.client.get(
                f"/projects/{project_id}/merge_requests/{mr_iid}/diffs"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch MR diff for {mr_iid}: {e}")
            raise
    
    async def post_merge_request_comment(
        self,
        project_id: int,
        mr_iid: int,
        comment: str
    ) -> Dict[str, Any]:
        """Post a comment on a merge request"""
        try:
            response = await self.client.post(
                f"/projects/{project_id}/merge_requests/{mr_iid}/notes",
                json={"body": comment}
            )
            response.raise_for_status()
            logger.info(f"Posted comment to MR {mr_iid}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to post comment to MR {mr_iid}: {e}")
            raise
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()