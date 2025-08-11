"""
GitLab API integration service
"""

import httpx
from typing import Dict, Any, List
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.settings import settings
from src.exceptions import GitLabAPIException

logger = logging.getLogger(__name__)


class GitLabService:
    """Service for interacting with GitLab API"""

    def __init__(self):
        self.base_url = f"{settings.gitlab_url}/api/v4"
        self.headers = {
            "PRIVATE-TOKEN": settings.gitlab_token,
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(
            base_url=self.base_url, headers=self.headers, timeout=30.0
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    )
    async def get_merge_request(self, project_id: int, mr_iid: int) -> Dict[str, Any]:
        """Fetch merge request details with retry logic"""
        try:
            response = await self.client.get(
                f"/projects/{project_id}/merge_requests/{mr_iid}"
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.RequestError as e:
            logger.error(f"Network error fetching MR {mr_iid}: {e}")
            raise GitLabAPIException(
                message=f"Network error fetching merge request {mr_iid}",
                details={"project_id": project_id, "mr_iid": mr_iid},
                original_error=e,
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                f"GitLab API error {e.response.status_code} for MR {mr_iid}: {e.response.text}"
            )
            raise GitLabAPIException(
                message=f"GitLab API error for merge request {mr_iid}",
                status_code=e.response.status_code,
                response_body=e.response.text,
                details={"project_id": project_id, "mr_iid": mr_iid},
                original_error=e,
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching MR {mr_iid}: {e}")
            raise GitLabAPIException(
                message=f"Unexpected error fetching merge request {mr_iid}",
                details={"project_id": project_id, "mr_iid": mr_iid},
                original_error=e,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    )
    async def get_merge_request_diff(
        self, project_id: int, mr_iid: int
    ) -> List[Dict[str, Any]]:
        """Fetch merge request diff with retry logic"""
        try:
            response = await self.client.get(
                f"/projects/{project_id}/merge_requests/{mr_iid}/diffs"
            )
            response.raise_for_status()
            result: list[dict[str, Any]] = response.json()
            return result
        except httpx.RequestError as e:
            logger.error(f"Network error fetching MR diff for {mr_iid}: {e}")
            raise GitLabAPIException(
                message=f"Network error fetching MR diff for {mr_iid}",
                details={"project_id": project_id, "mr_iid": mr_iid},
                original_error=e,
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                f"GitLab API error {e.response.status_code} for MR diff {mr_iid}: {e.response.text}"
            )
            raise GitLabAPIException(
                message=f"GitLab API error for MR diff {mr_iid}",
                status_code=e.response.status_code,
                response_body=e.response.text,
                details={"project_id": project_id, "mr_iid": mr_iid},
                original_error=e,
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching MR diff for {mr_iid}: {e}")
            raise GitLabAPIException(
                message=f"Unexpected error fetching MR diff for {mr_iid}",
                details={"project_id": project_id, "mr_iid": mr_iid},
                original_error=e,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    )
    async def post_merge_request_comment(
        self, project_id: int, mr_iid: int, comment: str
    ) -> Dict[str, Any]:
        """Post a comment on a merge request with retry logic"""
        try:
            response = await self.client.post(
                f"/projects/{project_id}/merge_requests/{mr_iid}/notes",
                json={"body": comment},
            )
            response.raise_for_status()
            logger.info(f"Posted comment to MR {mr_iid}")
            result: dict[str, Any] = response.json()
            return result
        except httpx.RequestError as e:
            logger.error(f"Network error posting comment to MR {mr_iid}: {e}")
            raise GitLabAPIException(
                message=f"Network error posting comment to MR {mr_iid}",
                details={
                    "project_id": project_id,
                    "mr_iid": mr_iid,
                    "comment_length": len(comment),
                },
                original_error=e,
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                f"GitLab API error {e.response.status_code} posting comment to MR {mr_iid}: {e.response.text}"
            )
            raise GitLabAPIException(
                message=f"GitLab API error posting comment to MR {mr_iid}",
                status_code=e.response.status_code,
                response_body=e.response.text,
                details={
                    "project_id": project_id,
                    "mr_iid": mr_iid,
                    "comment_length": len(comment),
                },
                original_error=e,
            )
        except Exception as e:
            logger.error(f"Unexpected error posting comment to MR {mr_iid}: {e}")
            raise GitLabAPIException(
                message=f"Unexpected error posting comment to MR {mr_iid}",
                details={
                    "project_id": project_id,
                    "mr_iid": mr_iid,
                    "comment_length": len(comment),
                },
                original_error=e,
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
            logger.info("GitLab service HTTP client closed")
