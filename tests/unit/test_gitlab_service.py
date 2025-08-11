"""
Unit tests for GitLab service
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from src.services.gitlab_service import GitLabService

@pytest.fixture
def mock_gitlab_service():
    """Create a mock GitLab service"""
    with patch('httpx.AsyncClient') as mock_client:
        service = GitLabService()
        service.client = mock_client.return_value
        return service

@pytest.mark.asyncio
async def test_get_merge_request(mock_gitlab_service):
    """Test fetching merge request details"""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": 123,
        "iid": 1,
        "title": "Test MR",
        "description": "Test description"
    }
    mock_response.raise_for_status = MagicMock()
    
    mock_gitlab_service.client.get = AsyncMock(return_value=mock_response)
    
    # Execute
    result = await mock_gitlab_service.get_merge_request(project_id=1, mr_iid=1)
    
    # Assertions
    assert result["id"] == 123
    assert result["title"] == "Test MR"
    mock_gitlab_service.client.get.assert_called_once_with("/projects/1/merge_requests/1")

@pytest.mark.asyncio
async def test_post_merge_request_comment(mock_gitlab_service):
    """Test posting comment to merge request"""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 456, "body": "Test comment"}
    mock_response.raise_for_status = MagicMock()
    
    mock_gitlab_service.client.post = AsyncMock(return_value=mock_response)
    
    # Execute
    result = await mock_gitlab_service.post_merge_request_comment(
        project_id=1, 
        mr_iid=1, 
        comment="Test comment"
    )
    
    # Assertions
    assert result["id"] == 456
    assert result["body"] == "Test comment"
    mock_gitlab_service.client.post.assert_called_once_with(
        "/projects/1/merge_requests/1/notes",
        json={"body": "Test comment"}
    )