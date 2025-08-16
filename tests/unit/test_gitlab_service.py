"""
Comprehensive unit tests for GitLab service
"""

from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from src.config.settings import Settings
from src.exceptions import GitLabAPIException
from src.services.gitlab_service import GitLabService


@pytest.fixture
def mock_settings():
    """Create mock settings"""
    return MagicMock(
        spec=Settings,
        gitlab_url="https://gitlab.example.com",
        gitlab_token="test-token",
        request_timeout=30.0,
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30.0,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_timeout=60,
    )


@pytest.fixture
def mock_gitlab_service(mock_settings):
    """Create a mock GitLab service with properly mocked client"""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_instance = AsyncMock()
        mock_client_class.return_value = mock_client_instance

        with patch("src.services.gitlab_service.settings", mock_settings), patch(
            "pybreaker.CircuitBreaker"
        ) as mock_breaker, patch(
            "src.api.middleware.get_correlation_id", return_value="test-correlation"
        ), patch(
            "src.api.middleware.get_request_id", return_value="test-request"
        ):
            mock_breaker.return_value = MagicMock()
            service = GitLabService()
            service.client = mock_client_instance
            return service


@pytest.fixture
def mock_response():
    """Create a mock HTTP response"""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock()
    response.text = "Success"
    return response


@pytest.fixture
def gitlab_responses(load_fixture):
    """Load GitLab API response fixtures"""
    return load_fixture("gitlab_responses.json")


@pytest.mark.asyncio
async def test_get_merge_request(mock_gitlab_service):
    """Test fetching merge request details"""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": 123,
        "iid": 1,
        "title": "Test MR",
        "description": "Test description",
    }
    mock_response.raise_for_status = MagicMock()

    mock_gitlab_service._make_protected_request = AsyncMock(return_value=mock_response)

    # Execute
    result = await mock_gitlab_service.get_merge_request(project_id=1, mr_iid=1)

    # Assertions
    assert result["id"] == 123
    assert result["title"] == "Test MR"
    mock_gitlab_service._make_protected_request.assert_called_once_with(
        "GET", "/projects/1/merge_requests/1"
    )


@pytest.mark.asyncio
async def test_get_merge_request_diff(mock_gitlab_service, gitlab_responses):
    """Test fetching merge request diff"""
    expected_diff = [
        {
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "diff": "@@ -1,5 +1,6 @@\n def hello():\n-    print('hello')\n+    print('hello world')",
        }
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = expected_diff
    mock_response.raise_for_status = MagicMock()
    mock_gitlab_service._make_protected_request = AsyncMock(return_value=mock_response)

    # Execute
    result = await mock_gitlab_service.get_merge_request_diff(project_id=1, mr_iid=1)

    # Assertions
    assert result == expected_diff
    mock_gitlab_service._make_protected_request.assert_called_once_with(
        "GET", "/projects/1/merge_requests/1/diffs"
    )


@pytest.mark.asyncio
async def test_post_merge_request_comment(mock_gitlab_service, gitlab_responses):
    """Test posting comment to merge request"""
    expected_comment = gitlab_responses["merge_request_comment"]

    mock_response = MagicMock()
    mock_response.json.return_value = expected_comment
    mock_response.raise_for_status = MagicMock()
    mock_gitlab_service._make_protected_request = AsyncMock(return_value=mock_response)

    # Execute
    result = await mock_gitlab_service.post_merge_request_comment(
        project_id=1, mr_iid=1, comment="Test comment"
    )

    # Assertions
    assert result["id"] == expected_comment["id"]
    assert result["body"] == expected_comment["body"]
    mock_gitlab_service._make_protected_request.assert_called_once_with(
        "POST", "/projects/1/merge_requests/1/notes", json={"body": "Test comment"}
    )


# ============================================================================
# Error Handling Tests for Existing Methods
# ============================================================================


@pytest.mark.asyncio
async def test_get_merge_request_network_error(mock_gitlab_service):
    """Test get_merge_request with network error"""
    mock_gitlab_service._make_protected_request = AsyncMock(
        side_effect=httpx.RequestError("Connection failed")
    )

    with pytest.raises(GitLabAPIException) as exc_info:
        await mock_gitlab_service.get_merge_request(project_id=1, mr_iid=1)

    assert "Network error fetching merge request 1" in str(exc_info.value)
    assert exc_info.value.details["project_id"] == 1
    assert exc_info.value.details["mr_iid"] == 1


@pytest.mark.asyncio
async def test_get_merge_request_http_error(mock_gitlab_service):
    """Test get_merge_request with HTTP error"""
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden"

    http_error = httpx.HTTPStatusError(
        "403 Forbidden", request=MagicMock(), response=mock_response
    )
    mock_gitlab_service._make_protected_request = AsyncMock(side_effect=http_error)

    with pytest.raises(GitLabAPIException) as exc_info:
        await mock_gitlab_service.get_merge_request(project_id=1, mr_iid=1)

    assert "GitLab API error for merge request 1" in str(exc_info.value)
    assert exc_info.value.status_code == 403
    assert exc_info.value.details["response_body"] == "Forbidden"


@pytest.mark.asyncio
async def test_get_merge_request_unexpected_error(mock_gitlab_service):
    """Test get_merge_request with unexpected error"""
    mock_gitlab_service._make_protected_request = AsyncMock(
        side_effect=ValueError("Unexpected error")
    )

    with pytest.raises(GitLabAPIException) as exc_info:
        await mock_gitlab_service.get_merge_request(project_id=1, mr_iid=1)

    assert "Unexpected error fetching merge request 1" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, ValueError)


@pytest.mark.asyncio
async def test_get_merge_request_diff_http_error(mock_gitlab_service):
    """Test get_merge_request_diff with HTTP error"""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    http_error = httpx.HTTPStatusError(
        "500 Internal Server Error", request=MagicMock(), response=mock_response
    )
    mock_gitlab_service._make_protected_request = AsyncMock(side_effect=http_error)

    with pytest.raises(GitLabAPIException) as exc_info:
        await mock_gitlab_service.get_merge_request_diff(project_id=1, mr_iid=1)

    assert "GitLab API error for MR diff 1" in str(exc_info.value)
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_post_merge_request_comment_network_error(mock_gitlab_service):
    """Test post_merge_request_comment with network error"""
    mock_gitlab_service._make_protected_request = AsyncMock(
        side_effect=httpx.RequestError("Network error")
    )

    with pytest.raises(GitLabAPIException) as exc_info:
        await mock_gitlab_service.post_merge_request_comment(
            project_id=1, mr_iid=1, comment="Test comment"
        )

    assert "Network error posting comment to MR 1" in str(exc_info.value)
    assert exc_info.value.details["comment_length"] == 12


@pytest.mark.asyncio
async def test_post_merge_request_comment_http_error(mock_gitlab_service):
    """Test post_merge_request_comment with HTTP error"""
    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.text = "Unprocessable Entity"

    http_error = httpx.HTTPStatusError(
        "422 Unprocessable Entity", request=MagicMock(), response=mock_response
    )
    mock_gitlab_service._make_protected_request = AsyncMock(side_effect=http_error)

    with pytest.raises(GitLabAPIException) as exc_info:
        await mock_gitlab_service.post_merge_request_comment(
            project_id=1, mr_iid=1, comment="Invalid comment"
        )

    assert "GitLab API error posting comment to MR 1" in str(exc_info.value)
    assert exc_info.value.status_code == 422


# ============================================================================
# Service Configuration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_gitlab_service_initialization(mock_settings):
    """Test GitLab service initialization with correct configuration"""
    with patch("httpx.AsyncClient") as mock_client_class, patch(
        "pybreaker.CircuitBreaker"
    ) as mock_breaker, patch(
        "src.api.middleware.get_correlation_id", return_value="test-correlation"
    ), patch(
        "src.api.middleware.get_request_id", return_value="test-request"
    ):
        mock_breaker.return_value = MagicMock()
        with patch("src.services.gitlab_service.settings", mock_settings):
            service = GitLabService()

            assert service.base_url == "https://gitlab.example.com/api/v4"
            assert service.headers["PRIVATE-TOKEN"] == "test-token"
            assert service.headers["Content-Type"] == "application/json"

            # Verify AsyncClient was called with correct parameters
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            assert call_args[1]["base_url"] == "https://gitlab.example.com/api/v4"
            assert call_args[1]["timeout"] == 30.0


@pytest.mark.asyncio
async def test_gitlab_service_context_manager(mock_gitlab_service):
    """Test GitLab service as async context manager"""
    # Test __aenter__
    result = await mock_gitlab_service.__aenter__()
    assert result is mock_gitlab_service

    # Test __aexit__
    # Mock the is_closed property
    mock_gitlab_service.client.is_closed = False
    await mock_gitlab_service.__aexit__(None, None, None)
    mock_gitlab_service.client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_gitlab_service_close(mock_gitlab_service):
    """Test GitLab service close method"""
    # Mock the is_closed property
    mock_gitlab_service.client.is_closed = False
    await mock_gitlab_service.close()
    mock_gitlab_service.client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_gitlab_service_close_with_none_client():
    """Test GitLab service close method when client is None"""
    with patch("src.services.gitlab_service.settings") as mock_settings:
        mock_settings.gitlab_url = "https://gitlab.example.com"
        mock_settings.gitlab_token = "test-token"

        with patch("httpx.AsyncClient"):
            service = GitLabService()
            service.client = None

            # Should not raise an exception
            await service.close()


# ============================================================================
# Rate Limiting and Authentication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limit_error_handling(mock_gitlab_service):
    """Test proper error handling for rate limit responses"""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"

    http_error = httpx.HTTPStatusError(
        "429 Too Many Requests", request=MagicMock(), response=mock_response
    )
    mock_gitlab_service._make_protected_request = AsyncMock(side_effect=http_error)

    with pytest.raises(GitLabAPIException) as exc_info:
        await mock_gitlab_service.get_merge_request(project_id=1, mr_iid=1)

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.details["response_body"]


@pytest.mark.asyncio
async def test_authentication_error_handling(mock_gitlab_service):
    """Test proper error handling for authentication failures"""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    http_error = httpx.HTTPStatusError(
        "401 Unauthorized", request=MagicMock(), response=mock_response
    )
    mock_gitlab_service._make_protected_request = AsyncMock(side_effect=http_error)

    with pytest.raises(GitLabAPIException) as exc_info:
        await mock_gitlab_service.get_merge_request(project_id=1, mr_iid=1)

    assert exc_info.value.status_code == 401
    assert "Unauthorized" in exc_info.value.details["response_body"]


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================


@pytest.mark.asyncio
async def test_empty_comment_posting(mock_gitlab_service):
    """Test posting empty comment"""
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 123, "body": ""}
    mock_response.raise_for_status = MagicMock()
    mock_gitlab_service._make_protected_request = AsyncMock(return_value=mock_response)

    result = await mock_gitlab_service.post_merge_request_comment(
        project_id=1, mr_iid=1, comment=""
    )

    assert result["body"] == ""
    mock_gitlab_service._make_protected_request.assert_called_once_with(
        "POST", "/projects/1/merge_requests/1/notes", json={"body": ""}
    )


@pytest.mark.asyncio
async def test_large_comment_posting(mock_gitlab_service):
    """Test posting very large comment"""
    large_comment = "x" * 10000  # 10KB comment

    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 123, "body": large_comment}
    mock_response.raise_for_status = MagicMock()
    mock_gitlab_service._make_protected_request = AsyncMock(return_value=mock_response)

    result = await mock_gitlab_service.post_merge_request_comment(
        project_id=1, mr_iid=1, comment=large_comment
    )

    assert len(result["body"]) == 10000


# ============================================================================
# Success Path Comprehensive Tests
# ============================================================================


@pytest.mark.asyncio
async def test_complete_workflow_success(mock_gitlab_service, gitlab_responses):
    """Test a complete workflow using core service methods"""
    # Setup responses for each method call
    mr_response = MagicMock()
    mr_response.json.return_value = gitlab_responses["merge_request"]
    mr_response.raise_for_status = MagicMock()

    diff_response = MagicMock()
    diff_response.json.return_value = [
        {
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "diff": "@@ -1,5 +1,6 @@\n def hello():\n-    print('hello')\n+    print('hello world')",
        }
    ]
    diff_response.raise_for_status = MagicMock()

    comment_response = MagicMock()
    comment_response.json.return_value = gitlab_responses["merge_request_comment"]
    comment_response.raise_for_status = MagicMock()

    # Configure mock to return different responses for different calls
    call_responses = [
        mr_response,  # get_merge_request
        diff_response,  # get_merge_request_diff
        comment_response,  # post_merge_request_comment
    ]

    def side_effect(*args, **kwargs):
        return call_responses.pop(0)

    mock_gitlab_service._make_protected_request = AsyncMock(side_effect=side_effect)

    # Execute complete workflow
    mr = await mock_gitlab_service.get_merge_request(project_id=100, mr_iid=10)
    diff = await mock_gitlab_service.get_merge_request_diff(project_id=100, mr_iid=10)
    comment = await mock_gitlab_service.post_merge_request_comment(
        project_id=100, mr_iid=10, comment="Review completed"
    )

    # Verify all operations succeeded
    assert mr["title"] == "Add new feature"
    assert len(diff) > 0
    assert diff[0]["old_path"] == "src/main.py"
    assert comment["body"] == "Code review comment"

    # Verify all expected API calls were made
    expected_calls = [
        call("GET", "/projects/100/merge_requests/10"),
        call("GET", "/projects/100/merge_requests/10/diffs"),
        call(
            "POST",
            "/projects/100/merge_requests/10/notes",
            json={"body": "Review completed"},
        ),
    ]
    mock_gitlab_service._make_protected_request.assert_has_calls(expected_calls)
