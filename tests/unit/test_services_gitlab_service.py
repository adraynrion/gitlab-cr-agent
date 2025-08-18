"""
Tests for src/services/gitlab_service.py
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.exceptions import GitLabAPIException
from src.services.gitlab_service import GitLabService


class TestGitLabService:
    """Test GitLabService class"""

    def test_gitlab_service_initialization(self):
        """Test GitLab service basic initialization"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0

        with patch(
            "src.services.gitlab_service.get_settings", return_value=mock_settings
        ):
            service = GitLabService()
            assert service.base_url == "https://gitlab.com/api/v4"
            assert service.headers["PRIVATE-TOKEN"] == "test-token"

    def test_gitlab_service_with_different_urls(self):
        """Test GitLab service with different URL formats"""
        urls = [
            "https://gitlab.com",
            "https://gitlab.example.com",
            "https://git.company.com",
        ]

        for url in urls:
            mock_settings = Mock()
            mock_settings.gitlab_url = url
            mock_settings.gitlab_token = "test-token"
            mock_settings.request_timeout = 30.0
            mock_settings.max_keepalive_connections = 20
            mock_settings.max_connections = 100
            mock_settings.keepalive_expiry = 30.0

            with patch(
                "src.services.gitlab_service.get_settings", return_value=mock_settings
            ):
                service = GitLabService()
                assert service.base_url == f"{url}/api/v4"

    @patch("src.services.gitlab_service.get_settings")
    async def test_get_merge_request_success(self, mock_get_settings):
        """Test successful merge request retrieval"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "iid": 1,
            "title": "Test MR",
            "source_branch": "feature",
            "target_branch": "main",
        }
        mock_response.raise_for_status = Mock()

        service = GitLabService()

        # Mock the client request method directly
        with patch.object(
            service.client, "request", return_value=mock_response
        ) as mock_request:
            result = await service.get_merge_request(1, 1)

            assert result["iid"] == 1
            assert result["title"] == "Test MR"
            assert result["source_branch"] == "feature"
            assert result["target_branch"] == "main"
            mock_request.assert_called_once()

    @patch("src.services.gitlab_service.get_settings")
    async def test_get_merge_request_not_found(self, mock_get_settings):
        """Test merge request not found scenario"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response
        )

        service = GitLabService()

        with patch.object(service.client, "request", return_value=mock_response):
            with pytest.raises(GitLabAPIException) as exc_info:
                await service.get_merge_request(1, 999)

            assert exc_info.value.status_code == 404

    @patch("src.services.gitlab_service.get_settings")
    async def test_post_merge_request_comment_success(self, mock_get_settings):
        """Test successful merge request comment posting"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": 123,
            "body": "Test comment",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_response.raise_for_status = Mock()

        service = GitLabService()

        with patch.object(
            service.client, "request", return_value=mock_response
        ) as mock_request:
            result = await service.post_merge_request_comment(1, 1, "Test comment")

            assert result["id"] == 123
            assert result["body"] == "Test comment"
            mock_request.assert_called_once()

    @patch("src.services.gitlab_service.get_settings")
    async def test_http_request_error_handling(self, mock_get_settings):
        """Test HTTP request error handling"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        service = GitLabService()

        # Mock network error
        with patch.object(
            service.client, "request", side_effect=httpx.RequestError("Network error")
        ):
            with pytest.raises(GitLabAPIException):
                await service.get_merge_request(1, 1)

    @patch("src.services.gitlab_service.get_settings")
    def test_gitlab_service_properties(self, mock_get_settings):
        """Test GitLab service properties and attributes"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        service = GitLabService()

        # Test that service has expected attributes
        assert hasattr(service, "base_url")
        assert hasattr(service, "headers")


class TestGitLabServiceIntegration:
    """Test GitLab service integration scenarios"""

    def test_service_import_availability(self):
        """Test that GitLab service can be imported"""
        from src.services.gitlab_service import GitLabService

        assert GitLabService is not None

    @patch("src.services.gitlab_service.get_settings")
    def test_service_with_empty_token(self, mock_get_settings):
        """Test service creation with empty token"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = ""
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        service = GitLabService()
        assert service.headers["PRIVATE-TOKEN"] == ""

    @patch("src.services.gitlab_service.get_settings")
    def test_service_with_none_token(self, mock_get_settings):
        """Test service creation with None token"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = None
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        service = GitLabService()
        assert service.headers["PRIVATE-TOKEN"] is None

    @patch("src.services.gitlab_service.get_settings")
    async def test_api_error_status_codes(self, mock_get_settings):
        """Test handling of different API error status codes"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        error_codes = [400, 401, 403, 404, 500, 502, 503]

        service = GitLabService()

        for status_code in error_codes:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.text = f"Error {status_code}"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"Error {status_code}", request=Mock(), response=mock_response
            )

            with patch.object(service.client, "request", return_value=mock_response):
                with pytest.raises(GitLabAPIException) as exc_info:
                    await service.get_merge_request(1, 1)

                assert exc_info.value.status_code == status_code


class TestGitLabServiceRetry:
    """Test retry logic in GitLabService"""

    @pytest.mark.asyncio
    @patch("src.services.gitlab_service.get_settings")
    async def test_retry_on_network_error(self, mock_get_settings):
        """Test that network errors are properly handled"""
        mock_get_settings.return_value = Mock(
            gitlab_url="https://gitlab.example.com",
            gitlab_token="test-token",
            request_timeout=30.0,
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )

        service = GitLabService()

        # Test that network errors are caught and wrapped in GitLabAPIException
        with patch.object(
            service.client, "request", side_effect=httpx.RequestError("Network error")
        ):
            with pytest.raises(GitLabAPIException) as exc_info:
                await service.get_merge_request(1, 1)

            assert "Network error" in str(exc_info.value)


class TestGitLabServiceMethods:
    """Test individual GitLabService methods comprehensively"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for GitLab service"""
        settings = Mock()
        settings.gitlab_url = "https://gitlab.example.com"
        settings.gitlab_token = "glpat-test-token"
        settings.request_timeout = 30.0
        settings.max_keepalive_connections = 20
        settings.max_connections = 100
        settings.keepalive_expiry = 30.0
        return settings

    @pytest.fixture
    def gitlab_service(self, mock_settings):
        """Create GitLabService with mocked settings"""
        with patch(
            "src.services.gitlab_service.get_settings", return_value=mock_settings
        ):
            return GitLabService()

    def test_get_headers_with_tracing_no_context(self, gitlab_service):
        """Test headers without tracing context"""
        with patch(
            "src.services.gitlab_service.get_correlation_id", return_value=None
        ), patch("src.services.gitlab_service.get_request_id", return_value=None):
            headers = gitlab_service._get_headers_with_tracing()

            assert headers["PRIVATE-TOKEN"] == "glpat-test-token"
            assert headers["Content-Type"] == "application/json"
            assert "X-Correlation-ID" not in headers
            assert "X-Request-ID" not in headers

    def test_get_headers_with_tracing_with_context(self, gitlab_service):
        """Test headers with tracing context"""
        with patch(
            "src.services.gitlab_service.get_correlation_id", return_value="corr-123"
        ), patch("src.services.gitlab_service.get_request_id", return_value="req-456"):
            headers = gitlab_service._get_headers_with_tracing()

            assert headers["PRIVATE-TOKEN"] == "glpat-test-token"
            assert headers["Content-Type"] == "application/json"
            assert headers["X-Correlation-ID"] == "corr-123"
            assert headers["X-Request-ID"] == "req-456"

    @pytest.mark.asyncio
    async def test_make_protected_request_success(self, gitlab_service):
        """Test successful protected request"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = Mock()

        with patch.object(
            gitlab_service.client, "request", return_value=mock_response
        ) as mock_request:
            result = await gitlab_service._make_protected_request(
                method="GET", url="/test", json={"key": "value"}
            )

            assert result == mock_response
            mock_request.assert_called_once()
            # _make_protected_request doesn't call raise_for_status, the calling methods do

    @pytest.mark.asyncio
    async def test_make_protected_request_http_error(self, gitlab_service):
        """Test protected request with HTTP error"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response
        )

        with patch.object(gitlab_service.client, "request", return_value=mock_response):
            result = await gitlab_service._make_protected_request("GET", "/test")

            # _make_protected_request just returns the response, doesn't handle errors
            assert result == mock_response
            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_make_protected_request_network_error(self, gitlab_service):
        """Test protected request with network error"""
        with patch.object(
            gitlab_service.client,
            "request",
            side_effect=httpx.RequestError("Network error"),
        ):
            # _make_protected_request doesn't catch exceptions, they bubble up
            with pytest.raises(httpx.RequestError) as exc_info:
                await gitlab_service._make_protected_request("GET", "/test")

            assert "Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_merge_request_success(self, gitlab_service):
        """Test successful merge request retrieval"""
        mock_data = {
            "iid": 123,
            "title": "Test MR",
            "source_branch": "feature",
            "target_branch": "main",
            "state": "opened",
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = Mock()

        with patch.object(
            gitlab_service.client, "request", return_value=mock_response
        ) as mock_request:
            result = await gitlab_service.get_merge_request(100, 123)

            assert result == mock_data
            mock_request.assert_called_once()
            mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_merge_request_diff_success(self, gitlab_service):
        """Test successful merge request diff retrieval"""
        mock_diff_data = [
            {
                "old_path": "file1.py",
                "new_path": "file1.py",
                "diff": "@@ -1,3 +1,3 @@\n-old line\n+new line",
            }
        ]

        mock_response = Mock()
        mock_response.json.return_value = mock_diff_data
        mock_response.raise_for_status = Mock()

        with patch.object(
            gitlab_service.client, "request", return_value=mock_response
        ) as mock_request:
            result = await gitlab_service.get_merge_request_diff(100, 123)

            assert result == mock_diff_data
            mock_request.assert_called_once()
            mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_merge_request_comment_success(self, gitlab_service):
        """Test successful merge request comment posting"""
        comment_body = "This is a test comment"
        mock_response_data = {
            "id": 789,
            "body": comment_body,
            "created_at": "2023-01-01T00:00:00Z",
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = Mock()

        with patch.object(
            gitlab_service.client, "request", return_value=mock_response
        ) as mock_request:
            result = await gitlab_service.post_merge_request_comment(
                100, 123, comment_body
            )

            assert result == mock_response_data
            mock_request.assert_called_once()
            mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, gitlab_service):
        """Test GitLabService as async context manager"""
        async with gitlab_service as service:
            assert service is gitlab_service
            assert hasattr(service, "client")

    @pytest.mark.asyncio
    async def test_close_method(self, gitlab_service):
        """Test GitLabService close method"""
        with patch.object(gitlab_service.client, "aclose") as mock_close:
            await gitlab_service.close()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self, gitlab_service):
        """Test that __aexit__ calls close"""
        with patch.object(gitlab_service, "close") as mock_close:
            await gitlab_service.__aexit__(None, None, None)
            mock_close.assert_called_once()


class TestGitLabServiceErrorScenarios:
    """Test GitLabService error handling scenarios"""

    @pytest.fixture
    def gitlab_service(self):
        """Create GitLabService for error testing"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.example.com"
        mock_settings.gitlab_token = "glpat-test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0

        with patch(
            "src.services.gitlab_service.get_settings", return_value=mock_settings
        ):
            return GitLabService()

    @pytest.mark.asyncio
    async def test_get_merge_request_not_found(self, gitlab_service):
        """Test merge request not found error"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        error = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response
        )

        with patch.object(gitlab_service.client, "request", side_effect=error):
            with pytest.raises(GitLabAPIException) as exc_info:
                await gitlab_service.get_merge_request(100, 999)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_merge_request_diff_unauthorized(self, gitlab_service):
        """Test merge request diff unauthorized error"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        error = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response
        )

        with patch.object(gitlab_service.client, "request", side_effect=error):
            with pytest.raises(GitLabAPIException) as exc_info:
                await gitlab_service.get_merge_request_diff(100, 123)

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_post_comment_forbidden(self, gitlab_service):
        """Test post comment forbidden error"""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        error = httpx.HTTPStatusError(
            "Forbidden", request=Mock(), response=mock_response
        )

        with patch.object(gitlab_service.client, "request", side_effect=error):
            with pytest.raises(GitLabAPIException) as exc_info:
                await gitlab_service.post_merge_request_comment(100, 123, "test")

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_server_error_handling(self, gitlab_service):
        """Test server error handling"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        error = httpx.HTTPStatusError(
            "Internal Server Error", request=Mock(), response=mock_response
        )

        with patch.object(gitlab_service.client, "request", side_effect=error):
            with pytest.raises(GitLabAPIException) as exc_info:
                await gitlab_service.get_merge_request(100, 123)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, gitlab_service):
        """Test timeout error handling"""
        with patch.object(
            gitlab_service.client,
            "request",
            side_effect=httpx.TimeoutException("Timeout"),
        ):
            with pytest.raises(GitLabAPIException) as exc_info:
                await gitlab_service.get_merge_request(100, 123)

            assert "Network error" in str(exc_info.value)


class TestGitLabServiceLogging:
    """Test GitLabService logging functionality"""

    @pytest.fixture
    def gitlab_service(self):
        """Create GitLabService for logging tests"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.example.com"
        mock_settings.gitlab_token = "glpat-test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0

        with patch(
            "src.services.gitlab_service.get_settings", return_value=mock_settings
        ):
            return GitLabService()

    @pytest.mark.asyncio
    async def test_request_logging_on_success(self, gitlab_service):
        """Test that successful requests are logged"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = Mock()

        with patch.object(
            gitlab_service.client, "request", return_value=mock_response
        ), patch("src.services.gitlab_service.logger") as mock_logger:
            await gitlab_service.get_merge_request(100, 123)

            # Check that info logging was called
            assert mock_logger.info.called

    @pytest.mark.asyncio
    async def test_request_logging_on_error(self, gitlab_service):
        """Test that failed requests are logged"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response
        )

        with patch.object(
            gitlab_service.client, "request", return_value=mock_response
        ), patch("src.services.gitlab_service.logger") as mock_logger:
            with pytest.raises(GitLabAPIException):
                await gitlab_service.get_merge_request(100, 123)

            # Check that error logging was called
            assert mock_logger.error.called

    @pytest.mark.asyncio
    @patch("src.services.gitlab_service.get_settings")
    async def test_retry_exhausted(self, mock_get_settings):
        """Test when retries are exhausted"""
        mock_get_settings.return_value = Mock(
            gitlab_url="https://gitlab.example.com",
            gitlab_token="test-token",
            request_timeout=30.0,
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )

        service = GitLabService()

        # Mock to always fail
        async def mock_request(*args, **kwargs):
            raise httpx.RequestError("Network error")

        service._make_protected_request = mock_request

        with pytest.raises(GitLabAPIException) as exc_info:
            await service.get_merge_request(1, 1)

        assert "Network error" in str(exc_info.value)


class TestGitLabServiceEdgeCases:
    """Test edge cases for GitLabService"""

    @pytest.mark.asyncio
    @patch("src.services.gitlab_service.get_settings")
    async def test_empty_diff(self, mock_get_settings):
        """Test handling empty diff"""
        mock_get_settings.return_value = Mock(
            gitlab_url="https://gitlab.example.com",
            gitlab_token="test-token",
            request_timeout=30.0,
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )

        service = GitLabService()
        mock_response = Mock()
        mock_response.json.return_value = []  # Empty diff list
        mock_response.raise_for_status = Mock()

        with patch.object(service.client, "request", return_value=mock_response):
            result = await service.get_merge_request_diff(1, 1)
            assert result == []

    @pytest.mark.asyncio
    @patch("src.services.gitlab_service.get_settings")
    async def test_special_characters_in_comment(self, mock_get_settings):
        """Test posting comment with special characters"""
        mock_get_settings.return_value = Mock(
            gitlab_url="https://gitlab.example.com",
            gitlab_token="test-token",
            request_timeout=30.0,
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )

        service = GitLabService()
        mock_response = Mock()
        mock_response.json.return_value = {"id": 1}
        mock_response.raise_for_status = Mock()

        service._make_protected_request = AsyncMock(return_value=mock_response)

        special_comment = "Test with special chars: \n\t'\"<>&"
        result = await service.post_merge_request_comment(1, 1, special_comment)

        assert result == {"id": 1}
        call_args = service._make_protected_request.call_args
        assert call_args[1]["json"]["body"] == special_comment


class TestGitLabServiceConfiguration:
    """Test GitLab service configuration scenarios"""

    @patch("src.services.gitlab_service.get_settings")
    def test_service_with_custom_headers(self, mock_get_settings):
        """Test that service can be configured"""
        mock_settings = Mock()
        mock_settings.gitlab_url = "https://gitlab.com"
        mock_settings.gitlab_token = "test-token"
        mock_settings.request_timeout = 30.0
        mock_settings.max_keepalive_connections = 20
        mock_settings.max_connections = 100
        mock_settings.keepalive_expiry = 30.0
        mock_get_settings.return_value = mock_settings

        service = GitLabService()
        # Basic test that service is properly configured
        assert service.base_url.startswith("https://")
        assert len(service.headers["PRIVATE-TOKEN"]) > 0

    @patch("src.services.gitlab_service.get_settings")
    def test_service_url_normalization(self, mock_get_settings):
        """Test URL handling and normalization"""
        test_cases = [
            ("https://gitlab.com", "https://gitlab.com/api/v4"),
            ("https://gitlab.com/", "https://gitlab.com//api/v4"),
            ("http://gitlab.local", "http://gitlab.local/api/v4"),
        ]

        for input_url, expected_url in test_cases:
            mock_settings = Mock()
            mock_settings.gitlab_url = input_url
            mock_settings.gitlab_token = "test-token"
            mock_settings.request_timeout = 30.0
            mock_settings.max_keepalive_connections = 20
            mock_settings.max_connections = 100
            mock_settings.keepalive_expiry = 30.0
            mock_get_settings.return_value = mock_settings

            service = GitLabService()
            # URL should have /api/v4 appended
            assert service.base_url == expected_url
