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
        service = GitLabService("https://gitlab.com", "test-token")
        assert service.base_url == "https://gitlab.com"
        assert service.token == "test-token"

    def test_gitlab_service_with_different_urls(self):
        """Test GitLab service with different URL formats"""
        urls = [
            "https://gitlab.com",
            "https://gitlab.example.com",
            "https://git.company.com",
        ]

        for url in urls:
            service = GitLabService(url, "test-token")
            assert service.base_url == url

    @patch("httpx.AsyncClient")
    async def test_get_merge_request_success(self, mock_client):
        """Test successful merge request retrieval"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "iid": 1,
            "title": "Test MR",
            "source_branch": "feature",
            "target_branch": "main",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        service = GitLabService("https://gitlab.com", "test-token")
        result = await service.get_merge_request("test/repo", 1)

        assert result["iid"] == 1
        assert result["title"] == "Test MR"
        assert result["source_branch"] == "feature"
        assert result["target_branch"] == "main"

    @patch("httpx.AsyncClient")
    async def test_get_merge_request_not_found(self, mock_client):
        """Test merge request not found scenario"""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        service = GitLabService("https://gitlab.com", "test-token")

        with pytest.raises(GitLabAPIException) as exc_info:
            await service.get_merge_request("test/repo", 999)

        assert exc_info.value.status_code == 404

    @patch("httpx.AsyncClient")
    async def test_post_merge_request_comment_success(self, mock_client):
        """Test successful merge request comment posting"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": 123,
            "body": "Test comment",
            "created_at": "2023-01-01T00:00:00Z",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        service = GitLabService("https://gitlab.com", "test-token")
        result = await service.post_merge_request_comment(
            "test/repo", 1, "Test comment"
        )

        assert result["id"] == 123
        assert result["body"] == "Test comment"

    @patch("httpx.AsyncClient")
    async def test_http_request_error_handling(self, mock_client):
        """Test HTTP request error handling"""
        # Mock network error
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = httpx.RequestError("Network error")
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        service = GitLabService("https://gitlab.com", "test-token")

        with pytest.raises(GitLabAPIException):
            await service.get_merge_request("test/repo", 1)

    def test_gitlab_service_properties(self):
        """Test GitLab service properties and attributes"""
        service = GitLabService("https://gitlab.com", "test-token")

        # Test that service has expected attributes
        assert hasattr(service, "base_url")
        assert hasattr(service, "token")


class TestGitLabServiceIntegration:
    """Test GitLab service integration scenarios"""

    def test_service_import_availability(self):
        """Test that GitLab service can be imported"""
        from src.services.gitlab_service import GitLabService

        assert GitLabService is not None

    def test_service_with_empty_token(self):
        """Test service creation with empty token"""
        service = GitLabService("https://gitlab.com", "")
        assert service.token == ""

    def test_service_with_none_token(self):
        """Test service creation with None token"""
        service = GitLabService("https://gitlab.com", None)
        assert service.token is None

    @patch("httpx.AsyncClient")
    async def test_api_error_status_codes(self, mock_client):
        """Test handling of different API error status codes"""
        error_codes = [400, 401, 403, 404, 500, 502, 503]

        for status_code in error_codes:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.text = f"Error {status_code}"

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            service = GitLabService("https://gitlab.com", "test-token")

            with pytest.raises(GitLabAPIException) as exc_info:
                await service.get_merge_request("test/repo", 1)

            assert exc_info.value.status_code == status_code


class TestGitLabServiceRetry:
    """Test retry logic in GitLabService"""

    @pytest.mark.asyncio
    @patch("src.services.gitlab_service.get_settings")
    async def test_retry_on_network_error(self, mock_get_settings):
        """Test retry on network errors"""
        mock_get_settings.return_value = Mock(
            gitlab_url="https://gitlab.example.com",
            gitlab_token="test-token",
            request_timeout=30.0,
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )

        service = GitLabService()

        # Mock to fail twice then succeed
        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.RequestError("Network error")
            mock_response = Mock()
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status = Mock()
            return mock_response

        service._make_protected_request = mock_request

        result = await service.get_merge_request(1, 1)

        assert result == {"success": True}
        assert call_count == 3  # Should retry twice before succeeding

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

        assert "Failed to fetch merge request" in str(exc_info.value)


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
        mock_response.text = ""
        mock_response.raise_for_status = Mock()

        service._make_protected_request = AsyncMock(return_value=mock_response)

        result = await service.get_merge_request_diff(1, 1)
        assert result == ""

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

    def test_service_with_custom_headers(self):
        """Test that service can be configured"""
        service = GitLabService("https://gitlab.com", "test-token")
        # Basic test that service is properly configured
        assert service.base_url.startswith("https://")
        assert len(service.token) > 0

    def test_service_url_normalization(self):
        """Test URL handling and normalization"""
        test_cases = [
            ("https://gitlab.com", "https://gitlab.com"),
            ("https://gitlab.com/", "https://gitlab.com/"),
            ("http://gitlab.local", "http://gitlab.local"),
        ]

        for input_url, expected_url in test_cases:
            service = GitLabService(input_url, "test-token")
            # URL should be stored as provided (no automatic normalization)
            assert service.base_url == expected_url
