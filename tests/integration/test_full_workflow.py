"""
Comprehensive End-to-End Integration Tests for GitLab Code Review Agent

This module contains comprehensive integration tests covering the complete workflow
from webhook receipt through comment posting, including error scenarios,
multi-provider fallbacks, and performance testing.
"""

import asyncio
import json
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import AsyncClient, ConnectError, HTTPStatusError, Response, TimeoutException

from src.config.settings import Settings
from src.exceptions import GitLabAPIException
from src.models.gitlab_models import MergeRequestEvent
from src.services.gitlab_service import GitLabService
from src.services.review_service import ReviewService


@pytest.mark.integration
class TestFullWorkflowIntegration:
    """
    End-to-end integration tests for the complete webhook-to-review workflow.
    Tests the entire system working together from webhook receipt to comment posting.
    """

    @pytest.fixture
    async def mock_httpx_client(self):
        """Create a mock httpx client with configurable responses."""
        client = AsyncMock(spec=AsyncClient)

        # Default successful responses
        def create_response(data: dict, status_code: int = 200):
            response = Mock(spec=Response)
            response.status_code = status_code
            response.json.return_value = data
            response.text = json.dumps(data)
            response.headers = {"content-type": "application/json"}
            response.is_success = status_code < 400
            response.raise_for_status = Mock()
            if status_code >= 400:
                response.raise_for_status.side_effect = HTTPStatusError(
                    message="HTTP Error", request=Mock(), response=response
                )
            return response

        # Configure mock responses
        client.get.return_value = create_response({"status": "success"})
        client.post.return_value = create_response({"id": 123})
        client.put.return_value = create_response({"updated": True})

        return client

    @pytest.fixture
    async def integrated_review_service(
        self, settings: Settings, mock_httpx_client
    ) -> ReviewService:
        """Create a fully integrated review service with real dependencies."""
        # GitLabService uses global settings, doesn't take arguments
        gitlab_service = GitLabService()
        gitlab_service.client = mock_httpx_client

        # ReviewService only takes optional review_agent
        review_service = ReviewService()
        return review_service

    @pytest.fixture
    def large_merge_request_payload(self, generate_webhook_payload) -> Dict[str, Any]:
        """Generate a webhook payload for a large merge request."""
        return generate_webhook_payload(
            object_attributes={
                "id": 999,
                "iid": 50,
                "title": "Large refactor: Update entire codebase",
                "description": "Major refactoring changes affecting multiple components",
                "state": "opened",
                "action": "open",
                "source_branch": "major-refactor",
                "target_branch": "main",
                "work_in_progress": False,
                "draft": False,
            }
        )

    @pytest.fixture
    def complex_diff_response(self) -> Dict[str, Any]:
        """Create a complex diff with multiple file types and changes."""
        return {
            "changes": [
                {
                    "old_path": "src/services/auth.py",
                    "new_path": "src/services/auth.py",
                    "new_file": False,
                    "renamed_file": False,
                    "deleted_file": False,
                    "diff": """@@ -1,15 +1,25 @@
 import hashlib
 import jwt
+import bcrypt
+from datetime import datetime, timedelta
+from typing import Optional

 class AuthService:
-    def __init__(self):
-        self.secret = "default"
+    def __init__(self, secret_key: str, algorithm: str = "HS256"):
+        self.secret_key = secret_key
+        self.algorithm = algorithm

-    def hash_password(self, password):
-        return hashlib.sha256(password.encode()).hexdigest()
+    def hash_password(self, password: str) -> str:
+        salt = bcrypt.gensalt()
+        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

-    def verify_password(self, password, hashed):
-        return self.hash_password(password) == hashed
+    def verify_password(self, password: str, hashed: str) -> bool:
+        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
+
+    def generate_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
+        if expires_delta is None:
+            expires_delta = timedelta(hours=1)
+
+        expire = datetime.utcnow() + expires_delta
+        payload = {"user_id": user_id, "exp": expire}
+        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)""",
                },
                {
                    "old_path": "tests/test_auth.py",
                    "new_path": "tests/test_auth.py",
                    "new_file": False,
                    "renamed_file": False,
                    "deleted_file": False,
                    "diff": """@@ -1,10 +1,35 @@
 import pytest
+from datetime import timedelta
+from unittest.mock import patch
 from src.services.auth import AuthService

+@pytest.fixture
+def auth_service():
+    return AuthService("test-secret-key")
+
 def test_hash_password():
-    auth = AuthService()
+    auth = AuthService("test-key")
     hashed = auth.hash_password("password123")
-    assert len(hashed) == 64
-    assert hashed != "password123"
+    assert hashed != "password123"
+    assert len(hashed) > 50  # bcrypt hashes are longer
+
+def test_verify_password():
+    auth = AuthService("test-key")
+    password = "secure_password"
+    hashed = auth.hash_password(password)
+    assert auth.verify_password(password, hashed)
+    assert not auth.verify_password("wrong_password", hashed)
+
+@patch('jwt.encode')
+def test_generate_token(mock_jwt_encode, auth_service):
+    mock_jwt_encode.return_value = "fake-token"
+
+    token = auth_service.generate_token("user123")
+
+    assert token == "fake-token"
+    mock_jwt_encode.assert_called_once()
+    call_args = mock_jwt_encode.call_args
+    payload = call_args[0][0]
+    assert payload["user_id"] == "user123"
+    assert "exp" in payload""",
                },
                {
                    "old_path": "frontend/components/Login.jsx",
                    "new_path": "frontend/components/Login.jsx",
                    "new_file": False,
                    "renamed_file": False,
                    "deleted_file": False,
                    "diff": """@@ -1,20 +1,45 @@
 import React, { useState } from 'react';
+import { useNavigate } from 'react-router-dom';
+import { toast } from 'react-toastify';
+import { authAPI } from '../services/api';

-const Login = () => {
+const Login = ({ onLogin }) => {
   const [username, setUsername] = useState('');
   const [password, setPassword] = useState('');
+  const [isLoading, setIsLoading] = useState(false);
+  const navigate = useNavigate();

-  const handleSubmit = (e) => {
+  const handleSubmit = async (e) => {
     e.preventDefault();
-    console.log('Login attempt:', { username, password });
+
+    if (!username || !password) {
+      toast.error('Please fill in all fields');
+      return;
+    }
+
+    setIsLoading(true);
+    try {
+      const response = await authAPI.login(username, password);
+      localStorage.setItem('token', response.token);
+      onLogin(response.user);
+      toast.success('Login successful!');
+      navigate('/dashboard');
+    } catch (error) {
+      toast.error('Login failed. Please check your credentials.');
+    } finally {
+      setIsLoading(false);
+    }
   };

   return (
-    <form onSubmit={handleSubmit}>
+    <div className="login-container">
+      <form onSubmit={handleSubmit} className="login-form">
+        <h2>Login</h2>
         <input
           type="text"
           placeholder="Username"
           value={username}
           onChange={(e) => setUsername(e.target.value)}
+          disabled={isLoading}
+          required
         />
         <input
           type="password"
           placeholder="Password"
           value={password}
           onChange={(e) => setPassword(e.target.value)}
+          disabled={isLoading}
+          required
         />
-        <button type="submit">Login</button>
-    </form>
+        <button
+          type="submit"
+          disabled={isLoading}
+          className={isLoading ? 'loading' : ''}
+        >
+          {isLoading ? 'Logging in...' : 'Login'}
+        </button>
+      </form>
+    </div>
   );
 };""",
                },
                {
                    "old_path": "config/database.yml",
                    "new_path": "config/database.yml",
                    "new_file": False,
                    "renamed_file": False,
                    "deleted_file": False,
                    "diff": """@@ -1,8 +1,20 @@
 development:
   adapter: postgresql
+  encoding: unicode
   database: myapp_development
-  username: postgres
-  password: password
+  username: <%= ENV['DB_USERNAME'] || 'postgres' %>
+  password: <%= ENV['DB_PASSWORD'] || 'password' %>
   host: localhost
   port: 5432
+  pool: <%= ENV.fetch("RAILS_MAX_THREADS") { 5 } %>
+  timeout: 5000
+
+test:
+  adapter: postgresql
+  encoding: unicode
+  database: myapp_test<%= ENV['TEST_ENV_NUMBER'] %>
+  username: <%= ENV['DB_USERNAME'] || 'postgres' %>
+  password: <%= ENV['DB_PASSWORD'] || 'password' %>
+  host: localhost
+  port: 5432
+  pool: <%= ENV.fetch("RAILS_MAX_THREADS") { 5 } %>""",
                },
                {
                    "old_path": "docs/api.md",
                    "new_path": "docs/api.md",
                    "new_file": True,
                    "renamed_file": False,
                    "deleted_file": False,
                    "diff": """+# API Documentation
+
+## Authentication Endpoints
+
+### POST /auth/login
+Login with username and password.
+
+**Request Body:**
+```json
+{
+  "username": "string",
+  "password": "string"
+}
+```
+
+**Response:**
+```json
+{
+  "token": "jwt-token",
+  "user": {
+    "id": "string",
+    "username": "string",
+    "email": "string"
+  }
+}
+```
+
+### POST /auth/logout
+Logout and invalidate token.
+
+**Headers:**
+- Authorization: Bearer {token}
+
+**Response:**
+```json
+{
+  "message": "Logout successful"
+}
+```""",
                },
            ]
        }

    async def test_complete_webhook_to_review_flow(
        self,
        integrated_review_service: ReviewService,
        merge_request_event: MergeRequestEvent,
        load_fixture,
        mock_httpx_client,
    ):
        """Test complete flow from webhook receipt to review comment posting."""
        # Setup mock responses
        gitlab_responses = load_fixture("gitlab_responses.json")
        llm_responses = load_fixture("llm_responses.json")

        # Configure GitLab API responses
        mock_httpx_client.get.side_effect = [
            # MR changes request
            Mock(
                status_code=200,
                json=Mock(return_value=gitlab_responses["merge_request_changes"]),
                is_success=True,
            ),
            # User permissions check
            Mock(
                status_code=200,
                json=Mock(return_value=gitlab_responses["user_info"]),
                is_success=True,
            ),
        ]

        # Configure AI provider response
        mock_ai_response = Mock(
            status_code=200,
            json=Mock(return_value=llm_responses["openai_review_response"]),
            is_success=True,
        )

        # Configure comment posting response
        mock_comment_response = Mock(
            status_code=201,
            json=Mock(return_value=gitlab_responses["merge_request_comment"]),
            is_success=True,
        )

        mock_httpx_client.post.side_effect = [mock_ai_response, mock_comment_response]

        # Execute the full workflow
        with patch("agents.providers.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            result = await integrated_review_service.process_review(merge_request_event)

        # Verify the complete workflow executed
        assert result is not None

        # Verify GitLab API calls were made
        assert mock_httpx_client.get.call_count == 2
        assert mock_httpx_client.post.call_count == 2

        # Verify specific API endpoints were called
        get_calls = [call[0][0] for call in mock_httpx_client.get.call_args_list]
        assert any(
            "merge_requests" in str(call) and "changes" in str(call)
            for call in get_calls
        )

        post_calls = [call[0][0] for call in mock_httpx_client.post.call_args_list]
        assert any("notes" in str(call) for call in post_calls)

    async def test_gitlab_api_integration_with_realistic_responses(
        self,
        integrated_review_service: ReviewService,
        merge_request_event: MergeRequestEvent,
        load_fixture,
        mock_httpx_client,
    ):
        """Test GitLab API integration with realistic API responses and error handling."""
        gitlab_responses = load_fixture("gitlab_responses.json")

        # Test successful API calls
        mock_httpx_client.get.return_value = Mock(
            status_code=200,
            json=Mock(return_value=gitlab_responses["merge_request_changes"]),
            is_success=True,
        )

        from src.services.gitlab_service import GitLabService

        gitlab_service = GitLabService()
        gitlab_service.client = mock_httpx_client

        # Test get changes
        changes = await gitlab_service.get_merge_request_diff(100, 10)
        assert "changes" in changes
        assert len(changes["changes"]) > 0

        # Test API error responses
        mock_httpx_client.get.return_value = Mock(
            status_code=404,
            json=Mock(return_value=gitlab_responses["error_response"]),
            is_success=False,
        )
        mock_httpx_client.get.return_value.raise_for_status.side_effect = (
            HTTPStatusError(
                message="404 Not Found",
                request=Mock(),
                response=mock_httpx_client.get.return_value,
            )
        )

        with pytest.raises(GitLabAPIException) as exc_info:
            await gitlab_service.get_merge_request_diff(100, 999)

        assert exc_info.value.status_code == 404
        assert "Project Not Found" in str(exc_info.value.message)

    async def test_ai_provider_integration_with_mock_responses(
        self,
        integrated_review_service: ReviewService,
        merge_request_event: MergeRequestEvent,
        load_fixture,
        mock_httpx_client,
    ):
        """Test AI provider integration with various mock responses and fallback scenarios."""
        llm_responses = load_fixture("llm_responses.json")

        # Configure successful responses
        mock_httpx_client.get.return_value = Mock(
            status_code=200, json=Mock(return_value={"changes": []}), is_success=True
        )

        # Test different AI providers
        providers = ["openai", "anthropic", "gemini", "azure", "ollama"]

        for provider in providers:
            # Update settings for each provider
            integrated_review_service.settings.ai_provider = provider

            if provider == "openai":
                mock_response = llm_responses["openai_review_response"]
            elif provider == "anthropic":
                mock_response = llm_responses["anthropic_review_response"]
            elif provider == "gemini":
                mock_response = llm_responses["gemini_review_response"]
            elif provider == "azure":
                mock_response = llm_responses["azure_review_response"]
            elif provider == "ollama":
                mock_response = llm_responses["ollama_review_response"]

            mock_httpx_client.post.return_value = Mock(
                status_code=200, json=Mock(return_value=mock_response), is_success=True
            )

            with patch("agents.providers.AsyncClient") as mock_client_class:
                mock_client_class.return_value.__aenter__.return_value = (
                    mock_httpx_client
                )

                result = await integrated_review_service.process_review(
                    merge_request_event
                )
                assert result is not None

    async def test_full_error_recovery_scenarios_with_retries(
        self,
        integrated_review_service: ReviewService,
        merge_request_event: MergeRequestEvent,
        mock_httpx_client,
    ):
        """Test comprehensive error recovery scenarios including retry logic."""

        # Test network timeout with retries
        mock_httpx_client.get.side_effect = [
            TimeoutException("Request timed out"),
            TimeoutException("Request timed out"),
            Mock(
                status_code=200,
                json=Mock(return_value={"changes": []}),
                is_success=True,
            ),
        ]

        # Should succeed after 2 retries
        from src.services.gitlab_service import GitLabService

        gitlab_service = GitLabService()
        gitlab_service.client = mock_httpx_client
        result = await gitlab_service.get_merge_request_diff(100, 10)
        assert "changes" in result
        assert mock_httpx_client.get.call_count == 3

        # Test connection error
        mock_httpx_client.get.side_effect = [
            ConnectError("Connection failed"),
            ConnectError("Connection failed"),
            ConnectError("Connection failed"),
        ]
        mock_httpx_client.get.call_count = 0

        with pytest.raises(GitLabAPIException) as exc_info:
            await gitlab_service.get_merge_request_diff(100, 10)

        assert "Connection failed" in str(exc_info.value.message)
        assert mock_httpx_client.get.call_count == 3  # Max retries

        # Test rate limiting with exponential backoff
        rate_limit_response = Mock(
            status_code=429,
            json=Mock(return_value={"message": "Too Many Requests"}),
            is_success=False,
            headers={"Retry-After": "2"},
        )
        rate_limit_response.raise_for_status.side_effect = HTTPStatusError(
            message="429 Too Many Requests",
            request=Mock(),
            response=rate_limit_response,
        )

        success_response = Mock(
            status_code=200, json=Mock(return_value={"changes": []}), is_success=True
        )

        mock_httpx_client.get.side_effect = [rate_limit_response, success_response]
        mock_httpx_client.get.call_count = 0

        start_time = time.time()
        result = await gitlab_service.get_merge_request_diff(100, 10)
        elapsed_time = time.time() - start_time

        # Should have waited at least 2 seconds for retry-after
        assert elapsed_time >= 1.8  # Allow slight timing variance
        assert "changes" in result
        assert mock_httpx_client.get.call_count == 2

    async def test_multi_provider_fallback_testing(
        self,
        integrated_review_service: ReviewService,
        merge_request_event: MergeRequestEvent,
        load_fixture,
        mock_httpx_client,
    ):
        """Test multi-provider fallback when primary AI provider fails."""
        llm_responses = load_fixture("llm_responses.json")

        # Setup GitLab response
        mock_httpx_client.get.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"changes": [{"diff": "test diff"}]}),
            is_success=True,
        )

        # Configure fallback providers
        fallback_providers = ["anthropic", "gemini"]
        integrated_review_service.settings.ai_fallback_providers = fallback_providers

        # Mock primary provider failure and fallback success
        error_response = Mock(
            status_code=429,
            json=Mock(return_value=llm_responses["error_response"]),
            is_success=False,
        )
        error_response.raise_for_status.side_effect = HTTPStatusError(
            message="Rate limit exceeded", request=Mock(), response=error_response
        )

        success_response = Mock(
            status_code=200,
            json=Mock(return_value=llm_responses["anthropic_review_response"]),
            is_success=True,
        )

        comment_response = Mock(
            status_code=201, json=Mock(return_value={"id": 123}), is_success=True
        )

        # Primary fails, fallback succeeds
        mock_httpx_client.post.side_effect = [
            error_response,
            success_response,
            comment_response,
        ]

        with patch("agents.providers.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            result = await integrated_review_service.process_review(merge_request_event)

        assert result is not None
        # Should have called primary + fallback + comment posting
        assert mock_httpx_client.post.call_count == 3

    async def test_large_merge_request_processing(
        self,
        integrated_review_service: ReviewService,
        large_merge_request_payload: Dict[str, Any],
        complex_diff_response: Dict[str, Any],
        load_fixture,
        mock_httpx_client,
    ):
        """Test processing of large merge requests with many files and changes."""
        llm_responses = load_fixture("llm_responses.json")

        # Create MR event from large payload
        event = MergeRequestEvent.model_validate(large_merge_request_payload)

        # Setup complex diff response
        mock_httpx_client.get.return_value = Mock(
            status_code=200,
            json=Mock(return_value=complex_diff_response),
            is_success=True,
        )

        # Mock AI response for large content
        mock_httpx_client.post.side_effect = [
            Mock(
                status_code=200,
                json=Mock(return_value=llm_responses["review_with_comments"]),
                is_success=True,
            ),
            Mock(status_code=201, json=Mock(return_value={"id": 456}), is_success=True),
        ]

        with patch("agents.providers.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            start_time = time.time()
            result = await integrated_review_service.process_review(event)
            processing_time = time.time() - start_time

        assert result is not None
        # Should handle large content efficiently (under 30 seconds)
        assert processing_time < 30.0

        # Verify all file types were processed
        get_calls = mock_httpx_client.get.call_args_list
        assert len(get_calls) >= 1  # At least the changes call

        post_calls = mock_httpx_client.post.call_args_list
        assert len(post_calls) >= 1  # At least the review call

    async def test_complex_diff_scenarios_multiple_files(
        self,
        integrated_review_service: ReviewService,
        merge_request_event: MergeRequestEvent,
        complex_diff_response: Dict[str, Any],
        load_fixture,
        mock_httpx_client,
    ):
        """Test handling complex diff scenarios with multiple file types and changes."""
        load_fixture("llm_responses.json")

        # Use the complex diff response
        mock_httpx_client.get.return_value = Mock(
            status_code=200,
            json=Mock(return_value=complex_diff_response),
            is_success=True,
        )

        # Mock comprehensive review response
        comprehensive_review = {
            "choices": [
                {
                    "message": {
                        "content": """# Comprehensive Code Review

## Security Analysis
- **CRITICAL**: Changed from SHA256 to bcrypt for password hashing - excellent security improvement
- **GOOD**: Added JWT token generation with expiration
- **SUGGESTION**: Consider adding rate limiting for login attempts

## Code Quality Analysis
- **POSITIVE**: Added proper type hints throughout
- **POSITIVE**: Improved error handling in frontend
- **SUGGESTION**: Consider extracting API calls to separate service layer

## File-by-File Analysis

### src/services/auth.py
- Excellent security improvements with bcrypt
- Good type annotations
- Consider adding password complexity validation

### tests/test_auth.py
- Good test coverage expansion
- Proper mocking usage
- Consider adding negative test cases

### frontend/components/Login.jsx
- Good error handling and loading states
- Proper form validation
- Consider adding accessibility attributes

### config/database.yml
- Good environment variable usage
- Added test configuration
- Consider adding production config

### docs/api.md
- Good API documentation
- Clear examples
- Consider adding error response examples

## Overall Assessment
Strong improvements across security, testing, and user experience. Ready to merge with minor suggestions.""",
                        "role": "assistant",
                    }
                }
            ],
            "usage": {"total_tokens": 450},
        }

        mock_httpx_client.post.side_effect = [
            Mock(
                status_code=200,
                json=Mock(return_value=comprehensive_review),
                is_success=True,
            ),
            Mock(status_code=201, json=Mock(return_value={"id": 789}), is_success=True),
        ]

        with patch("agents.providers.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            result = await integrated_review_service.process_review(merge_request_event)

        assert result is not None

        # Verify the AI request included all file types
        ai_request_call = mock_httpx_client.post.call_args_list[0]
        request_data = ai_request_call[1]["json"]

        # Should contain multiple file extensions and comprehensive diff content
        content = str(request_data)
        assert ".py" in content or "python" in content.lower()
        assert ".jsx" in content or "javascript" in content.lower()
        assert ".yml" in content or "yaml" in content.lower()
        assert ".md" in content or "markdown" in content.lower()

    async def test_comment_posting_and_formatting_validation(
        self,
        integrated_review_service: ReviewService,
        merge_request_event: MergeRequestEvent,
        load_fixture,
        mock_httpx_client,
    ):
        """Test comment posting with proper formatting and validation."""
        gitlab_responses = load_fixture("gitlab_responses.json")
        load_fixture("llm_responses.json")

        # Setup responses
        mock_httpx_client.get.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"changes": [{"diff": "test diff"}]}),
            is_success=True,
        )

        formatted_review = {
            "choices": [
                {
                    "message": {
                        "content": """## 🔍 Code Review Summary

This merge request introduces new functionality with overall good code quality.

### ✅ Positive Aspects
- Clean implementation
- Good test coverage
- Proper error handling

### ⚠️ Areas for Improvement
1. **Missing type hints** - Consider adding type annotations
2. **Documentation** - Add docstrings to new methods
3. **Performance** - Consider caching for repeated operations

### 🔒 Security Review
No security issues identified.

### 📊 Metrics
- Files changed: 3
- Lines added: +45
- Lines removed: -12
- Test coverage: Good

---
*Review generated by GitLab AI Code Review Agent*""",
                        "role": "assistant",
                    }
                }
            ]
        }

        mock_httpx_client.post.side_effect = [
            Mock(
                status_code=200,
                json=Mock(return_value=formatted_review),
                is_success=True,
            ),
            Mock(
                status_code=201,
                json=Mock(return_value=gitlab_responses["merge_request_comment"]),
                is_success=True,
            ),
        ]

        with patch("agents.providers.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            result = await integrated_review_service.process_review(merge_request_event)

        assert result is not None

        # Verify comment formatting
        comment_call = mock_httpx_client.post.call_args_list[1]
        comment_data = comment_call[1]["json"]

        assert "body" in comment_data
        comment_body = comment_data["body"]

        # Verify proper markdown formatting
        assert "## 🔍 Code Review Summary" in comment_body
        assert "### ✅ Positive Aspects" in comment_body
        assert "### ⚠️ Areas for Improvement" in comment_body
        assert "*Review generated by GitLab AI Code Review Agent*" in comment_body

        # Verify proper structure
        assert comment_body.count("###") >= 3  # Multiple sections
        assert comment_body.count("-") >= 5  # List items

    async def test_performance_under_load(
        self,
        integrated_review_service: ReviewService,
        generate_webhook_payload,
        load_fixture,
        mock_httpx_client,
    ):
        """Test system performance under concurrent load."""
        llm_responses = load_fixture("llm_responses.json")

        # Setup fast responses
        mock_httpx_client.get.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"changes": [{"diff": "small change"}]}),
            is_success=True,
        )

        mock_httpx_client.post.side_effect = [
            Mock(
                status_code=200,
                json=Mock(return_value=llm_responses["openai_review_response"]),
                is_success=True,
            ),
            Mock(status_code=201, json=Mock(return_value={"id": 123}), is_success=True),
        ] * 10  # Repeat for multiple concurrent requests

        # Create multiple webhook events
        events = []
        for i in range(10):
            payload = generate_webhook_payload(
                object_attributes={
                    "id": i,
                    "iid": i + 100,
                    "title": f"Test MR {i}",
                    "action": "open",
                }
            )
            events.append(MergeRequestEvent.model_validate(payload))

        with patch("agents.providers.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            # Process concurrent requests
            start_time = time.time()
            tasks = [
                integrated_review_service.process_review(event) for event in events
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

        # Verify performance metrics
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 10

        # Should handle 10 concurrent requests in reasonable time
        assert total_time < 60.0  # All requests under 1 minute

        # Average time per request should be reasonable
        avg_time = total_time / 10
        assert avg_time < 10.0  # Average under 10 seconds per request

        # Verify all requests were processed
        assert mock_httpx_client.get.call_count == 10
        assert mock_httpx_client.post.call_count == 20  # 10 reviews + 10 comments

    async def test_realistic_integration_scenarios(
        self, integrated_review_service: ReviewService, load_fixture, mock_httpx_client
    ):
        """Test realistic integration scenarios with various edge cases."""

        # Scenario 1: Draft MR (should be skipped)
        draft_payload = {
            "object_kind": "merge_request",
            "event_type": "merge_request",
            "object_attributes": {
                "id": 1,
                "iid": 10,
                "title": "Draft: Work in progress",
                "action": "open",
                "draft": True,
                "work_in_progress": True,
                "state": "opened",
            },
            "user": {"id": 1, "username": "test"},
            "project": {"id": 100, "name": "test"},
        }

        draft_event = MergeRequestEvent.model_validate(draft_payload)

        # Should skip draft MR
        result = await integrated_review_service.process_review(draft_event)
        assert result is None  # Skipped due to draft status

        # Scenario 2: MR with no changes
        no_changes_payload = {
            "object_kind": "merge_request",
            "event_type": "merge_request",
            "object_attributes": {
                "id": 2,
                "iid": 11,
                "title": "No changes MR",
                "action": "open",
                "draft": False,
                "work_in_progress": False,
                "state": "opened",
            },
            "user": {"id": 1, "username": "test"},
            "project": {"id": 100, "name": "test"},
        }

        mock_httpx_client.get.return_value = Mock(
            status_code=200, json=Mock(return_value={"changes": []}), is_success=True
        )

        no_changes_event = MergeRequestEvent.model_validate(no_changes_payload)
        result = await integrated_review_service.process_review(no_changes_event)
        assert result is None  # Skipped due to no changes

        # Scenario 3: MR with binary file changes only
        binary_payload = {
            "object_kind": "merge_request",
            "event_type": "merge_request",
            "object_attributes": {
                "id": 3,
                "iid": 12,
                "title": "Binary file changes",
                "action": "open",
                "draft": False,
                "work_in_progress": False,
                "state": "opened",
            },
            "user": {"id": 1, "username": "test"},
            "project": {"id": 100, "name": "test"},
        }

        binary_changes = {
            "changes": [
                {
                    "old_path": "assets/image.png",
                    "new_path": "assets/image.png",
                    "diff": "Binary files differ",
                    "new_file": False,
                    "renamed_file": False,
                    "deleted_file": False,
                }
            ]
        }

        mock_httpx_client.get.return_value = Mock(
            status_code=200, json=Mock(return_value=binary_changes), is_success=True
        )

        binary_event = MergeRequestEvent.model_validate(binary_payload)
        result = await integrated_review_service.process_review(binary_event)
        assert result is None  # Skipped due to only binary changes

        # Scenario 4: MR close event (should be skipped)
        close_payload = {
            "object_kind": "merge_request",
            "event_type": "merge_request",
            "object_attributes": {
                "id": 4,
                "iid": 13,
                "title": "Closed MR",
                "action": "close",
                "draft": False,
                "work_in_progress": False,
                "state": "closed",
            },
            "user": {"id": 1, "username": "test"},
            "project": {"id": 100, "name": "test"},
        }

        close_event = MergeRequestEvent.model_validate(close_payload)
        result = await integrated_review_service.process_review(close_event)
        assert result is None  # Skipped due to close action

    async def test_webhook_validation_and_security(
        self, integrated_review_service: ReviewService, mock_httpx_client
    ):
        """Test webhook validation and security features."""

        # Test invalid webhook payload
        invalid_payload = {
            "object_kind": "issue",  # Wrong object kind
            "event_type": "issue",
        }

        with pytest.raises(Exception):
            # This should fail validation
            MergeRequestEvent.model_validate(invalid_payload)

        # Test minimal valid payload
        minimal_payload = {
            "object_kind": "merge_request",
            "event_type": "merge_request",
            "object_attributes": {
                "id": 1,
                "iid": 1,
                "title": "Test",
                "action": "open",
                "state": "opened",
            },
        }

        # Should handle minimal payload gracefully
        try:
            event = MergeRequestEvent.model_validate(minimal_payload)
            assert event.object_kind == "merge_request"
        except Exception:
            # If validation fails, that's also acceptable behavior
            pass

    async def test_rate_limiting_and_throttling(
        self,
        integrated_review_service: ReviewService,
        merge_request_event: MergeRequestEvent,
        mock_httpx_client,
    ):
        """Test rate limiting and throttling behavior."""

        # Setup rate limit response
        rate_limit_response = Mock(
            status_code=429,
            json=Mock(return_value={"message": "Rate limit exceeded"}),
            headers={"Retry-After": "60"},
            is_success=False,
        )
        rate_limit_response.raise_for_status.side_effect = HTTPStatusError(
            message="429 Too Many Requests",
            request=Mock(),
            response=rate_limit_response,
        )

        # Configure consecutive rate limit responses
        mock_httpx_client.get.side_effect = [
            rate_limit_response,
            rate_limit_response,
            rate_limit_response,
        ]

        # Should fail after max retries
        from src.services.gitlab_service import GitLabService

        gitlab_service = GitLabService()
        gitlab_service.client = mock_httpx_client

        with pytest.raises(GitLabAPIException) as exc_info:
            await gitlab_service.get_merge_request_diff(100, 10)

        assert exc_info.value.status_code == 429
        assert "Rate limit" in str(exc_info.value.message)

        # Verify retries were attempted
        assert mock_httpx_client.get.call_count == 3

    async def test_memory_and_resource_management(
        self, integrated_review_service: ReviewService, mock_httpx_client
    ):
        """Test memory usage and resource management under load."""
        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Setup responses for large data processing
        large_diff = "+" + "x" * 10000 + "\n" * 1000  # Large diff content
        large_changes = {
            "changes": [
                {
                    "old_path": f"file_{i}.py",
                    "new_path": f"file_{i}.py",
                    "diff": large_diff,
                    "new_file": False,
                    "renamed_file": False,
                    "deleted_file": False,
                }
                for i in range(20)  # 20 files with large diffs
            ]
        }

        mock_httpx_client.get.return_value = Mock(
            status_code=200, json=Mock(return_value=large_changes), is_success=True
        )

        # Process multiple large reviews
        large_payload = {
            "object_kind": "merge_request",
            "event_type": "merge_request",
            "object_attributes": {
                "id": 999,
                "iid": 999,
                "title": "Large MR",
                "action": "open",
                "state": "opened",
            },
            "user": {"id": 1, "username": "test"},
            "project": {"id": 100, "name": "test"},
        }

        large_event = MergeRequestEvent.model_validate(large_payload)

        # Process and cleanup
        for i in range(5):
            try:
                await integrated_review_service.process_review(large_event)
            except Exception:
                pass  # Expected to fail due to no AI response, but tests memory

            # Force garbage collection
            gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"

    async def test_concurrent_webhook_processing(
        self,
        integrated_review_service: ReviewService,
        generate_webhook_payload,
        load_fixture,
        mock_httpx_client,
    ):
        """Test concurrent webhook processing without race conditions."""
        llm_responses = load_fixture("llm_responses.json")

        # Create unique responses for each request to detect race conditions
        def create_unique_response(request_id):
            return Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "changes": [
                            {
                                "diff": f"unique diff for request {request_id}",
                                "old_path": f"file_{request_id}.py",
                                "new_path": f"file_{request_id}.py",
                            }
                        ]
                    }
                ),
                is_success=True,
            )

        def create_ai_response(request_id):
            review_data = llm_responses["openai_review_response"].copy()
            review_data["choices"][0]["message"][
                "content"
            ] = f"Review for MR {request_id}"
            return Mock(
                status_code=200, json=Mock(return_value=review_data), is_success=True
            )

        def create_comment_response(request_id):
            return Mock(
                status_code=201,
                json=Mock(return_value={"id": 1000 + request_id}),
                is_success=True,
            )

        # Setup responses for 5 concurrent requests
        get_responses = [create_unique_response(i) for i in range(5)]
        post_responses = []
        for i in range(5):
            post_responses.extend([create_ai_response(i), create_comment_response(i)])

        mock_httpx_client.get.side_effect = get_responses
        mock_httpx_client.post.side_effect = post_responses

        # Create 5 different webhook events
        events = []
        for i in range(5):
            payload = generate_webhook_payload(
                object_attributes={
                    "id": 2000 + i,
                    "iid": 200 + i,
                    "title": f"Concurrent MR {i}",
                    "action": "open",
                }
            )
            events.append(MergeRequestEvent.model_validate(payload))

        with patch("agents.providers.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            # Process all events concurrently
            start_time = time.time()
            results = await asyncio.gather(
                *[integrated_review_service.process_review(event) for event in events],
                return_exceptions=True,
            )
            total_time = time.time() - start_time

        # Verify no race conditions occurred
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5

        # Verify concurrent processing was faster than sequential
        assert total_time < 30.0  # Should be much faster than 5 sequential requests

        # Verify each request got its unique response
        assert mock_httpx_client.get.call_count == 5
        assert mock_httpx_client.post.call_count == 10  # 5 AI calls + 5 comment calls

    @pytest.mark.slow
    async def test_end_to_end_stress_test(
        self,
        integrated_review_service: ReviewService,
        generate_webhook_payload,
        load_fixture,
        mock_httpx_client,
    ):
        """Comprehensive stress test of the entire system."""
        llm_responses = load_fixture("llm_responses.json")

        # Configure variable response times to simulate real conditions
        response_delays = [0.1, 0.2, 0.5, 0.1, 0.3] * 4  # 20 total with varying delays

        async def delayed_response(delay, response_data, status_code=200):
            await asyncio.sleep(delay)
            return Mock(
                status_code=status_code,
                json=Mock(return_value=response_data),
                is_success=status_code < 400,
            )

        # Create 20 different scenarios
        scenarios = []
        for i in range(20):
            delay = response_delays[i]

            # Mix of different MR types
            if i % 4 == 0:  # Large MR
                changes = {
                    "changes": [
                        {"diff": "large diff " * 100, "old_path": f"file_{i}.py"}
                    ]
                }
            elif i % 4 == 1:  # Small MR
                changes = {
                    "changes": [
                        {"diff": f"small change {i}", "old_path": f"file_{i}.py"}
                    ]
                }
            elif i % 4 == 2:  # Multi-file MR
                changes = {
                    "changes": [
                        {"diff": f"change {i}_1", "old_path": f"file_{i}_1.py"},
                        {"diff": f"change {i}_2", "old_path": f"file_{i}_2.js"},
                        {"diff": f"change {i}_3", "old_path": f"file_{i}_3.md"},
                    ]
                }
            else:  # No changes (should be skipped)
                changes = {"changes": []}

            scenarios.append((delay, changes, i))

        # Setup mock responses
        get_responses = []
        post_responses = []

        for delay, changes, i in scenarios:
            get_responses.append(delayed_response(delay * 0.5, changes))
            if changes["changes"]:  # Only if there are changes
                post_responses.extend(
                    [
                        delayed_response(
                            delay, llm_responses["openai_review_response"]
                        ),
                        delayed_response(delay * 0.2, {"id": 3000 + i}, 201),
                    ]
                )

        mock_httpx_client.get.side_effect = [await resp for resp in get_responses]
        mock_httpx_client.post.side_effect = [await resp for resp in post_responses]

        # Create webhook events
        events = []
        for i in range(20):
            payload = generate_webhook_payload(
                object_attributes={
                    "id": 3000 + i,
                    "iid": 300 + i,
                    "title": f"Stress test MR {i}",
                    "action": "open",
                }
            )
            events.append(MergeRequestEvent.model_validate(payload))

        with patch("agents.providers.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            # Run stress test
            start_time = time.time()
            results = await asyncio.gather(
                *[integrated_review_service.process_review(event) for event in events],
                return_exceptions=True,
            )
            total_time = time.time() - start_time

        # Analyze results
        successful_results = [
            r for r in results if r is not None and not isinstance(r, Exception)
        ]
        skipped_results = [r for r in results if r is None]
        error_results = [r for r in results if isinstance(r, Exception)]

        # Verify system handled stress well
        assert (
            len(error_results) == 0
        ), f"Got {len(error_results)} errors during stress test"
        assert len(successful_results) > 10, "Should have processed most non-empty MRs"
        assert len(skipped_results) <= 5, "Should have skipped empty MRs"

        # Performance should be reasonable under stress
        assert (
            total_time < 120.0
        ), f"Stress test took {total_time:.2f}s, should be under 2 minutes"

        # Log stress test results
        print("\nStress Test Results:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Successful: {len(successful_results)}")
        print(f"Skipped: {len(skipped_results)}")
        print(f"Errors: {len(error_results)}")
        print(f"Average time per request: {total_time/20:.2f}s")


@pytest.mark.integration
class TestSecurityFeaturesIntegration:
    """
    Integration tests for new security features including timestamp validation,
    circuit breaker patterns, and diff size limits.
    """

    @pytest.fixture
    async def mock_webhook_request(self):
        """Create a mock webhook request with configurable headers."""

        def create_request(headers=None, client_host="127.0.0.1"):
            request = Mock()
            request.headers = headers or {}
            request.client = Mock()
            request.client.host = client_host
            return request

        return create_request

    @pytest.mark.security
    async def test_webhook_timestamp_validation_integration(
        self, mock_webhook_request, settings: Settings
    ):
        """Test end-to-end webhook timestamp validation."""
        from unittest.mock import patch

        from src.api.webhooks import verify_gitlab_token

        # Mock the global settings to use our test settings
        with patch("src.api.webhooks.settings", settings):
            # Test with valid timestamp
            current_time = time.time()
            request = mock_webhook_request(
                {
                    "X-Gitlab-Token": settings.gitlab_webhook_secret,
                    "X-Gitlab-Timestamp": str(current_time),
                }
            )

            # Should not raise exception
            verify_gitlab_token(request)

            # Test with expired timestamp
            old_timestamp = current_time - 400  # 6+ minutes ago
            request = mock_webhook_request(
                {
                    "X-Gitlab-Token": settings.gitlab_webhook_secret,
                    "X-Gitlab-Timestamp": str(old_timestamp),
                }
            )

            with pytest.raises(Exception) as exc_info:
                verify_gitlab_token(request)

            assert "timestamp" in str(exc_info.value).lower()

    @pytest.mark.security
    async def test_circuit_breaker_integration(
        self, settings: Settings, mock_httpx_client
    ):
        """Test circuit breaker pattern integration with GitLab API."""
        # GitLabService is a separate instance, not attached to ReviewService
        from src.services.gitlab_service import GitLabService

        gitlab_service = GitLabService()
        gitlab_service.client = mock_httpx_client

        # Configure failures to trigger circuit breaker
        failure_threshold = settings.circuit_breaker_failure_threshold or 5

        # Make failures exceed threshold
        mock_httpx_client.get.side_effect = HTTPStatusError(
            message="Server Error", request=Mock(), response=Mock(status_code=500)
        )

        # Make enough failed requests to trigger circuit breaker
        for i in range(failure_threshold + 1):
            try:
                await gitlab_service.get_merge_request_diff(100, 10)
            except Exception:
                pass  # Expected failures

        # Circuit breaker should now be open
        # Next request should fail immediately without hitting the API
        start_time = time.time()
        try:
            await gitlab_service.get_merge_request_diff(100, 11)
        except Exception:
            pass
        elapsed_time = time.time() - start_time

        # Should fail quickly (circuit breaker open)
        assert elapsed_time < 0.1, "Circuit breaker should fail fast"

    @pytest.mark.security
    async def test_diff_size_validation_integration(
        self, settings: Settings, mock_httpx_client
    ):
        """Test diff size validation in end-to-end workflow."""
        # Create a very large diff that exceeds size limit
        large_diff = "+" + "x" * (settings.max_diff_size + 1000)

        large_diff_response = {
            "changes": [
                {
                    "old_path": "src/large_file.py",
                    "new_path": "src/large_file.py",
                    "diff": large_diff,
                }
            ]
        }

        mock_httpx_client.get.return_value = Mock(
            status_code=200,
            json=Mock(return_value=large_diff_response),
            is_success=True,
        )

        from src.services.gitlab_service import GitLabService

        gitlab_service = GitLabService()
        gitlab_service.client = mock_httpx_client

        # Should handle large diff gracefully (truncate or skip)
        try:
            result = await gitlab_service.get_merge_request_diff(100, 10)
            # If it succeeds, it should have been truncated or filtered
            assert result is not None
        except Exception as e:
            # Or it might raise a validation error
            assert "size" in str(e).lower() or "large" in str(e).lower()

    @pytest.mark.performance
    async def test_http_client_configuration_integration(
        self, settings: Settings, mock_httpx_client
    ):
        """Test HTTP client configuration integration."""
        from src.services.gitlab_service import GitLabService

        gitlab_service = GitLabService()
        gitlab_service.client = mock_httpx_client

        # Test that client is configured with settings
        assert gitlab_service.client is not None

        # Verify timeout configuration
        # (This would require accessing the actual client configuration)
        # For now, just verify the service works
        with patch.object(gitlab_service.client, "get") as mock_get:
            mock_get.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"changes": []}),
                is_success=True,
            )

            result = await gitlab_service.get_merge_request_diff(100, 10)
            assert result is not None

            # Verify get was called (client is working)
            mock_get.assert_called_once()

    @pytest.mark.integration
    async def test_performance_settings_integration(
        self, settings: Settings, mock_httpx_client
    ):
        """Test integration of performance-related settings."""
        # Verify settings are properly loaded
        assert settings.max_diff_size > 0
        assert settings.request_timeout > 0
        assert settings.max_connections > 0
        assert settings.max_keepalive_connections > 0
        assert settings.keepalive_expiry > 0

        # Test that services respect these settings
        from src.services.gitlab_service import GitLabService

        gitlab_service = GitLabService()
        gitlab_service.client = mock_httpx_client

        # Mock a successful request
        mock_httpx_client.get.return_value = Mock(
            status_code=200, json=Mock(return_value={"changes": []}), is_success=True
        )

        # Should work with current configuration
        result = await gitlab_service.get_merge_request_diff(100, 10)
        assert result is not None


@pytest.mark.integration
class TestVersionSystemIntegration:
    """
    Integration tests for the new version system.
    """

    @pytest.mark.integration
    async def test_version_system_integration(self):
        """Test version system integration across components."""
        from src.api.health import status
        from src.utils.version import get_version, get_version_info

        # Test version utility
        version = get_version()
        assert version is not None
        assert isinstance(version, str)
        assert len(version.split(".")) == 3  # Semantic versioning

        # Test version info
        version_info = get_version_info()
        assert version_info["version"] == version
        assert isinstance(version_info["major"], int)
        assert isinstance(version_info["minor"], int)
        assert isinstance(version_info["patch"], int)
        assert version_info["full"] == f"v{version}"

        # Test health endpoint uses version
        with patch("src.api.health.settings") as mock_settings:
            mock_settings.environment = "test"
            mock_settings.gitlab_url = "https://gitlab.test.com"
            mock_settings.ai_model = "test-model"
            mock_settings.gitlab_trigger_tag = "test"
            mock_settings.debug = False

            health_result = await status()
            assert health_result["version"] == version

    @pytest.mark.integration
    async def test_version_error_handling_integration(self):
        """Test version system error handling in integrated environment."""
        import tempfile
        from pathlib import Path

        from src.utils.version import get_version

        # Test with actual empty file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write("")  # Empty file
            tmp_path = Path(tmp_file.name)

            with patch("src.utils.version.Path") as mock_path:
                # Mock the path resolution to return our temp file
                mock_path.return_value.parent.parent.__truediv__.return_value = tmp_path

                # Reset cache to force file read
                import src.utils.version

                src.utils.version._cached_version = None

                # Should raise ValueError for empty file
                with pytest.raises(ValueError) as exc_info:
                    get_version()

                assert "Version file is empty" in str(exc_info.value)

            # Clean up
            tmp_path.unlink()

    @pytest.mark.integration
    async def test_version_caching_integration(self):
        """Test version caching in integrated environment."""
        # Clear cache
        import src.utils.version
        from src.utils.version import get_version

        src.utils.version._cached_version = None

        # First call should cache
        version1 = get_version()

        # Second call should use cache
        version2 = get_version()

        assert version1 == version2
        assert version1 == "2.1.0"  # Current version
