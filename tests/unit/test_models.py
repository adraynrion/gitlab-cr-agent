"""Unit tests for data models and validation."""

import json

import pytest
from pydantic import ValidationError

from src.models.gitlab_models import GitLabProject, GitLabUser, MergeRequestAttributes
from src.models.review_models import CodeIssue, ReviewContext, ReviewResult


class TestGitLabUser:
    """Test GitLabUser model validation."""

    def test_valid_user_creation(self):
        """Test creating a valid user."""
        user_data = {
            "id": 1,
            "username": "testuser",
            "name": "Test User",
            "email": "test@example.com",
            "avatar_url": "https://example.com/avatar.jpg",
        }

        user = GitLabUser(**user_data)
        assert user.id == 1
        assert user.username == "testuser"
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.avatar_url == "https://example.com/avatar.jpg"

    def test_user_with_minimal_data(self):
        """Test user creation with minimal required data."""
        user_data = {
            "id": 1,
            "username": "testuser",
            "name": "Test User",
            "email": "test@example.com",
        }

        user = GitLabUser(**user_data)
        assert user.id == 1
        assert user.username == "testuser"
        assert user.avatar_url is None

    def test_user_serialization(self):
        """Test user serialization to dict."""
        user_data = {
            "id": 1,
            "username": "testuser",
            "name": "Test User",
            "email": "test@example.com",
            "avatar_url": "https://example.com/avatar.jpg",
        }

        user = GitLabUser(**user_data)
        serialized = user.model_dump()

        assert serialized == user_data


class TestGitLabProject:
    """Test GitLabProject model validation."""

    def test_valid_project_creation(self):
        """Test creating a valid project."""
        project_data = {
            "id": 100,
            "name": "test-project",
            "description": "Test project description",
            "web_url": "https://gitlab.example.com/test-group/test-project",
            "avatar_url": "https://example.com/avatar.jpg",
            "git_ssh_url": "git@gitlab.example.com:test-group/test-project.git",
            "git_http_url": "https://gitlab.example.com/test-group/test-project.git",
            "namespace": "test-group",
            "visibility_level": 0,
            "path_with_namespace": "test-group/test-project",
            "default_branch": "main",
        }

        project = GitLabProject(**project_data)
        assert project.id == 100
        assert project.name == "test-project"
        assert project.namespace == "test-group"


class TestMergeRequestAttributes:
    """Test MergeRequestAttributes model validation."""

    def test_valid_mr_attributes_creation(self):
        """Test creating valid merge request attributes."""
        mr_data = {
            "id": 1,
            "iid": 10,
            "title": "Add new feature",
            "description": "This MR adds a new feature",
            "state": "opened",
            "created_at": "2024-08-15 12:00:00 UTC",
            "updated_at": "2024-08-15 12:30:00 UTC",
            "target_branch": "main",
            "source_branch": "feature-branch",
            "source_project_id": 100,
            "target_project_id": 100,
            "author_id": 1,
            "url": "https://gitlab.example.com/test-group/test-project/-/merge_requests/10",
            "source": {"id": 100, "name": "test-project"},
            "target": {"id": 100, "name": "test-project"},
            "last_commit": {"id": "abc123", "message": "Add feature"},
            "work_in_progress": False,
            "action": "open",
        }

        mr_attrs = MergeRequestAttributes(**mr_data)
        assert mr_attrs.id == 1
        assert mr_attrs.iid == 10
        assert mr_attrs.title == "Add new feature"
        assert mr_attrs.state == "opened"
        assert mr_attrs.work_in_progress is False


class TestCodeIssue:
    """Test CodeIssue model validation."""

    def test_valid_code_issue_creation(self):
        """Test creating a valid code issue."""
        issue_data = {
            "file_path": "src/main.py",
            "line_number": 42,
            "severity": "high",
            "category": "security",
            "description": "Potential SQL injection vulnerability",
            "suggestion": "Use parameterized queries instead",
            "code_example": "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
        }

        issue = CodeIssue(**issue_data)
        assert issue.file_path == "src/main.py"
        assert issue.line_number == 42
        assert issue.severity == "high"
        assert issue.category == "security"

    def test_code_issue_without_example(self):
        """Test code issue creation without code example."""
        issue_data = {
            "file_path": "src/utils.py",
            "line_number": 15,
            "severity": "medium",
            "category": "performance",
            "description": "Inefficient loop implementation",
            "suggestion": "Consider using list comprehension",
        }

        issue = CodeIssue(**issue_data)
        assert issue.code_example is None

    def test_code_issue_line_number_validation(self):
        """Test line number validation (must be >= 1)."""
        # Valid line numbers
        valid_line_numbers = [1, 10, 100, 1000, 999999]

        for line_num in valid_line_numbers:
            issue_data = {
                "file_path": "src/test.py",
                "line_number": line_num,
                "severity": "medium",
                "category": "style",
                "description": "Test issue",
                "suggestion": "Fix this",
            }

            issue = CodeIssue(**issue_data)
            assert issue.line_number == line_num

    def test_code_issue_invalid_line_numbers(self):
        """Test that invalid line numbers are rejected."""
        invalid_line_numbers = [0, -1, -10]

        for line_num in invalid_line_numbers:
            issue_data = {
                "file_path": "src/test.py",
                "line_number": line_num,
                "severity": "medium",
                "category": "style",
                "description": "Test issue",
                "suggestion": "Fix this",
            }

            with pytest.raises(ValidationError) as exc_info:
                CodeIssue(**issue_data)

            assert "line_number" in str(exc_info.value)

    def test_code_issue_line_number_edge_cases(self):
        """Test line number edge cases."""
        # Test line number 1 (minimum valid value)
        issue_data = {
            "file_path": "src/test.py",
            "line_number": 1,
            "severity": "low",
            "category": "style",
            "description": "First line issue",
            "suggestion": "Fix first line",
        }

        issue = CodeIssue(**issue_data)
        assert issue.line_number == 1

        # Test very large line number
        issue_data["line_number"] = 2147483647  # Max int32
        issue = CodeIssue(**issue_data)
        assert issue.line_number == 2147483647


class TestReviewResult:
    """Test ReviewResult model validation."""

    def test_valid_review_result_creation(self):
        """Test creating a valid review result."""
        result_data = {
            "overall_assessment": "approve_with_changes",
            "risk_level": "medium",
            "summary": "Good implementation with minor improvements needed",
            "issues": [
                {
                    "file_path": "src/main.py",
                    "line_number": 42,
                    "severity": "medium",
                    "category": "style",
                    "description": "Line too long",
                    "suggestion": "Break into multiple lines",
                }
            ],
            "positive_feedback": ["Good error handling", "Clear variable names"],
            "metrics": {"lines_reviewed": 150, "files_changed": 3},
        }

        result = ReviewResult(**result_data)
        assert result.overall_assessment == "approve_with_changes"
        assert result.risk_level == "medium"
        assert len(result.issues) == 1
        assert len(result.positive_feedback) == 2

    def test_review_result_with_empty_lists(self):
        """Test review result with empty issues and feedback."""
        result_data = {
            "overall_assessment": "approve",
            "risk_level": "low",
            "summary": "Excellent code quality",
        }

        result = ReviewResult(**result_data)
        assert result.issues == []
        assert result.positive_feedback == []
        assert result.metrics == {}


class TestReviewContext:
    """Test ReviewContext model validation."""

    def test_valid_review_context_creation(self):
        """Test creating a valid review context."""
        context_data = {
            "repository_url": "https://gitlab.example.com/test-group/test-project",
            "merge_request_iid": 10,
            "source_branch": "feature-branch",
            "target_branch": "main",
            "trigger_tag": "ai-review",
            "file_changes": [
                {
                    "old_path": "src/main.py",
                    "new_path": "src/main.py",
                    "diff": "@@ -1,5 +1,6 @@\n def hello():\n-    print('hello')\n+    print('hello world')",
                }
            ],
        }

        context = ReviewContext(**context_data)
        assert (
            context.repository_url
            == "https://gitlab.example.com/test-group/test-project"
        )
        assert context.merge_request_iid == 10
        assert context.source_branch == "feature-branch"
        assert context.target_branch == "main"
        assert len(context.file_changes) == 1


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_gitlab_user_json_roundtrip(self):
        """Test GitLabUser JSON serialization and deserialization."""
        user_data = {
            "id": 1,
            "username": "testuser",
            "name": "Test User",
            "email": "test@example.com",
            "avatar_url": "https://example.com/avatar.jpg",
        }

        # Create model instance
        user = GitLabUser(**user_data)

        # Serialize to JSON
        json_str = user.model_dump_json()

        # Deserialize back to dict
        deserialized_data = json.loads(json_str)

        # Create new instance from deserialized data
        user2 = GitLabUser(**deserialized_data)

        assert user.id == user2.id
        assert user.username == user2.username
        assert user.email == user2.email

    def test_review_result_json_roundtrip(self):
        """Test ReviewResult JSON serialization and deserialization."""
        result_data = {
            "overall_assessment": "needs_work",
            "risk_level": "high",
            "summary": "Several critical issues found",
            "issues": [
                {
                    "file_path": "src/auth.py",
                    "line_number": 25,
                    "severity": "critical",
                    "category": "security",
                    "description": "Hardcoded password found",
                    "suggestion": "Use environment variables for secrets",
                }
            ],
            "positive_feedback": [],
            "metrics": {"security_issues": 1, "total_lines": 50},
        }

        # Create model instance
        result = ReviewResult(**result_data)

        # Serialize to JSON
        json_str = result.model_dump_json()

        # Deserialize back to dict
        deserialized_data = json.loads(json_str)

        # Create new instance from deserialized data
        result2 = ReviewResult(**deserialized_data)

        assert result.overall_assessment == result2.overall_assessment
        assert result.risk_level == result2.risk_level
        assert len(result.issues) == len(result2.issues)
        assert result.issues[0].severity == result2.issues[0].severity


class TestModelValidation:
    """Test model validation edge cases."""

    def test_invalid_severity_level(self):
        """Test CodeIssue with invalid severity level."""
        issue_data = {
            "file_path": "src/main.py",
            "line_number": 42,
            "severity": "invalid",  # Invalid severity
            "category": "security",
            "description": "Test issue",
            "suggestion": "Fix it",
        }

        with pytest.raises(ValidationError) as exc_info:
            CodeIssue(**issue_data)

        assert "severity" in str(exc_info.value)

    def test_invalid_assessment_type(self):
        """Test ReviewResult with invalid assessment."""
        result_data = {
            "overall_assessment": "maybe",  # Invalid assessment
            "risk_level": "medium",
            "summary": "Test review",
        }

        with pytest.raises(ValidationError) as exc_info:
            ReviewResult(**result_data)

        assert "overall_assessment" in str(exc_info.value)

    def test_negative_line_number(self):
        """Test CodeIssue rejects negative line numbers."""
        issue_data = {
            "file_path": "src/main.py",
            "line_number": -1,  # Should be rejected
            "severity": "high",
            "category": "security",
            "description": "Test issue",
            "suggestion": "Fix it",
        }

        with pytest.raises(ValidationError) as exc_info:
            CodeIssue(**issue_data)

        assert "greater_than_equal" in str(exc_info.value)

    def test_zero_line_number(self):
        """Test CodeIssue rejects zero line numbers."""
        issue_data = {
            "file_path": "src/main.py",
            "line_number": 0,  # Should be rejected
            "severity": "high",
            "category": "security",
            "description": "Test issue",
            "suggestion": "Fix it",
        }

        with pytest.raises(ValidationError) as exc_info:
            CodeIssue(**issue_data)

        assert "greater_than_equal" in str(exc_info.value)

    def test_valid_line_number(self):
        """Test CodeIssue accepts valid positive line numbers."""
        issue_data = {
            "file_path": "src/main.py",
            "line_number": 1,  # Should be accepted
            "severity": "high",
            "category": "security",
            "description": "Test issue",
            "suggestion": "Fix it",
        }

        issue = CodeIssue(**issue_data)
        assert issue.line_number == 1
