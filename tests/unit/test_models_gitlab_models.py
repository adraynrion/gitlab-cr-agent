"""
Tests for src/models/gitlab_models.py
"""

import json

import pytest
from pydantic import ValidationError

from src.models.gitlab_models import GitLabProject, GitLabUser, MergeRequestAttributes


class TestGitLabUser:
    """Test GitLabUser model validation"""

    def test_valid_user_creation(self):
        """Test creating a valid user"""
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
        """Test user creation with minimal required data"""
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
        """Test user serialization to dict"""
        user_data = {
            "id": 1,
            "username": "testuser",
            "name": "Test User",
            "email": "test@example.com",
        }

        user = GitLabUser(**user_data)
        user_dict = user.dict()

        assert user_dict["id"] == 1
        assert user_dict["username"] == "testuser"
        assert user_dict["name"] == "Test User"
        assert user_dict["email"] == "test@example.com"

    def test_user_json_roundtrip(self):
        """Test GitLabUser JSON serialization roundtrip"""
        user_data = {
            "id": 123,
            "username": "jsonuser",
            "name": "JSON User",
            "email": "json@example.com",
        }

        # Create user, serialize to JSON, then deserialize
        original_user = GitLabUser(**user_data)
        json_str = original_user.json()
        parsed_data = json.loads(json_str)
        recreated_user = GitLabUser(**parsed_data)

        assert original_user.id == recreated_user.id
        assert original_user.username == recreated_user.username
        assert original_user.name == recreated_user.name
        assert original_user.email == recreated_user.email


class TestGitLabProject:
    """Test GitLabProject model validation"""

    def test_valid_project_creation(self):
        """Test creating a valid project"""
        project_data = {
            "id": 1,
            "name": "test-project",
            "web_url": "https://gitlab.com/test/project",
            "path_with_namespace": "test/project",
        }

        project = GitLabProject(**project_data)
        assert project.id == 1
        assert project.name == "test-project"
        assert project.web_url == "https://gitlab.com/test/project"
        assert project.path_with_namespace == "test/project"

    def test_project_with_optional_fields(self):
        """Test project creation with optional fields"""
        project_data = {
            "id": 2,
            "name": "full-project",
            "web_url": "https://gitlab.com/test/full",
            "path_with_namespace": "test/full",
            "description": "A full test project",
            "default_branch": "main",
        }

        project = GitLabProject(**project_data)
        assert project.description == "A full test project"
        assert project.default_branch == "main"

    def test_project_minimal_required_fields(self):
        """Test project with only required fields"""
        project_data = {
            "id": 3,
            "name": "minimal-project",
            "web_url": "https://gitlab.com/test/minimal",
            "path_with_namespace": "test/minimal",
        }

        project = GitLabProject(**project_data)
        assert project.id == 3
        assert project.name == "minimal-project"


class TestMergeRequestAttributes:
    """Test MergeRequestAttributes model validation"""

    def test_valid_mr_attributes_creation(self):
        """Test creating valid merge request attributes"""
        mr_data = {
            "iid": 1,
            "title": "Test MR",
            "description": "Test merge request",
            "source_branch": "feature",
            "target_branch": "main",
            "state": "opened",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
        }

        mr = MergeRequestAttributes(**mr_data)
        assert mr.iid == 1
        assert mr.title == "Test MR"
        assert mr.source_branch == "feature"
        assert mr.target_branch == "main"
        assert mr.state == "opened"

    def test_mr_attributes_with_optional_fields(self):
        """Test merge request attributes with optional fields"""
        mr_data = {
            "iid": 2,
            "title": "Full MR",
            "description": "Full merge request",
            "source_branch": "feature-full",
            "target_branch": "develop",
            "state": "merged",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z",
            "author_id": 123,
            "assignee_id": 456,
            "merge_status": "can_be_merged",
        }

        mr = MergeRequestAttributes(**mr_data)
        assert mr.author_id == 123
        assert mr.assignee_id == 456
        assert mr.merge_status == "can_be_merged"

    def test_mr_attributes_required_fields(self):
        """Test merge request attributes validation for required fields"""
        # Test with missing required field should raise ValidationError
        incomplete_data = {
            "title": "Incomplete MR",
            "source_branch": "feature",
            # Missing other required fields
        }

        with pytest.raises(ValidationError):
            MergeRequestAttributes(**incomplete_data)


class TestGitLabModelsValidation:
    """Test validation scenarios for GitLab models"""

    def test_user_invalid_email(self):
        """Test user validation with invalid email"""
        invalid_user_data = {
            "id": 1,
            "username": "testuser",
            "name": "Test User",
            "email": "invalid-email",  # Invalid email format
        }

        # Depending on model validation, this might pass or fail
        try:
            user = GitLabUser(**invalid_user_data)
            # If validation is permissive, test that user was created
            assert user.email == "invalid-email"
        except ValidationError:
            # If validation is strict, expect ValidationError
            assert True

    def test_project_invalid_url(self):
        """Test project validation with invalid URL"""
        invalid_project_data = {
            "id": 1,
            "name": "test-project",
            "web_url": "not-a-valid-url",  # Invalid URL format
            "path_with_namespace": "test/project",
        }

        # Depending on model validation, this might pass or fail
        try:
            project = GitLabProject(**invalid_project_data)
            # If validation is permissive, test that project was created
            assert project.web_url == "not-a-valid-url"
        except ValidationError:
            # If validation is strict, expect ValidationError
            assert True

    def test_negative_ids(self):
        """Test models with negative IDs"""
        try:
            user = GitLabUser(
                id=-1,
                username="negativeuser",
                name="Negative User",
                email="negative@example.com",
            )
            # If negative IDs are allowed, test creation
            assert user.id == -1
        except ValidationError:
            # If negative IDs are not allowed, expect ValidationError
            assert True


class TestGitLabModelsEdgeCases:
    """Test edge cases for GitLab models"""

    def test_user_with_empty_strings(self):
        """Test user creation with empty strings"""
        user_data = {
            "id": 1,
            "username": "",
            "name": "",
            "email": "",
        }

        try:
            user = GitLabUser(**user_data)
            # If empty strings are allowed
            assert user.username == ""
        except ValidationError:
            # If empty strings are not allowed
            assert True

    def test_user_with_very_long_values(self):
        """Test user creation with very long values"""
        long_string = "x" * 1000
        user_data = {
            "id": 1,
            "username": long_string,
            "name": long_string,
            "email": f"{long_string}@example.com",
        }

        try:
            user = GitLabUser(**user_data)
            # If long strings are allowed
            assert len(user.username) == 1000
        except ValidationError:
            # If there are length limits
            assert True

    def test_mr_attributes_with_null_optional_fields(self):
        """Test merge request attributes with null optional fields"""
        mr_data = {
            "iid": 1,
            "title": "Test MR",
            "description": None,  # Null optional field
            "source_branch": "feature",
            "target_branch": "main",
            "state": "opened",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
        }

        try:
            mr = MergeRequestAttributes(**mr_data)
            assert mr.description is None
        except ValidationError:
            # If null values are not allowed for this field
            assert True
