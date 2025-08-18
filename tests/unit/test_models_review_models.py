"""
Tests for src/models/review_models.py
"""

import json

import pytest
from pydantic import ValidationError

from src.models.review_models import CodeIssue, ReviewContext, ReviewResult


class TestCodeIssue:
    """Test CodeIssue model validation"""

    def test_valid_code_issue_creation(self):
        """Test creating a valid code issue"""
        issue_data = {
            "file_path": "src/test.py",
            "line_number": 42,
            "severity": "high",
            "category": "security",
            "description": "Potential SQL injection vulnerability",
            "suggestion": "Use parameterized queries",
            "code_example": "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
        }

        issue = CodeIssue(**issue_data)
        assert issue.file_path == "src/test.py"
        assert issue.line_number == 42
        assert issue.severity == "high"
        assert issue.category == "security"
        assert issue.description == "Potential SQL injection vulnerability"
        assert issue.suggestion == "Use parameterized queries"
        assert issue.code_example is not None

    def test_code_issue_without_example(self):
        """Test code issue creation without code example"""
        issue_data = {
            "file_path": "src/style.py",
            "line_number": 10,
            "severity": "low",
            "category": "style",
            "description": "Line too long",
            "suggestion": "Break line into multiple lines",
        }

        issue = CodeIssue(**issue_data)
        assert issue.file_path == "src/style.py"
        assert issue.line_number == 10
        assert issue.severity == "low"
        assert issue.category == "style"
        assert issue.code_example is None

    def test_code_issue_line_number_validation(self):
        """Test code issue line number validation"""
        issue_data = {
            "file_path": "src/test.py",
            "line_number": 1,  # Minimum valid line number
            "severity": "medium",
            "category": "correctness",
            "description": "Logic error",
            "suggestion": "Fix the logic",
        }

        issue = CodeIssue(**issue_data)
        assert issue.line_number == 1

    def test_code_issue_invalid_line_numbers(self):
        """Test code issue with invalid line numbers"""
        invalid_line_numbers = [0, -1, -10]

        for line_num in invalid_line_numbers:
            issue_data = {
                "file_path": "src/test.py",
                "line_number": line_num,
                "severity": "medium",
                "category": "correctness",
                "description": "Logic error",
                "suggestion": "Fix the logic",
            }

            with pytest.raises(ValidationError):
                CodeIssue(**issue_data)

    def test_code_issue_line_number_edge_cases(self):
        """Test code issue line number edge cases"""
        # Test very large line number
        issue_data = {
            "file_path": "src/large_file.py",
            "line_number": 999999,
            "severity": "low",
            "category": "style",
            "description": "Style issue",
            "suggestion": "Improve style",
        }

        issue = CodeIssue(**issue_data)
        assert issue.line_number == 999999

    def test_code_issue_severity_levels(self):
        """Test all valid severity levels"""
        severities = ["critical", "high", "medium", "low"]

        for severity in severities:
            issue_data = {
                "file_path": "src/test.py",
                "line_number": 1,
                "severity": severity,
                "category": "security",
                "description": "Test issue",
                "suggestion": "Fix it",
            }

            issue = CodeIssue(**issue_data)
            assert issue.severity == severity

    def test_code_issue_categories(self):
        """Test all valid categories"""
        categories = [
            "security",
            "performance",
            "correctness",
            "style",
            "maintainability",
        ]

        for category in categories:
            issue_data = {
                "file_path": "src/test.py",
                "line_number": 1,
                "severity": "medium",
                "category": category,
                "description": "Test issue",
                "suggestion": "Fix it",
            }

            issue = CodeIssue(**issue_data)
            assert issue.category == category

    def test_invalid_severity_level(self):
        """Test invalid severity level"""
        issue_data = {
            "file_path": "src/test.py",
            "line_number": 1,
            "severity": "invalid",  # Invalid severity
            "category": "security",
            "description": "Test issue",
            "suggestion": "Fix it",
        }

        with pytest.raises(ValidationError):
            CodeIssue(**issue_data)

    def test_invalid_category(self):
        """Test invalid category"""
        issue_data = {
            "file_path": "src/test.py",
            "line_number": 1,
            "severity": "medium",
            "category": "invalid",  # Invalid category
            "description": "Test issue",
            "suggestion": "Fix it",
        }

        with pytest.raises(ValidationError):
            CodeIssue(**issue_data)


class TestReviewResult:
    """Test ReviewResult model validation"""

    def test_valid_review_result_creation(self):
        """Test creating a valid review result"""
        issue = CodeIssue(
            file_path="src/test.py",
            line_number=10,
            severity="medium",
            category="correctness",
            description="Logic error",
            suggestion="Fix logic",
        )

        result_data = {
            "overall_assessment": "approve_with_changes",
            "risk_level": "medium",
            "summary": "Good code with minor issues",
            "issues": [issue],
            "positive_feedback": ["Clean code structure", "Good test coverage"],
            "metrics": {"lines_reviewed": 150, "issues_found": 1},
        }

        result = ReviewResult(**result_data)
        assert result.overall_assessment == "approve_with_changes"
        assert result.risk_level == "medium"
        assert result.summary == "Good code with minor issues"
        assert len(result.issues) == 1
        assert len(result.positive_feedback) == 2
        assert result.metrics["lines_reviewed"] == 150

    def test_review_result_with_empty_lists(self):
        """Test review result with empty lists"""
        result_data = {
            "overall_assessment": "approve",
            "risk_level": "low",
            "summary": "Perfect code",
            "issues": [],
            "positive_feedback": [],
            "metrics": {},
        }

        result = ReviewResult(**result_data)
        assert len(result.issues) == 0
        assert len(result.positive_feedback) == 0
        assert len(result.metrics) == 0

    def test_review_result_assessment_types(self):
        """Test all valid assessment types"""
        assessments = ["approve", "approve_with_changes", "needs_work", "reject"]

        for assessment in assessments:
            result_data = {
                "overall_assessment": assessment,
                "risk_level": "low",
                "summary": f"Review with {assessment}",
                "issues": [],
                "positive_feedback": [],
                "metrics": {},
            }

            result = ReviewResult(**result_data)
            assert result.overall_assessment == assessment

    def test_review_result_risk_levels(self):
        """Test all valid risk levels"""
        risk_levels = ["low", "medium", "high", "critical"]

        for risk_level in risk_levels:
            result_data = {
                "overall_assessment": "approve",
                "risk_level": risk_level,
                "summary": f"Review with {risk_level} risk",
                "issues": [],
                "positive_feedback": [],
                "metrics": {},
            }

            result = ReviewResult(**result_data)
            assert result.risk_level == risk_level

    def test_invalid_assessment_type(self):
        """Test invalid assessment type"""
        result_data = {
            "overall_assessment": "invalid_assessment",
            "risk_level": "low",
            "summary": "Test review",
            "issues": [],
            "positive_feedback": [],
            "metrics": {},
        }

        with pytest.raises(ValidationError):
            ReviewResult(**result_data)

    def test_invalid_risk_level(self):
        """Test invalid risk level"""
        result_data = {
            "overall_assessment": "approve",
            "risk_level": "invalid_risk",
            "summary": "Test review",
            "issues": [],
            "positive_feedback": [],
            "metrics": {},
        }

        with pytest.raises(ValidationError):
            ReviewResult(**result_data)


class TestReviewContext:
    """Test ReviewContext model validation"""

    def test_valid_review_context_creation(self):
        """Test creating a valid review context"""
        context_data = {
            "repository_url": "https://gitlab.com/test/repo",
            "merge_request_iid": 42,
            "source_branch": "feature-branch",
            "target_branch": "main",
            "trigger_tag": "@review",
            "file_changes": [
                {"path": "src/main.py", "diff": "..."},
                {"path": "tests/test_main.py", "diff": "..."},
            ],
        }

        context = ReviewContext(**context_data)
        assert context.repository_url == "https://gitlab.com/test/repo"
        assert context.merge_request_iid == 42
        assert context.source_branch == "feature-branch"
        assert context.target_branch == "main"
        assert context.trigger_tag == "@review"
        assert len(context.file_changes) == 2

    def test_review_context_with_minimal_data(self):
        """Test review context with minimal required data"""
        context_data = {
            "repository_url": "https://gitlab.com/minimal/repo",
            "merge_request_iid": 1,
            "source_branch": "feature",
            "target_branch": "main",
            "trigger_tag": "@review",
            "file_changes": [],
        }

        context = ReviewContext(**context_data)
        assert context.repository_url == "https://gitlab.com/minimal/repo"
        assert len(context.file_changes) == 0

    def test_review_context_different_branches(self):
        """Test review context with different branch names"""
        branch_combinations = [
            ("feature/new-feature", "develop"),
            ("hotfix/bug-123", "main"),
            ("release/v1.0.0", "master"),
            ("chore/update-deps", "staging"),
        ]

        for source, target in branch_combinations:
            context_data = {
                "repository_url": "https://gitlab.com/test/repo",
                "merge_request_iid": 1,
                "source_branch": source,
                "target_branch": target,
                "trigger_tag": "@review",
                "file_changes": [],
            }

            context = ReviewContext(**context_data)
            assert context.source_branch == source
            assert context.target_branch == target


class TestReviewModelsIntegration:
    """Test integration between review models"""

    def test_review_result_json_roundtrip(self):
        """Test ReviewResult JSON serialization roundtrip"""
        issue = CodeIssue(
            file_path="src/test.py",
            line_number=5,
            severity="high",
            category="security",
            description="Security issue",
            suggestion="Fix security",
        )

        result_data = {
            "overall_assessment": "needs_work",
            "risk_level": "high",
            "summary": "Security issues found",
            "issues": [issue],
            "positive_feedback": ["Good structure"],
            "metrics": {"security_issues": 1},
        }

        # Create result, serialize to JSON, then deserialize
        original_result = ReviewResult(**result_data)
        json_str = original_result.model_dump_json()
        parsed_data = json.loads(json_str)
        recreated_result = ReviewResult(**parsed_data)

        assert original_result.overall_assessment == recreated_result.overall_assessment
        assert original_result.risk_level == recreated_result.risk_level
        assert len(original_result.issues) == len(recreated_result.issues)
        assert (
            original_result.issues[0].file_path == recreated_result.issues[0].file_path
        )

    def test_complex_review_result_with_multiple_issues(self):
        """Test review result with multiple issues of different types"""
        issues = [
            CodeIssue(
                file_path="src/security.py",
                line_number=10,
                severity="critical",
                category="security",
                description="SQL injection vulnerability",
                suggestion="Use parameterized queries",
            ),
            CodeIssue(
                file_path="src/performance.py",
                line_number=25,
                severity="medium",
                category="performance",
                description="Inefficient loop",
                suggestion="Use list comprehension",
            ),
            CodeIssue(
                file_path="src/style.py",
                line_number=5,
                severity="low",
                category="style",
                description="Missing docstring",
                suggestion="Add function docstring",
            ),
        ]

        result_data = {
            "overall_assessment": "needs_work",
            "risk_level": "high",
            "summary": "Multiple issues found across different categories",
            "issues": issues,
            "positive_feedback": ["Good test coverage", "Clean architecture"],
            "metrics": {
                "total_issues": 3,
                "critical_issues": 1,
                "security_issues": 1,
                "performance_issues": 1,
                "style_issues": 1,
            },
        }

        result = ReviewResult(**result_data)
        assert len(result.issues) == 3
        assert result.risk_level == "high"
        assert result.metrics["total_issues"] == 3

        # Verify issue details
        security_issues = [i for i in result.issues if i.category == "security"]
        performance_issues = [i for i in result.issues if i.category == "performance"]
        style_issues = [i for i in result.issues if i.category == "style"]

        assert len(security_issues) == 1
        assert len(performance_issues) == 1
        assert len(style_issues) == 1
        assert security_issues[0].severity == "critical"
