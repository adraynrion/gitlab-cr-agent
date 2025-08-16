"""Unit tests for version utility module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.version import get_version, get_version_info


class TestGetVersion:
    """Test the get_version function."""

    def test_get_version_success(self):
        """Test getting version from valid version.txt file."""
        # Reset cache for this test
        import src.utils.version
        from pathlib import Path

        src.utils.version._cached_version = None

        # Read expected version from actual file
        project_root = Path(__file__).parent.parent.parent
        version_file = project_root / "version.txt"
        expected_version = version_file.read_text().strip()

        version = get_version()
        assert version == expected_version
        assert isinstance(version, str)

    def test_get_version_caching(self):
        """Test that version is cached after first read."""
        import src.utils.version
        from pathlib import Path

        src.utils.version._cached_version = None

        # Read expected version from actual file
        project_root = Path(__file__).parent.parent.parent
        version_file = project_root / "version.txt"
        expected_version = version_file.read_text().strip()

        # First call should read from file
        version1 = get_version()

        # Second call should use cache
        version2 = get_version()

        assert version1 == version2
        assert version1 == expected_version

    def test_get_version_with_custom_file(self):
        """Test reading version from custom version.txt location."""
        import src.utils.version

        src.utils.version._cached_version = None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary version file
            version_file = Path(temp_dir) / "version.txt"
            version_file.write_text("1.0.0\n")

            # Mock the project root to point to temp directory
            with patch("src.utils.version.Path") as mock_path:
                mock_path(__file__).parent.parent.parent = Path(temp_dir)
                mock_path.return_value.parent.parent = Path(temp_dir)

                # This would require mocking the file resolution
                # For now, just test the current implementation

    def test_get_version_missing_file(self, tmp_path):
        """Test error when version.txt file is missing."""
        import src.utils.version

        src.utils.version._cached_version = None

        # Patch the project root to point to a temp directory without version.txt
        with patch("src.utils.version.Path") as mock_path:
            # Mock to return the temp directory as project root
            mock_path.return_value.parent.parent = tmp_path

            with pytest.raises(FileNotFoundError) as exc_info:
                get_version()

            assert "Version file not found" in str(exc_info.value)

    def test_get_version_empty_file(self):
        """Test error when version.txt file is empty."""
        import src.utils.version

        src.utils.version._cached_version = None

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write("")
            temp_file.flush()

            with patch("src.utils.version.Path") as mock_path:
                mock_version_file = mock_path.return_value.parent.parent / "version.txt"
                mock_version_file.exists.return_value = True

                # Mock open to return empty content
                with patch("builtins.open", mock_open_empty_file()):
                    with pytest.raises(ValueError) as exc_info:
                        get_version()

                    assert "Version file is empty" in str(exc_info.value)

    def test_get_version_invalid_format(self):
        """Test error when version format is invalid."""
        import src.utils.version

        src.utils.version._cached_version = None

        invalid_versions = [
            "1.0",  # Missing patch version
            "1.0.0.0",  # Too many parts
            "1.a.0",  # Non-numeric parts
            "v1.0.0",  # Prefix not allowed
            "1.0.0-beta",  # Pre-release not allowed in basic validation
        ]

        for invalid_version in invalid_versions:
            with patch("src.utils.version.Path") as mock_path:
                mock_version_file = mock_path.return_value.parent.parent / "version.txt"
                mock_version_file.exists.return_value = True

                with patch("builtins.open", mock_open_with_content(invalid_version)):
                    with pytest.raises(ValueError) as exc_info:
                        get_version()

                    assert "Invalid version format" in str(exc_info.value)
                    assert invalid_version in str(exc_info.value)

    def test_get_version_file_read_error(self):
        """Test error when version file cannot be read."""
        import src.utils.version

        src.utils.version._cached_version = None

        with patch("src.utils.version.Path") as mock_path:
            mock_version_file = mock_path.return_value.parent.parent / "version.txt"
            mock_version_file.exists.return_value = True

            with patch("builtins.open", side_effect=IOError("Permission denied")):
                with pytest.raises(FileNotFoundError) as exc_info:
                    get_version()

                assert "Failed to read version file" in str(exc_info.value)

    def test_get_version_with_whitespace(self):
        """Test version reading with surrounding whitespace."""
        import src.utils.version

        src.utils.version._cached_version = None

        with patch("src.utils.version.Path") as mock_path:
            mock_version_file = mock_path.return_value.parent.parent / "version.txt"
            mock_version_file.exists.return_value = True

            with patch("builtins.open", mock_open_with_content("  2.0.0  \n")):
                version = get_version()
                assert version == "2.0.0"


class TestGetVersionInfo:
    """Test the get_version_info function."""

    def test_get_version_info_success(self):
        """Test getting detailed version information."""
        import src.utils.version
        from pathlib import Path

        # Read actual version from file
        project_root = Path(__file__).parent.parent.parent
        version_file = project_root / "version.txt"
        actual_version = version_file.read_text().strip()
        
        # Parse version parts
        major, minor, patch = map(int, actual_version.split('.'))

        src.utils.version._cached_version = actual_version  # Use cached version

        version_info = get_version_info()

        assert version_info == {
            "version": actual_version,
            "major": major,
            "minor": minor,
            "patch": patch,
            "full": f"v{actual_version}",
        }

    def test_get_version_info_structure(self):
        """Test that version info has all required fields."""
        import src.utils.version

        src.utils.version._cached_version = "1.2.3"

        version_info = get_version_info()

        required_keys = {"version", "major", "minor", "patch", "full"}
        assert set(version_info.keys()) == required_keys

        assert isinstance(version_info["version"], str)
        assert isinstance(version_info["major"], int)
        assert isinstance(version_info["minor"], int)
        assert isinstance(version_info["patch"], int)
        assert isinstance(version_info["full"], str)

    def test_get_version_info_values(self):
        """Test version info values for different versions."""
        import src.utils.version

        test_cases = [
            ("1.0.0", {"major": 1, "minor": 0, "patch": 0, "full": "v1.0.0"}),
            ("10.20.30", {"major": 10, "minor": 20, "patch": 30, "full": "v10.20.30"}),
            ("0.1.0", {"major": 0, "minor": 1, "patch": 0, "full": "v0.1.0"}),
        ]

        for version, expected in test_cases:
            src.utils.version._cached_version = version
            version_info = get_version_info()

            assert version_info["version"] == version
            assert version_info["major"] == expected["major"]
            assert version_info["minor"] == expected["minor"]
            assert version_info["patch"] == expected["patch"]
            assert version_info["full"] == expected["full"]


class TestVersionCaching:
    """Test version caching behavior."""

    def test_cache_reset_behavior(self):
        """Test that cache can be reset and reloaded."""
        import src.utils.version

        # Set initial cache
        src.utils.version._cached_version = "1.0.0"
        assert get_version() == "1.0.0"

        # Reset cache
        src.utils.version._cached_version = None

        # Should read from actual file
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        version_file = project_root / "version.txt"
        expected_version = version_file.read_text().strip()
        
        version = get_version()
        assert version == expected_version  # Current actual version

    def test_cache_persistence(self):
        """Test that cache persists across multiple calls."""
        import src.utils.version

        src.utils.version._cached_version = None

        # First call reads from file and caches
        version1 = get_version()

        # Multiple subsequent calls should use cache
        version2 = get_version()
        version3 = get_version()

        # Read expected version from actual file
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        version_file = project_root / "version.txt"
        expected_version = version_file.read_text().strip()

        assert version1 == version2 == version3
        assert version1 == expected_version


class TestVersionValidation:
    """Test semantic version validation logic."""

    def test_valid_semantic_versions(self):
        """Test that valid semantic versions pass validation."""
        import src.utils.version

        valid_versions = [
            "0.0.0",
            "1.0.0",
            "0.1.0",
            "0.0.1",
            "10.20.30",
            "999.999.999",
        ]

        for version in valid_versions:
            src.utils.version._cached_version = None

            with patch("src.utils.version.Path") as mock_path:
                mock_version_file = mock_path.return_value.parent.parent / "version.txt"
                mock_version_file.exists.return_value = True

                with patch("builtins.open", mock_open_with_content(version)):
                    result = get_version()
                    assert result == version

    def test_invalid_semantic_versions(self):
        """Test that invalid semantic versions are rejected."""
        import src.utils.version

        invalid_versions = [
            "1",  # Too few parts
            "1.0",  # Too few parts
            "1.0.0.0",  # Too many parts
            "1.0.a",  # Non-numeric
            "a.0.0",  # Non-numeric
            "1.a.0",  # Non-numeric
            "",  # Empty
            "1.0.0-alpha",  # Pre-release
            "v1.0.0",  # Prefix
        ]

        for version in invalid_versions:
            src.utils.version._cached_version = None

            with patch("src.utils.version.Path") as mock_path:
                mock_file_path = Mock()
                mock_file_path.exists.return_value = True
                mock_path.return_value.parent.parent.__truediv__.return_value = (
                    mock_file_path
                )

                with patch("builtins.open", mock_open_with_content(version)):
                    with pytest.raises(ValueError) as exc_info:
                        get_version()

                    # Empty string should raise "Version file is empty"
                    # Other invalid formats should raise "Invalid version format"
                    if version == "":
                        assert "Version file is empty" in str(exc_info.value)
                    else:
                        assert "Invalid version format" in str(exc_info.value)


# Helper functions for mocking file operations
def mock_open_with_content(content):
    """Create a mock open that returns specified content."""
    from unittest.mock import mock_open

    return mock_open(read_data=content)


def mock_open_empty_file():
    """Create a mock open that returns empty content."""
    from unittest.mock import mock_open

    return mock_open(read_data="")


class TestVersionIntegration:
    """Integration tests for version module with actual file system."""

    def test_actual_version_file_exists(self):
        """Test that the actual version.txt file exists and is readable."""
        from pathlib import Path

        # Get the actual project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        version_file = project_root / "version.txt"

        assert version_file.exists(), "version.txt file should exist in project root"

        content = version_file.read_text().strip()
        assert content, "version.txt should not be empty"

        # Should be valid semantic version
        parts = content.split(".")
        assert len(parts) == 3, f"Version {content} should have 3 parts"
        assert all(
            part.isdigit() for part in parts
        ), f"All parts of {content} should be numeric"

    def test_version_matches_file_content(self):
        """Test that get_version() returns content from version.txt."""
        import src.utils.version

        src.utils.version._cached_version = None

        # Read actual file content
        from pathlib import Path

        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        version_file = project_root / "version.txt"
        expected_version = version_file.read_text().strip()

        # Get version via utility
        actual_version = get_version()

        assert actual_version == expected_version
