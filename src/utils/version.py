"""
Version utility for reading application version from version.txt
"""

from pathlib import Path
from typing import Optional

# Cache for the version to avoid reading file multiple times
_cached_version: Optional[str] = None


def get_version() -> str:
    """
    Get application version from version.txt file.

    Returns:
        Version string (e.g., "2.0.0")

    Raises:
        FileNotFoundError: If version.txt file is not found
        ValueError: If version.txt file is empty or invalid
    """
    global _cached_version

    if _cached_version is not None:
        return _cached_version

    # Get the project root directory (parent of src)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    version_file = project_root / "version.txt"

    if not version_file.exists():
        raise FileNotFoundError(f"Version file not found: {version_file}")

    try:
        with open(version_file, "r", encoding="utf-8") as f:
            version = f.read().strip()

        if not version:
            raise ValueError("Version file is empty")

        # Basic validation - should follow semantic versioning pattern
        version_parts = version.split(".")
        if len(version_parts) != 3 or not all(part.isdigit() for part in version_parts):
            raise ValueError(
                f"Invalid version format: {version}. Expected semantic versioning (e.g., '2.0.0')"
            )

        _cached_version = version
        return version

    except (IOError, OSError) as e:
        raise FileNotFoundError(f"Failed to read version file: {e}")


def get_version_info() -> dict:
    """
    Get detailed version information.

    Returns:
        Dictionary containing version details
    """
    version = get_version()
    major, minor, patch = version.split(".")

    return {
        "version": version,
        "major": int(major),
        "minor": int(minor),
        "patch": int(patch),
        "full": f"v{version}",
    }
