"""
Language detection and routing system for multi-language code review
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ProgrammingLanguage(Enum):
    """Supported programming languages"""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    UNKNOWN = "unknown"


class LanguageDetector:
    """Detects programming languages based on file extensions and content"""

    # File extension to language mapping
    EXTENSION_MAP: Dict[str, ProgrammingLanguage] = {
        # Python
        ".py": ProgrammingLanguage.PYTHON,
        ".pyx": ProgrammingLanguage.PYTHON,
        ".pyi": ProgrammingLanguage.PYTHON,
        ".pyw": ProgrammingLanguage.PYTHON,
        # JavaScript
        ".js": ProgrammingLanguage.JAVASCRIPT,
        ".jsx": ProgrammingLanguage.JAVASCRIPT,
        ".mjs": ProgrammingLanguage.JAVASCRIPT,
        ".cjs": ProgrammingLanguage.JAVASCRIPT,
        # TypeScript
        ".ts": ProgrammingLanguage.TYPESCRIPT,
        ".tsx": ProgrammingLanguage.TYPESCRIPT,
        ".d.ts": ProgrammingLanguage.TYPESCRIPT,
        # Go
        ".go": ProgrammingLanguage.GO,
        # Rust
        ".rs": ProgrammingLanguage.RUST,
        # Java
        ".java": ProgrammingLanguage.JAVA,
        ".class": ProgrammingLanguage.JAVA,
        ".jar": ProgrammingLanguage.JAVA,
        # C#
        ".cs": ProgrammingLanguage.CSHARP,
        ".csx": ProgrammingLanguage.CSHARP,
        # C++
        ".cpp": ProgrammingLanguage.CPP,
        ".cc": ProgrammingLanguage.CPP,
        ".cxx": ProgrammingLanguage.CPP,
        ".c++": ProgrammingLanguage.CPP,
        ".hpp": ProgrammingLanguage.CPP,
        ".hh": ProgrammingLanguage.CPP,
        ".hxx": ProgrammingLanguage.CPP,
        ".h++": ProgrammingLanguage.CPP,
        # C
        ".c": ProgrammingLanguage.C,
        ".h": ProgrammingLanguage.C,
        # PHP
        ".php": ProgrammingLanguage.PHP,
        ".phtml": ProgrammingLanguage.PHP,
        ".php3": ProgrammingLanguage.PHP,
        ".php4": ProgrammingLanguage.PHP,
        ".php5": ProgrammingLanguage.PHP,
        ".php7": ProgrammingLanguage.PHP,
        # Ruby
        ".rb": ProgrammingLanguage.RUBY,
        ".rbw": ProgrammingLanguage.RUBY,
        # Kotlin
        ".kt": ProgrammingLanguage.KOTLIN,
        ".kts": ProgrammingLanguage.KOTLIN,
        # Swift
        ".swift": ProgrammingLanguage.SWIFT,
    }

    # Special filename patterns
    SPECIAL_FILES: Dict[str, ProgrammingLanguage] = {
        "Dockerfile": ProgrammingLanguage.UNKNOWN,
        "Makefile": ProgrammingLanguage.UNKNOWN,
        "requirements.txt": ProgrammingLanguage.PYTHON,
        "setup.py": ProgrammingLanguage.PYTHON,
        "pyproject.toml": ProgrammingLanguage.PYTHON,
        "package.json": ProgrammingLanguage.JAVASCRIPT,
        "tsconfig.json": ProgrammingLanguage.TYPESCRIPT,
        "go.mod": ProgrammingLanguage.GO,
        "go.sum": ProgrammingLanguage.GO,
        "Cargo.toml": ProgrammingLanguage.RUST,
        "Cargo.lock": ProgrammingLanguage.RUST,
        "pom.xml": ProgrammingLanguage.JAVA,
        "build.gradle": ProgrammingLanguage.JAVA,
        "composer.json": ProgrammingLanguage.PHP,
        "Gemfile": ProgrammingLanguage.RUBY,
    }

    @classmethod
    def detect_language(cls, file_path: str) -> ProgrammingLanguage:
        """
        Detect programming language from file path

        Args:
            file_path: Path to the file

        Returns:
            Detected programming language
        """
        path = Path(file_path)

        # Check special filenames first
        if path.name in cls.SPECIAL_FILES:
            return cls.SPECIAL_FILES[path.name]

        # Check file extension
        extension = path.suffix.lower()
        if extension in cls.EXTENSION_MAP:
            return cls.EXTENSION_MAP[extension]

        # Handle double extensions like .d.ts
        if len(path.suffixes) >= 2:
            double_extension = "".join(path.suffixes[-2:]).lower()
            if double_extension in cls.EXTENSION_MAP:
                return cls.EXTENSION_MAP[double_extension]

        return ProgrammingLanguage.UNKNOWN

    @classmethod
    def detect_languages_from_file_changes(
        cls, file_changes: List[Dict[str, Any]]
    ) -> Dict[ProgrammingLanguage, List[str]]:
        """
        Detect languages from a list of file changes

        Args:
            file_changes: List of file change dictionaries with 'path' key

        Returns:
            Dictionary mapping languages to list of file paths
        """
        language_files: Dict[ProgrammingLanguage, List[str]] = {}

        for file_change in file_changes:
            file_path = file_change.get("path", "")
            if not file_path:
                continue

            language = cls.detect_language(file_path)

            if language not in language_files:
                language_files[language] = []

            language_files[language].append(file_path)

        return language_files

    @classmethod
    def get_primary_language(
        cls, file_changes: List[Dict[str, Any]]
    ) -> ProgrammingLanguage:
        """
        Get the primary language from file changes (most common non-unknown language)

        Args:
            file_changes: List of file change dictionaries

        Returns:
            Primary programming language
        """
        language_files = cls.detect_languages_from_file_changes(file_changes)

        # Remove unknown files from consideration
        known_languages = {
            lang: files
            for lang, files in language_files.items()
            if lang != ProgrammingLanguage.UNKNOWN
        }

        if not known_languages:
            return ProgrammingLanguage.UNKNOWN

        # Return language with most files
        primary_language = max(
            known_languages.keys(), key=lambda lang: len(known_languages[lang])
        )
        return primary_language

    @classmethod
    def should_run_language_tools(
        cls, file_changes: List[Dict[str, Any]], target_language: ProgrammingLanguage
    ) -> bool:
        """
        Determine if language-specific tools should run for the target language

        Args:
            file_changes: List of file change dictionaries
            target_language: Language to check for

        Returns:
            True if tools for target language should run
        """
        language_files = cls.detect_languages_from_file_changes(file_changes)

        # Check if the target language has any files
        return (
            target_language in language_files
            and len(language_files[target_language]) > 0
        )

    @classmethod
    def get_language_statistics(
        cls, file_changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get detailed statistics about languages in the file changes

        Args:
            file_changes: List of file change dictionaries

        Returns:
            Dictionary with language statistics
        """
        language_files = cls.detect_languages_from_file_changes(file_changes)

        total_files = len(file_changes)
        stats = {
            "total_files": total_files,
            "languages_detected": len(language_files),
            "primary_language": cls.get_primary_language(file_changes).value,
            "language_breakdown": {
                lang.value: {
                    "file_count": len(files),
                    "percentage": (len(files) / total_files) * 100
                    if total_files > 0
                    else 0,
                    "files": files,
                }
                for lang, files in language_files.items()
            },
        }

        return stats


class LanguageRouter:
    """Routes tools based on detected languages"""

    def __init__(self):
        """Initialize the language router"""
        self.language_detector = LanguageDetector()

    def get_applicable_tools(
        self, file_changes: List[Dict[str, Any]], available_tools: List[str]
    ) -> Dict[str, List[str]]:
        """
        Get applicable tools for the detected languages

        Args:
            file_changes: List of file change dictionaries
            available_tools: List of available tool names

        Returns:
            Dictionary mapping tool categories to applicable tool names
        """
        language_files = self.language_detector.detect_languages_from_file_changes(
            file_changes
        )
        primary_language = self.language_detector.get_primary_language(file_changes)

        applicable_tools: Dict[str, List[str]] = {
            "universal": [],  # Tools that apply to all languages
            "language_specific": [],  # Tools specific to detected languages
        }

        # Universal tools (apply to all code changes)
        universal_patterns = [
            "ComplexityAnalysisTool",  # TODO: Will become language-agnostic
            "CodeQualityTool",  # TODO: Will become language-agnostic
        ]

        # Python-specific tools
        python_specific_patterns = [
            "PythonSecurityAnalysisTool",
            "PythonComplexityAnalysisTool",
            "PythonCodeQualityTool",
            "PythonPerformancePatternTool",
            "PythonAsyncPatternValidationTool",
            "PythonErrorHandlingTool",
            "PythonTypeHintValidationTool",
            "PythonFrameworkSpecificTool",
            "PythonDocumentationLookupTool",
            "PythonAPIUsageValidationTool",
            "PythonSecurityPatternValidationTool",
        ]

        # Categorize available tools
        for tool_name in available_tools:
            # Check if it's a universal tool
            if any(pattern in tool_name for pattern in universal_patterns):
                applicable_tools["universal"].append(tool_name)
            # Check if it's language-specific and we have that language
            elif any(pattern in tool_name for pattern in python_specific_patterns):
                if ProgrammingLanguage.PYTHON in language_files:
                    applicable_tools["language_specific"].append(tool_name)
            # Skip any tools that don't match our patterns - they're not supported

        logger.info(
            f"Language routing: Primary={primary_language.value}, "
            f"Languages={list(language_files.keys())}, "
            f"Universal tools={len(applicable_tools['universal'])}, "
            f"Language-specific tools={len(applicable_tools['language_specific'])}"
        )

        return applicable_tools

    def should_skip_tool(
        self, tool_name: str, file_changes: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if a tool should be skipped based on language detection

        Args:
            tool_name: Name of the tool
            file_changes: List of file change dictionaries

        Returns:
            True if tool should be skipped
        """
        language_files = self.language_detector.detect_languages_from_file_changes(
            file_changes
        )

        # Skip Python tools if no Python files
        python_tool_patterns = [
            "PythonSecurityAnalysisTool",
            "PythonComplexityAnalysisTool",
            "PythonCodeQualityTool",
            "PythonPerformancePatternTool",
            "PythonAsyncPatternValidationTool",
            "PythonErrorHandlingTool",
            "PythonTypeHintValidationTool",
            "PythonFrameworkSpecificTool",
            "PythonDocumentationLookupTool",
            "PythonAPIUsageValidationTool",
            "PythonSecurityPatternValidationTool",
        ]

        # If it's a Python tool and we have no Python files, skip it
        if any(pattern in tool_name for pattern in python_tool_patterns):
            return ProgrammingLanguage.PYTHON not in language_files

        # Don't skip by default
        return False

    def get_language_context(
        self, file_changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get language context for tools

        Args:
            file_changes: List of file change dictionaries

        Returns:
            Language context dictionary
        """
        stats = self.language_detector.get_language_statistics(file_changes)

        return {
            "language_statistics": stats,
            "primary_language": stats["primary_language"],
            "detected_languages": list(stats["language_breakdown"].keys()),
            "performance_optimized": True,  # Indicates language-aware routing is active
        }
