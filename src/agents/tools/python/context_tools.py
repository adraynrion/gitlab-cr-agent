"""
Context7 MCP integration tools for evidence-based code review
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.tools.base import (
    BaseTool,
    ToolCategory,
    ToolContext,
    ToolPriority,
    ToolResult,
)
from src.agents.tools.registry import register_tool

logger = logging.getLogger(__name__)


class Context7Client:
    """Async client wrapper for Context7 MCP integration"""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize Context7 client

        Args:
            settings: Optional configuration settings
        """
        self.settings = settings or {}
        self.cache_ttl = self.settings.get("cache_ttl", 3600)  # 1 hour default
        self._cache: Dict[str, Tuple[Any, float]] = {}

    async def resolve_library_id(self, library_name: str) -> Optional[str]:
        """Resolve a library name to Context7 library ID

        Args:
            library_name: Name of the library to resolve

        Returns:
            Context7 library ID or None if not found
        """
        # For demonstration, we'll simulate the resolution
        # In production, this would call the actual MCP tool
        library_mappings = {
            "python": "/websites/python-3",
            "fastapi": "/tiangolo/fastapi",
            "pydantic": "/pydantic/pydantic",
            "httpx": "/encode/httpx",
            "pytest": "/pytest-dev/pytest",
            "gitlab": "/python-gitlab/python-gitlab",
            "tenacity": "/jd/tenacity",
            "numpy": "/numpy/numpy",
            "pandas": "/pandas-dev/pandas",
        }

        # Simulate async operation
        await asyncio.sleep(0.1)

        # Try exact match first
        if library_name.lower() in library_mappings:
            return library_mappings[library_name.lower()]

        # Try partial match
        for key, value in library_mappings.items():
            if key in library_name.lower() or library_name.lower() in key:
                return value

        return None

    async def get_library_docs(
        self, library_id: str, topic: Optional[str] = None, max_tokens: int = 2000
    ) -> Optional[Dict[str, Any]]:
        """Get documentation for a library from Context7

        Args:
            library_id: Context7 library ID
            topic: Optional topic to focus on
            max_tokens: Maximum tokens to retrieve

        Returns:
            Documentation data or None if not found
        """
        # Check cache
        cache_key = f"{library_id}:{topic}:{max_tokens}"
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            import time

            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for library docs: {library_id}")
                return data if isinstance(data, dict) else None

        # Simulate async operation
        await asyncio.sleep(0.2)

        # For demonstration, return simulated documentation
        # In production, this would call the actual MCP tool
        docs = {
            "library_id": library_id,
            "topic": topic,
            "snippets": [
                {
                    "title": f"Best practices for {topic or 'general usage'}",
                    "description": "Documentation excerpt with best practices",
                    "code": "# Example code snippet",
                    "source": f"https://docs.example.com/{library_id}",
                }
            ],
            "references": [
                f"https://docs.example.com/{library_id}/api",
                f"https://docs.example.com/{library_id}/guide",
            ],
        }

        # Cache the result
        import time

        self._cache[cache_key] = (docs, time.time())

        return docs


@register_tool(enabled=True, name="PythonDocumentationLookupTool")
class PythonDocumentationLookupTool(BaseTool):
    """Look up documentation for Python libraries and frameworks used in the code"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DOCUMENTATION

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.HIGH

    @property
    def requires_network(self) -> bool:
        return True

    async def initialize(self, context: ToolContext) -> None:
        """Initialize the tool with Context7 client"""
        await super().initialize(context)
        self.client = Context7Client(context.settings.get("context7", {}))

    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute documentation lookup for imported libraries"""
        try:
            # Ensure client is initialized
            if not hasattr(self, "client"):
                await self.initialize(context)

            # Extract imports from the diff
            imports = self._extract_imports(context.diff_content)

            if not imports:
                return ToolResult(
                    tool_name=self.name,
                    category=self.category,
                    success=True,
                    positive_findings=["No new imports to validate"],
                    error_message=None,
                    execution_time_ms=0,
                )

            # Look up documentation for each import
            issues = []
            suggestions = []
            evidence = {}
            references = []

            for import_info in imports:
                library_name = import_info["library"]

                # Resolve library ID
                library_id = await self.client.resolve_library_id(library_name)
                if not library_id:
                    continue

                # Get documentation
                docs = await self.client.get_library_docs(
                    library_id, topic=import_info.get("module")
                )

                if docs:
                    # Analyze usage against documentation
                    analysis = self._analyze_usage(import_info, docs)

                    if analysis.get("issues"):
                        issues.extend(analysis["issues"])

                    if analysis.get("suggestions"):
                        suggestions.extend(analysis["suggestions"])

                    evidence[library_name] = docs.get("snippets", [])
                    references.extend(docs.get("references", []))

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                evidence=evidence,
                references=references,
                confidence_score=0.9 if evidence else 0.5,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Documentation lookup failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _extract_imports(self, diff_content: str) -> List[Dict[str, Any]]:
        """Extract import statements from diff content using fast string operations"""
        imports = []

        # Process only added lines (lines starting with +)
        added_lines = [
            line.strip()
            for line in diff_content.split("\n")
            if line.strip().startswith("+") and not line.strip().startswith("+++")
        ]

        for line in added_lines:
            # Remove the + prefix and normalize whitespace
            code_line = line[1:].strip()

            # Skip empty lines and comments
            if not code_line or code_line.startswith("#"):
                continue

            library = None
            module = None

            # Fast string-based import detection (replaces 3 regex patterns)
            if code_line.startswith("import "):
                # Simple import: "import library"
                parts = code_line.split()
                if len(parts) >= 2:
                    library = parts[1].split(".")[0]  # Get base library name
            elif code_line.startswith("from "):
                # From import: "from library import ..." or "from library.module import ..."
                if " import " in code_line:
                    from_part = code_line.split(" import ")[0]
                    library_path = from_part[5:]  # Remove "from "
                    if "." in library_path:
                        parts = library_path.split(".")
                        library = parts[0]
                        module = parts[1]
                    else:
                        library = library_path

            # Add valid library imports (skip standard library)
            if library and library not in ["os", "sys", "time", "json", "re", "typing"]:
                imports.append(
                    {
                        "library": library,
                        "module": module,
                        "line": code_line,  # Store the actual import line instead of regex pattern
                    }
                )

        return imports

    def _analyze_usage(
        self, import_info: Dict[str, Any], docs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze library usage against documentation"""
        analysis: Dict[str, Any] = {"issues": [], "suggestions": []}

        # Check for deprecated usage patterns
        for snippet in docs.get("snippets", []):
            if "deprecated" in snippet.get("description", "").lower():
                analysis["issues"].append(
                    {
                        "severity": "medium",
                        "description": f"Library {import_info['library']} may have deprecated features",
                        "evidence": snippet["description"],
                    }
                )

        # Add general suggestions based on documentation
        if docs.get("snippets"):
            analysis["suggestions"].append(
                f"Review {import_info['library']} documentation for best practices: "
                f"{docs.get('references', [''])[0]}"
            )

        return analysis


@register_tool(enabled=True, name="PythonAPIUsageValidationTool")
class PythonAPIUsageValidationTool(BaseTool):
    """Validate Python API usage against official documentation"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CORRECTNESS

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.HIGH

    @property
    def requires_network(self) -> bool:
        return True

    async def initialize(self, context: ToolContext) -> None:
        """Initialize the tool with Context7 client"""
        await super().initialize(context)
        self.client = Context7Client(context.settings.get("context7", {}))

    async def execute(self, context: ToolContext) -> ToolResult:
        """Validate API calls against documentation"""
        try:
            # Ensure client is initialized
            if not hasattr(self, "client"):
                await self.initialize(context)

            # Extract API calls from the diff
            api_calls = self._extract_api_calls(context.diff_content)

            if not api_calls:
                return ToolResult(
                    tool_name=self.name,
                    category=self.category,
                    success=True,
                    positive_findings=["No API calls to validate"],
                    error_message=None,
                    execution_time_ms=0,
                )

            issues = []
            suggestions = []
            evidence = {}

            for api_call in api_calls:
                # Look up the API documentation
                library_id = await self.client.resolve_library_id(api_call["library"])
                if not library_id:
                    continue

                docs = await self.client.get_library_docs(
                    library_id, topic=api_call["method"]
                )

                if docs:
                    # Validate the API usage
                    validation: Dict[str, Any] = self._validate_api_call(api_call, docs)

                    if validation.get("issues"):
                        issues.extend(validation["issues"])

                    if validation.get("suggestions"):
                        suggestions.extend(validation["suggestions"])

                    evidence[f"{api_call['library']}.{api_call['method']}"] = docs

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                evidence=evidence,
                confidence_score=0.85 if evidence else 0.3,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"API validation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _extract_api_calls(self, diff_content: str) -> List[Dict[str, Any]]:
        """Extract API calls from diff content"""
        api_calls = []

        # Pattern for method calls (simplified)
        patterns = [
            (r"\+\s*(\w+)\.(\w+)\(", 2),  # object.method(
            (r"\+\s*await\s+(\w+)\.(\w+)\(", 2),  # await object.method(
            (r"\+\s*\w+\s*=\s*(\w+)\.(\w+)\(", 2),  # var = object.method(
        ]

        for pattern, expected_groups in patterns:
            matches = re.findall(pattern, diff_content)
            for match in matches:
                if len(match) >= expected_groups:
                    api_calls.append(
                        {
                            "library": match[0],
                            "method": match[1] if expected_groups == 2 else match[2],
                            "async": "await" in pattern,
                        }
                    )

        return api_calls

    def _validate_api_call(
        self, api_call: Dict[str, Any], docs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate an API call against documentation"""
        validation: Dict[str, Any] = {"issues": [], "suggestions": []}

        # Check if method exists in documentation
        method_found = False
        for snippet in docs.get("snippets", []):
            if api_call["method"] in snippet.get("code", ""):
                method_found = True
                break

        if not method_found:
            validation["issues"].append(
                {
                    "severity": "medium",
                    "description": f"Method {api_call['method']} not found in standard documentation",
                    "file_path": "unknown",
                    "line_number": 0,
                }
            )

        # Check async usage
        if api_call.get("async"):
            validation["suggestions"].append(
                f"Ensure {api_call['method']} supports async operation"
            )

        return validation


@register_tool(enabled=True, name="PythonSecurityPatternValidationTool")
class PythonSecurityPatternValidationTool(BaseTool):
    """Validate Python security patterns against best practices documentation"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SECURITY

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.CRITICAL

    @property
    def requires_network(self) -> bool:
        return True

    async def initialize(self, context: ToolContext) -> None:
        """Initialize the tool with Context7 client"""
        await super().initialize(context)
        self.client = Context7Client(context.settings.get("context7", {}))

    async def execute(self, context: ToolContext) -> ToolResult:
        """Check for security issues using documentation-based patterns"""
        try:
            # Ensure client is initialized
            if not hasattr(self, "client"):
                await self.initialize(context)

            # Detect security-sensitive patterns
            patterns = self._detect_security_patterns(context.diff_content)

            if not patterns:
                return ToolResult(
                    tool_name=self.name,
                    category=self.category,
                    success=True,
                    positive_findings=["No security-sensitive patterns detected"],
                    error_message=None,
                    execution_time_ms=0,
                )

            issues = []
            suggestions = []
            evidence = {}
            references = []

            # Look up security best practices for each pattern
            for pattern in patterns:
                # Try to get security documentation
                library_id = await self.client.resolve_library_id("security")
                docs = None

                if library_id:
                    docs = await self.client.get_library_docs(
                        library_id, topic=pattern["type"]
                    )

                # Validate against security best practices (with or without docs)
                validation: Dict[str, Any] = self._validate_security_pattern(
                    pattern, docs or {}
                )

                if validation.get("issues"):
                    issues.extend(validation["issues"])

                if validation.get("suggestions"):
                    suggestions.extend(validation["suggestions"])

                if docs:
                    evidence[pattern["type"]] = docs
                    references.extend(docs.get("references", []))

            # Add OWASP references
            references.append("https://owasp.org/www-project-top-ten/")

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=True,
                issues=issues,
                suggestions=suggestions,
                evidence=evidence,
                references=references,
                confidence_score=0.95 if issues else 0.8,
                error_message=None,
                execution_time_ms=0,
            )

        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def _detect_security_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Detect security-sensitive patterns in the code"""
        patterns = []

        # Security pattern detection
        security_patterns = {
            "sql_injection": r'\+.*(?:query|execute|SELECT|INSERT|UPDATE|DELETE).*(?:%s|%d|\{.*\}|f".*\{.*\}|f\'.*\{.*\')',
            "hardcoded_secret": r'\+.*(?:password|secret|token|key)\s*=\s*["\'][^"\']+["\']',
            "weak_crypto": r"\+.*(?:md5|sha1)\s*\(",
            "eval_usage": r"\+.*(?:eval|exec)\s*\(",
            "file_traversal": r"\+.*(?:open|read)\s*\([^)]*\.\./[^)]*\)",
            "cors_wildcard": r'\+.*allow_origins\s*=\s*\["?\*"?\]',
        }

        for pattern_type, regex in security_patterns.items():
            matches = re.findall(regex, diff_content, re.IGNORECASE)
            for match in matches:
                patterns.append(
                    {
                        "type": pattern_type,
                        "match": match,
                        "severity": self._get_severity(pattern_type),
                    }
                )

        return patterns

    def _get_severity(self, pattern_type: str) -> str:
        """Get severity level for a security pattern"""
        severity_map = {
            "sql_injection": "critical",
            "hardcoded_secret": "critical",
            "weak_crypto": "high",
            "eval_usage": "high",
            "file_traversal": "high",
            "cors_wildcard": "medium",
        }
        return severity_map.get(pattern_type, "medium")

    def _validate_security_pattern(
        self, pattern: Dict[str, Any], docs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a security pattern against best practices"""
        validation: Dict[str, Any] = {"issues": [], "suggestions": []}

        # Create issue for the security pattern
        validation["issues"].append(
            {
                "severity": pattern["severity"],
                "category": "security",
                "description": f"Potential {self._format_pattern_name(pattern['type'])} vulnerability detected",
                "file_path": "unknown",
                "line_number": 0,
                "evidence": pattern["match"],
            }
        )

        # Add suggestions based on pattern type
        suggestions_map = {
            "sql_injection": "Use parameterized queries or prepared statements",
            "hardcoded_secret": "Use environment variables or secure secret management",
            "weak_crypto": "Use strong hashing algorithms like SHA-256 or bcrypt",
            "eval_usage": "Avoid eval/exec; use safer alternatives like ast.literal_eval",
            "file_traversal": "Validate and sanitize file paths",
            "cors_wildcard": "Specify allowed origins explicitly instead of using wildcards",
        }

        if pattern["type"] in suggestions_map:
            validation["suggestions"].append(suggestions_map[pattern["type"]])

        return validation

    def _format_pattern_name(self, pattern_type: str) -> str:
        """Format pattern type name for display"""
        # Special cases for proper capitalization
        if pattern_type == "sql_injection":
            return "SQL injection"
        elif pattern_type == "hardcoded_secret":
            return "hardcoded secret"
        elif pattern_type == "weak_crypto":
            return "weak cryptography"
        elif pattern_type == "eval_usage":
            return "eval usage"
        elif pattern_type == "file_traversal":
            return "file traversal"
        elif pattern_type == "cors_wildcard":
            return "CORS wildcard"
        else:
            return pattern_type.replace("_", " ")
