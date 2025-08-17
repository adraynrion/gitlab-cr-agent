"""
AST-based Python code parser for efficient code analysis

This module provides fast, accurate Python code analysis using the built-in AST module
instead of expensive regex patterns. It focuses on analyzing diff content to extract
meaningful code structures and patterns.
"""

import ast
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


class CompiledPatterns:
    """Cached compiled regex patterns for performance"""

    def __init__(self):
        self._patterns: Dict[str, Pattern] = {}

    @lru_cache(maxsize=64)
    def get(self, pattern: str, flags: int = 0) -> Pattern:
        """Get a compiled pattern with caching"""
        key = f"{pattern}:{flags}"
        if key not in self._patterns:
            self._patterns[key] = re.compile(pattern, flags)
        return self._patterns[key]

    def search(self, pattern: str, text: str, flags: int = 0) -> Optional[re.Match]:
        """Cached regex search"""
        return self.get(pattern, flags).search(text)

    def findall(self, pattern: str, text: str, flags: int = 0) -> List[str]:
        """Cached regex findall"""
        return self.get(pattern, flags).findall(text)

    def finditer(self, pattern: str, text: str, flags: int = 0):
        """Cached regex finditer"""
        return self.get(pattern, flags).finditer(text)


# Global compiled patterns instance
_compiled_patterns = CompiledPatterns()


class PythonCodeParser:
    """Fast Python code analysis using AST parsing"""

    def __init__(self):
        """Initialize the parser"""
        self.standard_libs = {
            "os",
            "sys",
            "time",
            "json",
            "re",
            "typing",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "urllib",
            "http",
            "logging",
            "unittest",
            "math",
            "random",
            "string",
            "hashlib",
            "uuid",
            "copy",
        }

    def extract_added_code_lines(self, diff_content: str) -> List[str]:
        """Extract only the added code lines from diff content"""
        added_lines = []
        for line in diff_content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("+") and not stripped.startswith("+++"):
                code_line = stripped[1:].strip()
                if code_line and not code_line.startswith(
                    "#"
                ):  # Skip empty lines and comments
                    added_lines.append(code_line)
        return added_lines

    def parse_code_safely(self, code: str) -> Optional[ast.AST]:
        """Safely parse Python code using AST"""
        try:
            return ast.parse(code)
        except SyntaxError:
            # Try to parse as expression if statement parsing fails
            try:
                return ast.parse(code, mode="eval")
            except SyntaxError:
                return None
        except Exception as e:
            logger.debug(f"AST parsing failed: {e}")
            return None

    def extract_functions(self, diff_content: str) -> List[Dict[str, Any]]:
        """Extract function definitions using AST parsing"""
        functions = []
        added_lines = self.extract_added_code_lines(diff_content)

        # Reconstruct code blocks for AST parsing
        code_block = "\n".join(added_lines)

        tree = self.parse_code_safely(code_block)
        if tree is None:
            # Fallback to line-by-line analysis for malformed code
            return self._extract_functions_line_by_line(added_lines)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [
                        self._get_decorator_name(d) for d in node.decorator_list
                    ],
                    "has_docstring": ast.get_docstring(node) is not None,
                    "returns": self._get_return_annotation(node),
                    "complexity_indicators": self._analyze_function_complexity(node),
                }
                functions.append(func_info)

        return functions

    def _extract_functions_line_by_line(
        self, added_lines: List[str]
    ) -> List[Dict[str, Any]]:
        """Fallback function extraction using simple line analysis"""
        functions = []
        for i, line in enumerate(added_lines):
            if line.strip().startswith("def ") or line.strip().startswith("async def "):
                # Basic function detection
                is_async = line.strip().startswith("async def ")
                func_name = line.split("(")[0].split()[-1] if "(" in line else "unknown"
                functions.append(
                    {
                        "name": func_name,
                        "line_number": i + 1,
                        "is_async": is_async,
                        "args": [],
                        "decorators": [],
                        "has_docstring": False,
                        "returns": None,
                        "complexity_indicators": {"estimated": True},
                    }
                )
        return functions

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from AST node"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return "unknown"

    def _get_return_annotation(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation"""
        if func_node.returns:
            try:
                return ast.unparse(func_node.returns)
            except (AttributeError, TypeError):
                return "annotated"
        return None

    def _analyze_function_complexity(
        self, func_node: ast.FunctionDef
    ) -> Dict[str, Any]:
        """Analyze function complexity indicators"""
        complexity = {
            "branches": 0,
            "loops": 0,
            "nested_functions": 0,
            "try_blocks": 0,
            "lines": len(func_node.body),
        }

        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.IfExp)):
                complexity["branches"] += 1
            elif isinstance(
                node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp)
            ):
                complexity["loops"] += 1
            elif isinstance(node, ast.FunctionDef) and node != func_node:
                complexity["nested_functions"] += 1
            elif isinstance(node, ast.Try):
                complexity["try_blocks"] += 1

        return complexity

    def extract_imports(self, diff_content: str) -> List[Dict[str, Any]]:
        """Extract import statements using AST parsing with line-based fallback"""
        imports = []
        added_lines = self.extract_added_code_lines(diff_content)

        # Use line-by-line analysis for imports (more reliable than AST for partial code)
        for line in added_lines:
            import_info = self._parse_import_line(line)
            if import_info:
                imports.append(import_info)

        return imports

    def _parse_import_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single import line"""
        line = line.strip()

        if line.startswith("import "):
            # Simple import: "import library" or "import library as alias"
            parts = line.split()
            if len(parts) >= 2:
                module_part = parts[1]
                library = module_part.split(".")[0]
                alias = parts[3] if len(parts) >= 4 and parts[2] == "as" else None

                if library not in self.standard_libs:
                    return {
                        "type": "import",
                        "library": library,
                        "module": module_part if "." in module_part else None,
                        "alias": alias,
                        "line": line,
                    }

        elif line.startswith("from "):
            # From import: "from library import item" or "from library.module import item"
            if " import " in line:
                from_part, import_part = line.split(" import ", 1)
                library_path = from_part[5:]  # Remove "from "

                if "." in library_path:
                    parts = library_path.split(".")
                    library = parts[0]
                    module = ".".join(parts[1:])
                else:
                    library = library_path
                    module = None

                # Extract imported items
                imported_items = [item.strip() for item in import_part.split(",")]

                if library not in self.standard_libs:
                    return {
                        "type": "from_import",
                        "library": library,
                        "module": module,
                        "items": imported_items,
                        "line": line,
                    }

        return None

    def extract_security_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Extract security-relevant patterns using AST analysis"""
        patterns = []
        added_lines = self.extract_added_code_lines(diff_content)

        # Combine lines to form code blocks for AST parsing
        code_block = "\n".join(added_lines)
        tree = self.parse_code_safely(code_block)

        if tree:
            patterns.extend(self._find_security_patterns_ast(tree))

        # Also do line-by-line analysis for patterns AST might miss
        patterns.extend(self._find_security_patterns_text(added_lines))

        return patterns

    def _find_security_patterns_ast(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find security patterns using AST analysis"""
        patterns = []

        for node in ast.walk(tree):
            # Detect eval/exec calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ["eval", "exec"]:
                    patterns.append(
                        {
                            "type": "code_injection",
                            "severity": "critical",
                            "description": f"Use of {node.func.id}() detected",
                            "line": node.lineno,
                            "evidence": node.func.id,
                        }
                    )

            # Detect string formatting in potential SQL contexts
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                # String % formatting
                if self._looks_like_sql_string(node.left):
                    patterns.append(
                        {
                            "type": "sql_injection",
                            "severity": "critical",
                            "description": "Potential SQL injection via string formatting",
                            "line": node.lineno,
                            "evidence": "String % formatting in SQL context",
                        }
                    )

            elif isinstance(node, ast.JoinedStr):
                # f-string formatting
                if self._contains_sql_keywords(node):
                    patterns.append(
                        {
                            "type": "sql_injection",
                            "severity": "critical",
                            "description": "Potential SQL injection via f-string",
                            "line": node.lineno,
                            "evidence": "f-string in SQL context",
                        }
                    )

        return patterns

    def _find_security_patterns_text(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find security patterns using text analysis"""
        patterns = []

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Hardcoded secrets detection
            if (
                any(
                    keyword in line_lower
                    for keyword in ["password", "secret", "token", "key"]
                )
                and "=" in line
            ):
                if not any(
                    secure in line_lower
                    for secure in ["getenv", "environ", "config.", "settings."]
                ):
                    patterns.append(
                        {
                            "type": "hardcoded_secrets",
                            "severity": "critical",
                            "description": "Potential hardcoded secret detected",
                            "line": i + 1,
                            "evidence": line.strip(),
                        }
                    )

            # Weak crypto detection
            if "md5" in line_lower or "sha1" in line_lower:
                patterns.append(
                    {
                        "type": "weak_crypto",
                        "severity": "high",
                        "description": "Weak cryptographic algorithm detected",
                        "line": i + 1,
                        "evidence": line.strip(),
                    }
                )

        return patterns

    def _looks_like_sql_string(self, node: ast.expr) -> bool:
        """Check if an AST node looks like a SQL string"""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            sql_keywords = ["select", "insert", "update", "delete", "from", "where"]
            return any(keyword in node.value.lower() for keyword in sql_keywords)
        return False

    def _contains_sql_keywords(self, node: ast.JoinedStr) -> bool:
        """Check if a JoinedStr contains SQL keywords"""
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                sql_keywords = ["select", "insert", "update", "delete", "from", "where"]
                if any(keyword in value.value.lower() for keyword in sql_keywords):
                    return True
        return False

    @lru_cache(maxsize=128)
    def get_complexity_score(self, code_snippet: str) -> int:
        """Calculate cyclomatic complexity using AST (cached)"""
        tree = self.parse_code_safely(code_snippet)
        if not tree:
            return 1  # Base complexity

        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.If,
                    ast.While,
                    ast.For,
                    ast.ExceptHandler,
                    ast.With,
                    ast.Assert,
                    ast.BoolOp,
                ),
            ):
                complexity += 1
            elif isinstance(node, ast.IfExp):  # Ternary operator
                complexity += 1

        return complexity

    def extract_performance_patterns(self, diff_content: str) -> List[Dict[str, Any]]:
        """Extract performance anti-patterns using AST and text analysis"""
        patterns = []
        added_lines = self.extract_added_code_lines(diff_content)

        # Check for string concatenation in loops using text analysis
        for i, line in enumerate(added_lines):
            if "+=" in line and any(
                str_type in line for str_type in ["str(", '"', "'"]
            ):
                patterns.append(
                    {
                        "type": "string_concatenation",
                        "severity": "medium",
                        "description": "String concatenation in loop detected",
                        "line": i + 1,
                        "evidence": line.strip(),
                        "suggestion": "Use ''.join() or list comprehension",
                    }
                )

        # Check for membership testing on lists
        for i, line in enumerate(added_lines):
            if " in [" in line or " in (" in line:
                patterns.append(
                    {
                        "type": "membership_test",
                        "severity": "medium",
                        "description": "Inefficient membership testing on list/tuple",
                        "line": i + 1,
                        "evidence": line.strip(),
                        "suggestion": "Use set for O(1) membership testing",
                    }
                )

        return patterns


# Create a global parser instance for reuse
_parser_instance = None


def get_parser() -> PythonCodeParser:
    """Get a shared parser instance (singleton pattern)"""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = PythonCodeParser()
    return _parser_instance


def get_compiled_patterns() -> CompiledPatterns:
    """Get the global compiled patterns instance"""
    return _compiled_patterns
