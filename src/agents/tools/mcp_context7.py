"""
Context7 MCP tools implementation for PydanticAI agent

These tools integrate Context7 MCP functionality via HTTP API calls,
making Context7's documentation database available as agent tools.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class LibraryResolutionResult(BaseModel):
    """Result of library ID resolution"""

    library_id: Optional[str] = Field(
        default=None, description="Context7-compatible library ID"
    )
    name: str = Field(description="Library name")
    description: Optional[str] = Field(default=None, description="Library description")
    trust_score: Optional[float] = Field(
        default=None, description="Trust score for the library"
    )
    versions: List[str] = Field(default_factory=list, description="Available versions")
    context7_available: bool = Field(
        default=True,
        description="Whether Context7 MCP was available for this resolution",
    )
    unavailability_reason: Optional[str] = Field(
        default=None, description="Reason why Context7 was unavailable"
    )


class CodeSnippet(BaseModel):
    """A code snippet from documentation"""

    title: str = Field(description="Title of the code snippet")
    description: str = Field(description="Description of what the snippet does")
    code: str = Field(description="The actual code")
    source: str = Field(description="Source URL or reference")
    language: str = Field(description="Programming language")


class QuestionAnswer(BaseModel):
    """A Q&A pair from documentation"""

    topic: str = Field(description="Topic category")
    question: str = Field(description="The question")
    answer: str = Field(description="The answer")
    source: str = Field(description="Source URL or reference")


class LibraryDocumentation(BaseModel):
    """Documentation retrieved for a library"""

    library_id: str = Field(description="Context7 library ID")
    topic: Optional[str] = Field(description="Specific topic requested")
    snippets: List[CodeSnippet] = Field(
        default_factory=list, description="Code snippets"
    )
    questions_answers: List[QuestionAnswer] = Field(
        default_factory=list, description="Q&A pairs"
    )
    references: List[str] = Field(default_factory=list, description="Reference URLs")
    context7_available: bool = Field(
        default=True, description="Whether Context7 MCP was available for this request"
    )
    unavailability_reason: Optional[str] = Field(
        default=None, description="Reason why Context7 was unavailable"
    )


async def _check_context7_available() -> bool:
    """Check if Context7 API is available"""
    try:
        settings = get_settings()
        if not settings.context7_enabled:
            return False

        # Simple health check to Context7 API
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.context7_api_url}/health")
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"Context7 API health check failed: {e}")
        return False


async def resolve_library_id(library_name: str) -> LibraryResolutionResult:
    """
    Resolve a library name to Context7-compatible library ID.

    This tool resolves a general library name into a Context7-compatible library ID
    and returns matching libraries with trust scores and available versions.

    Args:
        library_name: The name of the library to search for

    Returns:
        LibraryResolutionResult with availability status and library information
    """
    try:
        # Check if Context7 API is available
        if not await _check_context7_available():
            logger.warning(f"Context7 API not available for resolving {library_name}")
            return LibraryResolutionResult(
                library_id=None,
                name=library_name,
                description=None,
                trust_score=None,
                versions=[],
                context7_available=False,
                unavailability_reason="Context7 API not available or disabled",
            )

        # Call the Context7 API
        settings = get_settings()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.context7_api_url}/api/resolve-library-id",
                json={"libraryName": library_name},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.warning(
                    f"Context7 API error for {library_name}: {response.status_code}"
                )
                return LibraryResolutionResult(
                    library_id=None,
                    name=library_name,
                    description=None,
                    trust_score=None,
                    versions=[],
                    context7_available=False,
                    unavailability_reason=f"Context7 API error: {response.status_code}",
                )

            result = response.text

        # Parse the result to extract the best match
        if result and "Context7-compatible library ID:" in result:
            lines = result.split("\n")
            best_match = None
            current_match: Dict[str, Any] = {}

            for line in lines:
                line = line.strip()
                if line.startswith("- Title:"):
                    if current_match and current_match.get("library_id"):
                        if not best_match or current_match.get(
                            "trust_score", 0
                        ) > best_match.get("trust_score", 0):
                            best_match = current_match.copy()
                    current_match = {"name": line[8:].strip()}
                elif line.startswith("- Context7-compatible library ID:"):
                    current_match["library_id"] = line[34:].strip()
                elif line.startswith("- Description:"):
                    current_match["description"] = line[14:].strip()
                elif line.startswith("- Trust Score:"):
                    try:
                        current_match["trust_score"] = float(line[14:].strip())
                    except ValueError:
                        current_match["trust_score"] = 0.0
                elif line.startswith("- Versions:"):
                    versions_str = line[11:].strip()
                    current_match["versions"] = (
                        [v.strip() for v in versions_str.split(",")]
                        if versions_str
                        else []
                    )

            # Don't forget the last match
            if current_match and current_match.get("library_id"):
                if not best_match or current_match.get(
                    "trust_score", 0
                ) > best_match.get("trust_score", 0):
                    best_match = current_match

            if best_match:
                return LibraryResolutionResult(
                    library_id=best_match.get("library_id"),
                    name=best_match.get("name", library_name),
                    description=best_match.get("description"),
                    trust_score=best_match.get("trust_score"),
                    versions=best_match.get("versions", []),
                    context7_available=True,
                    unavailability_reason=None,
                )

        # Context7 available but no match found
        logger.info(f"Context7 MCP available but no match found for {library_name}")
        return LibraryResolutionResult(
            library_id=None,
            name=library_name,
            description=None,
            trust_score=None,
            versions=[],
            context7_available=True,
            unavailability_reason=None,
        )

    except Exception as e:
        logger.error(f"Error resolving library ID for {library_name}: {e}")
        return LibraryResolutionResult(
            library_id=None,
            name=library_name,
            description=None,
            trust_score=None,
            versions=[],
            context7_available=False,
            unavailability_reason=f"Error during Context7 resolution: {str(e)}",
        )


async def get_library_docs(
    context7_library_id: str, topic: Optional[str] = None, max_tokens: int = 2000
) -> LibraryDocumentation:
    """
    Get up-to-date documentation for a library from Context7.

    Fetches current documentation, code examples, and Q&A for the specified
    library from Context7's comprehensive database.

    Args:
        context7_library_id: Context7-compatible library ID (e.g., '/tiangolo/fastapi')
        topic: Optional topic to focus documentation on (e.g., 'authentication')
        max_tokens: Maximum number of tokens to retrieve (default: 2000)

    Returns:
        LibraryDocumentation with availability status and documentation content
    """
    try:
        # Check if Context7 API is available
        if not await _check_context7_available():
            logger.warning(
                f"Context7 API not available for getting docs for {context7_library_id}"
            )
            return LibraryDocumentation(
                library_id=context7_library_id,
                topic=topic,
                snippets=[],
                questions_answers=[],
                references=[],
                context7_available=False,
                unavailability_reason="Context7 API not available or disabled",
            )

        # Call the Context7 API
        settings = get_settings()
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "context7CompatibleLibraryID": context7_library_id,
                "tokens": max_tokens,
            }
            if topic:
                payload["topic"] = topic

            response = await client.post(
                f"{settings.context7_api_url}/api/get-library-docs",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.warning(
                    f"Context7 API error for {context7_library_id}: {response.status_code}"
                )
                return LibraryDocumentation(
                    library_id=context7_library_id,
                    topic=topic,
                    snippets=[],
                    questions_answers=[],
                    references=[],
                    context7_available=False,
                    unavailability_reason=f"Context7 API error: {response.status_code}",
                )

            result = response.text

        # Parse the result into structured format
        docs = LibraryDocumentation(
            library_id=context7_library_id,
            topic=topic,
            context7_available=True,
            unavailability_reason=None,
        )

        if not result:
            return docs

        # Parse code snippets
        if "CODE SNIPPETS" in result:
            snippet_section = result.split("CODE SNIPPETS")[1]
            if "QUESTIONS AND ANSWERS" in snippet_section:
                snippet_section = snippet_section.split("QUESTIONS AND ANSWERS")[0]

            # Split by separator lines
            snippets = snippet_section.split("----------------------------------------")

            for snippet_text in snippets:
                if "TITLE:" in snippet_text and "CODE:" in snippet_text:
                    snippet = _parse_code_snippet(snippet_text.strip())
                    if snippet:
                        docs.snippets.append(snippet)

                        # Add source to references
                        if snippet.source and snippet.source not in docs.references:
                            docs.references.append(snippet.source)

        # Parse Q&A section
        if "QUESTIONS AND ANSWERS" in result:
            qa_section = result.split("QUESTIONS AND ANSWERS")[1]
            qa_items = qa_section.split("----------------------------------------")

            for qa_text in qa_items:
                if "Q:" in qa_text and "A:" in qa_text:
                    qa = _parse_question_answer(qa_text.strip())
                    if qa:
                        docs.questions_answers.append(qa)

                        # Add source to references
                        if qa.source and qa.source not in docs.references:
                            docs.references.append(qa.source)

        return docs

    except Exception as e:
        logger.error(f"Error getting library docs for {context7_library_id}: {e}")
        return LibraryDocumentation(
            library_id=context7_library_id,
            topic=topic,
            snippets=[],
            questions_answers=[],
            references=[],
            context7_available=False,
            unavailability_reason=f"Error during Context7 documentation retrieval: {str(e)}",
        )


def _parse_code_snippet(snippet_text: str) -> Optional[CodeSnippet]:
    """Parse a single code snippet from Context7 result"""
    lines = snippet_text.split("\n")
    title = ""
    description = ""
    source = ""
    language = ""
    code_lines: List[str] = []

    capturing_code = False

    for line in lines:
        line = line.strip()
        if line.startswith("TITLE:"):
            title = line[6:].strip()
        elif line.startswith("DESCRIPTION:"):
            description = line[12:].strip()
        elif line.startswith("SOURCE:"):
            source = line[7:].strip()
        elif line.startswith("LANGUAGE:"):
            language = line[9:].strip()
        elif line.startswith("CODE:"):
            capturing_code = True
        elif capturing_code:
            if line.startswith("```"):
                if not code_lines:  # Start of code block
                    continue
                else:  # End of code block
                    break
            else:
                code_lines.append(line)

    if title and code_lines:
        return CodeSnippet(
            title=title,
            description=description,
            code="\n".join(code_lines),
            source=source,
            language=language,
        )

    return None


def _parse_question_answer(qa_text: str) -> Optional[QuestionAnswer]:
    """Parse a single Q&A item from Context7 result"""
    lines = qa_text.split("\n")
    topic = ""
    question = ""
    answer = ""
    source = ""

    for line in lines:
        line = line.strip()
        if line.startswith("TOPIC:"):
            topic = line[6:].strip()
        elif line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
        elif line.startswith("SOURCE:"):
            source = line[7:].strip()

    if question and answer:
        return QuestionAnswer(
            topic=topic, question=question, answer=answer, source=source
        )

    return None


async def search_documentation(
    query: str, libraries: Optional[List[str]] = None, max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for documentation across multiple libraries.

    This tool helps find relevant documentation and code examples
    across multiple libraries based on a search query.

    Args:
        query: Search query (e.g., "authentication", "async patterns")
        libraries: Optional list of library names to search in
        max_results: Maximum number of results to return

    Returns:
        List of relevant documentation results with Context7 availability status
    """
    results: List[Dict[str, Any]] = []

    # Return empty results if no libraries specified and Context7 is unavailable
    if not libraries:
        if not await _check_context7_available():
            logger.warning(
                "Context7 API not available and no specific libraries provided for search"
            )
            return []
        else:
            logger.info(
                "No specific libraries provided for search query, Context7 may provide suggestions"
            )
            return []

    for library_name in libraries[:max_results]:
        try:
            # Resolve library ID
            resolution = await resolve_library_id(library_name)

            # Track Context7 availability for this library
            context7_available = resolution.context7_available

            if not resolution.library_id:
                # Add unavailability info if Context7 was not available
                if not context7_available:
                    results.append(
                        {
                            "library": library_name,
                            "library_id": None,
                            "trust_score": None,
                            "snippets_count": 0,
                            "qa_count": 0,
                            "relevant_snippets": [],
                            "references": [],
                            "context7_available": False,
                            "unavailability_reason": resolution.unavailability_reason,
                        }
                    )
                continue

            # Get documentation for the query topic
            docs = await get_library_docs(
                resolution.library_id, topic=query, max_tokens=1000
            )

            result_entry = {
                "library": library_name,
                "library_id": resolution.library_id,
                "trust_score": resolution.trust_score,
                "snippets_count": len(docs.snippets),
                "qa_count": len(docs.questions_answers),
                "relevant_snippets": [
                    {
                        "title": snippet.title,
                        "description": snippet.description,
                        "code_preview": snippet.code[:200] + "..."
                        if len(snippet.code) > 200
                        else snippet.code,
                    }
                    for snippet in docs.snippets[:2]  # Top 2 snippets
                ],
                "references": docs.references[:3],  # Top 3 references
                "context7_available": docs.context7_available,
                "unavailability_reason": docs.unavailability_reason,
            }

            results.append(result_entry)

        except Exception as e:
            logger.warning(f"Error searching {library_name}: {e}")
            results.append(
                {
                    "library": library_name,
                    "library_id": None,
                    "trust_score": None,
                    "snippets_count": 0,
                    "qa_count": 0,
                    "relevant_snippets": [],
                    "references": [],
                    "context7_available": False,
                    "unavailability_reason": f"Error during search: {str(e)}",
                }
            )

    # Sort by trust score and relevance (None values go to end)
    results.sort(
        key=lambda x: (x.get("trust_score") or 0, x.get("snippets_count", 0)),
        reverse=True,
    )

    return results[:max_results]


async def validate_api_usage(
    library_name: str, code_snippet: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate API usage against official documentation.

    This tool checks if the provided code snippet follows the best practices
    and patterns documented in the official library documentation.

    Args:
        library_name: Name of the library being used
        code_snippet: Code snippet to validate
        context: Optional context about where/how the code is used

    Returns:
        Validation result with Context7 availability status
    """
    try:
        # Resolve library ID
        resolution = await resolve_library_id(library_name)

        if not resolution.context7_available:
            logger.warning(f"Context7 MCP not available for validating {library_name}")
            return {
                "valid": None,  # Cannot determine validity without Context7
                "issues": [],
                "suggestions": [],
                "references": [],
                "context7_available": False,
                "unavailability_reason": resolution.unavailability_reason,
                "library_name": library_name,
            }

        if not resolution.library_id:
            logger.info(f"No Context7 library ID found for {library_name}")
            return {
                "valid": None,  # Cannot validate without library ID
                "issues": [],
                "suggestions": [],
                "references": [],
                "context7_available": True,
                "unavailability_reason": None,
                "library_name": library_name,
            }

        # Get documentation
        docs = await get_library_docs(
            resolution.library_id, topic=context, max_tokens=3000
        )

        if not docs.context7_available:
            logger.warning(
                f"Context7 documentation retrieval failed for {library_name}"
            )
            return {
                "valid": None,
                "issues": [],
                "suggestions": [],
                "references": [],
                "context7_available": False,
                "unavailability_reason": docs.unavailability_reason,
                "library_name": library_name,
            }

        validation_result = {
            "valid": True,
            "issues": [],
            "suggestions": [],
            "references": docs.references,
            "context7_available": True,
            "unavailability_reason": None,
            "library_name": library_name,
        }

        # Check against code snippets for patterns
        for snippet in docs.snippets:
            # Look for similar patterns or anti-patterns
            if any(
                keyword in snippet.title.lower()
                for keyword in ["deprecated", "avoid", "don't"]
            ):
                # Check if code might be using deprecated patterns
                snippet_keywords = _extract_keywords(snippet.code)
                code_keywords = _extract_keywords(code_snippet)

                if any(keyword in code_keywords for keyword in snippet_keywords):
                    issues_list = validation_result.get("issues", [])
                    if isinstance(issues_list, list):
                        issues_list.append(
                            f"Potential deprecated pattern detected. See: {snippet.title}"
                        )

            elif any(
                keyword in snippet.title.lower()
                for keyword in ["best practice", "recommended", "example"]
            ):
                suggestions_list = validation_result.get("suggestions", [])
                if isinstance(suggestions_list, list):
                    suggestions_list.append(f"Best practice example: {snippet.title}")

        # Check Q&A for common issues
        for qa in docs.questions_answers:
            if any(
                keyword in qa.question.lower()
                for keyword in ["error", "issue", "problem", "warning"]
            ):
                suggestions_list = validation_result.get("suggestions", [])
                if isinstance(suggestions_list, list):
                    suggestions_list.append(f"Common issue: {qa.question}")

        return validation_result

    except Exception as e:
        logger.error(f"Error validating API usage for {library_name}: {e}")
        return {
            "valid": False,
            "issues": [f"Error during validation: {str(e)}"],
            "suggestions": [],
            "references": [],
            "context7_available": False,
            "unavailability_reason": f"Error during validation: {str(e)}",
            "library_name": library_name,
        }


def _extract_keywords(code: str) -> List[str]:
    """Extract keywords and function names from code"""
    import re

    # Simple keyword extraction - look for function calls and imports
    keywords = []

    # Find function calls: word(
    function_calls = re.findall(r"(\w+)\s*\(", code)
    keywords.extend(function_calls)

    # Find imports: from x import y, import x
    imports = re.findall(r"(?:from\s+(\w+)|import\s+(\w+))", code)
    for imp in imports:
        keywords.extend([x for x in imp if x])

    # Find class/method calls: object.method
    method_calls = re.findall(r"(\w+)\.(\w+)", code)
    for obj, method in method_calls:
        keywords.extend([obj, method])

    return list(set(keywords))  # Remove duplicates
