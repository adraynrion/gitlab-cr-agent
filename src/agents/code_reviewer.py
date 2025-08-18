"""
PydanticAI-based code review agent with multi-LLM support and tool system
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict
from typing import List as TypingList
from typing import Optional

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.providers import get_llm_model
from src.config.settings import get_settings
from src.exceptions import AIProviderException, ReviewProcessException
from src.models.review_models import ReviewContext, ReviewResult

logger = logging.getLogger(__name__)

# System prompt for code review with tool integration
CODE_REVIEW_SYSTEM_PROMPT = """
You are an expert software engineer conducting thorough, evidence-based code reviews.
Your role is to analyze code changes using specialized tools and provide constructive, actionable feedback.

REVIEW FRAMEWORK:
1. **Correctness**: Identify logic errors, edge cases, and algorithm issues
2. **Security**: Detect vulnerabilities, input validation issues, authentication flaws
3. **Performance**: Find bottlenecks, inefficient algorithms, resource usage problems
4. **Maintainability**: Assess code clarity, structure, documentation quality
5. **Best Practices**: Check language conventions, design patterns, testing

NATIVE MCP INTEGRATION:
You have direct access to Context7's comprehensive documentation database through native MCP tools:
- resolve-library-id(libraryName): Get Context7 library ID and metadata for a library
- get-library-docs(context7CompatibleLibraryID, topic, tokens): Fetch current documentation and examples
- validate-code-against-docs: Validate code usage patterns against official documentation

IMPORTANT: Context7 MCP may not always be available. When Context7 is unavailable:
- Clearly inform the user that Context7 documentation validation is not available
- Explain that the review is based on your knowledge only
- Still provide a thorough review using your training knowledge
- Be transparent about limitations without Context7 verification

Use these Context7 MCP tools to:
- Verify API usage patterns against official documentation
- Find relevant code examples and best practices
- Check for deprecated methods or patterns
- Get authoritative references for your recommendations
- Provide evidence-based analysis backed by current documentation

OUTPUT REQUIREMENTS:
- Integrate tool findings into your review
- Provide specific file paths and line numbers for each issue
- Categorize issues by severity (critical, high, medium, low)
- Include concrete suggestions backed by tool evidence
- Reference official documentation when available
- Show code examples when helpful
- Balance criticism with positive observations
- Consider the broader codebase context

EVIDENCE-BASED APPROACH:
- Prioritize issues backed by tool evidence
- Include references to official documentation
- Cite specific security guidelines (OWASP, etc.)
- Use quantitative metrics when available
- Distinguish between tool-detected issues and manual observations

TONE:
- Be constructive and educational
- Focus on substantial issues over style preferences
- Acknowledge good practices when observed
- Provide clear rationale for suggestions with supporting evidence
"""


@dataclass
class ReviewDependencies:
    """Dependencies for code review operations"""

    repository_url: str
    branch: str
    merge_request_iid: int
    gitlab_token: str
    diff_content: str
    file_changes: TypingList[Dict[str, Any]]
    review_trigger_tag: str


class CodeReviewAgent:
    """Main code review agent using PydanticAI with native MCP integration"""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the review agent with specified model and MCP toolsets"""
        settings = get_settings()
        self.model_name = model_name or settings.ai_model
        self.model = get_llm_model(self.model_name)

        # Initialize MCP toolsets
        toolsets = []
        if settings.context7_enabled:
            try:
                # Create Context7 MCP server with hardcoded configuration
                context7_server = MCPServerStdio(
                    command="npx",
                    args=["-y", "@upstash/context7-mcp@1.0.14"],
                    timeout=30.0,
                )
                toolsets.append(context7_server)
                logger.info("Context7 MCP server configured successfully")
            except Exception as e:
                logger.warning(f"Failed to configure Context7 MCP server: {e}")
                settings.context7_enabled = False

        # Create PydanticAI agent with MCP toolsets
        self.agent = Agent(
            model=self.model,
            output_type=ReviewResult,
            deps_type=ReviewDependencies,
            system_prompt=CODE_REVIEW_SYSTEM_PROMPT,
            retries=settings.ai_retries,
            toolsets=toolsets,
        )

        logger.info(
            f"Initialized CodeReviewAgent with model: {self.model_name}, MCP enabled: {settings.context7_enabled}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((Exception,)),  # Retry on most exceptions
    )
    async def review_merge_request(
        self, diff_content: str, context: ReviewContext
    ) -> ReviewResult:
        """Perform comprehensive code review using native MCP integration"""

        settings = get_settings()
        logger.info(f"Starting MCP-enhanced review for MR {context.merge_request_iid}")

        if settings.context7_enabled:
            logger.info("Context7 MCP integration enabled")
        else:
            logger.info("Context7 MCP disabled, proceeding with basic review")

        # Prepare dependencies for agent
        deps = ReviewDependencies(
            repository_url=context.repository_url,
            branch=context.target_branch,
            merge_request_iid=context.merge_request_iid,
            gitlab_token=settings.gitlab_token,
            diff_content=diff_content,
            file_changes=context.file_changes,
            review_trigger_tag=context.trigger_tag,
        )

        # Construct enhanced review prompt with MCP integration instructions
        review_prompt = f"""
        Please review the following code changes from a GitLab merge request.

        Repository: {context.repository_url}
        Target Branch: {context.target_branch}
        Source Branch: {context.source_branch}

        DIFF CONTENT:
        {diff_content}

        CONTEXT7 MCP INTEGRATION:
        {"You have access to Context7's documentation database via MCP tools. Use these tools to validate API usage, get current documentation, and provide evidence-based recommendations." if settings.context7_enabled else "Context7 MCP is disabled. Rely on your training knowledge for the review."}

        Available Context7 MCP tools:
        {"- resolve-library-id(libraryName): Get library metadata" + chr(10) + "- get-library-docs(libraryId, topic, tokens): Fetch documentation" + chr(10) + "- validate_code_against_docs: Validate against current docs" if settings.context7_enabled else "No Context7 tools available"}

        INSTRUCTIONS:
        1. Analyze the code changes for correctness, security, performance, and maintainability
        2. {"Use Context7 MCP tools to validate library usage against current documentation" if settings.context7_enabled else "Rely on your training knowledge since Context7 is unavailable"}
        3. Provide specific, actionable feedback with line numbers when possible
        4. {"Include references to official documentation when Context7 tools provide them" if settings.context7_enabled else "Note that recommendations are based on training knowledge without real-time documentation validation"}
        5. Highlight both issues and positive aspects of the code
        6. Focus on substantial issues over style preferences

        Provide a comprehensive review covering:
        1. Critical issues that must be addressed
        2. {"Library usage validation (using Context7 if available)" if settings.context7_enabled else "Library usage based on training knowledge"}
        3. Security vulnerabilities and best practices
        4. Performance considerations
        5. Code quality and maintainability
        6. Positive observations
        """

        try:
            # Run the MCP-enhanced review agent
            async with self.agent:  # Context manager for MCP server lifecycle
                result = await self.agent.run(review_prompt, deps=deps)

                # Log token usage for monitoring
                if hasattr(result, "usage"):
                    usage = result.usage()
                    logger.info(
                        f"Review completed. Tokens used: {usage.total_tokens if hasattr(usage, 'total_tokens') else 'unknown'}"
                    )

                # Validate result has expected output
                if not hasattr(result, "output"):
                    raise AIProviderException(
                        message="AI provider returned unexpected result structure",
                        provider=self.model_name,
                        details={"result_type": type(result).__name__},
                    )

                # The output should be a ReviewResult from PydanticAI
                return result.output  # type: ignore[return-value]

        except FileNotFoundError as e:
            # Handle MCP server startup failure (npx not found, etc.)
            logger.warning(f"Context7 MCP server failed to start: {e}")
            logger.info("Falling back to review without Context7 MCP integration")

            # Create fallback agent without MCP toolsets
            fallback_agent = Agent(
                model=self.model,
                output_type=ReviewResult,
                deps_type=ReviewDependencies,
                system_prompt=CODE_REVIEW_SYSTEM_PROMPT.replace(
                    "You have access to Context7's comprehensive documentation database via MCP tools:",
                    "Context7 MCP integration is unavailable. Review based on training knowledge:",
                ).replace(
                    "Use these Context7 MCP tools to:",
                    "Without Context7 MCP, focus on:",
                ),
                retries=settings.ai_retries,
            )

            # Run fallback review
            result = await fallback_agent.run(review_prompt, deps=deps)

            # Log token usage for monitoring
            if hasattr(result, "usage"):
                usage = result.usage()
                logger.info(
                    f"Fallback review completed. Tokens used: {usage.total_tokens if hasattr(usage, 'total_tokens') else 'unknown'}"
                )

            return result.output  # type: ignore[return-value]

        except AIProviderException:
            # Don't wrap AI provider exceptions
            raise
        except Exception as e:
            logger.error(f"Review failed for MR {context.merge_request_iid}: {e}")
            raise ReviewProcessException(
                message=f"Review process failed for MR {context.merge_request_iid}",
                merge_request_iid=context.merge_request_iid,
                details={"model_name": self.model_name},
                original_error=e,
            )


async def initialize_review_agent() -> CodeReviewAgent:
    """Factory function to initialize the review agent"""
    try:
        settings = get_settings()
        agent = CodeReviewAgent(model_name=settings.ai_model)
        logger.info(
            f"Review agent initialized successfully with model: {settings.ai_model}"
        )
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize review agent: {e}")
        settings = get_settings()
        raise ReviewProcessException(
            message="Failed to initialize AI review agent",
            details={"model_name": settings.ai_model},
            original_error=e,
        )
