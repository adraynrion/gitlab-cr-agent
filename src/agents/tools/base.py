"""
Base classes and interfaces for the tool system
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolPriority(Enum):
    """Priority levels for tool execution"""

    CRITICAL = 1  # Must run, affects security or correctness
    HIGH = 2  # Should run, important for quality
    MEDIUM = 3  # Nice to have, improves review depth
    LOW = 4  # Optional, adds minor insights


class ToolCategory(Enum):
    """Categories for tool classification"""

    DOCUMENTATION = "documentation"  # Documentation lookup and validation
    SECURITY = "security"  # Security analysis
    PERFORMANCE = "performance"  # Performance analysis
    CORRECTNESS = "correctness"  # Code correctness validation
    STYLE = "style"  # Code style and conventions
    MAINTAINABILITY = "maintainability"  # Code maintainability checks


@dataclass
class ToolContext:
    """Context passed to tools during execution"""

    # Code context
    diff_content: str
    file_changes: List[Dict[str, Any]]
    source_branch: str
    target_branch: str

    # Repository context
    repository_url: str
    project_id: Optional[int] = None
    merge_request_iid: Optional[int] = None

    # Configuration
    settings: Dict[str, Any] = field(default_factory=dict)

    # Caching
    cache: Dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolResult(BaseModel):
    """Standardized result from tool execution"""

    tool_name: str = Field(
        ..., description="Name of the tool that generated this result"
    )
    category: ToolCategory = Field(..., description="Category of the tool")
    success: bool = Field(..., description="Whether the tool executed successfully")

    # Findings
    issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of issues found"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    positive_findings: List[str] = Field(
        default_factory=list, description="Positive aspects identified"
    )

    # Evidence and references
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting evidence from documentation or best practices",
    )
    references: List[str] = Field(
        default_factory=list,
        description="External references (documentation links, etc.)",
    )

    # Metrics and scores
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Quantitative metrics from analysis"
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the findings (0-1)"
    )

    # Error handling
    error_message: Optional[str] = Field(
        None, description="Error message if tool failed"
    )
    partial_result: bool = Field(
        default=False, description="Whether this is a partial result due to errors"
    )

    # Metadata
    execution_time_ms: Optional[int] = Field(
        None, description="Tool execution time in milliseconds"
    )
    cached: bool = Field(
        default=False, description="Whether result was retrieved from cache"
    )


class BaseTool(ABC):
    """Abstract base class for all review tools"""

    def __init__(self, name: Optional[str] = None):
        """Initialize the tool

        Args:
            name: Optional custom name for the tool
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._initialized = False

    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        """Return the category of this tool"""

    @property
    @abstractmethod
    def priority(self) -> ToolPriority:
        """Return the priority of this tool"""

    @property
    def description(self) -> str:
        """Return a description of what this tool does"""
        return self.__doc__ or "No description available"

    @property
    def requires_network(self) -> bool:
        """Whether this tool requires network access"""
        return False

    @property
    def cacheable(self) -> bool:
        """Whether results from this tool can be cached"""
        return True

    @property
    def timeout_seconds(self) -> int:
        """Maximum execution time for this tool"""
        return 30

    async def initialize(self, context: ToolContext) -> None:
        """Initialize the tool with context

        Override this method to perform any async initialization

        Args:
            context: The tool execution context
        """
        self._initialized = True
        self.logger.debug(f"Initialized tool: {self.name}")

    @abstractmethod
    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute the tool and return results

        Args:
            context: The tool execution context

        Returns:
            ToolResult with findings and evidence
        """

    async def run(self, context: ToolContext) -> ToolResult:
        """Run the tool with initialization and error handling

        Args:
            context: The tool execution context

        Returns:
            ToolResult with findings or error information
        """
        import time

        start_time = time.time()

        try:
            # Initialize if needed
            if not self._initialized:
                await self.initialize(context)

            # Check cache if applicable
            if self.cacheable:
                cache_key = self._get_cache_key(context)
                if cache_key in context.cache:
                    self.logger.debug(f"Cache hit for tool: {self.name}")
                    cached_result: ToolResult = context.cache[cache_key]
                    cached_result.cached = True
                    return cached_result

            # Execute the tool
            self.logger.debug(f"Executing tool: {self.name}")
            result = await self.execute(context)

            # Add execution metadata
            execution_time = int((time.time() - start_time) * 1000)
            result.execution_time_ms = max(1, execution_time)  # Ensure at least 1ms
            result.tool_name = self.name
            result.category = self.category

            # Cache if applicable
            if self.cacheable and result.success:
                cache_key = self._get_cache_key(context)
                context.cache[cache_key] = result

            self.logger.debug(f"Tool {self.name} completed in {execution_time}ms")
            return result

        except Exception as e:
            self.logger.error(f"Tool {self.name} failed: {e}")
            execution_time = int((time.time() - start_time) * 1000)

            return ToolResult(
                tool_name=self.name,
                category=self.category,
                success=False,
                error_message=str(e),
                partial_result=True,
                execution_time_ms=max(1, execution_time),  # Ensure at least 1ms
            )

    def _get_cache_key(self, context: ToolContext) -> str:
        """Generate a cache key for the tool execution

        Args:
            context: The tool execution context

        Returns:
            A unique cache key string
        """
        import hashlib

        # Create a cache key based on tool name and relevant context
        key_parts = [
            self.name,
            context.source_branch,
            context.target_branch,
            hashlib.md5(context.diff_content.encode()).hexdigest()[:8],
        ]

        return ":".join(key_parts)

    def __repr__(self) -> str:
        """String representation of the tool"""
        return (
            f"<{self.__class__.__name__} "
            f"name='{self.name}' "
            f"category={self.category.value} "
            f"priority={self.priority.value}>"
        )
