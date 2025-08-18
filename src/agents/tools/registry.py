"""
Tool registry for dynamic tool management and discovery
"""

import logging
from typing import Any, Dict, List, Optional, Set, Type

from src.agents.tools.base import (
    BaseTool,
    ToolCategory,
    ToolContext,
    ToolPriority,
    ToolResult,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Singleton registry for managing review tools"""

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls) -> "ToolRegistry":
        """Ensure singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry"""
        if not self._initialized:
            self._tools: Dict[str, Type[BaseTool]] = {}
            self._tool_instances: Dict[str, BaseTool] = {}
            self._categories: Dict[ToolCategory, Set[str]] = {
                category: set() for category in ToolCategory
            }
            self._priorities: Dict[ToolPriority, Set[str]] = {
                priority: set() for priority in ToolPriority
            }
            self._enabled_tools: Set[str] = set()
            self._disabled_tools: Set[str] = set()
            self._initialized = True
            logger.info("Tool registry initialized")

    def register(self, tool_class: Type[BaseTool], enabled: bool = True) -> None:
        """Register a tool class

        Args:
            tool_class: The tool class to register
            enabled: Whether the tool should be enabled by default
        """
        # Create a temporary instance to get metadata
        temp_instance = tool_class()
        tool_name = temp_instance.name

        # Register the tool class
        self._tools[tool_name] = tool_class

        # Update category and priority indices
        self._categories[temp_instance.category].add(tool_name)
        self._priorities[temp_instance.priority].add(tool_name)

        # Set enabled status
        if enabled:
            self._enabled_tools.add(tool_name)
        else:
            self._disabled_tools.add(tool_name)

        logger.debug(
            f"Registered tool: {tool_name} "
            f"(category={temp_instance.category.value}, "
            f"priority={temp_instance.priority.value}, "
            f"enabled={enabled})"
        )

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool

        Args:
            tool_name: Name of the tool to unregister
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool not found for unregistration: {tool_name}")
            return

        # Get tool instance for metadata
        tool_class = self._tools[tool_name]
        temp_instance = tool_class()

        # Remove from all indices
        del self._tools[tool_name]
        self._categories[temp_instance.category].discard(tool_name)
        self._priorities[temp_instance.priority].discard(tool_name)
        self._enabled_tools.discard(tool_name)
        self._disabled_tools.discard(tool_name)

        # Remove cached instance if exists
        if tool_name in self._tool_instances:
            del self._tool_instances[tool_name]

        logger.debug(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool instance by name

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not found
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool not found: {tool_name}")
            return None

        # Return cached instance or create new one
        if tool_name not in self._tool_instances:
            self._tool_instances[tool_name] = self._tools[tool_name]()

        return self._tool_instances[tool_name]

    def get_tools_by_category(
        self, category: ToolCategory, enabled_only: bool = True
    ) -> List[BaseTool]:
        """Get all tools in a category

        Args:
            category: The tool category
            enabled_only: Whether to return only enabled tools

        Returns:
            List of tool instances
        """
        tool_names = self._categories.get(category, set())

        if enabled_only:
            tool_names = tool_names.intersection(self._enabled_tools)

        tools = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)

        return tools

    def get_tools_by_priority(
        self, priority: ToolPriority, enabled_only: bool = True
    ) -> List[BaseTool]:
        """Get all tools with a specific priority

        Args:
            priority: The tool priority
            enabled_only: Whether to return only enabled tools

        Returns:
            List of tool instances
        """
        tool_names = self._priorities.get(priority, set())

        if enabled_only:
            tool_names = tool_names.intersection(self._enabled_tools)

        tools = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)

        return tools

    def get_enabled_tools(self) -> List[BaseTool]:
        """Get all enabled tools

        Returns:
            List of enabled tool instances
        """
        tools = []
        for name in self._enabled_tools:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)

        return tools

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools

        Returns:
            List of all tool instances
        """
        tools = []
        for name in self._tools:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)

        return tools

    def enable_tool(self, tool_name: str) -> None:
        """Enable a tool

        Args:
            tool_name: Name of the tool to enable
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool not found: {tool_name}")
            return

        self._enabled_tools.add(tool_name)
        self._disabled_tools.discard(tool_name)
        logger.debug(f"Enabled tool: {tool_name}")

    def disable_tool(self, tool_name: str) -> None:
        """Disable a tool

        Args:
            tool_name: Name of the tool to disable
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool not found: {tool_name}")
            return

        self._disabled_tools.add(tool_name)
        self._enabled_tools.discard(tool_name)
        logger.debug(f"Disabled tool: {tool_name}")

    def is_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled

        Args:
            tool_name: Name of the tool

        Returns:
            True if the tool is enabled
        """
        return tool_name in self._enabled_tools

    def configure_from_settings(self, settings: Dict[str, Any]) -> None:
        """Configure registry from settings dictionary

        Args:
            settings: Configuration dictionary
        """
        # Enable/disable tools based on settings
        enabled_tools = settings.get("enabled_tools", [])
        disabled_tools = settings.get("disabled_tools", [])

        for tool_name in enabled_tools:
            self.enable_tool(tool_name)

        for tool_name in disabled_tools:
            self.disable_tool(tool_name)

        # Configure by category
        enabled_categories = settings.get("enabled_categories", [])
        disabled_categories = settings.get("disabled_categories", [])

        for category_name in enabled_categories:
            try:
                category = ToolCategory(category_name)
                for tool_name in self._categories[category]:
                    self.enable_tool(tool_name)
            except ValueError:
                logger.warning(f"Invalid category in settings: {category_name}")

        for category_name in disabled_categories:
            try:
                category = ToolCategory(category_name)
                for tool_name in self._categories[category]:
                    self.disable_tool(tool_name)
            except ValueError:
                logger.warning(f"Invalid category in settings: {category_name}")

        logger.info(
            f"Registry configured: {len(self._enabled_tools)} tools enabled, "
            f"{len(self._disabled_tools)} tools disabled"
        )

    async def execute_tools(
        self,
        context: ToolContext,
        categories: Optional[List[ToolCategory]] = None,
        priorities: Optional[List[ToolPriority]] = None,
        parallel: bool = True,
    ) -> List[ToolResult]:
        """Execute multiple tools and collect results with language-aware routing

        Args:
            context: The tool execution context
            categories: Optional list of categories to execute
            priorities: Optional list of priorities to execute
            parallel: Whether to execute tools in parallel

        Returns:
            List of tool results
        """
        import asyncio

        # Determine which tools to execute
        tools_to_run = set()

        if categories:
            for category in categories:
                tools_to_run.update(
                    self.get_tools_by_category(category, enabled_only=True)
                )

        if priorities:
            for priority in priorities:
                tools_to_run.update(
                    self.get_tools_by_priority(priority, enabled_only=True)
                )

        # Default to all enabled tools if no filters
        if not categories and not priorities:
            tools_to_run = set(self.get_enabled_tools())

        # Language-aware filtering is simplified in the Context7-based architecture
        # The unified Context7 tool handles language detection internally via import parsing
        logger.debug(
            f"Executing {len(tools_to_run)} tools in simplified Context7 architecture"
        )

        if not tools_to_run:
            logger.warning("No tools to execute after language filtering")
            return []

        logger.info(f"Executing {len(tools_to_run)} language-appropriate tools")

        # Execute tools
        if parallel:
            # Run tools in parallel
            tasks = [tool.run(context) for tool in tools_to_run]
            gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and log them
            valid_results: List[ToolResult] = []
            for result in gathered_results:
                if isinstance(result, Exception):
                    logger.error(f"Tool execution failed: {result}")
                elif isinstance(result, ToolResult):
                    valid_results.append(result)

            return valid_results
        else:
            # Run tools sequentially
            sequential_results: List[ToolResult] = []
            for tool in tools_to_run:
                try:
                    result = await tool.run(context)
                    sequential_results.append(result)
                except Exception as e:
                    logger.error(f"Tool {tool.name} failed: {e}")
                    # Create a failed ToolResult for the failed tool
                    failed_result = ToolResult(
                        tool_name=tool.name,
                        category=tool.category,
                        success=False,
                        error_message=str(e),
                        partial_result=True,
                        execution_time_ms=0,
                    )
                    sequential_results.append(failed_result)

            return sequential_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics

        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_tools": len(self._tools),
            "enabled_tools": len(self._enabled_tools),
            "disabled_tools": len(self._disabled_tools),
            "tools_by_category": {
                category.value: len(tools)
                for category, tools in self._categories.items()
            },
            "tools_by_priority": {
                str(priority.value): len(tools)
                for priority, tools in self._priorities.items()
            },
            "cached_instances": len(self._tool_instances),
        }

    def clear_cache(self) -> None:
        """Clear all cached tool instances"""
        self._tool_instances.clear()
        logger.debug("Cleared tool instance cache")

    def __repr__(self) -> str:
        """String representation of the registry"""
        stats = self.get_statistics()
        return (
            f"<ToolRegistry "
            f"total={stats['total_tools']} "
            f"enabled={stats['enabled_tools']} "
            f"disabled={stats['disabled_tools']}>"
        )


# Decorator for automatic tool registration
def register_tool(
    enabled_or_name=True, enabled: bool = True, name: Optional[str] = None
):
    """Decorator to automatically register a tool class

    Args:
        enabled_or_name: Can be bool (enabled) or str (name) for flexible usage
        enabled: Whether the tool should be enabled by default (when using named params)
        name: Optional custom name for the tool

    Usage:
        @register_tool
        @register_tool(enabled=True)
        @register_tool("CustomName")
        @register_tool(name="CustomName", enabled=False)
        class MyTool(BaseTool):
            ...
    """

    # Handle different calling patterns
    if callable(enabled_or_name):
        # Called as @register_tool (without parentheses)
        # In this case, enabled_or_name is actually the class
        cls = enabled_or_name
        registry = ToolRegistry()
        registry.register(cls, enabled=True)
        return cls
    elif isinstance(enabled_or_name, str):
        # Called as @register_tool("CustomName")
        actual_name = enabled_or_name
        actual_enabled = enabled  # Use default or passed enabled value
    elif isinstance(enabled_or_name, bool) and enabled_or_name is False:
        # Called as @register_tool(False) - explicit boolean
        actual_enabled = enabled_or_name
        actual_name = name or ""
    else:
        # Default case - @register_tool() or @register_tool(enabled=X, name=Y)
        # For @register_tool(enabled=False), enabled_or_name is True (default)
        # but the 'enabled' keyword parameter is False
        actual_enabled = enabled
        actual_name = name or ""

    def decorator(cls: Type[BaseTool]) -> Type[BaseTool]:
        # Set custom name if provided
        if actual_name:
            original_init = cls.__init__

            def new_init(self, *args, **kwargs):
                original_init(self, name=actual_name, *args, **kwargs)

            # Use setattr to avoid method assignment type error
            setattr(cls, "__init__", new_init)

        # Register the tool
        registry = ToolRegistry()
        registry.register(cls, enabled=actual_enabled)

        return cls

    return decorator
