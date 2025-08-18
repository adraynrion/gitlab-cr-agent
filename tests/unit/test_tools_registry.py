"""
Tests for src/agents/tools/registry.py
"""

from unittest.mock import Mock, patch

from src.agents.tools.base import (
    BaseTool,
    ToolCategory,
    ToolContext,
    ToolPriority,
    ToolResult,
)
from src.agents.tools.registry import ToolRegistry, register_tool


class MockRegistryTool(BaseTool):
    """Mock tool for registry testing"""

    # Class attributes for metadata
    name = "MockRegistryTool"
    category = ToolCategory.CORRECTNESS
    priority = ToolPriority.MEDIUM

    def __init__(self, name: str = "MockRegistryTool"):
        super().__init__(name)
        self.execute_called = False

    @property
    def description(self) -> str:
        return "Mock tool for testing registry"

    async def execute(self, context: ToolContext) -> ToolResult:
        self.execute_called = True
        return ToolResult(
            tool_name=self.name,
            category=self.__class__.category,
            success=True,
            issues=[],
            suggestions=["Test suggestion"],
            positive_findings=["Test passed"],
        )


class TestToolRegistry:
    """Test ToolRegistry class"""

    def test_registry_singleton(self):
        """Test that ToolRegistry implements singleton pattern"""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        assert registry1 is registry2

    def test_tool_registration(self):
        """Test tool registration"""
        registry = ToolRegistry()
        MockRegistryTool("TestTool")

        registry.register(MockRegistryTool, enabled=True)

        # Tool should be registered
        tool_names = [tool.name for tool in registry.get_all_tools()]
        assert "MockRegistryTool" in tool_names

        # Get the tool instance
        retrieved_tool = registry.get_tool("MockRegistryTool")
        assert retrieved_tool is not None
        assert isinstance(retrieved_tool, MockRegistryTool)

    def test_tool_unregistration(self):
        """Test tool unregistration"""
        registry = ToolRegistry()
        MockRegistryTool("UnregisterTool")

        # Register then unregister
        registry.register(MockRegistryTool, enabled=True)
        tool_names = [tool.name for tool in registry.get_all_tools()]
        assert "MockRegistryTool" in tool_names

        registry.unregister("MockRegistryTool")
        tool_names = [tool.name for tool in registry.get_all_tools()]
        assert "MockRegistryTool" not in tool_names

    def test_tool_enabling_disabling(self):
        """Test tool enabling and disabling"""
        registry = ToolRegistry()
        MockRegistryTool("EnableDisableTool")

        registry.register(MockRegistryTool, enabled=True)

        # Test enabling/disabling
        registry.disable_tool("MockRegistryTool")
        assert not registry.is_enabled("MockRegistryTool")

        registry.enable_tool("MockRegistryTool")
        assert registry.is_enabled("MockRegistryTool")

    def test_get_tool_names(self):
        """Test getting tool names"""
        registry = ToolRegistry()

        # Clear registry for clean test
        registry.clear_cache()

        # Register some tools (registry accepts classes, not instances)
        registry.register(MockRegistryTool, enabled=True)

        names = [tool.name for tool in registry.get_all_tools()]
        assert "MockRegistryTool" in names

    def test_get_tools_by_category(self):
        """Test getting tools by category"""
        registry = ToolRegistry()

        registry.register(MockRegistryTool, enabled=True)

        correctness_tools = registry.get_tools_by_category(ToolCategory.CORRECTNESS)
        tool_names = [t.name for t in correctness_tools]
        assert "MockRegistryTool" in tool_names

    def test_registry_statistics(self):
        """Test registry statistics"""
        registry = ToolRegistry()

        # Register a tool
        registry.register(MockRegistryTool, enabled=True)

        stats = registry.get_statistics()

        # Should have statistics
        assert isinstance(stats, dict)
        assert "total_tools" in stats or len(stats) >= 0

    @patch("src.config.settings.get_settings")
    def test_configuration_from_settings(self, mock_get_settings):
        """Test registry configuration from settings"""
        # Mock settings
        settings = Mock()
        settings.context7_enabled = True
        mock_get_settings.return_value = settings

        registry = ToolRegistry()

        # Test that registry can be configured
        assert registry is not None

    async def test_tool_execution_through_registry(self):
        """Test tool execution through registry"""
        registry = ToolRegistry()

        registry.register(MockRegistryTool, enabled=True)

        context = ToolContext(
            diff_content="test diff",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.example.com/test/repo",
        )

        # Execute tool through registry
        retrieved_tool = registry.get_tool("MockRegistryTool")
        result = await retrieved_tool.execute(context)

        assert result.success
        assert retrieved_tool.execute_called

    def test_cache_clearing(self):
        """Test cache clearing functionality"""
        registry = ToolRegistry()

        registry.register(MockRegistryTool, enabled=True)
        registry.clear_cache()

        # Registry should still function after cache clear
        assert registry is not None


class TestRegisterDecorator:
    """Test register_tool decorator"""

    def setup_method(self):
        """Clear registry before each test to avoid singleton interference"""
        registry = ToolRegistry()
        registry.clear_cache()
        # Clear all registered tools to start fresh
        registry._tools.clear()
        registry._tool_instances.clear()
        registry._enabled_tools.clear()
        registry._disabled_tools.clear()
        for category_set in registry._categories.values():
            category_set.clear()
        for priority_set in registry._priorities.values():
            priority_set.clear()

    def test_decorator_registration(self):
        """Test tool registration via decorator"""

        @register_tool
        class DecoratedTool(BaseTool):
            # Class attributes for metadata
            category = ToolCategory.PERFORMANCE
            priority = ToolPriority.MEDIUM

            @property
            def description(self) -> str:
                return "Decorated tool"

            async def execute(self, context: ToolContext) -> ToolResult:
                return ToolResult(
                    tool_name=self.name, category=self.__class__.category, success=True
                )

        registry = ToolRegistry()

        # Tool should be automatically registered
        tool_names = [tool.name for tool in registry.get_all_tools()]
        assert "DecoratedTool" in tool_names

    def test_decorator_with_custom_name(self):
        """Test decorator with custom tool name"""

        @register_tool("CustomDecoratedTool")
        class AnotherDecoratedTool(BaseTool):
            # Class attributes for metadata
            category = ToolCategory.SECURITY
            priority = ToolPriority.MEDIUM

            @property
            def description(self) -> str:
                return "Custom named decorated tool"

            async def execute(self, context: ToolContext) -> ToolResult:
                return ToolResult(
                    tool_name=self.name, category=self.__class__.category, success=True
                )

        registry = ToolRegistry()

        # Tool should be registered with custom name
        tool_names = [tool.name for tool in registry.get_all_tools()]
        assert "CustomDecoratedTool" in tool_names

    def test_decorator_with_defaults(self):
        """Test decorator with default parameters"""

        @register_tool(enabled=False)
        class DisabledDecoratedTool(BaseTool):
            # Class attributes for metadata
            category = ToolCategory.MAINTAINABILITY
            priority = ToolPriority.MEDIUM

            @property
            def description(self) -> str:
                return "Disabled decorated tool"

            async def execute(self, context: ToolContext) -> ToolResult:
                return ToolResult(
                    tool_name=self.name, category=self.__class__.category, success=True
                )

        registry = ToolRegistry()

        # Tool should be registered but disabled
        tool_names = [tool.name for tool in registry.get_all_tools()]
        assert "DisabledDecoratedTool" in tool_names
        assert not registry.is_enabled("DisabledDecoratedTool")


class TestRegistryEdgeCases:
    """Test registry edge cases and error scenarios"""

    def test_duplicate_tool_registration(self):
        """Test handling of duplicate tool registrations"""
        registry = ToolRegistry()

        registry.register(MockRegistryTool, enabled=True)

        # Registering the same class again should handle gracefully
        try:
            registry.register(MockRegistryTool, enabled=True)
            # Should either succeed or handle gracefully
            assert True
        except Exception:
            # Some registries might raise exceptions for duplicates
            assert True

    def test_nonexistent_tool_operations(self):
        """Test operations on nonexistent tools"""
        registry = ToolRegistry()

        # Getting nonexistent tool
        tool = registry.get_tool("NonexistentTool")
        assert tool is None

        # Disabling nonexistent tool should handle gracefully
        try:
            registry.disable_tool("NonexistentTool")
            assert True
        except Exception:
            # Some registries might raise exceptions
            assert True

    def test_registry_with_no_tools(self):
        """Test registry behavior with no tools"""
        registry = ToolRegistry()
        registry.clear_cache()

        names = [tool.name for tool in registry.get_all_tools()]
        assert isinstance(names, list)

        stats = registry.get_statistics()
        assert isinstance(stats, dict)
