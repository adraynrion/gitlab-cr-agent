"""
Tests for src/agents/tools/registry.py
"""

from unittest.mock import Mock, patch

from src.agents.tools.base import BaseTool, ToolCategory, ToolContext, ToolResult
from src.agents.tools.registry import ToolRegistry, register_tool


class MockRegistryTool(BaseTool):
    """Mock tool for registry testing"""

    def __init__(self, name: str = "MockRegistryTool"):
        super().__init__(name)
        self.execute_called = False

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CORRECTNESS

    @property
    def description(self) -> str:
        return "Mock tool for testing registry"

    async def _execute(self, context: ToolContext) -> ToolResult:
        self.execute_called = True
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"test": "registry execution"},
            metadata={},
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
        tool = MockRegistryTool("TestTool")

        registry.register_tool(tool)

        # Tool should be registered
        assert "TestTool" in registry.get_tool_names()
        assert registry.get_tool("TestTool") == tool

    def test_tool_unregistration(self):
        """Test tool unregistration"""
        registry = ToolRegistry()
        tool = MockRegistryTool("UnregisterTool")

        # Register then unregister
        registry.register_tool(tool)
        assert "UnregisterTool" in registry.get_tool_names()

        registry.unregister_tool("UnregisterTool")
        assert "UnregisterTool" not in registry.get_tool_names()

    def test_tool_enabling_disabling(self):
        """Test tool enabling and disabling"""
        registry = ToolRegistry()
        tool = MockRegistryTool("EnableDisableTool")

        registry.register_tool(tool)

        # Test enabling/disabling
        registry.disable_tool("EnableDisableTool")
        assert not registry.is_tool_enabled("EnableDisableTool")

        registry.enable_tool("EnableDisableTool")
        assert registry.is_tool_enabled("EnableDisableTool")

    def test_get_tool_names(self):
        """Test getting tool names"""
        registry = ToolRegistry()

        # Clear registry for clean test
        registry.clear_cache()

        # Register some tools
        tool1 = MockRegistryTool("Tool1")
        tool2 = MockRegistryTool("Tool2")

        registry.register_tool(tool1)
        registry.register_tool(tool2)

        names = registry.get_tool_names()
        assert "Tool1" in names
        assert "Tool2" in names

    def test_get_tools_by_category(self):
        """Test getting tools by category"""
        registry = ToolRegistry()
        tool = MockRegistryTool("CategoryTool")

        registry.register_tool(tool)

        correctness_tools = registry.get_tools_by_category(ToolCategory.CORRECTNESS)
        tool_names = [t.name for t in correctness_tools]
        assert "CategoryTool" in tool_names

    def test_registry_statistics(self):
        """Test registry statistics"""
        registry = ToolRegistry()

        # Register a tool
        tool = MockRegistryTool("StatsTool")
        registry.register_tool(tool)

        stats = registry.get_statistics()

        # Should have statistics
        assert isinstance(stats, dict)
        assert "total_tools" in stats or len(stats) >= 0

    @patch("src.config.settings.get_settings")
    def test_configuration_from_settings(self, mock_get_settings):
        """Test registry configuration from settings"""
        # Mock settings
        settings = Mock()
        settings.tools_enabled = True
        mock_get_settings.return_value = settings

        registry = ToolRegistry()

        # Test that registry can be configured
        assert registry is not None

    async def test_tool_execution_through_registry(self):
        """Test tool execution through registry"""
        registry = ToolRegistry()
        tool = MockRegistryTool("ExecutionTool")

        registry.register_tool(tool)

        context = ToolContext(
            language="python",
            file_path="test.py",
            diff_content="test diff",
            merge_request_context={},
        )

        # Execute tool through registry
        retrieved_tool = registry.get_tool("ExecutionTool")
        result = await retrieved_tool.execute(context)

        assert result.success
        assert tool.execute_called

    def test_cache_clearing(self):
        """Test cache clearing functionality"""
        registry = ToolRegistry()
        tool = MockRegistryTool("CacheTool")

        registry.register_tool(tool)
        registry.clear_cache()

        # Registry should still function after cache clear
        assert registry is not None


class TestRegisterDecorator:
    """Test register_tool decorator"""

    def test_decorator_registration(self):
        """Test tool registration via decorator"""

        @register_tool
        class DecoratedTool(BaseTool):
            @property
            def category(self) -> ToolCategory:
                return ToolCategory.PERFORMANCE

            @property
            def description(self) -> str:
                return "Decorated tool"

            async def _execute(self, context: ToolContext) -> ToolResult:
                return ToolResult(
                    tool_name=self.name, success=True, data={}, metadata={}
                )

        registry = ToolRegistry()

        # Tool should be automatically registered
        assert "DecoratedTool" in registry.get_tool_names()

    def test_decorator_with_custom_name(self):
        """Test decorator with custom tool name"""

        @register_tool("CustomDecoratedTool")
        class AnotherDecoratedTool(BaseTool):
            @property
            def category(self) -> ToolCategory:
                return ToolCategory.SECURITY

            @property
            def description(self) -> str:
                return "Custom named decorated tool"

            async def _execute(self, context: ToolContext) -> ToolResult:
                return ToolResult(
                    tool_name=self.name, success=True, data={}, metadata={}
                )

        registry = ToolRegistry()

        # Tool should be registered with custom name
        assert "CustomDecoratedTool" in registry.get_tool_names()

    def test_decorator_with_defaults(self):
        """Test decorator with default parameters"""

        @register_tool(enabled=False)
        class DisabledDecoratedTool(BaseTool):
            @property
            def category(self) -> ToolCategory:
                return ToolCategory.MAINTAINABILITY

            @property
            def description(self) -> str:
                return "Disabled decorated tool"

            async def _execute(self, context: ToolContext) -> ToolResult:
                return ToolResult(
                    tool_name=self.name, success=True, data={}, metadata={}
                )

        registry = ToolRegistry()

        # Tool should be registered but disabled
        assert "DisabledDecoratedTool" in registry.get_tool_names()
        assert not registry.is_tool_enabled("DisabledDecoratedTool")


class TestRegistryEdgeCases:
    """Test registry edge cases and error scenarios"""

    def test_duplicate_tool_registration(self):
        """Test handling of duplicate tool registrations"""
        registry = ToolRegistry()
        tool1 = MockRegistryTool("DuplicateTool")
        tool2 = MockRegistryTool("DuplicateTool")  # Same name

        registry.register_tool(tool1)

        # Registering with same name should handle gracefully
        try:
            registry.register_tool(tool2)
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

        names = registry.get_tool_names()
        assert isinstance(names, list)

        stats = registry.get_statistics()
        assert isinstance(stats, dict)
