"""
Unit tests for the tool system framework
"""


import pytest

from src.agents.tools.base import (
    BaseTool,
    ToolCategory,
    ToolContext,
    ToolPriority,
    ToolResult,
)
from src.agents.tools.registry import ToolRegistry, register_tool


class MockTool(BaseTool):
    """Mock tool for testing"""

    def __init__(self, name: str = "MockTool", should_fail: bool = False):
        super().__init__(name)
        self.should_fail = should_fail
        self.execute_called = False
        self.initialize_called = False

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CORRECTNESS

    @property
    def priority(self) -> ToolPriority:
        return ToolPriority.MEDIUM

    async def initialize(self, context: ToolContext) -> None:
        await super().initialize(context)
        self.initialize_called = True

    async def execute(self, context: ToolContext) -> ToolResult:
        self.execute_called = True

        if self.should_fail:
            raise Exception("Mock tool failure")

        return ToolResult(
            tool_name=self.name,
            category=self.category,
            success=True,
            issues=[
                {
                    "severity": "medium",
                    "category": "correctness",
                    "description": "Mock issue",
                    "file_path": "test.py",
                    "line_number": 1,
                }
            ],
            suggestions=["Mock suggestion"],
            positive_findings=["Mock positive finding"],
            evidence={"mock_evidence": "evidence_value"},
            references=["https://example.com/docs"],
            metrics={"mock_metric": 42},
            confidence_score=0.8,
        )


class TestToolContext:
    """Test ToolContext class"""

    def test_tool_context_creation(self):
        """Test creating a tool context"""
        context = ToolContext(
            diff_content="+ print('hello')",
            file_changes=[{"path": "test.py", "action": "added"}],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
            merge_request_iid=123,
        )

        assert context.diff_content == "+ print('hello')"
        assert context.source_branch == "feature"
        assert context.target_branch == "main"
        assert context.repository_url == "https://gitlab.com/test/repo"
        assert context.merge_request_iid == 123
        assert context.cache == {}
        assert context.metadata == {}


class TestToolResult:
    """Test ToolResult model"""

    def test_tool_result_creation(self):
        """Test creating a tool result"""
        result = ToolResult(
            tool_name="TestTool",
            category=ToolCategory.SECURITY,
            success=True,
            issues=[{"description": "test issue"}],
            suggestions=["test suggestion"],
            confidence_score=0.9,
        )

        assert result.tool_name == "TestTool"
        assert result.category == ToolCategory.SECURITY
        assert result.success is True
        assert len(result.issues) == 1
        assert result.issues[0]["description"] == "test issue"
        assert result.suggestions == ["test suggestion"]
        assert result.confidence_score == 0.9
        assert result.cached is False
        assert result.partial_result is False

    def test_tool_result_validation(self):
        """Test tool result validation"""
        # Test confidence score validation
        with pytest.raises(ValueError):
            ToolResult(
                tool_name="TestTool",
                category=ToolCategory.SECURITY,
                success=True,
                confidence_score=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValueError):
            ToolResult(
                tool_name="TestTool",
                category=ToolCategory.SECURITY,
                success=True,
                confidence_score=-0.1,  # Invalid: < 0.0
            )


class TestBaseTool:
    """Test BaseTool abstract class"""

    @pytest.fixture
    def tool_context(self):
        """Create a test tool context"""
        return ToolContext(
            diff_content="+ print('hello')",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

    @pytest.mark.asyncio
    async def test_tool_execution(self, tool_context):
        """Test basic tool execution"""
        tool = MockTool("TestTool")

        result = await tool.run(tool_context)

        assert tool.initialize_called is True
        assert tool.execute_called is True
        assert result.success is True
        assert result.tool_name == "TestTool"
        assert result.category == ToolCategory.CORRECTNESS
        assert len(result.issues) == 1
        assert result.issues[0]["description"] == "Mock issue"
        assert result.suggestions == ["Mock suggestion"]
        assert result.positive_findings == ["Mock positive finding"]
        assert result.evidence == {"mock_evidence": "evidence_value"}
        assert result.references == ["https://example.com/docs"]
        assert result.confidence_score == 0.8
        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, tool_context):
        """Test tool error handling"""
        tool = MockTool("FailingTool", should_fail=True)

        result = await tool.run(tool_context)

        assert result.success is False
        assert result.tool_name == "FailingTool"
        assert result.error_message == "Mock tool failure"
        assert result.partial_result is True
        assert result.execution_time_ms is not None

    @pytest.mark.asyncio
    async def test_tool_caching(self, tool_context):
        """Test tool result caching"""
        tool = MockTool("CacheableTool")

        # First execution - not cached
        result1 = await tool.run(tool_context)
        assert result1.cached is False

        # Second execution - should be cached
        result2 = await tool.run(tool_context)
        assert result2.cached is True
        assert result2.tool_name == result1.tool_name

    def test_tool_properties(self):
        """Test tool property defaults"""
        tool = MockTool()

        assert tool.requires_network is False
        assert tool.cacheable is True
        assert tool.timeout_seconds == 30
        assert tool.category == ToolCategory.CORRECTNESS
        assert tool.priority == ToolPriority.MEDIUM
        assert (
            "Mock tool for testing" in tool.description
            or tool.description == "No description available"
        )

    def test_cache_key_generation(self, tool_context):
        """Test cache key generation"""
        tool = MockTool()

        cache_key = tool._get_cache_key(tool_context)

        assert isinstance(cache_key, str)
        assert "MockTool" in cache_key
        assert "feature" in cache_key  # source branch
        assert "main" in cache_key  # target branch
        # Should contain a hash of the diff content
        assert len(cache_key.split(":")) >= 4


class TestToolRegistry:
    """Test ToolRegistry class"""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing"""
        # Create a new instance bypassing singleton for testing
        registry = object.__new__(ToolRegistry)
        registry._initialized = False
        registry.__init__()
        return registry

    def test_registry_singleton(self):
        """Test registry singleton behavior"""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()

        assert registry1 is registry2

    def test_tool_registration(self, registry):
        """Test tool registration"""
        tool_class = MockTool

        registry.register(tool_class, enabled=True)

        # Check tool is registered
        tools = registry.get_all_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], MockTool)

        # Check tool is enabled
        assert registry.is_enabled("MockTool")

        # Check category indexing
        correctness_tools = registry.get_tools_by_category(ToolCategory.CORRECTNESS)
        assert len(correctness_tools) == 1

        # Check priority indexing
        medium_tools = registry.get_tools_by_priority(ToolPriority.MEDIUM)
        assert len(medium_tools) == 1

    def test_tool_enabling_disabling(self, registry):
        """Test tool enabling and disabling"""
        registry.register(MockTool, enabled=True)

        assert registry.is_enabled("MockTool")

        registry.disable_tool("MockTool")
        assert not registry.is_enabled("MockTool")

        registry.enable_tool("MockTool")
        assert registry.is_enabled("MockTool")

    def test_tool_unregistration(self, registry):
        """Test tool unregistration"""
        registry.register(MockTool, enabled=True)
        assert len(registry.get_all_tools()) == 1

        registry.unregister("MockTool")
        assert len(registry.get_all_tools()) == 0
        assert not registry.is_enabled("MockTool")

    def test_registry_statistics(self, registry):
        """Test registry statistics"""
        # Clear any existing registrations from other tests
        all_tools = list(registry._tools.keys())
        for tool_name in all_tools:
            registry.unregister(tool_name)

        # Register our test tool
        registry.register(MockTool, enabled=True)

        stats = registry.get_statistics()

        assert stats["total_tools"] == 1
        assert stats["enabled_tools"] == 1
        assert stats["disabled_tools"] == 0
        assert stats["tools_by_category"]["correctness"] == 1
        assert stats["tools_by_priority"]["3"] == 1  # MEDIUM = 3

    def test_configuration_from_settings(self, registry):
        """Test registry configuration from settings (simplified Context7-only)"""
        registry.register(
            MockTool, enabled=True
        )  # In Context7-only, tools are enabled by default

        settings = {
            "context7": {
                "enabled": True,
                "api_url": "http://context7:8080",
                "max_tokens": 2000,
            }
        }

        registry.configure_from_settings(settings)

        # In simplified Context7 setup, registered tools are available
        assert len(registry._tools) > 0

    @pytest.mark.asyncio
    async def test_tool_execution_parallel(self, registry):
        """Test parallel tool execution"""

        # Register multiple tools
        class Tool1(MockTool):
            def __init__(self):
                super().__init__("Tool1")

        class Tool2(MockTool):
            def __init__(self):
                super().__init__("Tool2")

        registry.register(Tool1, enabled=True)
        registry.register(Tool2, enabled=True)

        context = ToolContext(
            diff_content="+ print('test')",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        results = await registry.execute_tools(context, parallel=True)

        assert len(results) == 2
        tool_names = [r.tool_name for r in results]
        assert "Tool1" in tool_names
        assert "Tool2" in tool_names

        for result in results:
            assert result.success is True

    @pytest.mark.asyncio
    async def test_tool_execution_sequential(self, registry):
        """Test sequential tool execution"""
        registry.register(MockTool, enabled=True)

        context = ToolContext(
            diff_content="+ print('test')",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        results = await registry.execute_tools(context, parallel=False)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].tool_name == "MockTool"

    @pytest.mark.asyncio
    async def test_tool_execution_with_failures(self, registry):
        """Test tool execution with failures"""

        class FailingTool(MockTool):
            def __init__(self):
                super().__init__("FailingTool", should_fail=True)

        registry.register(MockTool, enabled=True)
        registry.register(FailingTool, enabled=True)

        context = ToolContext(
            diff_content="+ print('test')",
            file_changes=[],
            source_branch="feature",
            target_branch="main",
            repository_url="https://gitlab.com/test/repo",
        )

        results = await registry.execute_tools(context, parallel=True)

        # Should get results from successful tool, failed tool should be filtered out in parallel mode
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        assert len(successful_results) >= 1
        assert len(failed_results) >= 1

    def test_cache_clearing(self, registry):
        """Test cache clearing"""
        registry.register(MockTool, enabled=True)

        # Get a tool instance (this caches it)
        tool = registry.get_tool("MockTool")
        assert tool is not None

        stats_before = registry.get_statistics()
        assert stats_before["cached_instances"] == 1

        registry.clear_cache()

        stats_after = registry.get_statistics()
        assert stats_after["cached_instances"] == 0


class TestRegisterDecorator:
    """Test the register_tool decorator"""

    def test_decorator_registration(self):
        """Test automatic registration with decorator"""
        # Get the singleton registry
        registry = ToolRegistry()

        # Clear any existing registrations
        registry.clear_cache()

        @register_tool(enabled=True, name="DecoratedTool")
        class DecoratedTool(BaseTool):
            @property
            def category(self) -> ToolCategory:
                return ToolCategory.SECURITY

            @property
            def priority(self) -> ToolPriority:
                return ToolPriority.HIGH

            async def execute(self, context: ToolContext) -> ToolResult:
                return ToolResult(
                    tool_name=self.name, category=self.category, success=True
                )

        # Tool should be automatically registered
        tools = registry.get_all_tools()
        decorated_tools = [t for t in tools if t.name == "DecoratedTool"]
        assert len(decorated_tools) == 1
        assert registry.is_enabled("DecoratedTool")

        tool = decorated_tools[0]
        assert tool.category == ToolCategory.SECURITY
        assert tool.priority == ToolPriority.HIGH

    def test_decorator_with_defaults(self):
        """Test decorator with default settings"""
        # Get the singleton registry
        registry = ToolRegistry()

        @register_tool()
        class DefaultTool(BaseTool):
            @property
            def category(self) -> ToolCategory:
                return ToolCategory.PERFORMANCE

            @property
            def priority(self) -> ToolPriority:
                return ToolPriority.LOW

            async def execute(self, context: ToolContext) -> ToolResult:
                return ToolResult(
                    tool_name=self.name, category=self.category, success=True
                )

        # Should be registered with default name
        tool = registry.get_tool("DefaultTool")
        assert tool is not None
        assert registry.is_enabled("DefaultTool")  # Default enabled=True
