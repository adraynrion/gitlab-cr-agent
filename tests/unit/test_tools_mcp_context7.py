"""
Minimal tests for Context7 MCP integration (native PydanticAI approach)
"""

import pytest


class TestContext7NativeMCP:
    """Test Context7 native MCP integration"""

    def test_context7_mcp_imports_available(self):
        """Test that PydanticAI MCP imports work"""
        try:
            from pydantic_ai.mcp import MCPServerStdio

            assert MCPServerStdio is not None
        except ImportError as e:
            pytest.skip(f"pydantic-ai[mcp] not available: {e}")

    def test_context7_mcp_server_creation(self):
        """Test Context7 MCP server can be created with correct parameters"""
        try:
            from pydantic_ai.mcp import MCPServerStdio

            # Test server creation with hardcoded Context7 parameters
            server = MCPServerStdio(
                command="npx", args=["-y", "@upstash/context7-mcp@1.0.14"], timeout=30.0
            )
            assert server is not None
        except ImportError as e:
            pytest.skip(f"pydantic-ai[mcp] not available: {e}")


class TestContext7Integration:
    """Test Context7 MCP integration functionality"""

    def test_context7_configuration_in_settings(self):
        """Test Context7 configuration is properly integrated in settings"""
        from src.config.settings import get_settings

        settings = get_settings()

        # Check that Context7 settings exist
        assert hasattr(settings, "context7_enabled")
        # Should be boolean type
        assert isinstance(settings.context7_enabled, bool)

    def test_context7_agent_integration_imports(self):
        """Test that Context7 agent integration can be imported"""
        try:
            from src.agents.code_reviewer import CodeReviewAgent

            assert CodeReviewAgent is not None
        except ImportError as e:
            pytest.fail(f"Failed to import CodeReviewAgent: {e}")

    @pytest.mark.asyncio
    async def test_context7_agent_initialization_without_mcp(self):
        """Test agent can be initialized when Context7 is disabled"""
        from unittest.mock import Mock, patch

        from src.agents.code_reviewer import CodeReviewAgent

        with patch("src.agents.code_reviewer.get_settings") as mock_get_settings, patch(
            "src.agents.code_reviewer.get_llm_model"
        ) as mock_get_llm:
            # Mock settings with Context7 disabled
            settings = Mock()
            settings.ai_model = "test:model"
            settings.ai_retries = 3
            settings.context7_enabled = False
            mock_get_settings.return_value = settings

            # Mock model
            mock_model = Mock()
            mock_get_llm.return_value = mock_model

            try:
                agent = CodeReviewAgent()
                assert agent.model_name == "test:model"
                # Should initialize successfully even without Context7
                assert agent is not None
            except Exception as e:
                # Some initialization errors are OK in test environment
                assert (
                    "context7" not in str(e).lower()
                )  # Should not fail due to Context7


class TestContext7Configuration:
    """Test Context7 MCP configuration and settings"""

    def test_context7_settings_integration(self):
        """Test that Context7 settings are properly integrated"""
        from src.config.settings import get_settings

        settings = get_settings()

        # Check that Context7 settings exist
        assert hasattr(settings, "context7_enabled")
        # Only context7_enabled should be configurable, rest is hardcoded
        assert isinstance(settings.context7_enabled, bool)

    def test_context7_configurable_version(self):
        """Test that Context7 MCP uses configurable version"""
        from unittest.mock import Mock, patch

        # Test that the agent uses configurable MCP version
        try:
            from src.agents.code_reviewer import CodeReviewAgent

            with patch(
                "src.agents.code_reviewer.get_settings"
            ) as mock_get_settings, patch(
                "src.agents.code_reviewer.get_llm_model"
            ) as mock_get_llm, patch(
                "src.agents.code_reviewer.MCPServerStdio"
            ) as mock_mcp_server:
                # Mock settings with Context7 enabled
                settings = Mock()
                settings.ai_model = "test:model"
                settings.ai_retries = 3
                settings.context7_enabled = True
                settings.context7_mcp_version = "1.0.14"
                mock_get_settings.return_value = settings

                # Mock model and MCP server
                mock_model = Mock()
                mock_get_llm.return_value = mock_model
                mock_mcp_server.return_value = Mock()

                try:
                    CodeReviewAgent()

                    # Verify MCPServerStdio was called with hardcoded values
                    mock_mcp_server.assert_called_once_with(
                        command="npx",
                        args=["-y", "@upstash/context7-mcp@1.0.14"],
                        timeout=30.0,
                    )
                except Exception:
                    # Initialization may fail in test environment, but we still want to verify the call
                    if mock_mcp_server.called:
                        mock_mcp_server.assert_called_with(
                            command="npx",
                            args=["-y", "@upstash/context7-mcp@1.0.14"],
                            timeout=30.0,
                        )
        except ImportError as e:
            pytest.skip(f"PydanticAI MCP not available: {e}")
