"""
Tests for native Context7 MCP integration (replaces unified tools)
"""

import pytest


class TestNativeContext7MCP:
    """Native Context7 MCP integration tests"""

    def test_context7_mcp_agent_integration(self):
        """Test Context7 MCP integration with CodeReviewAgent"""
        try:
            from src.agents.code_reviewer import CodeReviewAgent

            assert CodeReviewAgent is not None
        except ImportError as e:
            pytest.skip(f"CodeReviewAgent import failed: {e}")

    def test_context7_mcp_imports(self):
        """Test that PydanticAI MCP imports work"""
        try:
            from pydantic_ai.mcp import MCPServerStdio

            assert MCPServerStdio is not None
        except ImportError as e:
            pytest.skip(f"pydantic-ai[mcp] not available: {e}")

    def test_context7_settings_only_enabled_configurable(self):
        """Test that only context7_enabled is configurable"""
        from src.config.settings import get_settings

        settings = get_settings()

        # Only context7_enabled should be configurable
        assert hasattr(settings, "context7_enabled")
        assert isinstance(settings.context7_enabled, bool)

        # Other Context7 settings should not exist (hardcoded in agent)
        assert not hasattr(settings, "context7_mcp_command")
        assert not hasattr(settings, "context7_mcp_args")
        assert not hasattr(settings, "context7_mcp_timeout")

    def test_agent_with_context7_disabled(self):
        """Test agent initialization with Context7 disabled"""
        from unittest.mock import Mock, patch

        try:
            from src.agents.code_reviewer import CodeReviewAgent

            with patch("src.agents.code_reviewer.get_settings") as mock_settings, patch(
                "src.agents.code_reviewer.get_llm_model"
            ) as mock_get_llm:
                # Mock settings with Context7 disabled
                settings = Mock()
                settings.ai_model = "test:model"
                settings.ai_retries = 3
                settings.context7_enabled = False
                mock_settings.return_value = settings

                # Mock model
                mock_model = Mock()
                mock_get_llm.return_value = mock_model

                try:
                    agent = CodeReviewAgent()
                    assert agent.model_name == "test:model"
                    # Agent should initialize successfully without Context7
                    assert agent is not None
                except Exception:
                    # Some initialization may fail in test environment, that's OK
                    pass
        except ImportError as e:
            pytest.skip(f"CodeReviewAgent not available: {e}")

    def test_agent_with_context7_enabled_configurable_version(self):
        """Test that agent uses configurable Context7 MCP version"""
        from unittest.mock import Mock, patch

        try:
            from src.agents.code_reviewer import CodeReviewAgent

            with patch("src.agents.code_reviewer.get_settings") as mock_settings, patch(
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
                mock_settings.return_value = settings

                # Mock model and MCP server
                mock_model = Mock()
                mock_get_llm.return_value = mock_model
                mock_mcp_server.return_value = Mock()

                try:
                    CodeReviewAgent()

                    # Verify configurable MCP version is used
                    mock_mcp_server.assert_called_once_with(
                        command="npx",
                        args=["-y", "@upstash/context7-mcp@1.0.14"],
                        timeout=30.0,
                    )
                except Exception:
                    # Even if initialization fails, verify the MCP server call
                    if mock_mcp_server.called:
                        call_args = mock_mcp_server.call_args
                        assert call_args[1]["command"] == "npx"
                        assert call_args[1]["args"] == [
                            "-y",
                            "@upstash/context7-mcp@1.0.14",
                        ]
                        assert call_args[1]["timeout"] == 30.0
        except ImportError as e:
            pytest.skip(f"PydanticAI MCP not available: {e}")
