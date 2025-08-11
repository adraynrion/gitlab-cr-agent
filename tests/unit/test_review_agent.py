"""
Unit tests for the code review agent
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.code_reviewer import CodeReviewAgent
from src.models.review_models import ReviewContext, ReviewResult

@pytest.fixture
def mock_agent():
    """Create a mock review agent"""
    with patch('src.agents.code_reviewer.get_llm_model') as mock_model:
        mock_model.return_value = MagicMock()
        agent = CodeReviewAgent(model_name="openai:gpt-4o")
        return agent

@pytest.mark.asyncio
async def test_review_merge_request(mock_agent):
    """Test merge request review functionality"""
    
    # Prepare test data
    diff_content = """
    diff --git a/src/example.py b/src/example.py
    index 123..456 100644
    --- a/src/example.py
    +++ b/src/example.py
    @@ -1,5 +1,7 @@
     def calculate(a, b):
    -    return a + b
    +    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
    +        raise ValueError("Inputs must be numbers")
    +    result = a + b
    +    return result
    """
    
    context = ReviewContext(
        repository_url="https://gitlab.example.com/test/repo",
        merge_request_iid=123,
        source_branch="feature/validation",
        target_branch="main",
        trigger_tag="ai-review",
        file_changes=[]
    )
    
    # Mock the agent response
    mock_result = ReviewResult(
        overall_assessment="approve_with_changes",
        risk_level="low",
        summary="Added input validation to calculate function",
        issues=[],
        positive_feedback=["Good input validation added"],
        metrics={"files_reviewed": 1}
    )
    
    mock_agent.agent.run = AsyncMock(return_value=MagicMock(data=mock_result))
    
    # Execute review
    result = await mock_agent.review_merge_request(diff_content, context)
    
    # Assertions
    assert result.overall_assessment == "approve_with_changes"
    assert result.risk_level == "low"
    assert len(result.positive_feedback) > 0

@pytest.mark.asyncio
async def test_security_analysis_tool():
    """Test security analysis tool functionality"""
    agent = CodeReviewAgent()
    
    # Create mock context
    ctx = MagicMock()
    
    # Test dangerous code detection
    dangerous_code = "result = eval(user_input)"
    
    # Get the security analysis tool
    security_tool = None
    for tool in agent.agent._tools:
        if tool.name == "analyze_security_patterns":
            security_tool = tool
            break
    
    assert security_tool is not None
    
    # Test the tool
    result = await security_tool.func(ctx, dangerous_code)
    assert "eval" in result.lower()