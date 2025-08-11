"""
PydanticAI-based code review agent with multi-LLM support
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import logging

from src.models.review_models import (
    CodeIssue, ReviewResult, ReviewContext
)
from src.agents.providers import get_llm_model
from src.config.settings import settings

logger = logging.getLogger(__name__)

# System prompt for code review
CODE_REVIEW_SYSTEM_PROMPT = """
You are an expert software engineer conducting thorough code reviews.
Your role is to analyze code changes and provide constructive, actionable feedback.

REVIEW FRAMEWORK:
1. **Correctness**: Identify logic errors, edge cases, and algorithm issues
2. **Security**: Detect vulnerabilities, input validation issues, authentication flaws
3. **Performance**: Find bottlenecks, inefficient algorithms, resource usage problems
4. **Maintainability**: Assess code clarity, structure, documentation quality
5. **Best Practices**: Check language conventions, design patterns, testing

OUTPUT REQUIREMENTS:
- Provide specific file paths and line numbers for each issue
- Categorize issues by severity (critical, high, medium, low)
- Include concrete suggestions for improvements
- Show code examples when helpful
- Balance criticism with positive observations
- Consider the broader codebase context

TONE:
- Be constructive and educational
- Focus on substantial issues over style preferences
- Acknowledge good practices when observed
- Provide clear rationale for suggestions
"""

@dataclass
class ReviewDependencies:
    """Dependencies for code review operations"""
    repository_url: str
    branch: str
    merge_request_iid: int
    gitlab_token: str
    diff_content: str
    file_changes: List[Dict[str, Any]]
    review_trigger_tag: str

class CodeReviewAgent:
    """Main code review agent using PydanticAI"""
    
    def __init__(self, model_name: str = None):
        """Initialize the review agent with specified model"""
        self.model_name = model_name or settings.ai_model
        self.model = get_llm_model(self.model_name)
        
        # Create PydanticAI agent
        self.agent = Agent(
            model=self.model,
            output_type=ReviewResult,
            deps_type=ReviewDependencies,
            system_prompt=CODE_REVIEW_SYSTEM_PROMPT,
            retries=settings.ai_retries
        )
        
        # Register tools
        self._register_tools()
        
        logger.info(f"Initialized CodeReviewAgent with model: {self.model_name}")
    
    def _register_tools(self):
        """Register agent tools for enhanced analysis"""
        
        @self.agent.tool
        async def analyze_security_patterns(
            ctx: RunContext[ReviewDependencies],
            code_snippet: str
        ) -> str:
            """Analyze code for common security vulnerabilities"""
            security_checks = [
                "SQL injection risks",
                "XSS vulnerabilities", 
                "Authentication bypass",
                "Insecure cryptography",
                "Sensitive data exposure",
                "Input validation issues"
            ]
            
            findings = []
            # Simplified security analysis (would be more sophisticated in production)
            if "eval(" in code_snippet or "exec(" in code_snippet:
                findings.append("Dangerous use of eval/exec - potential code injection")
            if "password" in code_snippet.lower() and "plain" in code_snippet.lower():
                findings.append("Potential plaintext password storage")
            
            return f"Security analysis complete. Findings: {', '.join(findings) if findings else 'No issues detected'}"
        
        @self.agent.tool
        async def check_code_complexity(
            ctx: RunContext[ReviewDependencies],
            function_code: str
        ) -> Dict[str, Any]:
            """Calculate cyclomatic complexity and other metrics"""
            # Simplified complexity calculation
            lines = function_code.split('\n')
            complexity_score = 1  # Base complexity
            
            for line in lines:
                if any(keyword in line for keyword in ['if ', 'elif ', 'for ', 'while ', 'except']):
                    complexity_score += 1
            
            return {
                "cyclomatic_complexity": complexity_score,
                "lines_of_code": len(lines),
                "recommendation": "Consider refactoring" if complexity_score > 10 else "Acceptable complexity"
            }
        
        @self.agent.tool  
        async def suggest_improvements(
            ctx: RunContext[ReviewDependencies],
            issue_description: str
        ) -> str:
            """Generate specific improvement suggestions"""
            # Context-aware suggestions based on the issue
            suggestions_map = {
                "error handling": "Add try-except blocks with specific exception types",
                "type hints": "Add type annotations for function parameters and return values",
                "documentation": "Add docstrings following Google or NumPy style",
                "testing": "Create unit tests covering edge cases and error conditions"
            }
            
            for keyword, suggestion in suggestions_map.items():
                if keyword in issue_description.lower():
                    return suggestion
            
            return "Consider refactoring for better readability and maintainability"
    
    async def review_merge_request(
        self,
        diff_content: str,
        context: ReviewContext
    ) -> ReviewResult:
        """Perform comprehensive code review on merge request"""
        
        logger.info(f"Starting review for MR {context.merge_request_iid}")
        
        # Prepare dependencies
        deps = ReviewDependencies(
            repository_url=context.repository_url,
            branch=context.target_branch,
            merge_request_iid=context.merge_request_iid,
            gitlab_token=settings.gitlab_token,
            diff_content=diff_content,
            file_changes=context.file_changes,
            review_trigger_tag=context.trigger_tag
        )
        
        # Construct review prompt
        review_prompt = f"""
        Please review the following code changes from a GitLab merge request.
        
        Repository: {context.repository_url}
        Target Branch: {context.target_branch}
        Source Branch: {context.source_branch}
        
        DIFF CONTENT:
        {diff_content}
        
        Provide a comprehensive review focusing on:
        1. Critical issues that must be fixed
        2. Security vulnerabilities
        3. Performance concerns
        4. Code quality and maintainability
        5. Positive aspects worth highlighting
        """
        
        try:
            # Run the review agent
            result = await self.agent.run(review_prompt, deps=deps)
            
            # Log token usage for monitoring
            usage = result.usage()
            logger.info(f"Review completed. Tokens used: {usage.total_tokens}")
            
            return result.output
                
        except Exception as e:
            logger.error(f"Review failed for MR {context.merge_request_iid}: {e}")
            raise

async def initialize_review_agent() -> CodeReviewAgent:
    """Factory function to initialize the review agent"""
    return CodeReviewAgent(model_name=settings.ai_model)