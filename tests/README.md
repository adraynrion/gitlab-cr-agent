# Testing Guide

## Running Tests

### Unit Tests
Unit tests can be run without any external dependencies:

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test modules
python -m pytest tests/unit/test_tool_system.py -v
python -m pytest tests/unit/test_context_tools.py -v
```

### Integration Tests

Integration tests require an OpenRouter API key to test the complete AI-powered review workflow.

#### Setup for Integration Tests

1. **Get an OpenRouter API Key**:
   - Visit [OpenRouter](https://openrouter.ai/)
   - Sign up for a free account
   - Get your API key from the dashboard

2. **Set Environment Variable**:
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key-here"
   ```

3. **Run Integration Tests**:
   ```bash
   # Run all integration tests
   python -m pytest tests/integration/ -v
   
   # Run specific integration tests
   python -m pytest tests/integration/test_tool_workflow.py::TestToolWorkflowIntegration::test_complete_review_workflow -v
   ```

#### Why OpenRouter?

Integration tests use OpenRouter with the free model `openai/gpt-oss-20b:free` because:
- It provides free API access for testing
- No quota limits for the free tier model
- Compatible with OpenAI API format
- Reliable for CI/CD environments

#### Test Configuration

The integration tests automatically configure:
- **Base URL**: `https://openrouter.ai/api/v1`
- **Model**: `openai/gpt-oss-20b:free` 
- **API Key**: From `OPENROUTER_API_KEY` environment variable

If no API key is configured, integration tests will be skipped automatically.

## Test Coverage

Current test coverage includes:
- **Tool System Framework**: 20/20 unit tests passing
- **Context7 Integration**: 19/19 unit tests passing  
- **Integration Workflows**: Tests for complete review workflows
- **Error Handling**: Comprehensive error scenario testing

## CI/CD Integration

For continuous integration, set the `OPENROUTER_API_KEY` environment variable in your CI/CD pipeline:

```yaml
# GitHub Actions example
env:
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

# GitLab CI example  
variables:
  OPENROUTER_API_KEY: $OPENROUTER_API_KEY
```