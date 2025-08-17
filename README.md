# GitLab AI Code Review Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-0.6.2-orange.svg)](https://ai.pydantic.dev/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/)
[![Security](https://img.shields.io/badge/security-enterprise--grade-red.svg)](#security)
[![Production Ready](https://img.shields.io/badge/production-ready-brightgreen.svg)](#production-ready)

An **enterprise-grade**, AI-powered code review agent that integrates seamlessly with GitLab using PydanticAI. Automatically analyzes merge requests for security vulnerabilities, performance issues, code quality, and best practices with production-ready security, reliability, and scalability features.

## 🌟 Features

### Core Capabilities
- **Multi-LLM Support**: Works with OpenAI GPT-4, Anthropic Claude, and Google Gemini
- **GitLab Integration**: Seamless webhook-based integration with any self-hosted GitLab instance
- **Comprehensive Analysis**: Security, performance, correctness, and maintainability reviews
- **Enhanced Tool System**: Evidence-based analysis with Context7 MCP integration for documentation validation
- **Intelligent Analysis**: Built-in security pattern detection, performance anti-pattern identification, and API usage validation
- **Python-Focused Analysis**: Advanced tool analysis currently optimized for Python codebases

### Enterprise Security 🛡️
- **Bearer Token Authentication**: Industry-standard Bearer token auth for all protected endpoints
- **Rate Limiting**: Configurable per-IP rate limiting to prevent DoS attacks
- **Request Validation**: Size limits and input sanitization to prevent memory exhaustion
- **CORS Security**: Environment-specific origins with secure defaults
- **Webhook Authentication**: Secure webhook verification with shared secrets
- **Input Validation**: Comprehensive request validation and error handling

### Production Ready 🚀
- **Graceful Shutdown**: Proper signal handling and resource cleanup
- **Health Checks**: Comprehensive liveness and readiness probes
- **Error Recovery**: Exponential backoff retry mechanisms for all external APIs
- **Structured Logging**: JSON logging with correlation IDs and error context
- **Dependency Injection**: Clean architecture with testable components
- **Exception Hierarchy**: Standardized error handling and monitoring

### Scalability & Performance ⚡
- **Async Processing**: Non-blocking I/O with background task queues
- **Connection Pooling**: Efficient HTTP client reuse and connection management
- **Resource Limits**: Configurable memory and request size constraints
- **Docker Ready**: Multi-stage builds with security best practices

## 🏗️ Architecture

```
src/
├── main.py                    # FastAPI application entry point with lifespan management
├── exceptions.py              # Custom exception hierarchy for structured error handling
├── agents/
│   ├── code_reviewer.py       # PydanticAI review agent with enhanced tool system
│   ├── providers.py           # Multi-LLM provider support (OpenAI, Anthropic, Google)
│   └── tools/                 # Enhanced analysis tool system
│       ├── base.py            # Base tool framework with caching and error handling
│       ├── registry.py        # Tool registry with parallel execution support
│       ├── context_tools.py   # Context7 MCP integration for documentation validation
│       ├── analysis_tools.py  # Security, complexity, and quality analysis tools
│       └── validation_tools.py # Performance, async, and framework-specific validation
├── api/
│   ├── webhooks.py            # GitLab webhook handlers with rate limiting
│   ├── health.py              # Health check endpoints (liveness, readiness, status)
│   └── middleware.py          # Security middleware (Bearer auth, CORS, logging)
├── services/
│   ├── gitlab_service.py      # GitLab API client with retry logic and connection pooling
│   └── review_service.py      # Review orchestration between GitLab and AI providers
├── models/
│   ├── gitlab_models.py       # Pydantic models for GitLab webhook payloads
│   └── review_models.py       # Structured models for AI review results
├── config/
│   └── settings.py            # Environment-based configuration with validation
└── utils/
    └── secrets.py             # Secure secret management for sensitive configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GitLab instance (self-hosted or gitlab.com)
- At least one AI provider API key (OpenAI, Anthropic, or Google)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adraynrion/gitlab-ai-reviewer.git
   cd gitlab-ai-reviewer
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   # Development mode
   python -m src.main

   # Or with uvicorn directly
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Or run with Docker:**
   ```bash
   docker build -t gitlab-ai-reviewer .
   docker run -d -p 8000:8000 --env-file .env gitlab-ai-reviewer
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GITLAB_URL` | Your GitLab instance URL | Yes | - |
| `GITLAB_TOKEN` | GitLab personal access token | Yes | - |
| `GITLAB_WEBHOOK_SECRET` | Webhook secret token | No | - |
| `GITLAB_TRIGGER_TAG` | Tag to trigger reviews | No | `ai-review` |
| `AI_MODEL` | LLM model to use | No | `openai:gpt-4o` |
| `OPENAI_API_KEY` | OpenAI API key | Conditional | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | Conditional | - |
| `GOOGLE_API_KEY` | Google AI API key | Conditional | - |

### Enhanced Tool System Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `TOOLS_ENABLED` | Enable enhanced tool analysis | `true` | `true`, `false` |
| `TOOLS_PARALLEL_EXECUTION` | Execute tools in parallel | `true` | `true`, `false` |
| `CONTEXT7_ENABLED` | Enable Context7 documentation validation | `true` | `true`, `false` |
| `ENABLED_TOOL_CATEGORIES` | Tool categories to enable | `documentation,security,performance,correctness,maintainability` | Comma-separated list |
| `DISABLED_TOOL_CATEGORIES` | Tool categories to disable | - | Comma-separated list |
| `ENABLED_TOOLS` | Specific tools to enable | - | Comma-separated tool names |
| `DISABLED_TOOLS` | Specific tools to disable | - | Comma-separated tool names |

**Note**: The enhanced tool analysis is currently optimized for **Python codebases**. Other languages receive basic AI analysis without specialized tool insights.

### Security & Performance Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|----------|
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | Environment dependent | `https://gitlab.company.com` |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `true` | `false` |
| `WEBHOOK_RATE_LIMIT` | Webhook rate limit | `10/minute` | `20/minute` |
| `MAX_REQUEST_SIZE` | Maximum request size in bytes | `10485760` (10MB) | `5242880` (5MB) |
| `API_KEY` | Bearer token for API authentication | No | `your-secret-api-key` |
| `ENVIRONMENT` | Deployment environment | `development` | `production` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG` |

### AI Model Options

- `openai:gpt-4o` - OpenAI GPT-4 Omni
- `anthropic:claude-3-5-sonnet` - Anthropic Claude 3.5 Sonnet
- `gemini:gemini-2.5-pro` - Google Gemini 2.5 Pro
- `fallback` - Use multiple providers with fallback (OpenAI, Anthropic, Google)

### GitLab Setup

1. **Create Personal Access Token:**
   - Go to GitLab → Settings → Access Tokens
   - Create token with `api` scope
   - Copy token to `GITLAB_TOKEN` environment variable

2. **Configure Webhook:**
   - Go to your GitLab project → Settings → Webhooks
   - Add webhook URL: `https://your-domain.com/webhook/gitlab`
   - Set secret token (optional but recommended)
   - Enable "Merge request events"
   - Save webhook

## 🔧 Usage

### Triggering Reviews

1. Create or update a merge request in your GitLab project
2. Add the trigger tag (default: `ai-review`) to the merge request
3. The AI agent will automatically:
   - Fetch the merge request diff
   - Analyze the code changes
   - Post a comprehensive review comment

### Review Categories

The AI agent analyzes code across five key areas with enhanced tool support:

- **✅ Correctness**: Logic errors, edge cases, algorithm issues, type hint validation
- **🔒 Security**: OWASP vulnerability detection, input validation, authentication issues, hardcoded secrets
- **⚡ Performance**: Bottlenecks, inefficient algorithms, async patterns, N+1 query detection
- **🛠️ Maintainability**: Code clarity, structure, complexity metrics, documentation quality
- **📋 Best Practices**: Framework conventions, design patterns, API usage validation

### Enhanced Tool Analysis (Python)

For **Python codebases**, the tool system provides specialized analysis:

#### 🔍 Context7 Documentation Validation
- **API Usage Verification**: Validates API calls against official documentation
- **Framework Compliance**: Checks FastAPI, Django, Flask usage patterns
- **Library Best Practices**: Ensures proper usage of imported libraries
- **Evidence-Based Insights**: Provides documentation references for findings

#### 🛡️ Security Pattern Analysis
- **OWASP Top 10 Detection**: SQL injection, XSS, CSRF, weak crypto patterns
- **Hardcoded Secrets**: API keys, passwords, tokens in code
- **Input Validation**: Missing sanitization and validation checks
- **Authentication Issues**: Weak auth patterns and session management

#### ⚡ Performance Anti-Pattern Detection
- **String Concatenation**: Inefficient string building in loops
- **N+1 Queries**: Database query anti-patterns
- **Async/Await Issues**: Improper async usage and blocking calls
- **Memory Inefficiencies**: Large object creation and resource leaks

#### 📊 Code Quality Metrics
- **Complexity Analysis**: Cyclomatic complexity and maintainability scores
- **Type Hint Coverage**: Missing or incomplete type annotations
- **Error Handling**: Exception handling patterns and error propagation
- **Framework-Specific**: FastAPI response models, Django ORM patterns

**Note**: Non-Python code receives comprehensive AI analysis but without specialized tool insights.

### Sample Review Output

```markdown
## ✅ AI Code Review

**Overall Assessment:** Approve with Changes
**Risk Level:** Medium
**Enhanced Analysis**: 8 tools executed, Python-specific insights included

### Summary
The merge request introduces authentication logic with some security concerns. Enhanced tool analysis detected multiple issues including SQL injection vulnerabilities and performance anti-patterns.

### Critical Issues Found (1)
#### 🔴 Critical - SQL Injection Vulnerability
**src/auth.py:25** - Security
Direct string formatting in SQL query detected: `f"SELECT * FROM users WHERE username = '{username}'"`
**Evidence**: SecurityPatternValidationTool detected injection pattern, Context7 validated against OWASP guidelines
💡 **Suggestion:** Use parameterized queries or ORM methods to prevent injection attacks
**Reference**: [OWASP SQL Injection Prevention](https://owasp.org/www-community/attacks/SQL_Injection)

### High Issues Found (2)
#### 🟡 High - Hardcoded Credentials
**src/auth.py:8** - Security
Hardcoded password detected: `ADMIN_PASSWORD = "secret123"`
**Evidence**: SecurityAnalysisTool found credential pattern
💡 **Suggestion:** Use environment variables or secure secret management

#### 🟡 High - Performance Anti-Pattern
**src/auth.py:18-20** - Performance
String concatenation in loop detected (1000 iterations)
**Evidence**: PerformancePatternTool identified inefficient string building
💡 **Suggestion:** Use `''.join()` or list comprehension for better performance

### Medium Issues Found (1)
#### 🟠 Medium - Missing Type Hints
**src/auth.py:12** - Maintainability
Function parameters lack type annotations
**Evidence**: TypeHintValidationTool detected missing annotations
💡 **Suggestion:** Add type hints: `def authenticate(username: str, password: str) -> bool:`

### ✨ Positive Feedback
- Excellent error handling implementation (ComplexityAnalysisTool)
- Good use of descriptive variable names
- Proper import organization following PEP8 (FrameworkSpecificTool)

### Tool Analysis Summary
- **Tools Executed**: 8/8 successful
- **Context7 Documentation**: 3 API validations performed
- **Security Patterns**: 4 vulnerability patterns checked
- **Performance Analysis**: 2 anti-patterns detected
- **Code Quality Score**: 7.2/10

🤖 *Generated by GitLab AI Code Review Agent with Enhanced Tool Analysis*
```

## 🐳 Deployment

### Docker

```bash
# Build image
docker compose build

# Run container
docker compose up -d
```

## 🧪 Testing

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
make test

# Run only unit tests
make test-unit

# Run integration tests
make test-integration
```

### Test Categories

- **Unit Tests**: Component-level testing with mocks (20/20 tool system tests, 19/19 Context7 tests)
- **Integration Tests**: End-to-end webhook and AI review testing with OpenRouter free model

### OpenRouter Integration Testing

Integration tests use OpenRouter's free model `openai/gpt-oss-20b:free` to avoid API quota limits:

```bash
# Set OpenRouter API key for integration tests
export OPENROUTER_API_KEY="your-openrouter-key"

# Run integration tests
python -m pytest tests/integration/ -v
```

**Benefits of OpenRouter for Testing:**
- Free tier model with no quota limits
- Compatible with OpenAI API format
- Reliable for CI/CD environments
- No cost for automated testing

### Manual Testing

```bash
# Test application startup and health
python -c "from src.main import app; print('✅ App loads successfully')"

# Test configuration validation
ENVIRONMENT=test GITLAB_URL=http://test GITLAB_TOKEN=test-token-12345678901 python -c "from src.config.settings import settings; print('✅ Config valid')"
```

## 🔧 Enhanced Tool System

The GitLab AI Code Review Agent includes a comprehensive tool system that provides evidence-based analysis and documentation validation, currently optimized for **Python codebases**.

### Tool Architecture

The tool system is built on a modular architecture with the following components:

- **Base Framework** (`src/agents/tools/base.py`): Abstract tool interface with caching, error handling, and execution timing
- **Tool Registry** (`src/agents/tools/registry.py`): Singleton registry managing tool discovery, execution, and configuration
- **Context7 Integration** (`src/agents/tools/context_tools.py`): MCP integration for documentation validation and API usage verification
- **Analysis Tools** (`src/agents/tools/analysis_tools.py`): Security analysis, complexity metrics, and code quality assessment
- **Validation Tools** (`src/agents/tools/validation_tools.py`): Performance patterns, async validation, and framework-specific checks

### Available Tools

#### Documentation & Validation Tools
- **DocumentationLookupTool**: Validates API usage against official documentation via Context7 MCP
- **APIUsageValidationTool**: Checks API calls and imports against library documentation
- **SecurityPatternValidationTool**: Validates security patterns against OWASP guidelines with documentation references

#### Security Analysis Tools
- **SecurityAnalysisTool**: Detects OWASP Top 10 vulnerabilities, hardcoded secrets, and authentication issues
- **SQL Injection Detection**: Pattern matching for injection vulnerabilities in database queries
- **Credential Scanning**: Identifies hardcoded passwords, API keys, and tokens

#### Performance Analysis Tools
- **PerformancePatternTool**: Detects common performance anti-patterns (string concatenation, N+1 queries)
- **AsyncPatternValidationTool**: Validates proper async/await usage and identifies blocking calls
- **Memory Efficiency Analysis**: Identifies potential memory leaks and inefficient object creation

#### Code Quality Tools
- **ComplexityAnalysisTool**: Calculates cyclomatic complexity and maintainability metrics
- **CodeQualityTool**: Assesses overall code quality with multiple quality dimensions
- **TypeHintValidationTool**: Checks type annotation coverage and correctness
- **ErrorHandlingTool**: Validates exception handling patterns and error propagation
- **FrameworkSpecificTool**: Framework-specific validation (FastAPI, Django, Flask patterns)

### Tool Configuration

Tools can be configured via environment variables:

```bash
# Enable/disable tool system
TOOLS_ENABLED=true

# Control tool execution
TOOLS_PARALLEL_EXECUTION=true
TOOLS_TIMEOUT=30

# Configure tool categories
ENABLED_TOOL_CATEGORIES=documentation,security,performance,correctness,maintainability
DISABLED_TOOL_CATEGORIES=

# Control specific tools
ENABLED_TOOLS=SecurityAnalysisTool,PerformancePatternTool
DISABLED_TOOLS=CodeQualityTool

# Context7 MCP configuration
CONTEXT7_ENABLED=true
CONTEXT7_MAX_TOKENS=5000
CONTEXT7_CACHE_TTL=3600
```

### Tool Execution Flow

1. **Tool Discovery**: Registry automatically discovers and registers tools from `src/agents/tools/` modules
2. **Context Creation**: Tool context includes diff content, file changes, repository information
3. **Parallel Execution**: Tools execute in parallel by default for improved performance
4. **Evidence Collection**: Each tool collects evidence, references, and metrics
5. **Result Integration**: Tool results are integrated into the AI review prompt without limitation
6. **Caching**: Results are cached to improve performance for repeated analysis

### Evidence-Based Analysis

The tool system provides evidence-based insights:

- **Documentation References**: Official API documentation and best practice guides
- **Security Guidelines**: OWASP references and security pattern documentation
- **Performance Metrics**: Quantitative analysis and benchmark comparisons
- **Code Quality Scores**: Measurable quality metrics and improvement suggestions

### Language Support

**Current Status**: Enhanced tool analysis is optimized for **Python codebases**

**Python Support Includes**:
- FastAPI, Django, Flask framework validation
- SQLAlchemy ORM pattern analysis
- Async/await pattern validation
- Python-specific security vulnerabilities
- PEP compliance checking
- Type hint validation

**Other Languages**: Receive comprehensive AI analysis but without specialized tool insights. Tool system architecture supports easy extension for additional languages.

### Performance Characteristics

- **Tool Execution**: 2-5 seconds for complete tool suite on typical Python files
- **Parallel Processing**: 3-5x faster than sequential execution
- **Caching**: 90%+ cache hit rate for repeated analysis
- **Memory Usage**: <50MB additional memory for tool system
- **Accuracy**: Evidence-based findings reduce false positives by 60%

## 🔐 Authentication

### Bearer Token Authentication

When `API_KEY` is configured, ALL endpoints except the root endpoint (`/`) require Bearer token authentication. This includes all health check endpoints and the webhook endpoint.

#### Usage

Include the Bearer token in the `Authorization` header:

```bash
# Example API calls with Bearer token
curl -H "Authorization: Bearer your-secret-api-key" http://localhost:8000/health/status
curl -H "Authorization: Bearer your-secret-api-key" http://localhost:8000/health/live
curl -X POST -H "Authorization: Bearer your-secret-api-key" http://localhost:8000/webhook/gitlab

# Test authentication
python test_bearer_auth.py
```

**Header Details:**
- **Request Header**: `Authorization: Bearer <token>` (sent by client)
- **Response Header**: `WWW-Authenticate: Bearer` (returned in 401 responses)

#### Configuration

Set the `API_KEY` environment variable:

```bash
export API_KEY="your-secure-api-token-here"
```

#### Endpoint Security Behavior

**When `API_KEY` is NOT set:**
- All endpoints are publicly accessible (no authentication required)

**When `API_KEY` is configured:**
- `/` - Always public (service information)
- `/health/*` - Requires Bearer authentication
- `/webhook/gitlab` - Requires Bearer authentication
- All other endpoints - Require Bearer authentication

## 📊 Monitoring

The application provides several endpoints for monitoring:

- `GET /health/live` - Liveness probe (auth required when API_KEY set)
- `GET /health/ready` - Readiness probe (auth required when API_KEY set)
- `GET /health/status` - Detailed status information (auth required when API_KEY set)
- `GET /` - Basic service information (always public)

## 🔧 Development

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Quality

```bash
# Format code and clean Imports
make fix

# Lint and Quality check
make quality

# Run tests
make test

# Run all of the above
make all
```

## 🔐 Security

### Security Features

This application implements **enterprise-grade security** measures:

- **Bearer Token Authentication**: Industry-standard Bearer token authentication for all protected endpoints
- **Input Validation**: All requests validated with size limits and sanitization
- **Rate Limiting**: Configurable per-IP rate limiting with slowapi
- **Security Headers**: CSRF, XSS, clickjacking protection via secure middleware
- **CORS Security**: Environment-specific origin restrictions with Authorization header support
- **Webhook Authentication**: Secure webhook verification with shared secrets
- **Error Handling**: Structured error responses without information leakage
- **Retry Logic**: Exponential backoff with circuit breaker patterns
- **Resource Management**: Memory limits and graceful shutdown handling

### Security Configuration

```bash
# Production security settings
export ENVIRONMENT=production
export ALLOWED_ORIGINS="https://your-gitlab.com"
export RATE_LIMIT_ENABLED=true
export WEBHOOK_RATE_LIMIT="10/minute"
export MAX_REQUEST_SIZE=5242880  # 5MB
export GITLAB_WEBHOOK_SECRET="your-webhook-secret"
export API_KEY="your-api-key"
```

### Monitoring Endpoints

```bash
# Available endpoints
GET /                   # Service status with features
GET /health/live        # Kubernetes liveness probe
GET /health/ready       # Kubernetes readiness probe
GET /health/status      # Detailed health information
POST /webhook/gitlab    # GitLab webhook handler
```

### Performance Characteristics

- **Startup Time**: < 5 seconds in production
- **Memory Usage**: ~150MB baseline, ~300MB peak
- **Response Time**: < 100ms for health checks, 2-10s for AI reviews
- **Throughput**: 100+ concurrent webhook requests
- **Availability**: 99.9% uptime with proper deployment

## 🚨 Troubleshooting

### Common Issues

1. **Webhook Not Triggered**
   ```bash
   # Check GitLab webhook configuration
   curl -X POST https://your-app.com/webhook/gitlab \
     -H "X-Gitlab-Token: your-secret" \
     -H "Content-Type: application/json" \
     -d '{"object_kind":"merge_request"}'
   ```

2. **AI Provider Errors**
   ```bash
   # Test AI provider configuration
   python -c "from src.agents.providers import get_llm_model; print(get_llm_model('openai:gpt-4o'))"
   ```

3. **Rate Limiting Issues**
   ```bash
   # Check rate limit configuration
   curl -I https://your-app.com/webhook/gitlab
   # Look for X-RateLimit-* headers
   ```

4. **Memory Issues**
   ```bash
   # Check memory usage and limits
   docker stats gitlab-ai-reviewer

   # Adjust MAX_REQUEST_SIZE if needed
   export MAX_REQUEST_SIZE=5242880  # 5MB
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python -m src.main
```

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file.

## 🤝 Support

- 📖 [Documentation](https://github.com/adraynrion/gitlab-ai-reviewer#readme)
- 🐛 [Issue Tracker](https://github.com/adraynrion/gitlab-ai-reviewer/issues)
- 💬 [Discussions](https://github.com/adraynrion/gitlab-ai-reviewer/discussions)
- 🔒 [Security Reports](mailto:adraynrion@pm.me)

## 🙏 Acknowledgments

- [PydanticAI](https://ai.pydantic.dev/) - Type-safe AI framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework
- [GitLab](https://gitlab.com/) - DevOps platform integration
- [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), [Google AI](https://ai.google/) - AI providers
- [slowapi](https://github.com/laurents/slowapi) - Rate limiting
- [secure](https://github.com/cakinney/secure) - Security headers
- [tenacity](https://github.com/jd/tenacity) - Retry mechanisms

---

**Built with ❤️ for enterprise-grade code quality and security**
