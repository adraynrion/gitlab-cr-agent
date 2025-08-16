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
- **Intelligent Tools**: Built-in security pattern detection, complexity analysis, and improvement suggestions

### Enterprise Security 🛡️
- **Bearer Token Authentication**: Industry-standard Bearer token auth for all protected endpoints
- **Rate Limiting**: Configurable per-IP rate limiting to prevent DoS attacks
- **Request Validation**: Size limits and input sanitization to prevent memory exhaustion
- **Security Headers**: Full CSRF, XSS, and clickjacking protection
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
- **Monitoring**: Prometheus metrics and observability hooks

## 🏗️ Architecture

```
├── FastAPI Application (src/main.py)
├── PydanticAI Review Agent (src/agents/)
│   ├── Multi-LLM Provider Support
│   └── Security Analysis Tools
├── GitLab Integration (src/services/)
│   ├── Webhook Handler
│   ├── API Client
│   └── Comment Formatter
├── Review Orchestration (src/services/review_service.py)
└── Configuration & Models (src/config/, src/models/)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GitLab instance (self-hosted or GitLab.com)
- At least one AI provider API key (OpenAI, Anthropic, or Google)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/gitlab-ai-reviewer.git
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
- `gemini:gemini-1.5-pro` - Google Gemini 1.5 Pro
- `fallback` - Use multiple providers with fallback

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

The AI agent analyzes code across five key areas:

- **🔒 Security**: Vulnerability detection, input validation, authentication issues
- **⚡ Performance**: Bottlenecks, inefficient algorithms, resource usage
- **✅ Correctness**: Logic errors, edge cases, algorithm issues
- **🛠️ Maintainability**: Code clarity, structure, documentation quality
- **📋 Best Practices**: Language conventions, design patterns, testing

### Sample Review Output

```markdown
## ✅ AI Code Review

**Overall Assessment:** Approve with Changes
**Risk Level:** Medium

### Summary
The merge request introduces input validation to the calculate function, which is a good security practice. However, there are a few areas that need attention.

### Issues Found (2)

#### 🟡 High Issues
**src/calculator.py:15** - Security
Missing input sanitization could lead to injection attacks.
💡 **Suggestion:** Add proper input validation and sanitization

#### 🔵 Low Issues
**src/calculator.py:8** - Style
Function lacks proper type hints.
💡 **Suggestion:** Add type annotations for better code clarity

### ✨ Positive Feedback
- Excellent error handling implementation
- Good use of descriptive variable names
- Proper function documentation

🤖 *Generated by GitLab AI Code Review Agent*
```

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -t gitlab-ai-reviewer:latest .

# Run container
docker run -d \
  --name gitlab-ai-reviewer \
  -p 8000:8000 \
  --env-file .env \
  gitlab-ai-reviewer:latest
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

### Production Checklist

#### Security 🔒
- [ ] Configure proper secrets management (HashiCorp Vault, AWS Secrets Manager)
- [ ] Set up SSL/TLS certificates with automatic renewal
- [ ] Configure CORS origins for your specific GitLab domain
- [ ] Set appropriate rate limits for your traffic patterns
- [ ] Enable webhook secret verification
- [ ] Configure firewall rules and network security groups
- [ ] Set up API key authentication for admin endpoints

#### Reliability 🛠️
- [ ] Configure log aggregation (ELK stack, Splunk, or similar)
- [ ] Set up monitoring and alerts (Prometheus + Grafana)
- [ ] Configure health check endpoints for load balancers
- [ ] Test graceful shutdown behavior
- [ ] Validate retry mechanisms and circuit breaker patterns
- [ ] Set up distributed tracing (Jaeger, Zipkin)

#### Operations 🚀
- [ ] Test webhook connectivity and authentication
- [ ] Validate AI provider quotas and rate limits
- [ ] Configure horizontal pod autoscaling (if using Kubernetes)
- [ ] Set up backup strategies for configuration
- [ ] Document runbooks for common operational tasks
- [ ] Configure alerting for critical errors and performance degradation

## 🧪 Testing

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run only unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run security-focused tests
pytest tests/security/ -v
```

### Test Categories

- **Unit Tests**: Component-level testing with mocks
- **Integration Tests**: End-to-end webhook and API testing
- **Security Tests**: Rate limiting, input validation, and auth testing
- **Performance Tests**: Load testing and response time validation

### Manual Testing

```bash
# Test application startup and health
python -c "from src.main import app; print('✅ App loads successfully')"

# Test configuration validation
ENVIRONMENT=test GITLAB_URL=http://test GITLAB_TOKEN=test-token-12345678901 python -c "from src.config.settings import settings; print('✅ Config valid')"
```

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

### Project Structure

```
src/
├── main.py                   # FastAPI application with lifespan management
├── exceptions.py             # Standardized exception hierarchy
├── agents/
│   ├── code_reviewer.py      # PydanticAI review agent with tools
│   └── providers.py          # Multi-LLM configuration with error handling
├── api/
│   ├── webhooks.py          # GitLab webhook handlers with rate limiting
│   ├── health.py            # Comprehensive health check endpoints
│   └── middleware.py        # Security, authentication & logging
├── services/
│   ├── gitlab_service.py    # GitLab API client with retry logic
│   └── review_service.py    # Review orchestration with error recovery
├── models/
│   ├── gitlab_models.py     # GitLab webhook models with validation
│   └── review_models.py     # Review result models
└── config/
    └── settings.py          # Configuration management with validation
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Remove unused imports and variables (dry run)
autoflake --remove-all-unused-imports --remove-unused-variables --check --recursive src/ tests/

# Remove unused imports and variables (in place)
autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive src/ tests/

# Type check
mypy src/
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
GET /                    # Service status with features
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- 📖 [Documentation](https://github.com/your-org/gitlab-ai-reviewer#readme)
- 🐛 [Issue Tracker](https://github.com/your-org/gitlab-ai-reviewer/issues)
- 💬 [Discussions](https://github.com/your-org/gitlab-ai-reviewer/discussions)
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
