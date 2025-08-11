# GitLab AI Code Review Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-0.7+-orange.svg)](https://ai.pydantic.dev/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/)

A production-ready AI-powered code review agent that integrates with GitLab using PydanticAI. Automatically analyzes merge requests for security vulnerabilities, performance issues, code quality, and best practices.

## 🌟 Features

- **Multi-LLM Support**: Works with OpenAI GPT-4, Anthropic Claude, and Google Gemini
- **GitLab Integration**: Seamless webhook-based integration with any self-hosted GitLab instance  
- **Comprehensive Analysis**: Security, performance, correctness, and maintainability reviews
- **Production Ready**: Docker containerization, health checks, structured logging
- **Secure**: Token-based authentication, webhook verification, secure configuration
- **Scalable**: Async processing, background tasks, Kubernetes deployment ready
- **Extensible**: Clean architecture with pluggable components

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

3. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

4. **Or run locally:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   python -m src.main
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

- [ ] Configure proper secrets management
- [ ] Set up SSL/TLS certificates
- [ ] Configure log aggregation
- [ ] Set up monitoring and alerts
- [ ] Configure backup strategies
- [ ] Test webhook connectivity
- [ ] Validate AI provider quotas

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/
```

## 📊 Monitoring

The application provides several endpoints for monitoring:

- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe  
- `GET /health/status` - Detailed status information
- `GET /` - Basic service information

### Metrics

The service exposes Prometheus metrics for:
- Request latency and throughput
- AI model usage and costs
- GitLab API response times
- Review success/failure rates

## 🔧 Development

### Project Structure

```
src/
├── main.py                   # FastAPI application
├── agents/                   
│   ├── code_reviewer.py      # PydanticAI review agent
│   └── providers.py          # Multi-LLM configuration
├── api/
│   ├── webhooks.py          # GitLab webhook handlers
│   ├── health.py            # Health check endpoints  
│   └── middleware.py        # Authentication & logging
├── services/
│   ├── gitlab_service.py    # GitLab API client
│   └── review_service.py    # Review orchestration
├── models/
│   ├── gitlab_models.py     # GitLab webhook models
│   └── review_models.py     # Review result models
└── config/
    └── settings.py          # Configuration management
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

# Type check
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- 📖 [Documentation](https://github.com/your-org/gitlab-ai-reviewer#readme)
- 🐛 [Issue Tracker](https://github.com/your-org/gitlab-ai-reviewer/issues)
- 💬 [Discussions](https://github.com/your-org/gitlab-ai-reviewer/discussions)

## 🙏 Acknowledgments

- [PydanticAI](https://ai.pydantic.dev/) - Type-safe AI framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [GitLab](https://gitlab.com/) - DevOps platform
- [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), [Google AI](https://ai.google/) - AI providers

---

**Built with ❤️ for better code quality**