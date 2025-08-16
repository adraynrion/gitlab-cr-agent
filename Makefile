# GitLab Code Review Agent - Makefile
# Development automation for testing, linting, and code quality

.PHONY: help install test lint format typecheck security clean coverage html-cov all check ci fix quality dev-install

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
VENV_PATH := .venv
SRC_DIRS := src tests
SRC_FILES := $(shell find $(SRC_DIRS) -name "*.py" 2>/dev/null || echo "")
COVERAGE_MIN := 80

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# Help target - shows available commands
help: ## Show this help message
	@echo "$(BLUE)GitLab Code Review Agent - Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Common workflows:$(RESET)"
	@echo "  $(GREEN)make install$(RESET)     - Set up development environment"
	@echo "  $(GREEN)make check$(RESET)       - Run all quality checks (test + lint + type + format)"
	@echo "  $(GREEN)make fix$(RESET)         - Auto-fix formatting and import issues"
	@echo "  $(GREEN)make ci$(RESET)          - Run full CI pipeline (for automated builds)"

# Development Environment Setup
install: ## Install development dependencies and set up environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(RESET)"; \
		$(PYTHON) -m venv $(VENV_PATH); \
	fi
	@echo "$(YELLOW)Installing dependencies...$(RESET)"
	@$(VENV_PATH)/bin/pip install --upgrade pip
	@$(VENV_PATH)/bin/pip install -e ".[dev,test]"
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"

dev-install: install ## Alias for install

# Testing
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(RESET)"
	@$(VENV_PATH)/bin/$(PYTHON) -m pytest tests/ -v

test-unit: ## Run only unit tests
	@echo "$(BLUE)Running unit tests...$(RESET)"
	@$(VENV_PATH)/bin/$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run only integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	@$(VENV_PATH)/bin/$(PYTHON) -m pytest tests/integration/ -v

test-fast: ## Run tests with minimal output
	@echo "$(BLUE)Running fast tests...$(RESET)"
	@$(VENV_PATH)/bin/$(PYTHON) -m pytest tests/ -q

test-failed: ## Run only previously failed tests
	@echo "$(BLUE)Running previously failed tests...$(RESET)"
	@$(VENV_PATH)/bin/$(PYTHON) -m pytest --lf -v

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	@$(VENV_PATH)/bin/ptw -- tests/

# Coverage
coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	@$(VENV_PATH)/bin/$(PYTHON) -m pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-fail-under=$(COVERAGE_MIN)

html-cov: ## Generate HTML coverage report
	@echo "$(BLUE)Generating HTML coverage report...$(RESET)"
	@$(VENV_PATH)/bin/$(PYTHON) -m pytest tests/ \
		--cov=src \
		--cov-report=html \
		--cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(RESET)"

coverage-xml: ## Generate XML coverage report (for CI)
	@echo "$(BLUE)Generating XML coverage report...$(RESET)"
	@$(VENV_PATH)/bin/$(PYTHON) -m pytest tests/ \
		--cov=src \
		--cov-report=xml \
		--cov-report=term-missing

# Code Quality - Linting
lint: ## Run flake8 linting
	@echo "$(BLUE)Running flake8 linting...$(RESET)"
	@$(VENV_PATH)/bin/flake8 $(SRC_DIRS) --count --select=E9,F63,F7,F82 --show-source --statistics || \
		(echo "$(RED)✗ Critical linting errors found$(RESET)" && exit 1)
	@$(VENV_PATH)/bin/flake8 $(SRC_DIRS) --count --max-line-length=88 --extend-ignore=E203,W503 --statistics || \
		(echo "$(RED)✗ Linting issues found$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Linting passed$(RESET)"

lint-report: ## Generate flake8 HTML report
	@echo "$(BLUE)Generating flake8 HTML report...$(RESET)"
	@mkdir -p reports
	@$(VENV_PATH)/bin/flake8 $(SRC_DIRS) --format=html --htmldir=reports/flake8 --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(GREEN)✓ Linting report generated: reports/flake8/index.html$(RESET)"

# Code Quality - Type Checking
typecheck: ## Run mypy type checking
	@echo "$(BLUE)Running mypy type checking...$(RESET)"
	@$(VENV_PATH)/bin/mypy src/ --ignore-missing-imports --disallow-untyped-defs || \
		(echo "$(RED)✗ Type checking failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Type checking passed$(RESET)"

typecheck-report: ## Generate mypy HTML report
	@echo "$(BLUE)Generating mypy HTML report...$(RESET)"
	@mkdir -p reports
	@$(VENV_PATH)/bin/mypy src/ --ignore-missing-imports --html-report reports/mypy
	@echo "$(GREEN)✓ Type checking report generated: reports/mypy/index.html$(RESET)"

# Code Formatting
format: ## Format code with black
	@echo "$(BLUE)Formatting code with black...$(RESET)"
	@$(VENV_PATH)/bin/black $(SRC_DIRS) --line-length=88
	@echo "$(GREEN)✓ Code formatted$(RESET)"

format-check: ## Check if code is formatted correctly
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	@$(VENV_PATH)/bin/black $(SRC_DIRS) --line-length=88 --check || \
		(echo "$(RED)✗ Code formatting issues found. Run 'make format' to fix.$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Code formatting is correct$(RESET)"

format-diff: ## Show formatting differences
	@echo "$(BLUE)Showing formatting differences...$(RESET)"
	@$(VENV_PATH)/bin/black $(SRC_DIRS) --line-length=88 --diff

# Import Management
imports: ## Sort and clean up imports with isort and autoflake
	@echo "$(BLUE)Cleaning up imports...$(RESET)"
	@$(VENV_PATH)/bin/autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive $(SRC_DIRS)
	@$(VENV_PATH)/bin/isort $(SRC_DIRS)
	@echo "$(GREEN)✓ Imports cleaned and sorted$(RESET)"

imports-check: ## Check for unused imports and import ordering
	@echo "$(BLUE)Checking for unused imports and import ordering...$(RESET)"
	@$(VENV_PATH)/bin/autoflake --remove-all-unused-imports --remove-unused-variables --check --recursive $(SRC_DIRS) || \
		(echo "$(RED)✗ Unused imports found. Run 'make imports' to fix.$(RESET)" && exit 1)
	@$(VENV_PATH)/bin/isort --check-only $(SRC_DIRS) || \
		(echo "$(RED)✗ Import ordering issues found. Run 'make imports' to fix.$(RESET)" && exit 1)
	@echo "$(GREEN)✓ No unused imports and import order is correct$(RESET)"

imports-sort: ## Sort imports only with isort
	@echo "$(BLUE)Sorting imports with isort...$(RESET)"
	@$(VENV_PATH)/bin/isort $(SRC_DIRS)
	@echo "$(GREEN)✓ Imports sorted$(RESET)"

# Security
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	@if command -v bandit >/dev/null 2>&1; then \
		$(VENV_PATH)/bin/bandit -r src/ -ll; \
	else \
		echo "$(YELLOW)⚠ Bandit not installed. Install with: pip install bandit$(RESET)"; \
	fi

# Composite Commands
fix: ## Auto-fix code issues (formatting + imports)
	@echo "$(BLUE)Auto-fixing code issues...$(RESET)"
	@$(MAKE) imports
	@$(MAKE) format
	@echo "$(GREEN)✓ Code issues fixed$(RESET)"

quality: typecheck lint format-check imports-check ## Run all quality checks
	@echo "$(GREEN)✓ All quality checks passed$(RESET)"

check: test quality ## Run tests and quality checks
	@echo "$(GREEN)✓ All checks passed$(RESET)"

ci: ## Run full CI pipeline (tests + quality + coverage)
	@echo "$(BLUE)Running full CI pipeline...$(RESET)"
	@$(MAKE) format-check
	@$(MAKE) imports-check
	@$(MAKE) lint
	@$(MAKE) typecheck
	@$(MAKE) coverage
	@echo "$(GREEN)✓ CI pipeline completed successfully$(RESET)"

all: fix quality test ## Fix code issues, run quality checks, and test
	@echo "$(GREEN)✓ All tasks completed$(RESET)"

# Development Server
serve: ## Start development server
	@echo "$(BLUE)Starting development server...$(RESET)"
	@$(VENV_PATH)/bin/uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

serve-prod: ## Start production server
	@echo "$(BLUE)Starting production server...$(RESET)"
	@$(VENV_PATH)/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# Cleanup
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(RESET)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ reports/ dist/ build/
	@echo "$(GREEN)✓ Cleanup completed$(RESET)"

clean-venv: ## Remove virtual environment
	@echo "$(BLUE)Removing virtual environment...$(RESET)"
	@rm -rf $(VENV_PATH)
	@echo "$(GREEN)✓ Virtual environment removed$(RESET)"

# Documentation
docs: ## Generate documentation (if available)
	@echo "$(BLUE)Generating documentation...$(RESET)"
	@if [ -f "docs/Makefile" ]; then \
		$(MAKE) -C docs html; \
	else \
		echo "$(YELLOW)⚠ No documentation setup found$(RESET)"; \
	fi

# Docker (if Dockerfile exists)
docker-build: ## Build Docker image
	@if [ -f "Dockerfile" ]; then \
		echo "$(BLUE)Building Docker image...$(RESET)"; \
		docker build -t gitlab-code-review-agent .; \
	else \
		echo "$(YELLOW)⚠ No Dockerfile found$(RESET)"; \
	fi

docker-run: ## Run Docker container
	@if [ -f "Dockerfile" ]; then \
		echo "$(BLUE)Running Docker container...$(RESET)"; \
		docker run -p 8000:8000 gitlab-code-review-agent; \
	else \
		echo "$(YELLOW)⚠ No Dockerfile found$(RESET)"; \
	fi

# Utility
show-env: ## Show environment information
	@echo "$(BLUE)Environment Information:$(RESET)"
	@echo "Python: $(shell $(VENV_PATH)/bin/$(PYTHON) --version 2>/dev/null || echo 'Not found')"
	@echo "Virtual Environment: $(VENV_PATH)"
	@echo "Source Directories: $(SRC_DIRS)"
	@echo "Coverage Minimum: $(COVERAGE_MIN)%"
	@echo "Git Branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "Git Status: $(shell git status --porcelain 2>/dev/null | wc -l || echo 'N/A') files changed"

deps: ## Show dependency tree
	@echo "$(BLUE)Dependency tree:$(RESET)"
	@$(VENV_PATH)/bin/pip list

upgrade-deps: ## Upgrade all dependencies
	@echo "$(BLUE)Upgrading dependencies...$(RESET)"
	@$(VENV_PATH)/bin/pip install --upgrade pip
	@$(VENV_PATH)/bin/pip install --upgrade -e ".[dev,test]"

# Quick development workflow aliases
t: test ## Alias for test
f: format ## Alias for format
l: lint ## Alias for lint
c: check ## Alias for check
