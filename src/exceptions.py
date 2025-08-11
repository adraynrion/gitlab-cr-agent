"""
Custom exception hierarchy for GitLab AI Code Review Agent
"""

from typing import Optional, Dict, Any


class GitLabReviewerException(Exception):
    """Base exception for all GitLab reviewer errors"""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class GitLabAPIException(GitLabReviewerException):
    """GitLab API related errors"""
    
    def __init__(
        self, 
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if status_code:
            details['status_code'] = status_code
        if response_body:
            details['response_body'] = response_body
        
        super().__init__(message, details, kwargs.get('original_error'))
        self.status_code = status_code


class AIProviderException(GitLabReviewerException):
    """AI provider related errors"""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if provider:
            details['provider'] = provider
        if model:
            details['model'] = model
            
        super().__init__(message, details, kwargs.get('original_error'))
        self.provider = provider
        self.model = model


class WebhookValidationException(GitLabReviewerException):
    """Webhook payload validation errors"""
    
    def __init__(
        self,
        message: str,
        payload_section: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if payload_section:
            details['payload_section'] = payload_section
            
        super().__init__(message, details, kwargs.get('original_error'))


class ConfigurationException(GitLabReviewerException):
    """Configuration validation errors"""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
            
        super().__init__(message, details, kwargs.get('original_error'))


class ReviewProcessException(GitLabReviewerException):
    """Review process execution errors"""
    
    def __init__(
        self,
        message: str,
        merge_request_iid: Optional[int] = None,
        project_id: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if merge_request_iid:
            details['merge_request_iid'] = merge_request_iid
        if project_id:
            details['project_id'] = project_id
            
        super().__init__(message, details, kwargs.get('original_error'))
        self.merge_request_iid = merge_request_iid
        self.project_id = project_id


class SecurityException(GitLabReviewerException):
    """Security-related errors"""
    
    def __init__(
        self,
        message: str,
        security_context: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if security_context:
            details['security_context'] = security_context
            
        super().__init__(message, details, kwargs.get('original_error'))


class RateLimitException(GitLabReviewerException):
    """Rate limiting errors"""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after'] = retry_after
            
        super().__init__(message, details, kwargs.get('original_error'))
        self.retry_after = retry_after