"""
Circuit breaker implementation for AI provider calls
"""

import logging
from typing import Any, Callable, Optional, Type

from aiocircuitbreaker import CircuitBreaker

from src.config.settings import get_settings
from src.exceptions import AIProviderException

logger = logging.getLogger(__name__)


class AIProviderCircuitBreaker:
    """Circuit breaker for AI provider calls with configurable thresholds"""

    def __init__(
        self,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[int] = None,
        expected_exceptions: Optional[tuple[Type[Exception], ...]] = None,
    ):
        """
        Initialize circuit breaker for AI providers

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exceptions: Tuple of exceptions that trigger the circuit breaker
        """
        settings = get_settings()

        # Use settings or defaults
        self.failure_threshold = (
            failure_threshold or settings.circuit_breaker_failure_threshold
        )
        self.recovery_timeout = recovery_timeout or settings.circuit_breaker_timeout

        # Define expected exceptions
        if expected_exceptions is None:
            expected_exceptions = (
                AIProviderException,
                ConnectionError,
                OSError,
                TimeoutError,
            )

        # Create the circuit breaker
        self.breaker = CircuitBreaker(
            failure_threshold=self.failure_threshold,
            recovery_timeout=self.recovery_timeout,
            expected_exception=expected_exceptions,
        )

        logger.info(
            f"Circuit breaker initialized: "
            f"failure_threshold={self.failure_threshold}, "
            f"recovery_timeout={self.recovery_timeout}s"
        )

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Call a function with circuit breaker protection

        Args:
            func: Async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function call

        Raises:
            AIProviderException: When circuit is open
            Exception: Original exception if circuit is closed
        """
        try:
            # Use the circuit breaker to protect the call
            result = await self.breaker(func)(*args, **kwargs)

            logger.debug("Circuit breaker call succeeded")

            return result

        except Exception as e:
            # Check if this is a circuit breaker open exception
            if (
                "CircuitBreakerOpenException" in str(type(e))
                or "circuit" in str(e).lower()
            ):
                logger.error(
                    f"Circuit breaker is OPEN after {self.failure_threshold} failures. "
                    f"Will retry in {self.recovery_timeout}s"
                )
                raise AIProviderException(
                    message=f"AI provider circuit breaker is open. Too many failures ({self.failure_threshold}). "
                    f"Service will retry in {self.recovery_timeout} seconds.",
                    provider_name="unknown",
                    original_error=e,
                )
            else:
                logger.warning(f"Circuit breaker failure: {e}")
                raise

    @property
    def state(self) -> str:
        """Get current circuit breaker state"""
        # aiocircuitbreaker doesn't expose state directly, so we estimate
        if (
            hasattr(self.breaker, "_failure_count")
            and self.breaker._failure_count >= self.failure_threshold
        ):
            return "open"
        return "closed"

    @property
    def failure_count(self) -> int:
        """Get current failure count"""
        return getattr(self.breaker, "_failure_count", 0)

    def reset(self) -> None:
        """Manually reset the circuit breaker"""
        # aiocircuitbreaker doesn't have a direct reset, but we can work around it
        if hasattr(self.breaker, "_failure_count"):
            self.breaker._failure_count = 0
        if hasattr(self.breaker, "_failure_timestamp"):
            self.breaker._failure_timestamp = None
        logger.info("Circuit breaker manually reset")


# Global circuit breaker instance for AI providers
_ai_circuit_breaker: Optional[AIProviderCircuitBreaker] = None


def get_ai_circuit_breaker() -> AIProviderCircuitBreaker:
    """Get or create the global AI provider circuit breaker"""
    global _ai_circuit_breaker
    if _ai_circuit_breaker is None:
        _ai_circuit_breaker = AIProviderCircuitBreaker()
    return _ai_circuit_breaker


def reset_circuit_breaker() -> None:
    """Reset the global circuit breaker"""
    global _ai_circuit_breaker
    if _ai_circuit_breaker:
        _ai_circuit_breaker.reset()
