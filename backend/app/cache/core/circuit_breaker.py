"""
Circuit Breaker for Redis operations.

Provides protection against cascading failures by monitoring Redis operation
failures and temporarily disabling Redis access when failure threshold is reached.
"""
import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional

from redis.exceptions import ConnectionError, TimeoutError
from app.exceptions import CircuitBreakerError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker for Redis operations.

    Monitors Redis operation failures and opens the circuit when failure
    threshold is reached, preventing cascading failures.

    States:
    - CLOSED: Normal operation, all calls pass through
    - OPEN: Circuit is open, all calls fail fast
    - HALF_OPEN: Testing recovery, limited calls allowed
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exceptions: tuple = None,
        success_threshold: int = 3
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exceptions: Exceptions that count as failures
            success_threshold: Successful calls needed to close circuit from half-open
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        # Default exceptions that indicate Redis failures
        self.expected_exceptions = expected_exceptions or (
            ConnectionError,
            TimeoutError,
            OSError,
            Exception  # Catch-all for Redis-related errors
        )

        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

        # Metrics
        self._metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_opens': 0,
            'circuit_closes': 0,
            'fast_failures': 0,
            'last_state_change': time.time()
        }

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: When circuit is open
            Original exception: When function fails
        """
        self._metrics['total_calls'] += 1

        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self._metrics['fast_failures'] += 1
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. "
                    f"Failure count: {self.failure_count}, "
                    f"Last failure: {self.last_failure_time}"
                )

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._on_success()
            return result

        except self.expected_exceptions as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        logger.info("Circuit breaker transitioning to HALF_OPEN state")
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self._metrics['last_state_change'] = time.time()

    def _on_success(self):
        """Handle successful operation."""
        self._metrics['successful_calls'] += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.success_threshold:
                self._close_circuit()

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed operation."""
        self._metrics['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state immediately opens circuit
            self._open_circuit()

        elif (self.state == CircuitState.CLOSED and
              self.failure_count >= self.failure_threshold):
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit."""
        logger.warning(
            f"Circuit breaker OPENING. "
            f"Failure count: {self.failure_count}, "
            f"Threshold: {self.failure_threshold}"
        )
        self.state = CircuitState.OPEN
        self._metrics['circuit_opens'] += 1
        self._metrics['last_state_change'] = time.time()

    def _close_circuit(self):
        """Close the circuit."""
        logger.info("Circuit breaker CLOSING - service recovered")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._metrics['circuit_closes'] += 1
        self._metrics['last_state_change'] = time.time()

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        return {
            **self._metrics,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'failure_threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout,
            'last_failure_time': self.last_failure_time,
            'time_since_last_failure': (
                time.time() - self.last_failure_time
                if self.last_failure_time else None
            )
        }

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        logger.info("Circuit breaker manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._metrics['last_state_change'] = time.time()

    def is_healthy(self) -> bool:
        """Check if circuit breaker is in healthy state."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
