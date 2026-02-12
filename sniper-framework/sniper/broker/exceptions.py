"""Broker exception hierarchy."""


class BrokerError(Exception):
    """Base broker error."""


class BrokerConnectionError(BrokerError):
    """Connection to broker failed."""


class OrderRejectedError(BrokerError):
    """Order was rejected by broker."""


class InsufficientFundsError(BrokerError):
    """Insufficient funds for order."""


class RateLimitExceeded(BrokerError):
    """Rate limit exceeded."""


class InvalidOrderError(BrokerError):
    """Invalid order parameters."""
