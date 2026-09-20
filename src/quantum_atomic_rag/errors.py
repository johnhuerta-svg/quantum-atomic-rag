"""Application errors with stable categories for callers and observability."""


class QuantumAtomicRAGError(Exception):
    """Base exception for expected application failures."""


class ModelTransportError(QuantumAtomicRAGError):
    """The model endpoint could not be reached or returned a retryable error."""


class ModelResponseError(QuantumAtomicRAGError):
    """The model returned an unusable response."""


class ModelValidationError(QuantumAtomicRAGError):
    """The model response did not satisfy the requested schema."""
