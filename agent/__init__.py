def __getattr__(name: str):
    """Lazy import to avoid circular dependency with tools module."""
    if name == "SupportAgent":
        from .core import SupportAgent
        return SupportAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SupportAgent"]
