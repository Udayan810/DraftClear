"""
Observability Utility (Disabled)
Langfuse dependencies have been fully removed. This provides dummy decorators.
"""
import logging
from functools import wraps

logger = logging.getLogger(__name__)
logger.info("Observability is disabled. Langfuse has been removed.")

class DummyContext:
    def update_current_trace(self, *args, **kwargs):
        pass

langfuse_context = DummyContext()

def observe(*args, **kwargs):
    """Dummy decorator that does nothing"""
    def decorator(func):
        @wraps(func)
        def wrapper(*f_args, **f_kwargs):
            return func(*f_args, **f_kwargs)
        return wrapper
    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    return decorator

def update_trace_metadata(filename: str = None, iteration: int = None, page: int = None):
    """Dummy function"""
    pass

__all__ = ["observe", "langfuse_context", "update_trace_metadata"]
