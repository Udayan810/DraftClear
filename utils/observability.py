"""
Langfuse Observability Utility
Provides the @observe() decorator for tracing DraftClear agents
"""
import os
import logging
from langfuse.decorators import observe, langfuse_context

logger = logging.getLogger(__name__)

# Verify Langfuse configuration
public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
base_url = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

if not public_key or not secret_key:
    logger.warning("Langfuse credentials missing in environment. Observability will be disabled.")
else:
    logger.info(f"Langfuse initialized with base URL: {base_url}")

def update_trace_metadata(filename: str = None, iteration: int = None, page: int = None):
    """Update current trace with metadata for better filtering in Langfuse"""
    try:
        if filename:
            langfuse_context.update_current_trace(
                name=f"Process: {filename}",
                metadata={"filename": filename}
            )
        if iteration is not None:
            langfuse_context.update_current_trace(
                metadata={"max_iterations": iteration}
            )
        if page is not None:
            langfuse_context.update_current_trace(
                metadata={"page_number": page}
            )
    except Exception as e:
        logger.debug(f"Failed to update Langfuse metadata: {e}")

# Export the decorator
__all__ = ["observe", "langfuse_context", "update_trace_metadata"]
