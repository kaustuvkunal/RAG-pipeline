"""Centralized logging with structured format & sensitive data masking."""
import logging
import sys
from contextlib import contextmanager

def mask_sensitive(text: str, show_chars: int = 4) -> str:
    """Mask API keys/tokens while preserving first few chars for debugging."""
    if not text or len(text) < 8:
        return "***"
    return f"{text[:show_chars]}{'*' * (len(text) - show_chars)}"

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure app-wide structured logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | RAG.%(name)s:%(funcName)-15s | %(message)s"
    
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logging.basicConfig(level=log_level, handlers=[handler], force=True)
        
    return logging.getLogger("RAG")

def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger instance."""
    return logging.getLogger(f"RAG.{name}")

@contextmanager
def log_sequence(step: str):
    """Context manager for tracking invocation sequences with clear indentation."""
    logger = get_logger("sequence")
    logger.info(f"▶️ [STEP] {step}")
    yield
