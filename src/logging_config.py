"""Centralized logging with structured format & sensitive data masking."""
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

def mask_sensitive(text: str, show_chars: int = 4) -> str:
    """Mask API keys/tokens while preserving first few chars for debugging."""
    if not text or len(text) < 8:
        return "***"
    return f"{text[:show_chars]}{'*' * (len(text) - show_chars)}"

def setup_logging(level: str = "INFO", log_dir: str = "logs", console_output: bool = True, file_output: bool = True) -> logging.Logger:
    """Configure app-wide structured logging with timestamped log files."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | RAG.%(name)s:%(funcName)-15s | %(message)s"
    
    # Create handlers list
    handlers = []
    
    # Console handler (if enabled)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt))
        handlers.append(console_handler)
    
    # File handler (if enabled)
    if file_output:
        # Create logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"app_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(fmt))
        handlers.append(file_handler)
    
    if not logging.getLogger().handlers:
        logging.basicConfig(level=log_level, handlers=handlers, force=True)
        
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
