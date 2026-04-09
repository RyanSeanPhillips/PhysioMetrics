"""
Structured startup logging for PhysioMetrics.

Writes timestamped phase markers to %APPDATA%/PhysioMetrics/startup.log
so field debugging is possible even from console=False builds.

Usage:
    from core.startup_logger import setup_startup_logging, log_phase

    logger = setup_startup_logging(console=True)
    with log_phase("MainWindow init"):
        w = MainWindow()
"""

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from logging.handlers import RotatingFileHandler


_start_time = None
_logger = None


def setup_startup_logging(console: bool = False, log_dir: Path = None) -> logging.Logger:
    """
    Configure startup logger with file + optional console output.

    Args:
        console: If True, also print to stdout (for --console mode)
        log_dir: Directory for startup.log. Defaults to %APPDATA%/PhysioMetrics/

    Returns:
        Configured logger instance
    """
    global _start_time, _logger
    _start_time = time.perf_counter()

    if log_dir is None:
        if sys.platform == 'win32':
            log_dir = Path(__import__('os').environ.get('APPDATA', '')) / 'PhysioMetrics'
        else:
            log_dir = Path.home() / '.config' / 'PhysioMetrics'
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('physiometrics.startup')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('[%(elapsed)s] %(message)s')

    # File handler — rotating, keeps last 3 launches (1MB each)
    fh = RotatingFileHandler(
        log_dir / 'startup.log',
        maxBytes=1_000_000,
        backupCount=2,
        encoding='utf-8',
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.addFilter(_ElapsedFilter())
    logger.addHandler(fh)

    # Console handler (only if --console or running from source)
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        ch.addFilter(_ElapsedFilter())
        logger.addHandler(ch)

    logger.info(f"PhysioMetrics startup — Python {sys.version.split()[0]}, platform={sys.platform}")
    _logger = logger
    return logger


class _ElapsedFilter(logging.Filter):
    """Injects elapsed time since startup into log records."""
    def filter(self, record):
        if _start_time is not None:
            elapsed = time.perf_counter() - _start_time
            record.elapsed = f"{elapsed:6.2f}s"
        else:
            record.elapsed = "  ?.??s"
        return True


@contextmanager
def log_phase(name: str):
    """
    Context manager that logs the start and duration of a startup phase.

    Usage:
        with log_phase("ML model loading"):
            load_models()
        # Logs: "[  1.23s] ML model loading... done (0.45s)"
    """
    logger = _logger or logging.getLogger('physiometrics.startup')
    logger.info(f"{name}...")
    t0 = time.perf_counter()
    try:
        yield
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"{name}... FAILED ({elapsed:.2f}s): {e}")
        raise
    else:
        elapsed = time.perf_counter() - t0
        logger.info(f"{name}... done ({elapsed:.2f}s)")
