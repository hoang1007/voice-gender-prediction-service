import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_configured = False


class _RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    fmt = "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")
    rid_filter = _RequestIDFilter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(rid_filter)
    stdout_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [stdout_handler]

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"gender-{os.getpid()}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.addFilter(rid_filter)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
