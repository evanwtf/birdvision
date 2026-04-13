"""Helpers for optional file-backed logging in long-running scripts."""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
LOG_LEVEL = logging.INFO
DEFAULT_LOG_ENV = "BIRDVISION_SCRIPT_LOG_DIR"

_OPEN_LOG_STREAMS: list[io.TextIOBase] = []


class TeeStream(io.TextIOBase):
    """Mirror writes to the terminal and a log file."""

    def __init__(self, primary: io.TextIOBase, mirror: io.TextIOBase) -> None:
        self.primary = primary
        self.mirror = mirror

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.mirror.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.mirror.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write stdout/stderr and logger output to this file as well as the terminal",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help=(
            "Create a timestamped log file in this directory. "
            f"If omitted, {DEFAULT_LOG_ENV} is used when set."
        ),
    )


def resolve_log_file(
    script_stem: str,
    log_file: Path | None,
    log_dir: Path | None,
) -> Path | None:
    if log_file and log_dir:
        raise SystemExit("Use only one of --log-file or --log-dir")

    if log_file:
        return log_file

    if log_dir is None:
        env_value = os.environ.get(DEFAULT_LOG_ENV)
        if env_value:
            log_dir = Path(env_value)

    if log_dir is None:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{script_stem}_{timestamp}.log"


def configure_logging(
    script_stem: str,
    *,
    log_file: Path | None = None,
    log_dir: Path | None = None,
) -> Path | None:
    resolved = resolve_log_file(script_stem, log_file, log_dir)

    if resolved is not None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        sink = resolved.open("a", encoding="utf-8", buffering=1)
        _OPEN_LOG_STREAMS.append(sink)
        sys.stdout = TeeStream(sys.stdout, sink)
        sys.stderr = TeeStream(sys.stderr, sink)

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout, force=True)
    return resolved


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def estimate_remaining(elapsed_seconds: float, completed: int, total: int) -> float | None:
    if completed <= 0 or total <= completed:
        return 0.0 if total <= completed else None
    avg_seconds = elapsed_seconds / completed
    return avg_seconds * (total - completed)
