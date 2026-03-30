#!/usr/bin/env python3
"""
BirdVision web server entry point.

Usage:
    uv run scripts/serve.py
    uv run scripts/serve.py --config config.yaml --port 3587
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("birdvision.serve")


def main():
    parser = argparse.ArgumentParser(description="BirdVision web server")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3587)
    args = parser.parse_args()

    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config not found: {args.config} — using defaults")

    templates_dir = str(Path(__file__).parent.parent / "templates")

    from src.webapp import create_app
    app = create_app(config, templates_dir=templates_dir, config_path=config_path)

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "datefmt": "%H:%M:%S",
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }

    logger.info(f"Starting BirdVision web server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_config=log_config)


if __name__ == "__main__":
    main()
