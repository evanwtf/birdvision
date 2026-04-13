"""
Real-time bird identification pipeline for Raspberry Pi 5 + Hailo-8.

Entry point for the Pi streaming pipeline. Reads from V4L2 capture device,
runs Hailo-accelerated detection + classification, and logs top species.

Usage:
    uv run scripts/realtime_identify.py --config config.pi.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdVision real-time Pi pipeline")
    parser.add_argument("--config", required=True, help="Path to config.pi.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = yaml.safe_load(config_path.read_text())
    logger.info("Loaded config: %s", config_path)

    from src.realtime_pipeline import RealtimePipeline

    pipeline = RealtimePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
