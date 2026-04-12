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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdVision real-time Pi pipeline")
    parser.add_argument("--config", required=True, help="Path to config.pi.yaml")
    args = parser.parse_args()

    logger.info("BirdVision real-time pipeline starting (config: %s)", args.config)
    logger.warning("Pipeline not yet implemented — see issues #76, #78, #79")
    sys.exit(1)


if __name__ == "__main__":
    main()
