#!/usr/bin/env python3
"""WebSocket test client — streams a video file to the Pi sidecar server.

Simulates a phone connection for development without needing a physical
device.  Sends JPEG-compressed frames over WebSocket and prints detection
results as they arrive.

Usage:
    uv run --no-project --with websockets,opencv-python-headless \\
        scripts/ws_test_client.py test_video.mp4 \\
        --server ws://pi-ip:8765/ws --fps 5

    # With fake GPS coordinates:
    uv run --no-project --with websockets,opencv-python-headless \\
        scripts/ws_test_client.py test_video.mp4 --lat 40.7 --lon -73.5
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time

import cv2
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def stream_video(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Could not open video: %s", args.video)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        "Video: %s  %dx%d  %.1f fps  %d frames",
        args.video,
        width,
        height,
        src_fps,
        total_frames,
    )

    interval = 1.0 / args.fps
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(args.quality * 100)]

    frames_sent = 0
    results_received = 0
    latencies: list[float] = []
    send_times: dict[int, float] = {}
    stop = False

    def handle_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    t_start = time.monotonic()

    try:
        async with websockets.connect(args.server, max_size=10 * 1024 * 1024) as ws:
            logger.info("Connected to %s", args.server)

            async def receive_results():
                nonlocal results_received
                try:
                    async for msg in ws:
                        result = json.loads(msg)
                        results_received += 1
                        t_recv = time.monotonic()
                        fid = result.get("frame_id")
                        if fid is not None and fid in send_times:
                            latencies.append(t_recv - send_times.pop(fid))

                        if not args.quiet:
                            dets = result.get("detections", [])
                            if dets:
                                for d in dets:
                                    species = d.get("species") or "?"
                                    score = d.get("species_score") or 0
                                    logger.info(
                                        "  frame=%s  track=%s  %s  %.3f",
                                        result.get("frame_id"),
                                        d.get("track_id"),
                                        species,
                                        score,
                                    )
                            else:
                                logger.info(
                                    "  frame=%s  no detections  fps=%.1f",
                                    result.get("frame_id"),
                                    result.get("fps", 0),
                                )
                except websockets.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receive_results())

            while not stop:
                ok, frame = cap.read()
                if not ok:
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                if args.width and frame.shape[1] != args.width:
                    scale = args.width / frame.shape[1]
                    new_h = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (args.width, new_h))

                ok, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                metadata = {"frame_id": frames_sent}
                if args.lat is not None and args.lon is not None:
                    metadata["lat"] = args.lat
                    metadata["lon"] = args.lon

                send_times[frames_sent] = time.monotonic()
                await ws.send(json.dumps(metadata))
                await ws.send(jpeg_buf.tobytes())
                frames_sent += 1

                await asyncio.sleep(interval)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    except websockets.ConnectionClosed:
        logger.warning("Server closed connection")
    except ConnectionRefusedError:
        logger.error("Could not connect to %s", args.server)
        sys.exit(1)
    finally:
        cap.release()

    elapsed = time.monotonic() - t_start
    logger.info("--- Summary ---")
    logger.info("  Frames sent:      %d", frames_sent)
    logger.info("  Results received:  %d", results_received)
    logger.info("  Elapsed:           %.1fs", elapsed)
    if frames_sent > 0:
        logger.info("  Send rate:         %.1f fps", frames_sent / elapsed)
    if latencies:
        avg_ms = sum(latencies) / len(latencies) * 1000
        min_ms = min(latencies) * 1000
        max_ms = max(latencies) * 1000
        logger.info(
            "  Round-trip:        avg=%.0fms  min=%.0fms  max=%.0fms  (%d samples)",
            avg_ms,
            min_ms,
            max_ms,
            len(latencies),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a video file to the BirdVision WebSocket server",
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--server",
        default="ws://localhost:8765/ws",
        help="WebSocket URL (default: ws://localhost:8765/ws)",
    )
    parser.add_argument("--fps", type=float, default=5.0, help="Send rate (default: 5)")
    parser.add_argument(
        "--quality",
        type=float,
        default=0.65,
        help="JPEG quality 0.0-1.0 (default: 0.65)",
    )
    parser.add_argument("--width", type=int, default=None, help="Resize width (default: no resize)")
    parser.add_argument("--lat", type=float, default=None, help="GPS latitude")
    parser.add_argument("--lon", type=float, default=None, help="GPS longitude")
    parser.add_argument("--loop", action="store_true", help="Loop video continuously")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    args = parser.parse_args()

    asyncio.run(stream_video(args))


if __name__ == "__main__":
    main()
