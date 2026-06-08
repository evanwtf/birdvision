"""V4L2 live frame source for the Pi real-time pipeline.

Wraps cv2.VideoCapture with the CAP_V4L2 backend to read from the
Elgato Cam Link 4K (or any V4L2 capture device).

Usage:
    src = V4L2FrameSource(device="/dev/video0", width=1920, height=1080, fps=60)
    for frame_no, bgr in src.frames():
        process(bgr)

Cam Link 4K supported formats on Pi (from v4l2-ctl --list-formats-ext):
    YUYV 4:2:2, NV12, YU12 — all at 1920×1080 @ 59.94 fps. No MJPEG.
"""

import logging
import signal
import time
from typing import Iterator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class V4L2FrameSource:
    """Reads frames from a V4L2 capture device and yields (frame_no, bgr) pairs.

    Args:
        device:   V4L2 device path, e.g. "/dev/video0"
        width:    Capture width in pixels (default 1920)
        height:   Capture height in pixels (default 1080)
        fps:      Requested frame rate (default 60)
        fourcc:   FourCC format string (default "YUYV" — Cam Link 4K native)
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        fourcc: str = "YUYV",
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self._stop = False

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_number, bgr_frame) until stop() is called or an error occurs."""
        cap = self._open()
        frame_no = 0
        empty_streak = 0
        max_empty = 30  # consecutive empty reads before warning

        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok or frame is None:
                    empty_streak += 1
                    if empty_streak == max_empty:
                        logger.warning(
                            "%d consecutive empty reads from %s — camera may have disconnected",
                            empty_streak,
                            self.device,
                        )
                    time.sleep(0.01)
                    continue

                if empty_streak:
                    logger.info("Camera read resumed after %d empty reads", empty_streak)
                    empty_streak = 0

                yield frame_no, frame
                frame_no += 1

        finally:
            cap.release()
            logger.info("V4L2FrameSource released %s after %d frames", self.device, frame_no)

    def stop(self) -> None:
        """Signal the frames() iterator to exit cleanly after the current frame."""
        self._stop = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        logger.info(
            "Opening capture device %s  %dx%d @ %d fps  fourcc=%s",
            self.device,
            self.width,
            self.height,
            self.fps,
            self.fourcc,
        )
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open capture device: {self.device}")

        # Request YUYV (or caller-specified format); cv2 converts to BGR
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc.ljust(4))
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read back what the driver actually gave us
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("Capture device opened: %dx%d @ %.2f fps", actual_w, actual_h, actual_fps)

        if actual_w != self.width or actual_h != self.height:
            logger.warning(
                "Requested %dx%d but got %dx%d — check device capabilities",
                self.width,
                self.height,
                actual_w,
                actual_h,
            )

        return cap
