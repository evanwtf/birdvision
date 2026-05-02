"""File video frame source for the Pi real-time pipeline.

Reads a video file via cv2.VideoCapture and yields (frame_no, bgr) pairs —
same iterator interface as V4L2FrameSource and WebSocketFrameSource.

Useful for debugging the Hailo pipeline with known test footage, eliminating
phone/WebSocket/TV-screen variables.
"""

import logging
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FileFrameSource:
    """Reads frames from a video file and yields (frame_no, bgr) pairs.

    Args:
        file_path:   Path to the video file
        loop:        If True, restart from beginning when file ends (default False)
        fps_limit:   Max frames per second to yield; None = no limit (realtime).
                     Set to the video's native FPS for realtime playback, or
                     None / 0 to process as fast as possible.
    """

    def __init__(
        self,
        file_path: str,
        loop: bool = False,
        fps_limit: Optional[float] = None,
    ):
        self._file_path = Path(file_path)
        self._loop = loop
        self._fps_limit = fps_limit
        self._stop = False

        if not self._file_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self._file_path}")

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_number, bgr_frame) from the video file."""
        frame_no = 0

        while not self._stop:
            cap = cv2.VideoCapture(str(self._file_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self._file_path}")

            native_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                "Opened %s  %dx%d  %.2f fps  %d frames",
                self._file_path.name, width, height, native_fps, total_frames,
            )

            min_interval = 1.0 / self._fps_limit if self._fps_limit else 0.0
            t_last = time.monotonic()

            try:
                while not self._stop:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    if min_interval > 0:
                        now = time.monotonic()
                        sleep_for = min_interval - (now - t_last)
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        t_last = time.monotonic()

                    yield frame_no, frame
                    frame_no += 1
            finally:
                cap.release()

            if not self._loop:
                break
            logger.info("Looping video file from beginning")

        logger.info("FileFrameSource finished after %d frames", frame_no)

    def stop(self) -> None:
        """Signal the frames() iterator to exit cleanly."""
        self._stop = True
