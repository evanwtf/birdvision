"""Framebuffer display overlay for the Pi Touch Display 2.

Writes the live camera feed plus a closed-caption-style species label
directly to ``/dev/fb0``. No X11/Wayland.

Ownership model:
    - Pipeline thread calls :meth:`post` with the latest frame + state.
    - A background daemon thread owned by :class:`DisplayOverlay` pulls the
      most recent posted frame and writes to the framebuffer at ~30 Hz.
      The fb write never blocks detection/classification.

Verified facts carried from the display test in #89 (see pi/display_test/):
    - Framebuffer is 32bpp XRGB8888; use FBIOGET_{V,F}SCREENINFO ioctls
      for dimensions/stride, not sysfs virtual_size.
    - Caption is drawn in the pre-rotation (landscape) frame so that text
      sits at the visual bottom regardless of rotation angle.
    - Device is owned by group ``video`` (already covered by group_add for
      /dev/video0), so no extra group entry is needed.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import mmap
import struct
import threading
import time
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FBIOGET_VSCREENINFO = 0x4600
FBIOGET_FSCREENINFO = 0x4602

_CV_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAPTION_SCALE = 1.1
_CAPTION_THICK = 2
_CAPTION_MARGIN = 12
_BOX_COLOR = (0, 255, 0)
_BOX_THICK = 2


class DisplayOverlay:
    """Live-feed + caption framebuffer writer.

    Args:
        device: framebuffer device path (e.g. ``/dev/fb0``)
        rotate: rotation in degrees clockwise (0/90/180/270)
        show_fps: draw small FPS counter in a corner
        show_boxes: draw detection bounding boxes
        target_fps: max fb writes per second from the background thread
    """

    def __init__(
        self,
        device: str = "/dev/fb0",
        rotate: int = 0,
        show_fps: bool = False,
        show_boxes: bool = True,
        target_fps: float = 30.0,
    ) -> None:
        self._enabled = False
        self._device = device
        self._rotate = rotate if rotate in _CV_ROTATIONS or rotate == 0 else 0
        self._show_fps = show_fps
        self._show_boxes = show_boxes
        self._min_dt = 1.0 / max(1.0, target_fps)

        self._mm: mmap.mmap | None = None
        self._fd = None
        self._fb_w = 0
        self._fb_h = 0
        self._bpp = 32

        self._lock = threading.Lock()
        self._state: tuple | None = None  # (frame, bboxes, caption, fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        try:
            self._open()
            self._enabled = True
        except (OSError, PermissionError, ValueError) as exc:
            logger.warning("Display overlay disabled — cannot open %s: %s", device, exc)
            return

        self._thread = threading.Thread(target=self._run, name="display-overlay", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Framebuffer setup
    # ------------------------------------------------------------------

    def _open(self) -> None:
        self._fd = open(self._device, "r+b")  # noqa: SIM115  long-lived framebuffer handle

        vinfo = fcntl.ioctl(self._fd, FBIOGET_VSCREENINFO, b"\x00" * 160)
        xres, yres, _xv, _yv, _xo, _yo, bpp = struct.unpack_from("7I", vinfo, 0)

        finfo = fcntl.ioctl(self._fd, FBIOGET_FSCREENINFO, b"\x00" * 80)
        stride = struct.unpack_from("I", finfo, 48)[0]
        if stride == 0:
            stride = xres * (bpp // 8)

        if bpp not in (16, 32):
            raise ValueError(f"Unsupported framebuffer depth: {bpp}bpp")

        fb_size = stride * yres
        self._mm = mmap.mmap(self._fd.fileno(), fb_size)
        self._fb_w = xres
        self._fb_h = yres
        self._bpp = bpp

        logger.info(
            "Display overlay: %s %dx%d %dbpp rotate=%d°",
            self._device,
            xres,
            yres,
            bpp,
            self._rotate,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post(
        self,
        frame: np.ndarray,
        bboxes: Sequence[Iterable[float]] = (),
        top_species: str | None = None,
        top_score: float | None = None,
        fps: float = 0.0,
    ) -> None:
        """Hand the newest frame + state to the background writer.

        Cheap: copies lightweight references under a short lock. The
        frame is NOT copied — the caller must not mutate it after posting.
        """
        if not self._enabled:
            return
        caption = None
        if top_species and top_score is not None:
            caption = f"{top_species}  {top_score:.0%}"
        with self._lock:
            self._state = (frame, list(bboxes), caption, fps)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._mm is not None:
            with contextlib.suppress(Exception):
                self._mm.close()
            self._mm = None
        if self._fd is not None:
            with contextlib.suppress(Exception):
                self._fd.close()
            self._fd = None
        self._enabled = False

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            state = None
            with self._lock:
                if self._state is not None:
                    state = self._state
            if state is not None:
                try:
                    self._render(*state)
                except Exception:
                    logger.exception("Display overlay render failed")
            dt = time.monotonic() - t0
            remaining = self._min_dt - dt
            if remaining > 0:
                self._stop.wait(remaining)

    def _render(
        self,
        frame: np.ndarray,
        bboxes: Sequence[Iterable[float]],
        caption: str | None,
        fps: float,
    ) -> None:
        cv_rotate = _CV_ROTATIONS.get(self._rotate)
        if cv_rotate in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            resize_w, resize_h = self._fb_h, self._fb_w
        else:
            resize_w, resize_h = self._fb_w, self._fb_h

        src_h, src_w = frame.shape[:2]
        display = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        if self._show_boxes and bboxes and src_w > 0 and src_h > 0:
            sx = resize_w / src_w
            sy = resize_h / src_h
            for bb in bboxes:
                x1, y1, x2, y2 = (float(v) for v in bb)
                cv2.rectangle(
                    display,
                    (int(x1 * sx), int(y1 * sy)),
                    (int(x2 * sx), int(y2 * sy)),
                    _BOX_COLOR,
                    _BOX_THICK,
                )

        _draw_caption(display, caption if caption else "No Bird Detected")
        if self._show_fps:
            _draw_fps(display, fps)

        if cv_rotate is not None:
            display = cv2.rotate(display, cv_rotate)

        buf = _frame_to_fb(display, self._bpp)
        if self._mm is None:
            return
        self._mm.seek(0)
        self._mm.write(buf)


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------


def _draw_caption(frame: np.ndarray, label: str) -> None:
    (tw, th), baseline = cv2.getTextSize(label, _FONT, _CAPTION_SCALE, _CAPTION_THICK)
    h, w = frame.shape[:2]
    bar_h = th + baseline + _CAPTION_MARGIN * 2
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (_CAPTION_MARGIN, h - _CAPTION_MARGIN - baseline),
        _FONT,
        _CAPTION_SCALE,
        (255, 255, 255),
        _CAPTION_THICK,
        cv2.LINE_AA,
    )


def _draw_fps(frame: np.ndarray, fps: float) -> None:
    label = f"{fps:.0f} fps"
    (fw, fh), _ = cv2.getTextSize(label, _FONT, 0.6, 1)
    cv2.rectangle(frame, (0, 0), (fw + 12, fh + 10), (0, 0, 0), -1)
    cv2.putText(frame, label, (6, fh + 5), _FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)


def _frame_to_fb(frame: np.ndarray, bpp: int) -> bytes:
    if bpp == 32:
        h, w = frame.shape[:2]
        x = np.zeros((h, w, 1), dtype=np.uint8)
        return np.concatenate([frame, x], axis=2).tobytes()
    if bpp == 16:
        rgb = frame[:, :, ::-1].astype(np.uint16)
        r5 = (rgb[:, :, 0] >> 3).astype(np.uint16)
        g6 = (rgb[:, :, 1] >> 2).astype(np.uint16)
        b5 = (rgb[:, :, 2] >> 3).astype(np.uint16)
        return ((r5 << 11) | (g6 << 5) | b5).astype(np.uint16).tobytes()
    raise ValueError(f"Unsupported framebuffer depth: {bpp}bpp")
