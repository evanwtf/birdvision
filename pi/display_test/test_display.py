"""Framebuffer display test for Raspberry Pi Touch Display 2.

Captures live video from /dev/video0, scales it to the framebuffer
resolution, and overlays a rotating fake bird-ID caption at the bottom.

No Hailo, no ML — pure plumbing test for the display pipeline.

Usage:
    python test_display.py [--device /dev/video0] [--fb /dev/fb0]
                           [--rotate {0,90,180,270}] [--debug]
"""

import argparse
import fcntl
import logging
import mmap
import random
import signal
import struct
import sys
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FAKE_SPECIES = [
    ("Mourning Dove", 0.91),
    ("American Robin", 0.87),
    ("House Sparrow", 0.73),
    ("Blue Jay", 0.95),
    ("Northern Cardinal", 0.88),
    ("Black-capped Chickadee", 0.79),
    ("Downy Woodpecker", 0.82),
    ("White-breasted Nuthatch", 0.68),
    ("Song Sparrow", 0.71),
    ("European Starling", 0.84),
    ("American Goldfinch", 0.90),
    ("Red-tailed Hawk", 0.76),
    ("Cooper's Hawk", 0.65),
    ("Dark-eyed Junco", 0.83),
    ("Tufted Titmouse", 0.78),
]

CAPTION_INTERVAL = 3.0
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 1.1
FONT_THICKNESS   = 2
CAPTION_MARGIN   = 12

# Linux framebuffer ioctls
FBIOGET_VSCREENINFO = 0x4600
FBIOGET_FSCREENINFO = 0x4602

CV_ROTATIONS = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def open_framebuffer(device: str):
    """Open /dev/fbN and return (mmap, fd, width, height, bpp, stride).

    Uses FBIOGET_VSCREENINFO / FBIOGET_FSCREENINFO ioctls for reliable
    dimensions rather than guessing from sysfs virtual_size.
    """
    fd = open(device, "r+b")

    # struct fb_var_screeninfo — first fields we care about:
    #   u32 xres, yres, xres_virtual, yres_virtual, xoffset, yoffset, bits_per_pixel
    vinfo = fcntl.ioctl(fd, FBIOGET_VSCREENINFO, b"\x00" * 160)
    xres, yres, _xv, _yv, _xo, _yo, bpp = struct.unpack_from("7I", vinfo, 0)

    # struct fb_fix_screeninfo — line_length lives at offset 48 on aarch64 64-bit:
    #   char id[16], ulong smem_start (8B), u32 smem_len, type, type_aux,
    #   visual, u16 xpanstep, ypanstep, ywrapstep, <pad>, u32 line_length
    finfo = fcntl.ioctl(fd, FBIOGET_FSCREENINFO, b"\x00" * 80)
    stride = struct.unpack_from("I", finfo, 48)[0]

    if stride == 0:
        stride = xres * (bpp // 8)

    fb_size = stride * yres
    logger.info(
        "Framebuffer %s: %dx%d  %dbpp  stride=%d  mmap_size=%d",
        device, xres, yres, bpp, stride, fb_size,
    )

    mm = mmap.mmap(fd.fileno(), fb_size)
    return mm, fd, xres, yres, bpp, stride


def frame_to_fb(frame: np.ndarray, bpp: int) -> bytes:
    if bpp == 32:
        h, w = frame.shape[:2]
        x = np.zeros((h, w, 1), dtype=np.uint8)
        return np.concatenate([frame, x], axis=2).tobytes()
    elif bpp == 16:
        rgb = frame[:, :, ::-1].astype(np.uint16)
        r5 = (rgb[:, :, 0] >> 3).astype(np.uint16)
        g6 = (rgb[:, :, 1] >> 2).astype(np.uint16)
        b5 = (rgb[:, :, 2] >> 3).astype(np.uint16)
        return ((r5 << 11) | (g6 << 5) | b5).astype(np.uint16).tobytes()
    else:
        raise ValueError(f"Unsupported framebuffer depth: {bpp}bpp")


def draw_caption(frame: np.ndarray, species: str, confidence: float) -> None:
    label = f"{species}  {confidence:.0%}"
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    h, w = frame.shape[:2]
    bar_h = th + baseline + CAPTION_MARGIN * 2
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, label, (CAPTION_MARGIN, h - CAPTION_MARGIN - baseline),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    label = f"{fps:.0f} fps"
    (fw, fh), _ = cv2.getTextSize(label, FONT, 0.6, 1)
    cv2.rectangle(frame, (0, 0), (fw + 12, fh + 10), (0, 0, 0), -1)
    cv2.putText(frame, label, (6, fh + 5), FONT, 0.6,
                (180, 180, 180), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--fb",     default="/dev/fb0")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="Rotate frame before display (degrees clockwise)")
    ap.add_argument("--debug",  action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        mm, fb_fd, fb_w, fb_h, bpp, stride = open_framebuffer(args.fb)
    except (OSError, PermissionError) as e:
        logger.error("Cannot open framebuffer %s: %s", args.fb, e)
        sys.exit(1)

    logger.info("Opening capture device %s", args.device)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        logger.error("Cannot open %s", args.device)
        sys.exit(1)
    logger.info(
        "Capture: %.0fx%.0f @ %.0f fps  |  Display: %dx%d %dbpp  rotate=%d°",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FPS),
        fb_w, fb_h, bpp, args.rotate,
    )

    stop = False

    def _sig(signum, _frame):
        nonlocal stop
        logger.info("Signal %d — stopping", signum)
        stop = True

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    cv_rotate = CV_ROTATIONS.get(args.rotate)
    species, confidence = random.choice(FAKE_SPECIES)
    t_caption   = time.monotonic()
    t_fps       = time.monotonic()
    frame_count = 0
    fps         = 0.0

    logger.info("Running — Ctrl-C to stop")

    while not stop:
        ok, frame = cap.read()
        if not ok:
            logger.warning("Frame grab failed — retrying")
            time.sleep(0.01)
            continue

        frame_count += 1
        now = time.monotonic()

        if now - t_caption >= CAPTION_INTERVAL:
            species, confidence = random.choice(FAKE_SPECIES)
            logger.info("Caption → %s  %.0f%%", species, confidence * 100)
            t_caption = now

        elapsed = now - t_fps
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            t_fps = now
            logger.debug("fps=%.1f", fps)

        # Resize to the dimensions that will become fb_w × fb_h *after* rotation.
        # Draw caption / FPS in the natural (pre-rotation) orientation so "bottom"
        # means visual bottom on the physical display — regardless of how many
        # degrees the display rotates the fb content.
        if cv_rotate in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            resize_w, resize_h = fb_h, fb_w
        else:
            resize_w, resize_h = fb_w, fb_h

        display = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        draw_caption(display, species, confidence)
        draw_fps(display, fps)

        if cv_rotate is not None:
            display = cv2.rotate(display, cv_rotate)

        mm.seek(0)
        mm.write(frame_to_fb(display, bpp))

    cap.release()
    mm.close()
    fb_fd.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
