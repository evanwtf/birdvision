"""
Extract date/time, GPS coordinates, and basic media metadata using ExifTool.
"""

import contextlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import exiftool

logger = logging.getLogger(__name__)

_et_lock = threading.RLock()
_et_instance: exiftool.ExifToolHelper | None = None


def _get_exiftool() -> exiftool.ExifToolHelper:
    global _et_instance
    with _et_lock:
        if _et_instance is None or not _et_instance.running:
            if _et_instance is not None:
                with contextlib.suppress(Exception):
                    _et_instance.terminate()
            _et_instance = exiftool.ExifToolHelper()
            _et_instance.run()
        return _et_instance


DATETIME_FORMATS = [
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
]


@dataclass
class VideoMetadata:
    recorded_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    focal_length: str | None = None

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def camera_info(self) -> str | None:
        """Human-readable camera description, e.g. 'Apple iPhone 15 Pro, 77mm'."""
        parts = []
        if self.camera_make and self.camera_model:
            # Avoid doubling up "Apple Apple iPhone" etc.
            if self.camera_model.lower().startswith(self.camera_make.lower()):
                parts.append(self.camera_model)
            else:
                parts.append(f"{self.camera_make} {self.camera_model}")
        elif self.camera_model:
            parts.append(self.camera_model)
        if self.focal_length:
            parts.append(self.focal_length)
        return ", ".join(parts) if parts else None

    @property
    def osm_url(self) -> str | None:
        if not self.has_gps:
            return None
        return (
            f"https://www.openstreetmap.org/?mlat={self.latitude:.6f}"
            f"&mlon={self.longitude:.6f}#map=15/{self.latitude:.6f}/{self.longitude:.6f}"
        )


@dataclass
class MediaMetadata:
    recorded_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    focal_length: str | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    video_codec: str | None = None
    metadata_error: str | None = None

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def camera_info(self) -> str | None:
        return VideoMetadata(
            recorded_at=self.recorded_at,
            latitude=self.latitude,
            longitude=self.longitude,
            camera_make=self.camera_make,
            camera_model=self.camera_model,
            focal_length=self.focal_length,
        ).camera_info


def inspect_media(path: str) -> MediaMetadata:
    meta = MediaMetadata()
    try:
        with _et_lock:
            tags = _get_exiftool().get_metadata(path)[0]

        # Date — prefer QuickTime CreateDate, fall back to other fields
        for key in ("QuickTime:CreateDate", "EXIF:DateTimeOriginal", "File:FileModifyDate"):
            raw = tags.get(key)
            if raw:
                for fmt in DATETIME_FORMATS:
                    try:
                        meta.recorded_at = datetime.strptime(raw[:19], fmt)
                        break
                    except ValueError:
                        continue
            if meta.recorded_at:
                break

        # GPS — ExifTool exposes composite Latitude/Longitude in decimal degrees
        lat = tags.get("Composite:GPSLatitude") or tags.get("EXIF:GPSLatitude")
        lon = tags.get("Composite:GPSLongitude") or tags.get("EXIF:GPSLongitude")
        if lat is not None and lon is not None:
            meta.latitude = float(lat)
            meta.longitude = float(lon)

        # Camera info
        meta.camera_make = tags.get("EXIF:Make") or tags.get("QuickTime:Make")
        meta.camera_model = tags.get("EXIF:Model") or tags.get("QuickTime:Model")
        fl = tags.get("EXIF:FocalLengthIn35mmFormat") or tags.get("EXIF:FocalLength")
        if fl is not None:
            # ExifTool may return a float (mm) or a string like "77 mm"
            try:
                meta.focal_length = f"{float(fl):.0f}mm"
            except (ValueError, TypeError):
                meta.focal_length = str(fl)

    except Exception as e:
        meta.metadata_error = str(e)
        logger.warning(f"Could not extract metadata from {Path(path).name}: {e}")

    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}:
        _fill_video_tech_metadata(path, meta)
    else:
        _fill_image_dimensions(path, meta)

    return meta


def _fill_video_tech_metadata(path: str, meta: MediaMetadata):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        meta.width = width or None
        meta.height = height or None
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        if fourcc:
            meta.video_codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        if fps > 0:
            meta.fps = round(fps, 3)
            if frame_count > 0:
                meta.duration_s = round(frame_count / fps, 3)
    except Exception as e:
        logger.warning(f"Could not read video properties from {Path(path).name}: {e}")
    finally:
        cap.release()


def _fill_image_dimensions(path: str, meta: MediaMetadata):
    try:
        frame = cv2.imread(path)
        if frame is not None:
            meta.height = int(frame.shape[0])
            meta.width = int(frame.shape[1])
    except Exception as e:
        logger.warning(f"Could not read image dimensions from {Path(path).name}: {e}")


def extract(video_path: str) -> VideoMetadata:
    detailed = inspect_media(video_path)
    return VideoMetadata(
        recorded_at=detailed.recorded_at,
        latitude=detailed.latitude,
        longitude=detailed.longitude,
        camera_make=detailed.camera_make,
        camera_model=detailed.camera_model,
        focal_length=detailed.focal_length,
    )
