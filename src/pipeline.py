import bisect
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .classifier import BirdClassifier
from .detector import BirdDetector
from .metadata import MetadataPrior
from .tracker import BirdTracker, Track
from .video_metadata import extract as extract_media_metadata

logger = logging.getLogger(__name__)

SWAN_SPECIES = {"Mute Swan", "Tundra Swan"}


def resolution_warning_text(
    *,
    media_type: str,
    width: Optional[int],
    height: Optional[int],
) -> Optional[str]:
    if width is None or height is None:
        return None
    long_edge = max(width, height)
    short_edge = min(width, height)
    if media_type == "video" and (long_edge < 1280 or short_edge < 720):
        return "Low-resolution video can reduce bird detection recall, especially for small or distant birds."
    if media_type == "image" and (long_edge < 1600 or short_edge < 900):
        return "Low-resolution photos can reduce bird detection recall, especially for small or distant birds."
    return None


def compact_path(path: str | Path, *, keep_parts: int = 2) -> str:
    path_obj = Path(path)
    parts = path_obj.parts
    if len(parts) <= keep_parts:
        return str(path_obj)

    tail = Path(*parts[-keep_parts:])
    prefix = "/" if path_obj.is_absolute() else ""
    return f"{prefix}{{...}}/{tail}"


class BirdIdentificationPipeline:
    def __init__(self, config: dict):
        det = config.get("detector", {})
        cls = config.get("classifier", {})
        trk = config.get("tracker", {})
        meta = config.get("metadata", {})
        sp = config.get("species", {})

        self.detector = BirdDetector(
            model_path=det.get("model", "yolov8n.pt"),
            confidence=det.get("confidence", 0.3),
            device=det.get("device", "cuda"),
        )
        self.classifier = BirdClassifier(
            model_name=cls.get("model", "hf-hub:imageomics/bioclip"),
            device=cls.get("device", "cuda"),
            top_k=cls.get("top_k", 5),
        )
        self.tracker = BirdTracker(
            max_disappeared=trk.get("max_disappeared", 30),
            iou_threshold=trk.get("iou_threshold", 0.3),
            centroid_max_distance=trk.get("centroid_max_distance", 0.15),
        )
        self.prior_db_path = meta.get("ebird_db")
        self.prior_fips = meta.get("ebird_fips")
        self.prior = self._build_prior(
            latitude=meta.get("latitude"),
            longitude=meta.get("longitude"),
        )

        self.classify_every_n = cls.get("classify_every_n_frames", 15)
        self.crop_padding_ratio = cls.get("crop_padding_ratio", 0.12)
        self.crop_padding_ratio_min = cls.get("crop_padding_ratio_min", 0.04)
        self.crop_closeup_area_ratio = cls.get("crop_closeup_area_ratio", 0.10)
        self.min_crop_area = cls.get("min_crop_area", 2500)
        self.min_event_confidence = cls.get("min_event_confidence", 0.25)
        self.min_frames_to_report = trk.get("min_frames_to_report", 3)
        self.min_confidence_to_report = trk.get("min_confidence_to_report", 0.6)
        self.center_weight_strength = config.get("scoring", {}).get("center_weight_strength", 2.0)
        self.prompt_template = sp.get("prompt_template", "a photo of a {species}")
        self.results_dir = config.get("output", {}).get("results_dir", "results/")
        self.enable_small_bird_zoom_fallback = det.get("enable_small_bird_zoom_fallback", True)
        self.small_bird_fallback_every_n_frames = det.get("small_bird_fallback_every_n_frames", 5)
        self.verbose_runtime_logs = True
        self.print_video_summary = True
        self.compact_log_paths = False

    def _display_path(self, path: str | Path, *, keep_parts: int = 2) -> str:
        if not self.compact_log_paths:
            return str(path)
        return compact_path(path, keep_parts=keep_parts)

    def _build_prior(
        self,
        *,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        fips: Optional[str] = None,
        use_default_fips: bool = True,
    ) -> MetadataPrior:
        return MetadataPrior(
            latitude=latitude,
            longitude=longitude,
            db_path=self.prior_db_path,
            fips=self.prior_fips if use_default_fips and fips is None else fips,
        )

    def apply_config(self, config: dict):
        """Hot-reload mutable settings from config. Model/device changes require restart."""
        det = config.get("detector", {})
        cls = config.get("classifier", {})
        trk = config.get("tracker", {})
        meta = config.get("metadata", {})
        sp = config.get("species", {})

        new_confidence = det.get("confidence", 0.3)
        if self.detector.confidence != new_confidence:
            logger.info(f"Config reload: detector.confidence {self.detector.confidence} → {new_confidence}")
            self.detector.confidence = new_confidence

        new_top_k = cls.get("top_k", 5)
        if self.classifier.top_k != new_top_k:
            logger.info(f"Config reload: classifier.top_k {self.classifier.top_k} → {new_top_k}")
            self.classifier.top_k = new_top_k

        new_classify_every_n = cls.get("classify_every_n_frames", 15)
        if self.classify_every_n != new_classify_every_n:
            logger.info(f"Config reload: classify_every_n {self.classify_every_n} → {new_classify_every_n}")
            self.classify_every_n = new_classify_every_n

        new_crop_padding_ratio = cls.get("crop_padding_ratio", 0.12)
        if self.crop_padding_ratio != new_crop_padding_ratio:
            logger.info(f"Config reload: crop_padding_ratio {self.crop_padding_ratio} → {new_crop_padding_ratio}")
            self.crop_padding_ratio = new_crop_padding_ratio

        new_crop_padding_ratio_min = cls.get("crop_padding_ratio_min", 0.04)
        if self.crop_padding_ratio_min != new_crop_padding_ratio_min:
            logger.info(
                f"Config reload: crop_padding_ratio_min {self.crop_padding_ratio_min} → "
                f"{new_crop_padding_ratio_min}"
            )
            self.crop_padding_ratio_min = new_crop_padding_ratio_min

        new_crop_closeup_area_ratio = cls.get("crop_closeup_area_ratio", 0.10)
        if self.crop_closeup_area_ratio != new_crop_closeup_area_ratio:
            logger.info(
                f"Config reload: crop_closeup_area_ratio {self.crop_closeup_area_ratio} → "
                f"{new_crop_closeup_area_ratio}"
            )
            self.crop_closeup_area_ratio = new_crop_closeup_area_ratio

        new_min_crop_area = cls.get("min_crop_area", 2500)
        if self.min_crop_area != new_min_crop_area:
            logger.info(f"Config reload: min_crop_area {self.min_crop_area} → {new_min_crop_area}")
            self.min_crop_area = new_min_crop_area

        new_min_event_confidence = cls.get("min_event_confidence", 0.25)
        if self.min_event_confidence != new_min_event_confidence:
            logger.info(
                f"Config reload: min_event_confidence {self.min_event_confidence} → {new_min_event_confidence}"
            )
            self.min_event_confidence = new_min_event_confidence

        new_max_disappeared = trk.get("max_disappeared", 30)
        if self.tracker.max_disappeared != new_max_disappeared:
            logger.info(f"Config reload: tracker.max_disappeared {self.tracker.max_disappeared} → {new_max_disappeared}")
            self.tracker.max_disappeared = new_max_disappeared

        new_iou = trk.get("iou_threshold", 0.3)
        if self.tracker.iou_threshold != new_iou:
            logger.info(f"Config reload: tracker.iou_threshold {self.tracker.iou_threshold} → {new_iou}")
            self.tracker.iou_threshold = new_iou

        new_centroid_max_distance = trk.get("centroid_max_distance", 0.15)
        if self.tracker.centroid_max_distance != new_centroid_max_distance:
            logger.info(
                "Config reload: tracker.centroid_max_distance "
                f"{self.tracker.centroid_max_distance} → {new_centroid_max_distance}"
            )
            self.tracker.centroid_max_distance = new_centroid_max_distance

        new_min_frames = trk.get("min_frames_to_report", 3)
        if self.min_frames_to_report != new_min_frames:
            logger.info(f"Config reload: min_frames_to_report {self.min_frames_to_report} → {new_min_frames}")
            self.min_frames_to_report = new_min_frames

        new_min_conf = trk.get("min_confidence_to_report", 0.6)
        if self.min_confidence_to_report != new_min_conf:
            logger.info(f"Config reload: min_confidence_to_report {self.min_confidence_to_report} → {new_min_conf}")
            self.min_confidence_to_report = new_min_conf

        new_cw = config.get("scoring", {}).get("center_weight_strength", 2.0)
        if self.center_weight_strength != new_cw:
            logger.info(f"Config reload: center_weight_strength {self.center_weight_strength} → {new_cw}")
            self.center_weight_strength = new_cw

        new_small_bird_fallback = det.get("enable_small_bird_zoom_fallback", True)
        if self.enable_small_bird_zoom_fallback != new_small_bird_fallback:
            logger.info(
                "Config reload: enable_small_bird_zoom_fallback "
                f"{self.enable_small_bird_zoom_fallback} → {new_small_bird_fallback}"
            )
            self.enable_small_bird_zoom_fallback = new_small_bird_fallback

        new_small_bird_every_n = det.get("small_bird_fallback_every_n_frames", 5)
        if self.small_bird_fallback_every_n_frames != new_small_bird_every_n:
            logger.info(
                "Config reload: small_bird_fallback_every_n_frames "
                f"{self.small_bird_fallback_every_n_frames} → {new_small_bird_every_n}"
            )
            self.small_bird_fallback_every_n_frames = new_small_bird_every_n

        new_lat = meta.get("latitude")
        new_lon = meta.get("longitude")
        new_db_path = meta.get("ebird_db")
        new_fips = meta.get("ebird_fips")
        if (
            new_lat != self.prior.latitude
            or new_lon != self.prior.longitude
            or new_db_path != self.prior_db_path
            or new_fips != self.prior_fips
        ):
            logger.info(f"Config reload: location {new_lat}, {new_lon}")
            self.prior_db_path = new_db_path
            self.prior_fips = new_fips
            self.prior = self._build_prior(latitude=new_lat, longitude=new_lon)

        new_results_dir = config.get("output", {}).get("results_dir", "results/")
        if self.results_dir != new_results_dir:
            logger.info(
                "Config reload: results_dir %s → %s",
                self._display_path(self.results_dir),
                self._display_path(new_results_dir),
            )
            self.results_dir = new_results_dir

        new_species_file = sp.get("list_file")
        new_prompt = sp.get("prompt_template", "a photo of a {species}")
        if new_prompt != self.prompt_template:
            logger.info(f"Config reload: prompt_template changed, re-computing embeddings")
            self.prompt_template = new_prompt
            self.load_species(new_species_file)

    def load_species(self, species_file: Optional[str]):
        if species_file and os.path.exists(species_file):
            with open(species_file) as f:
                species = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            logger.info(f"Loaded {len(species)} species from {species_file}")
        else:
            from .pipeline_defaults import COMMON_NA_BIRDS
            logger.warning("No species file found — using built-in NA bird list")
            species = COMMON_NA_BIRDS
        self.classifier.set_species(species, prompt_template=self.prompt_template)

    def _expanded_crop(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        box_w = max(0, int(bbox[2] - bbox[0]))
        box_h = max(0, int(bbox[3] - bbox[1]))
        box_area = box_w * box_h
        if box_area < self.min_crop_area:
            return None

        frame_area = max(1, frame.shape[0] * frame.shape[1])
        area_ratio = box_area / frame_area

        max_padding = self.crop_padding_ratio
        min_padding = min(self.crop_padding_ratio_min, max_padding)
        if self.crop_closeup_area_ratio > 0:
            closeup_progress = min(area_ratio / self.crop_closeup_area_ratio, 1.0)
        else:
            closeup_progress = 0.0
        padding_ratio = max_padding - (max_padding - min_padding) * closeup_progress

        pad_x = int(round(box_w * padding_ratio))
        pad_y = int(round(box_h * padding_ratio))
        x1 = max(0, int(bbox[0]) - pad_x)
        y1 = max(0, int(bbox[1]) - pad_y)
        x2 = min(frame.shape[1], int(bbox[2]) + pad_x)
        y2 = min(frame.shape[0], int(bbox[3]) + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _build_video_predictions(self, track_predictions: List[List[Tuple[str, float]]]) -> List[dict]:
        """Aggregate track-level predictions into a video-level presence summary."""
        species_scores: Dict[str, float] = {}
        supporting_tracks: Dict[str, int] = {}

        for preds in track_predictions:
            seen_species = set()
            for species, prob in preds[:5]:
                species_scores[species] = max(species_scores.get(species, 0.0), prob)
                if species not in seen_species:
                    supporting_tracks[species] = supporting_tracks.get(species, 0) + 1
                    seen_species.add(species)

        ranked = sorted(
            species_scores.items(),
            key=lambda item: (-item[1], -supporting_tracks[item[0]], item[0]),
        )
        return [
            {
                "species": species,
                "presence_probability": round(prob, 4),
                "supporting_tracks": supporting_tracks[species],
            }
            for species, prob in ranked[:5]
        ]

    def _build_species_summary(
        self,
        weighted_scores: Dict[str, float],
        raw_scores: Dict[str, float],
    ) -> List[dict]:
        ranked = sorted(weighted_scores.items(), key=lambda item: -item[1])
        return [
            {
                "species": species,
                "probability": round(prob, 4),
                "raw_probability": round(raw_scores.get(species, 0.0), 4),
            }
            for species, prob in ranked[:5]
        ]

    def _apply_waterbird_shape_adjustment(
        self,
        preds: List[Tuple[str, float]],
        *,
        bbox: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> List[Tuple[str, float]]:
        if not preds:
            return preds

        has_swan = any(species in SWAN_SPECIES for species, _ in preds)
        has_gull = any("Gull" in species for species, _ in preds)
        if not (has_swan and has_gull):
            return preds

        box_w = max(float(bbox[2] - bbox[0]), 1.0)
        box_h = max(float(bbox[3] - bbox[1]), 1.0)
        aspect_ratio = box_w / box_h
        relative_height = box_h / max(float(frame_height), 1.0)
        relative_width = box_w / max(float(frame_width), 1.0)

        # Wide, low-profile detections on water are much more likely to be gulls than swans.
        if aspect_ratio < 1.15 or relative_height > 0.23 or relative_width < 0.03:
            return preds

        adjusted = []
        total = 0.0
        for species, prob in preds:
            score = prob
            if "Gull" in species:
                score *= 1.8
            elif species in SWAN_SPECIES:
                score *= 0.4
            adjusted.append((species, score))
            total += score

        if total <= 0:
            return preds
        return sorted(
            [(species, score / total) for species, score in adjusted],
            key=lambda item: -item[1],
        )

    def _select_video_gallery_plan(
        self,
        candidates: List[dict],
        *,
        total_frames: int,
        fps: float,
        min_frames: int = 3,
        max_frames: int = 6,
    ) -> List[dict]:
        if total_frames <= 0:
            return []

        selected: List[dict] = []
        min_gap = max(int(total_frames / 10), int((fps or 1) * 1.5), 1)
        for candidate in sorted(candidates, key=lambda item: (-item["score"], item["frame"])):
            if any(abs(candidate["frame"] - existing["frame"]) < min_gap for existing in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_frames:
                break

        if len(selected) < min_frames:
            desired = max(min_frames, min(max_frames, 4))
            fallback_frames = [
                int(round((idx + 1) * total_frames / (desired + 1)))
                for idx in range(desired)
            ]
            for frame_no in fallback_frames:
                if any(abs(frame_no - existing["frame"]) < min_gap for existing in selected):
                    continue
                selected.append({
                    "frame": min(max(frame_no, 0), max(total_frames - 1, 0)),
                    "timestamp_s": round(frame_no / fps, 2) if fps else None,
                    "score": 0.0,
                    "track_id": None,
                    "species": None,
                })
                if len(selected) >= min_frames:
                    break

        return sorted(selected, key=lambda item: item["frame"])[:max_frames]

    def _write_video_gallery_frames(
        self,
        video_path: str,
        gallery_plan: List[dict],
        crops_dir: Path,
    ) -> List[dict]:
        if not gallery_plan:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not reopen video for gallery snapshots: {video_path}")
            return []

        gallery = []
        try:
            for idx, item in enumerate(gallery_plan, start=1):
                frame_no = int(item["frame"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                if not ok or frame is None:
                    logger.warning(f"Could not read gallery frame {frame_no} from {Path(video_path).name}")
                    continue
                filename = f"frame_{idx:02d}_{frame_no:06d}.jpg"
                cv2.imwrite(str(crops_dir / filename), frame)
                gallery.append({
                    "frame": frame_no,
                    "timestamp_s": item.get("timestamp_s"),
                    "file": filename,
                    "track_id": item.get("track_id"),
                    "species": item.get("species"),
                })
        finally:
            cap.release()

        return gallery

    def _draw_image_annotations(
        self,
        frame: np.ndarray,
        detections: List[dict],
    ) -> np.ndarray:
        annotated = frame.copy()
        box_color = (32, 32, 220)
        label_bg_color = (245, 245, 245)
        label_text_color = (20, 20, 20)
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            label = str(det["detection_index"])
            top = det["species"][0] if det["species"] else None
            if top:
                label = f"{label}: {top['species']}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 4)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            text_x = max(0, x1)
            text_y = max(text_h + 8, y1 - 8)
            cv2.rectangle(
                annotated,
                (text_x, text_y - text_h - 8),
                (text_x + text_w + 12, text_y + baseline - 4),
                label_bg_color,
                -1,
            )
            cv2.rectangle(
                annotated,
                (text_x, text_y - text_h - 8),
                (text_x + text_w + 12, text_y + baseline - 4),
                box_color,
                2,
            )
            cv2.putText(
                annotated,
                label,
                (text_x + 6, text_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                label_text_color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _extract_video_stills(
        self,
        video_path: str,
        *,
        duration_s: float,
        fps: float,
        width: int,
        height: int,
        frames_with_detections: set,
        crops_dir: Path,
        video_date: Optional[datetime],
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> List[dict]:
        """Extract evenly-spaced stills from a video, snapped to tracked frames,
        and annotate them with detection boxes like photo jobs."""
        if not frames_with_detections or duration_s <= 0:
            return []

        num_stills = min(10, max(1, int(duration_s / 2)))
        sorted_detected = sorted(frames_with_detections)
        window = int(fps)  # +/-1 second

        # Compute evenly-spaced target frames, snap each to nearest detected frame
        selected_frames: List[int] = []
        for i in range(num_stills):
            t = (i + 0.5) * duration_s / num_stills
            target_frame = int(t * fps)
            idx = bisect.bisect_left(sorted_detected, target_frame)
            best = None
            best_dist = float("inf")
            for candidate_idx in (idx - 1, idx):
                if 0 <= candidate_idx < len(sorted_detected):
                    dist = abs(sorted_detected[candidate_idx] - target_frame)
                    if dist <= window and dist < best_dist:
                        best = sorted_detected[candidate_idx]
                        best_dist = dist
            if best is not None and best not in selected_frames:
                selected_frames.append(best)

        if not selected_frames:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Could not reopen video for stills: %s", video_path)
            return []

        stills: List[dict] = []
        try:
            for still_idx, frame_no in enumerate(selected_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                detections = self.detector.detect(frame)
                if not detections:
                    continue

                crops = []
                crop_indices = []
                for det_idx, det in enumerate(detections):
                    crop = self._expanded_crop(frame, det.bbox)
                    if crop is not None:
                        crops.append(crop)
                        crop_indices.append(det_idx)

                det_results: List[dict] = []
                if crops:
                    batch_preds = self.classifier.classify_batch(crops)
                    for i, (preds, crop) in enumerate(zip(batch_preds, crops)):
                        raw_preds = preds[:]
                        raw_top_conf = raw_preds[0][1] if raw_preds else 0.0
                        if raw_top_conf < self.min_event_confidence:
                            continue

                        preds = self.prior.apply(
                            preds,
                            dt=video_date,
                            latitude=latitude,
                            longitude=longitude,
                        )

                        det_idx_val = crop_indices[i]
                        det_results.append({
                            "detection_index": len(det_results) + 1,
                            "bbox": [float(x) for x in detections[det_idx_val].bbox],
                            "species": [
                                {"species": s, "probability": round(p, 4)}
                                for s, p in preds[:5]
                            ],
                            "raw_species": [
                                {"species": s, "probability": round(p, 4)}
                                for s, p in raw_preds[:5]
                            ],
                        })

                if not det_results:
                    continue

                timestamp_s = round(frame_no / fps, 2)
                annotated_file = f"still_{still_idx:02d}_{frame_no:06d}_annotated.jpg"
                annotated = self._draw_image_annotations(frame, det_results)
                cv2.imwrite(str(crops_dir / annotated_file), annotated)

                stills.append({
                    "frame": frame_no,
                    "timestamp_s": timestamp_s,
                    "annotated_file": annotated_file,
                    "image_width": width,
                    "image_height": height,
                    "detections": det_results,
                })
                logger.info(
                    "  Still %d/%d @ %.1fs: %d detection(s)",
                    still_idx + 1, len(selected_frames), timestamp_s, len(det_results),
                )
        finally:
            cap.release()

        return stills

    def process_video(
        self,
        video_path: str,
        video_date: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        result_stem: Optional[str] = None,
        source_filename: Optional[str] = None,
        display_name: Optional[str] = None,
        asset_records: Optional[list[dict]] = None,
    ) -> dict:
        """
        Process a single video. Returns a summary dict and writes a JSON results file.
        """
        self.tracker.next_id = 0
        self.tracker.tracks.clear()
        self.tracker.completed_tracks.clear()

        # Use per-video GPS if provided, otherwise fall back to config prior
        if latitude is not None and longitude is not None:
            self.prior = self._build_prior(
                latitude=latitude,
                longitude=longitude,
                fips=None,
                use_default_fips=False,
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "%s: %sx%s @ %.1ffps, ~%s frames",
            self._display_path(video_path, keep_parts=1),
            width,
            height,
            fps,
            total_frames,
        )

        frame_cx = width / 2.0
        frame_cy = height / 2.0

        frame_idx = 0
        # track_id -> list of per-event dicts
        track_events: Dict[int, List[dict]] = {}
        # track_id -> best crop (highest top-1 confidence seen so far)
        best_crops: Dict[int, Tuple[np.ndarray, float]] = {}  # tid -> (crop_bgr, confidence)
        gallery_candidates: List[dict] = []
        frames_with_detections: set = set()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect(frame)
                should_try_small_bird_fallback = (
                    not detections
                    and self.enable_small_bird_zoom_fallback
                    and self.small_bird_fallback_every_n_frames > 0
                    and frame_idx % self.small_bird_fallback_every_n_frames == 0
                )
                if should_try_small_bird_fallback:
                    detections = self.detector.detect_zoomed(frame)
                    if detections:
                        if self.verbose_runtime_logs:
                            logger.info(
                                f"  [{frame_idx / fps:.1f}s] recovered {len(detections)} detection(s) "
                                "via center zoom fallback"
                            )
                tracks = self.tracker.update(detections, frame_size=(width, height))

                if any(t.disappeared == 0 for t in tracks.values()):
                    frames_with_detections.add(frame_idx)

                # Map active tracks to their matching detection (for crop access)
                det_for_track: Dict[int, object] = {}
                for tid, track in tracks.items():
                    if track.disappeared > 0 or track.matched_detection_idx is None:
                        continue
                    det_for_track[tid] = detections[track.matched_detection_idx]

                # Collect tracks due for classification
                to_classify_ids = []
                to_classify_crops = []
                for tid, track in tracks.items():
                    if track.disappeared > 0:
                        continue
                    if (frame_idx - track.last_classified_frame) >= self.classify_every_n:
                        if tid in det_for_track:
                            crop = self._expanded_crop(frame, det_for_track[tid].bbox)
                            if crop is None:
                                track.last_classified_frame = frame_idx
                                continue
                            to_classify_ids.append(tid)
                            to_classify_crops.append(crop)

                if to_classify_crops:
                    batch_results = self.classifier.classify_batch(to_classify_crops)
                    for tid, preds, crop in zip(to_classify_ids, batch_results, to_classify_crops):
                        raw_preds = preds[:]  # save pre-prior visual scores
                        raw_top_conf = raw_preds[0][1] if raw_preds else 0.0
                        tracks[tid].last_classified_frame = frame_idx
                        if raw_top_conf < self.min_event_confidence:
                            if self.verbose_runtime_logs:
                                logger.info(
                                    f"  [{frame_idx / fps:.1f}s] track#{tid} skipped "
                                    f"low-confidence visual event ({raw_top_conf:.1%})"
                                )
                            continue

                        bbox = tracks[tid].bbox
                        preds = self._apply_waterbird_shape_adjustment(
                            preds,
                            bbox=bbox,
                            frame_width=width,
                            frame_height=height,
                        )
                        preds = self.prior.apply(preds, dt=video_date, latitude=latitude, longitude=longitude)

                        # Weight by proximity to frame center (Gaussian falloff)
                        bbox_cx = (bbox[0] + bbox[2]) / 2.0
                        bbox_cy = (bbox[1] + bbox[3]) / 2.0
                        dx = (bbox_cx - frame_cx) / frame_cx
                        dy = (bbox_cy - frame_cy) / frame_cy
                        dist_sq = dx * dx + dy * dy
                        center_weight = float(np.exp(-dist_sq * self.center_weight_strength))

                        tracks[tid].prediction_history.append(preds)
                        tracks[tid].raw_prediction_history.append(raw_preds)
                        tracks[tid].prediction_weights.append(center_weight)

                        if raw_top_conf > best_crops.get(tid, (None, -1.0))[1]:
                            best_crops[tid] = (crop.copy(), raw_top_conf)

                        timestamp_s = frame_idx / fps
                        event = {
                            "frame": frame_idx,
                            "timestamp_s": round(timestamp_s, 2),
                            "track_id": tid,
                            "predictions": [{"species": s, "probability": round(p, 4)} for s, p in preds],
                        }
                        track_events.setdefault(tid, []).append(event)
                        top_species = preds[0][0] if preds else None
                        gallery_candidates.append({
                            "frame": frame_idx,
                            "timestamp_s": round(timestamp_s, 2),
                            "score": float(raw_top_conf * center_weight),
                            "track_id": tid,
                            "species": top_species,
                        })
                        self._log_event(event)

                frame_idx += 1
                if self.verbose_runtime_logs and frame_idx % 500 == 0:
                    logger.info(f"  ...frame {frame_idx}/{total_frames}")

        finally:
            cap.release()

        # Save best crop per track
        crops_dir = Path(self.results_dir) / (Path(video_path).stem + "_crops")
        crops_dir.mkdir(parents=True, exist_ok=True)
        saved_crops: Dict[int, str] = {}
        for tid, (crop_bgr, _) in best_crops.items():
            crop_path = crops_dir / f"track_{tid}.jpg"
            cv2.imwrite(str(crop_path), crop_bgr)
            saved_crops[tid] = crop_path.name

        gallery_plan = self._select_video_gallery_plan(
            gallery_candidates,
            total_frames=frame_idx,
            fps=fps,
        )
        saved_gallery = self._write_video_gallery_frames(video_path, gallery_plan, crops_dir)

        duration_s = frame_idx / fps if fps else 0
        video_stills = self._extract_video_stills(
            video_path,
            duration_s=duration_s,
            fps=fps,
            width=width,
            height=height,
            frames_with_detections=frames_with_detections,
            crops_dir=crops_dir,
            video_date=video_date,
            latitude=latitude,
            longitude=longitude,
        )

        # Build per-track summaries using averaged predictions
        track_summaries = []
        video_track_predictions: List[List[Tuple[str, float]]] = []
        video_raw_predictions: List[List[Tuple[str, float]]] = []
        for tid, track in {**self.tracker.completed_tracks, **self.tracker.tracks}.items():
            best = track.best_prediction
            top_conf = best[0][1] if best else 0.0
            if track.frame_count < self.min_frames_to_report and top_conf < self.min_confidence_to_report:
                continue
            if best:
                raw = track.best_raw_prediction
                explanation = self._build_explanation(best, raw, video_date)
                video_track_predictions.append(best)
                if raw:
                    video_raw_predictions.append(raw)
                track_summaries.append({
                    "track_id": tid,
                    "frames_tracked": track.frame_count,
                    "classifications_made": len(track.prediction_history),
                    "averaged_predictions": [
                        {"species": s, "probability": round(p, 4)} for s, p in best[:5]
                    ],
                    "raw_predictions": [
                        {"species": s, "probability": round(p, 4)} for s, p in raw[:5]
                    ] if raw else [],
                    "explanation": explanation,
                    "crop": saved_crops.get(tid),
                })

        video_predictions = self._build_video_predictions(video_track_predictions)
        raw_video_predictions = self._build_video_predictions(video_raw_predictions)
        # Merge raw probabilities into video_predictions for side-by-side display
        raw_lookup = {p["species"]: p["presence_probability"] for p in raw_video_predictions}
        for p in video_predictions:
            p["raw_presence_probability"] = raw_lookup.get(p["species"], 0.0)
        summary = {
            "video": str(video_path),
            "source_filename": source_filename or Path(video_path).name,
            "display_name": display_name or source_filename or Path(video_path).name,
            "date": video_date.isoformat() if video_date else None,
            "latitude": latitude,
            "longitude": longitude,
            "asset_records": asset_records or [],
            "resolution_warning": resolution_warning_text(
                media_type="video",
                width=width,
                height=height,
            ),
            "video_info": {
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "total_frames": total_frames,
                "frames_processed": frame_idx,
                "duration_s": round(duration_s, 2),
            },
            "frames_processed": frame_idx,
            "fps": fps,
            "video_predictions": video_predictions,
            "frame_gallery": saved_gallery,
            "tracks": track_summaries,
            "video_stills": video_stills,
            "all_events": [e for events in track_events.values() for e in events],
        }

        # Write JSON results
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(self.results_dir) / ((result_stem or Path(video_path).stem) + "_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", self._display_path(out_path))

        if self.print_video_summary:
            self._print_summary(summary)
        return summary

    def _build_explanation(
        self,
        best: List[Tuple[str, float]],
        raw: Optional[List[Tuple[str, float]]],
        video_date: Optional[datetime],
    ) -> str:
        if not raw:
            return ""

        top_final = best[0][0] if best else "Unknown"
        raw_top3 = raw[:3]

        # Visual summary
        visual_parts = [f"{s} ({p:.0%})" for s, p in raw_top3]
        visual_str = ", ".join(visual_parts[:-1]) + f", or {visual_parts[-1]}" if len(visual_parts) > 1 else visual_parts[0]

        if not hasattr(self.prior, '_con') or self.prior._con is None or video_date is None:
            return f"Visually resembles {visual_str}. No location/date priors applied."

        # Get eBird frequencies for the raw top candidates
        raw_species = [s for s, _ in raw_top3]
        priors = self.prior.get_priors(raw_species, dt=video_date)

        month_name = video_date.strftime("%B")
        county = "this area"
        try:
            rows = self.prior._con.execute("SELECT name FROM counties").fetchall()
            if rows:
                county = ", ".join(r[0] for r in rows)
        except Exception:
            pass

        prior_parts = [f"{s}: {priors.get(s, 0):.1%}" for s, _ in raw_top3]
        prior_str = ", ".join(prior_parts)

        # Check if priors meaningfully changed the ranking
        raw_top = raw_top3[0][0]
        if raw_top != top_final:
            # Find how much more likely top_final is vs raw_top visually
            raw_final_prob = next((p for s, p in raw_top3 if s == top_final), None)
            raw_top_freq = priors.get(raw_top, 0.01)
            final_freq = priors.get(top_final, 0.01)
            if raw_top_freq > 0 and final_freq > raw_top_freq:
                ratio = final_freq / raw_top_freq
                return (
                    f"Visually resembles {visual_str}. "
                    f"In {county} in {month_name}, eBird frequency — {prior_str}. "
                    f"{top_final} is {ratio:.0f}× more likely than {raw_top} "
                    f"for this location and time of year."
                )

        return (
            f"Visually resembles {visual_str}. "
            f"In {county} in {month_name}, eBird frequency — {prior_str}. "
            f"Visual and location data agree on {top_final}."
        )

    def process_images(
        self,
        image_paths: list[str],
        *,
        source_filenames: Optional[list[str]] = None,
        video_date: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        job_id: Optional[str] = None,
        result_stem: Optional[str] = None,
        display_name: Optional[str] = None,
        asset_records: Optional[list[dict]] = None,
    ) -> dict:
        """
        Classify birds in one or more photos. Returns a summary dict and writes
        a JSON results file.  No tracker is used — each image is independent.
        """
        if latitude is not None and longitude is not None:
            self.prior = self._build_prior(
                latitude=latitude,
                longitude=longitude,
                fips=None,
                use_default_fips=False,
            )

        stem = result_stem or job_id or Path(image_paths[0]).stem
        crops_dir = Path(self.results_dir) / ((job_id or stem) + "_crops")
        crops_dir.mkdir(parents=True, exist_ok=True)

        image_results = []

        for img_idx, img_path in enumerate(image_paths):
            source_filename = (
                source_filenames[img_idx]
                if source_filenames and img_idx < len(source_filenames)
                else Path(img_path).name
            )
            # Extract per-image EXIF metadata
            img_meta = extract_media_metadata(img_path)
            img_date = img_meta.recorded_at or video_date  # fall back to job-level date
            img_latitude = img_meta.latitude if img_meta.latitude is not None else latitude
            img_longitude = img_meta.longitude if img_meta.longitude is not None else longitude
            img_ebird_region = self.prior.resolve_county_name(
                latitude=img_latitude,
                longitude=img_longitude,
            )

            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning(f"Cannot read image: {img_path}")
                image_results.append({
                    "filename": source_filename,
                    "date": img_meta.recorded_at.isoformat() if img_meta.recorded_at else None,
                    "latitude": img_meta.latitude,
                    "longitude": img_meta.longitude,
                    "camera": img_meta.camera_info,
                    "ebird_region": img_ebird_region,
                    "image_width": None,
                    "image_height": None,
                    "annotated_file": None,
                    "species_summary": [],
                    "detections": [],
                })
                continue

            detections = self.detector.detect(frame)
            logger.info(f"{source_filename}: {len(detections)} detection(s)")

            crops = []
            crop_indices = []  # which detection index each crop came from
            for det_idx, det in enumerate(detections):
                crop = self._expanded_crop(frame, det.bbox)
                if crop is not None:
                    crops.append(crop)
                    crop_indices.append(det_idx)

            det_results = []
            image_species_scores: Dict[str, float] = {}
            image_raw_species_scores: Dict[str, float] = {}
            if crops:
                batch_preds = self.classifier.classify_batch(crops)
                for i, (preds, crop) in enumerate(zip(batch_preds, crops)):
                    raw_preds = preds[:]
                    raw_top_conf = raw_preds[0][1] if raw_preds else 0.0

                    if raw_top_conf < self.min_event_confidence:
                        logger.info(
                            f"  {source_filename} bird#{crop_indices[i]} skipped "
                            f"low-confidence ({raw_top_conf:.1%})"
                        )
                        continue

                    preds = self.prior.apply(
                        preds,
                        dt=img_date,
                        latitude=img_latitude,
                        longitude=img_longitude,
                    )

                    crop_file = f"img{img_idx}_bird{crop_indices[i]}.jpg"
                    cv2.imwrite(str(crops_dir / crop_file), crop)

                    for species, prob in preds[:5]:
                        image_species_scores[species] = max(
                            image_species_scores.get(species, 0.0), prob
                        )
                    for species, prob in raw_preds[:5]:
                        image_raw_species_scores[species] = max(
                            image_raw_species_scores.get(species, 0.0), prob
                        )

                    det_idx = crop_indices[i]
                    det_results.append({
                        "detection_index": len(det_results) + 1,
                        "bbox": [float(x) for x in detections[det_idx].bbox],
                        "species": [
                            {"species": s, "probability": round(p, 4)}
                            for s, p in preds[:5]
                        ],
                        "raw_species": [
                            {"species": s, "probability": round(p, 4)}
                            for s, p in raw_preds[:5]
                        ],
                        "crop_file": crop_file,
                    })

            annotated_file = None
            if det_results:
                annotated_file = f"img{img_idx}_annotated.jpg"
                annotated = self._draw_image_annotations(frame, det_results)
                cv2.imwrite(str(crops_dir / annotated_file), annotated)

            image_results.append({
                "filename": source_filename,
                "date": img_meta.recorded_at.isoformat() if img_meta.recorded_at else None,
                "latitude": img_meta.latitude,
                "longitude": img_meta.longitude,
                "camera": img_meta.camera_info,
                "ebird_region": img_ebird_region,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "annotated_file": annotated_file,
                "species_summary": self._build_species_summary(image_species_scores, image_raw_species_scores),
                "detections": det_results,
            })

        summary = {
            "type": "images",
            "display_name": display_name or (source_filenames[0] if source_filenames else Path(image_paths[0]).name),
            "date": video_date.isoformat() if video_date else None,
            "latitude": latitude,
            "longitude": longitude,
            "asset_records": asset_records or [],
            "image_info": {
                "count": len(image_paths),
                "filenames": source_filenames or [Path(p).name for p in image_paths],
            },
            "images": image_results,
        }

        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(self.results_dir) / (stem + "_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", self._display_path(out_path))

        self._print_image_summary(summary)
        return summary

    def _print_image_summary(self, summary: dict):
        total_det = sum(len(img["detections"]) for img in summary["images"])
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            f"Images: {summary['image_info']['count']} | "
            f"Birds detected: {total_det}"
        )
        for img in summary["images"]:
            n = len(img["detections"])
            logger.info(f"  {img['filename']}: {n} bird(s)")
            if img["species_summary"]:
                for rank, pred in enumerate(img["species_summary"], 1):
                    logger.info(f"    {rank}. {pred['species']:<33} {pred['probability']:.1%}")
            for det in img["detections"]:
                top = det["species"][0] if det["species"] else None
                if top:
                    logger.info(f"    → {top['species']} ({top['probability']:.1%})")
        if not total_det:
            logger.info("No birds detected.")
        logger.info("=" * 60)

    def _log_event(self, event: dict):
        if not self.verbose_runtime_logs:
            return
        top = event["predictions"][0]
        others = ", ".join(
            f"{p['species']} {p['probability']:.1%}"
            for p in event["predictions"][1:3]
        )
        logger.info(
            f"  [{event['timestamp_s']:.1f}s] track#{event['track_id']} → "
            f"{top['species']} ({top['probability']:.1%})"
            + (f"  | also: {others}" if others else "")
        )

    def _print_summary(self, summary: dict):
        print(f"\n{'='*60}")
        print(f"Video: {Path(summary['video']).name}")
        print(f"Frames processed: {summary['frames_processed']}")
        if summary.get("video_predictions"):
            print("Likely birds in this video:")
            for rank, p in enumerate(summary["video_predictions"], 1):
                print(f"  {rank}. {p['species']:<35} {p['presence_probability']:.1%}")
        else:
            print("Likely birds in this video: none")
        print()
        print(f"Tracking details (nerd stuff): {len(summary['tracks'])} track(s)")
        print()
        for t in summary["tracks"]:
            print(f"  Track #{t['track_id']} ({t['frames_tracked']} frames, "
                  f"{t['classifications_made']} classifications)")
            for rank, p in enumerate(t["averaged_predictions"], 1):
                print(f"    {rank}. {p['species']:<35} {p['probability']:.1%}")
        print(f"{'='*60}\n")
