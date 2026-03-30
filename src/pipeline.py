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
from .tracker import BirdTracker, Track, iou

logger = logging.getLogger(__name__)


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
        )
        self.prior = MetadataPrior(
            latitude=meta.get("latitude"),
            longitude=meta.get("longitude"),
        )

        self.classify_every_n = cls.get("classify_every_n_frames", 15)
        self.prompt_template = sp.get("prompt_template", "a photo of a {species}")
        self.results_dir = config.get("output", {}).get("results_dir", "results/")

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

    def process_video(
        self,
        video_path: str,
        video_date: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> dict:
        """
        Process a single video. Returns a summary dict and writes a JSON results file.
        """
        self.tracker.tracks.clear()
        self.tracker.completed_tracks.clear()

        # Use per-video GPS if provided, otherwise fall back to config prior
        if latitude is not None and longitude is not None:
            self.prior = MetadataPrior(latitude=latitude, longitude=longitude)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"{Path(video_path).name}: {width}x{height} @ {fps:.1f}fps, ~{total_frames} frames")

        frame_idx = 0
        # track_id -> list of per-event dicts
        track_events: Dict[int, List[dict]] = {}
        # track_id -> best crop (highest top-1 confidence seen so far)
        best_crops: Dict[int, Tuple[np.ndarray, float]] = {}  # tid -> (crop_bgr, confidence)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections)

                # Map active tracks to their matching detection (for crop access)
                det_for_track: Dict[int, object] = {}
                for tid, track in tracks.items():
                    if track.disappeared > 0:
                        continue
                    for det in detections:
                        if iou(track.bbox, det.bbox) > 0.5:
                            det_for_track[tid] = det
                            break

                # Collect tracks due for classification
                to_classify_ids = []
                to_classify_crops = []
                for tid, track in tracks.items():
                    if track.disappeared > 0:
                        continue
                    if (frame_idx - track.last_classified_frame) >= self.classify_every_n:
                        if tid in det_for_track:
                            to_classify_ids.append(tid)
                            to_classify_crops.append(det_for_track[tid].crop)

                if to_classify_crops:
                    batch_results = self.classifier.classify_batch(to_classify_crops)
                    for tid, preds, crop in zip(to_classify_ids, batch_results, to_classify_crops):
                        preds = self.prior.apply(preds, dt=video_date)
                        tracks[tid].prediction_history.append(preds)
                        tracks[tid].last_classified_frame = frame_idx

                        top_conf = preds[0][1] if preds else 0.0
                        if top_conf > best_crops.get(tid, (None, -1.0))[1]:
                            best_crops[tid] = (crop.copy(), top_conf)

                        timestamp_s = frame_idx / fps
                        event = {
                            "frame": frame_idx,
                            "timestamp_s": round(timestamp_s, 2),
                            "track_id": tid,
                            "predictions": [{"species": s, "probability": round(p, 4)} for s, p in preds],
                        }
                        track_events.setdefault(tid, []).append(event)
                        self._log_event(event)

                frame_idx += 1
                if frame_idx % 500 == 0:
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

        # Build per-track summaries using averaged predictions
        track_summaries = []
        for tid, track in {**self.tracker.completed_tracks, **self.tracker.tracks}.items():
            best = track.best_prediction
            if best:
                track_summaries.append({
                    "track_id": tid,
                    "frames_tracked": track.frame_count,
                    "classifications_made": len(track.prediction_history),
                    "averaged_predictions": [
                        {"species": s, "probability": round(p, 4)} for s, p in best[:5]
                    ],
                    "crop": saved_crops.get(tid),
                })

        summary = {
            "video": str(video_path),
            "date": video_date.isoformat() if video_date else None,
            "latitude": latitude,
            "longitude": longitude,
            "frames_processed": frame_idx,
            "fps": fps,
            "tracks": track_summaries,
            "all_events": [e for events in track_events.values() for e in events],
        }

        # Write JSON results
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(self.results_dir) / (Path(video_path).stem + "_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results written to {out_path}")

        self._print_summary(summary)
        return summary

    def _log_event(self, event: dict):
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
        print(f"Tracks found: {len(summary['tracks'])}")
        print()
        for t in summary["tracks"]:
            print(f"  Track #{t['track_id']} ({t['frames_tracked']} frames, "
                  f"{t['classifications_made']} classifications)")
            for rank, p in enumerate(t["averaged_predictions"], 1):
                print(f"    {rank}. {p['species']:<35} {p['probability']:.1%}")
        print(f"{'='*60}\n")
