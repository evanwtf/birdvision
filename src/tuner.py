import copy
import json
import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .pipeline import BirdIdentificationPipeline

logger = logging.getLogger(__name__)

BASELINE_ASSET_SHA256 = "e84e85beb30cf97e7ccce5a1fe0a6b1bd705b5e81b282cd9a30008c9860cc3c6"
BASELINE_ASSET_FILENAME = f"{BASELINE_ASSET_SHA256}.mov"
BASELINE_TARGET_SPECIES = "Herring Gull"

TUNING_SPACE = OrderedDict(
    {
        "detector.confidence": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "detector.enable_small_bird_zoom_fallback": [True, False],
        "detector.small_bird_fallback_every_n_frames": [3, 5, 8],
        "classifier.classify_every_n_frames": [5, 10, 15, 20],
        "classifier.crop_padding_ratio": [0.08, 0.12, 0.18, 0.24],
        "classifier.crop_padding_ratio_min": [0.02, 0.04, 0.06],
        "classifier.crop_closeup_area_ratio": [0.06, 0.10, 0.14],
        "classifier.min_crop_area": [1600, 2500, 3600],
        "classifier.min_event_confidence": [0.20, 0.25, 0.30, 0.35],
        "tracker.max_disappeared": [60, 120, 180],
        "tracker.iou_threshold": [0.15, 0.20, 0.30],
        "tracker.centroid_max_distance": [0.12, 0.18, 0.24],
        "tracker.min_frames_to_report": [4, 8, 12],
        "tracker.min_confidence_to_report": [0.50, 0.60, 0.70],
        "scoring.center_weight_strength": [0.5, 1.0, 2.0, 3.0],
    }
)

TUNING_DEFAULTS = {
    "detector.confidence": 0.30,
    "detector.enable_small_bird_zoom_fallback": True,
    "detector.small_bird_fallback_every_n_frames": 5,
    "classifier.classify_every_n_frames": 15,
    "classifier.crop_padding_ratio": 0.12,
    "classifier.crop_padding_ratio_min": 0.04,
    "classifier.crop_closeup_area_ratio": 0.10,
    "classifier.min_crop_area": 2500,
    "classifier.min_event_confidence": 0.25,
    "tracker.max_disappeared": 30,
    "tracker.iou_threshold": 0.30,
    "tracker.centroid_max_distance": 0.15,
    "tracker.min_frames_to_report": 3,
    "tracker.min_confidence_to_report": 0.60,
    "scoring.center_weight_strength": 2.0,
}


def default_baseline_video_path() -> str:
    candidates = (
        Path("/data/videos/assets") / BASELINE_ASSET_FILENAME,
        Path("videos/assets") / BASELINE_ASSET_FILENAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _get_nested(config: dict, path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict):
            return TUNING_DEFAULTS[path]
        if part not in current:
            return TUNING_DEFAULTS[path]
        current = current[part]
    return current


def _set_nested(config: dict, path: str, value: Any) -> None:
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _ordered_unique(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _summarize_params(config: dict) -> dict[str, Any]:
    return {path: _get_nested(config, path) for path in TUNING_SPACE}


@dataclass
class TrialResult:
    trial_index: int
    reason: str
    elapsed_s: float
    results_dir: str
    results_json: Optional[str]
    target_species: str
    target_confidence: float
    raw_target_confidence: float
    target_rank: Optional[int]
    success_species: Optional[str]
    success_confidence: float
    top_species: Optional[str]
    top_confidence: float
    reached_target: bool
    changed_params: dict[str, Any]
    tuned_params: dict[str, Any]
    error: Optional[str] = None


class SingleVideoTuningRunner:
    def __init__(
        self,
        *,
        config: dict,
        video_path: str,
        target_species: str,
        success_species_contains: Optional[str] = None,
        stop_confidence: float,
        time_budget_s: float,
        results_dir: Optional[str] = None,
        max_trials: Optional[int] = None,
        video_date: Optional[datetime] = None,
    ):
        self.base_config = copy.deepcopy(config)
        self.video_path = str(Path(video_path))
        self.target_species = target_species
        self.success_species_contains = success_species_contains
        self.stop_confidence = stop_confidence
        self.time_budget_s = max(0.0, time_budget_s)
        self.max_trials = max_trials
        self.video_date = video_date

        root_results_dir = Path(
            results_dir or self.base_config.get("output", {}).get("results_dir", "results")
        )
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"tune_{Path(self.video_path).stem}_{timestamp}"
        self.run_dir = root_results_dir / "tuning" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.run_dir / "tuning_summary.json"

        pipeline_config = copy.deepcopy(self.base_config)
        pipeline_config.setdefault("output", {})["results_dir"] = str(self.run_dir / "bootstrap")
        self.pipeline = BirdIdentificationPipeline(pipeline_config)
        self.pipeline.verbose_runtime_logs = False
        self.pipeline.print_video_summary = False
        self.pipeline.compact_log_paths = True
        species_file = self.base_config.get("species", {}).get("list_file")
        self.pipeline.load_species(species_file)

        self.started_at = time.monotonic()
        self.trials: list[TrialResult] = []
        self.seen_configs: set[str] = set()
        self.stop_reason = "not_started"
        self.initial_top_species: Optional[str] = None

    def _elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def _time_budget_reached(self) -> bool:
        return self._elapsed_s() >= self.time_budget_s

    def _max_trials_reached(self) -> bool:
        return self.max_trials is not None and len(self.trials) >= self.max_trials

    def _make_trial_dir(self, trial_index: int) -> Path:
        trial_dir = self.run_dir / f"trial_{trial_index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir

    def _config_fingerprint(self, config: dict) -> str:
        return json.dumps(_summarize_params(config), sort_keys=True, separators=(",", ":"))

    def _changed_params(self, config: dict) -> dict[str, Any]:
        changed: dict[str, Any] = {}
        for path, value in _summarize_params(config).items():
            if value != _get_nested(self.base_config, path):
                changed[path] = value
        return changed

    def _lookup_target_metrics(self, summary: dict) -> tuple[float, float, Optional[int]]:
        predictions = summary.get("video_predictions", [])
        for idx, prediction in enumerate(predictions, start=1):
            if prediction.get("species") == self.target_species:
                return (
                    float(prediction.get("presence_probability", 0.0)),
                    float(prediction.get("raw_presence_probability", 0.0)),
                    idx,
                )
        return 0.0, 0.0, None

    def _lookup_success_metrics(self, summary: dict) -> tuple[Optional[str], float]:
        if not self.success_species_contains:
            return self.target_species, self._lookup_target_metrics(summary)[0]

        predictions = summary.get("video_predictions", [])
        needle = self.success_species_contains.lower()
        best_species = None
        best_confidence = 0.0
        for prediction in predictions:
            species = str(prediction.get("species", ""))
            confidence = float(prediction.get("presence_probability", 0.0))
            if needle in species.lower() and confidence > best_confidence:
                best_species = species
                best_confidence = confidence
        return best_species, best_confidence

    def _record_summary(self, best_trial: Optional[TrialResult]) -> None:
        payload = {
            "run_id": self.run_id,
            "video_path": self.video_path,
            "target_species": self.target_species,
            "success_species_contains": self.success_species_contains,
            "stop_confidence": self.stop_confidence,
            "time_budget_s": round(self.time_budget_s, 2),
            "elapsed_s": round(self._elapsed_s(), 2),
            "max_trials": self.max_trials,
            "stop_reason": self.stop_reason,
            "initial_top_species": self.initial_top_species,
            "best_trial": asdict(best_trial) if best_trial else None,
            "trials": [asdict(trial) for trial in self.trials],
        }
        with open(self.summary_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _run_trial(self, config: dict, *, reason: str) -> TrialResult:
        trial_index = len(self.trials) + 1
        trial_dir = self._make_trial_dir(trial_index)
        trial_config = copy.deepcopy(config)
        trial_config.setdefault("output", {})["results_dir"] = str(trial_dir)
        self.pipeline.apply_config(trial_config)

        logger.info(
            "Trial %03d: %s",
            trial_index,
            ", ".join(f"{key}={value}" for key, value in self._changed_params(trial_config).items()) or "baseline",
        )

        try:
            summary = self.pipeline.process_video(
                self.video_path,
                video_date=self.video_date,
                result_stem=Path(self.video_path).stem,
            )
            target_confidence, raw_target_confidence, target_rank = self._lookup_target_metrics(summary)
            success_species, success_confidence = self._lookup_success_metrics(summary)
            video_predictions = summary.get("video_predictions", [])
            top_prediction = video_predictions[0] if video_predictions else {}
            results_json = trial_dir / f"{Path(self.video_path).stem}_results.json"
            trial = TrialResult(
                trial_index=trial_index,
                reason=reason,
                elapsed_s=round(self._elapsed_s(), 2),
                results_dir=str(trial_dir),
                results_json=str(results_json) if results_json.exists() else None,
                target_species=self.target_species,
                target_confidence=round(target_confidence, 4),
                raw_target_confidence=round(raw_target_confidence, 4),
                target_rank=target_rank,
                success_species=success_species,
                success_confidence=round(success_confidence, 4),
                top_species=top_prediction.get("species"),
                top_confidence=round(float(top_prediction.get("presence_probability", 0.0)), 4),
                reached_target=success_confidence >= self.stop_confidence,
                changed_params=self._changed_params(trial_config),
                tuned_params=_summarize_params(trial_config),
            )
            logger.info(
                "Trial %03d result: target=%s %.1f%% (rank=%s), success=%s %.1f%%, top=%s %.1f%%",
                trial_index,
                self.target_species,
                trial.target_confidence * 100.0,
                trial.target_rank if trial.target_rank is not None else "-",
                trial.success_species or "none",
                trial.success_confidence * 100.0,
                trial.top_species or "none",
                trial.top_confidence * 100.0,
            )
        except Exception as exc:
            trial = TrialResult(
                trial_index=trial_index,
                reason=reason,
                elapsed_s=round(self._elapsed_s(), 2),
                results_dir=str(trial_dir),
                results_json=None,
                target_species=self.target_species,
                target_confidence=0.0,
                raw_target_confidence=0.0,
                target_rank=None,
                success_species=None,
                success_confidence=0.0,
                top_species=None,
                top_confidence=0.0,
                reached_target=False,
                changed_params=self._changed_params(trial_config),
                tuned_params=_summarize_params(trial_config),
                error=str(exc),
            )
            logger.exception("Trial %03d failed", trial_index)

        self.trials.append(trial)
        self.seen_configs.add(self._config_fingerprint(trial_config))
        return trial

    def _is_better(self, candidate: TrialResult, incumbent: Optional[TrialResult]) -> bool:
        if candidate.error:
            return False
        if incumbent is None or incumbent.error:
            return True

        candidate_rank = candidate.target_rank if candidate.target_rank is not None else 9999
        incumbent_rank = incumbent.target_rank if incumbent.target_rank is not None else 9999
        return (
            candidate.target_confidence,
            -candidate_rank,
            candidate.top_species == self.target_species,
            candidate.top_confidence,
        ) > (
            incumbent.target_confidence,
            -incumbent_rank,
            incumbent.top_species == self.target_species,
            incumbent.top_confidence,
        )

    def run(self) -> dict[str, Any]:
        best_config = copy.deepcopy(self.base_config)
        best_trial = self._run_trial(best_config, reason="baseline")
        if best_trial.error:
            self.stop_reason = "baseline_failed"
            self._record_summary(best_trial)
            raise RuntimeError(f"Baseline trial failed: {best_trial.error}")

        self.initial_top_species = best_trial.top_species
        self._record_summary(best_trial)
        if best_trial.reached_target:
            self.stop_reason = "target_reached"
            self._record_summary(best_trial)
            return self._final_payload(best_trial)

        round_index = 1
        while True:
            if self._time_budget_reached():
                self.stop_reason = "time_budget_reached"
                break
            if self._max_trials_reached():
                self.stop_reason = "max_trials_reached"
                break

            round_best_trial: Optional[TrialResult] = None
            round_best_config: Optional[dict] = None

            for path, values in TUNING_SPACE.items():
                current_value = _get_nested(best_config, path)
                for candidate_value in _ordered_unique([*values, current_value]):
                    if candidate_value == current_value:
                        continue
                    if self._time_budget_reached():
                        self.stop_reason = "time_budget_reached"
                        break
                    if self._max_trials_reached():
                        self.stop_reason = "max_trials_reached"
                        break

                    candidate_config = copy.deepcopy(best_config)
                    _set_nested(candidate_config, path, candidate_value)
                    fingerprint = self._config_fingerprint(candidate_config)
                    if fingerprint in self.seen_configs:
                        continue

                    trial = self._run_trial(
                        candidate_config,
                        reason=f"round_{round_index}:{path}={candidate_value}",
                    )
                    if trial.reached_target:
                        self.stop_reason = "target_reached"
                        if self._is_better(trial, best_trial):
                            best_trial = trial
                            best_config = candidate_config
                        self._record_summary(best_trial)
                        return self._final_payload(best_trial)

                    if self._is_better(trial, round_best_trial):
                        round_best_trial = trial
                        round_best_config = candidate_config

                if self.stop_reason in {"time_budget_reached", "max_trials_reached"}:
                    break

            if round_best_trial is not None and self._is_better(round_best_trial, best_trial):
                best_trial = round_best_trial
                best_config = round_best_config or best_config
                self._record_summary(best_trial)

            if self.stop_reason in {"time_budget_reached", "max_trials_reached"}:
                break

            if round_best_trial is None or not self._is_better(round_best_trial, best_trial):
                self.stop_reason = "no_improving_trials"
                break

            best_trial = round_best_trial
            best_config = round_best_config or best_config
            round_index += 1
            self._record_summary(best_trial)

        self._record_summary(best_trial)
        return self._final_payload(best_trial)

    def _final_payload(self, best_trial: TrialResult) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "summary_path": str(self.summary_path),
            "target_species": self.target_species,
            "success_species_contains": self.success_species_contains,
            "stop_confidence": self.stop_confidence,
            "stop_reason": self.stop_reason,
            "elapsed_s": round(self._elapsed_s(), 2),
            "trial_count": len(self.trials),
            "initial_top_species": self.initial_top_species,
            "best_trial": asdict(best_trial),
            "beat_initial_target_confidence": best_trial.target_confidence > self.trials[0].target_confidence,
            "flipped_top_result": best_trial.top_species is not None and best_trial.top_species != self.initial_top_species,
        }
