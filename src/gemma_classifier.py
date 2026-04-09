"""
Gemma 4 vision-language model classifier.

Wraps a Gemma 4 instruction-tuned model as a drop-in alternative to
BirdClassifier. Given a BGR crop, it prompts the model with the full
species list and returns a scores dict where the matched species gets 1.0
and all others get 0.0.

Usage:
    classifier = GemmaClassifier(
        model_name="google/gemma-4-E4B-it",
        species_list=species_list,
    )
    scores = classifier.classify(crop_bgr)  # dict[str, float]
"""

import logging
import time
import warnings

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class GemmaClassifier:
    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it",
        species_list: list[str] = (),
        device: str = "cuda",
        location_hint: str = "",
    ):
        """
        Args:
            model_name: HuggingFace model ID.
            species_list: Full list of candidate species names. Must be set
                before calling classify().
            device: Ignored — device_map="auto" is used; kept for interface
                compatibility with BirdClassifier.
            location_hint: Optional natural-language location appended to the
                prompt, e.g. "Nassau County, Long Island, New York". Helps the
                model apply ecological context.
        """
        from transformers import pipeline as hf_pipeline

        # Suppress transformers generation-config conflict messages.
        # These are emitted via the transformers logger (not warnings.warn),
        # so PYTHONWARNINGS and warnings.filterwarnings have no effect on them.
        # The behaviour is correct (max_new_tokens wins); this is pure noise.
        import re as _re
        class _GenerationNoiseFilter(logging.Filter):
            _patterns = [
                _re.compile(r"Both `max_new_tokens`.*and `max_length`.*seem to have been set"),
                _re.compile(r"Passing `generation_config` together with generation-related"),
                _re.compile(r"You seem to be using the pipelines sequentially on GPU"),
            ]
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                return not any(p.search(msg) for p in self._patterns)

        _filter = _GenerationNoiseFilter()
        for _name in ("transformers.generation.utils", "transformers.pipelines.base"):
            logging.getLogger(_name).addFilter(_filter)

        logger.info("Loading Gemma classifier: %s (4-bit quantization)", model_name)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        # Pass device_map inside model_kwargs, not as a top-level pipeline arg.
        # When device_map is a top-level arg, the pipeline sets self.device and
        # calls model.to(device) after loading, which OOMs on a quantized model.
        # Keeping it in model_kwargs lets the model load directly onto the right
        # devices without a second move.
        self._pipe = hf_pipeline(
            task="image-text-to-text",
            model=model_name,
            model_kwargs={
                "quantization_config": bnb_config,
                "device_map": "auto",
            },
        )
        # Gemma 4's bundled generation_config has max_length=20. Passing
        # max_new_tokens at call time triggers a deprecation warning because
        # the pipeline also forwards generation_config. Set max_new_tokens
        # on the config and clear max_length so no call-site kwarg is needed.
        self._pipe.model.generation_config.max_new_tokens = 32
        self._pipe.model.generation_config.max_length = None
        self.species_list = list(species_list)
        self._species_set = set(species_list)
        self._species_lower = {s.lower(): s for s in species_list}
        self._location_hint = location_hint
        self._prompt: str = ""
        if species_list:
            self._build_prompt()
        logger.info("Gemma classifier ready: %s", model_name)

    def set_species(self, species_list: list[str]) -> None:
        """Set (or replace) the candidate species list and rebuild the prompt."""
        self.species_list = list(species_list)
        self._species_set = set(species_list)
        self._species_lower = {s.lower(): s for s in species_list}
        self._build_prompt()

    def _build_prompt(self) -> None:
        location = (
            f" photographed in {self._location_hint}"
            if self._location_hint
            else ""
        )
        # Don't embed the full species list in the prompt — 238 species makes
        # the context huge and causes the model to fixate on list structure
        # rather than the image. Ask for the species name and fuzzy-match
        # the output against our list instead.
        region = self._location_hint or "the northeastern United States"
        self._prompt = (
            f"This is a photo from a backyard bird feeder camera in {region}. "
            f"A red bounding box marks the bird to identify. "
            f"What species is inside the red box? "
            f"Reply with only the North American common name, nothing else."
        )

    def classify(self, crop_bgr: np.ndarray) -> dict[str, float]:
        """
        Run inference on one BGR crop.

        Returns a dict mapping every species to a score. The matched species
        gets 1.0; all others get 0.0. Returns all-zeros if the model output
        cannot be matched to any species in the list.
        """
        if not self.species_list:
            raise RuntimeError("Call set_species() before classifying.")

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._prompt},
            ],
        }]

        t0 = time.monotonic()
        result = self._pipe(messages, max_new_tokens=16, return_full_text=False)
        elapsed = time.monotonic() - t0

        raw = result[0]["generated_text"].strip()
        logger.debug("Gemma raw output %r  (%.2fs)", raw, elapsed)

        matched = self._match_species(raw)
        if matched is None:
            logger.warning("Gemma output %r did not match any species in the list", raw)

        scores = {s: 0.0 for s in self.species_list}
        if matched:
            scores[matched] = 1.0
        return scores

    @staticmethod
    def _clean_output(raw: str) -> str:
        """Strip Gemma special tokens and whitespace from model output.

        Gemma 4 appends chat-template delimiters like '<turn|>' to its
        generated text. Strip anything of the form <...> before matching.
        """
        import re
        return re.sub(r"<[^>]*>", "", raw).strip()

    def _match_species(self, raw: str) -> str | None:
        """Exact → case-insensitive → fuzzy → None."""
        import difflib
        cleaned = self._clean_output(raw)
        if not cleaned:
            return None
        # Exact
        if cleaned in self._species_set:
            return cleaned
        # Case-insensitive
        if hit := self._species_lower.get(cleaned.lower()):
            return hit
        # Fuzzy: whole-string similarity
        matches = difflib.get_close_matches(cleaned, self.species_list, n=1, cutoff=0.6)
        if matches:
            logger.debug("Fuzzy match %r -> %r", cleaned, matches[0])
            return matches[0]
        # Fuzzy: try case-folded
        matches = difflib.get_close_matches(
            cleaned.lower(),
            [s.lower() for s in self.species_list],
            n=1, cutoff=0.6,
        )
        if matches:
            canonical = self._species_lower[matches[0]]
            logger.debug("Fuzzy match (lower) %r -> %r", cleaned, canonical)
            return canonical
        return None
