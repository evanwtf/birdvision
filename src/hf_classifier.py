"""
Generic HuggingFace image-classification backend.

Wraps any AutoModelForImageClassification model (e.g. EfficientNet trained on
BIRDS-525) for use in the eval pipeline. Because these models have a fixed
output vocabulary that may not match our species list exactly, this class
builds a label mapping at init time and aggregates scores through it.

Mapping strategy (applied in order):
  1. Exact case-insensitive match
  2. Manual overrides (NAME_OVERRIDES dict below)
  3. Fuzzy match via difflib (cutoff configurable)
  4. No match — model's score for that label is discarded

Labels that map to a species not in our list are also discarded.

Usage:
    clf = HFImageClassifier(
        model_name="chriamue/bird-species-classifier",
        species_list=species_list,
    )
    scores = clf.classify(crop_bgr)   # dict[str, float]
"""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Manual name overrides: model label (upper) -> our species list name (exact).
# Add entries here whenever a model uses a different common name than we do.
NAME_OVERRIDES: dict[str, str] = {
    "MALLARD DUCK": "Mallard",
    "COMMON STARLING": "European Starling",
    "SUPERB STARLING": None,  # exotic, not in our list
    "CASPIAN TERN": "Caspian Tern",
    "YELLOW WARBLER": "Yellow Warbler",
    "COMMON YELLOWTHROAT": "Common Yellowthroat",
    "DARK EYED JUNCO": "Dark-eyed Junco",
    "RUBY THROATED HUMMINGBIRD": "Ruby-throated Hummingbird",
    "AMERICAN GOLDEN PLOVER": "American Golden-Plover",
    "BLACK BELLIED PLOVER": "Black-bellied Plover",
    "SEMIPALMATED PLOVER": "Semipalmated Plover",
    "GREATER YELLOWLEGS": "Greater Yellowlegs",
    "LESSER YELLOWLEGS": "Lesser Yellowlegs",
    "DUNLIN": "Dunlin",
    "LEAST SANDPIPER": "Least Sandpiper",
    "SEMIPALMATED SANDPIPER": "Semipalmated Sandpiper",
    "WHIMBREL": "Whimbrel",
    "LONG BILLED DOWITCHER": "Long-billed Dowitcher",
    "SHORT BILLED DOWITCHER": "Short-billed Dowitcher",
    "WILSON SNIPE": "Wilson's Snipe",
    "LAUGHING GULL": "Laughing Gull",
    "RING BILLED GULL": "Ring-billed Gull",
    "HERRING GULL": "Herring Gull",
    "GREAT BLACK BACKED GULL": "Great Black-backed Gull",
    "LEAST TERN": "Least Tern",
    "COMMON TERN": "Common Tern",
    "FORSTERS TERN": "Forster's Tern",
    "BLACK SKIMMER": "Black Skimmer",
    "RUBY CROWNED KINGLET": "Ruby-crowned Kinglet",
    "GOLDEN CROWNED KINGLET": "Golden-crowned Kinglet",
    "RED BREASTED NUTHATCH": "Red-breasted Nuthatch",
    "WHITE BREASTED NUTHATCH": "White-breasted Nuthatch",
    "BROWN HEADED COWBIRD": "Brown-headed Cowbird",
    "RED WINGED BLACKBIRD": "Red-winged Blackbird",
    "YELLOW RUMPED WARBLER": "Yellow-rumped Warbler",
    "BLACKPOLL WARBLER": "Blackpoll Warbler",
    "BLACK AND WHITE WARBLER": "Black-and-white Warbler",
    "COMMON YELLOWTHROAT": "Common Yellowthroat",
    "AMERICAN REDSTART": "American Redstart",
    "CHIPPING SPARROW": "Chipping Sparrow",
    "FIELD SPARROW": "Field Sparrow",
    "WHITE THROATED SPARROW": "White-throated Sparrow",
    "WHITE CROWNED SPARROW": "White-crowned Sparrow",
    "FOX SPARROW": "Fox Sparrow",
    "SWAMP SPARROW": "Swamp Sparrow",
    "EASTERN TOWHEE": "Eastern Towhee",
    "BALTIMORE ORIOLE": "Baltimore Oriole",
    "ORCHARD ORIOLE": "Orchard Oriole",
    "ROSE BREASTED GROSBEAK": "Rose-breasted Grosbeak",
    "INDIGO BUNTING": "Indigo Bunting",
    "HOUSE FINCH": "House Finch",
    "PURPLE FINCH": "Purple Finch",
    "PINE SISKIN": "Pine Siskin",
    "AMERICAN GOLDFINCH": "American Goldfinch",
    "CEDAR WAXWING": "Cedar Waxwing",
    "BELTED KINGFISHER": "Belted Kingfisher",
    "RED BELLIED WOODPECKER": "Red-bellied Woodpecker",
    "HAIRY WOODPECKER": "Hairy Woodpecker",
    "NORTHERN FLICKER": "Northern Flicker",
    "PILEATED WOODPECKER": "Pileated Woodpecker",
    "EASTERN WOOD PEWEE": "Eastern Wood-Pewee",
    "EASTERN PHOEBE": "Eastern Phoebe",
    "GREAT CRESTED FLYCATCHER": "Great Crested Flycatcher",
    "EASTERN KINGBIRD": "Eastern Kingbird",
    "TREE SWALLOW": "Tree Swallow",
    "BARN SWALLOW": "Barn Swallow",
    "CLIFF SWALLOW": "Cliff Swallow",
    "PURPLE MARTIN": "Purple Martin",
    "BLUE GRAY GNATCATCHER": "Blue-gray Gnatcatcher",
    "HOUSE WREN": "House Wren",
    "CAROLINA WREN": "Carolina Wren",
    "GRAY CATBIRD": "Gray Catbird",
    "BROWN THRASHER": "Brown Thrasher",
    "NORTHERN MOCKINGBIRD": "Northern Mockingbird",
    "EASTERN BLUEBIRD": "Eastern Bluebird",
    "WOOD THRUSH": "Wood Thrush",
    "HERMIT THRUSH": "Hermit Thrush",
    "SWAINSON THRUSH": "Swainson's Thrush",
    "VEERY": "Veery",
}


class HFImageClassifier:
    def __init__(
        self,
        model_name: str,
        species_list: list[str],
        device: str = "cuda",
        fuzzy_cutoff: float = 0.75,
        top_k: int = 10,
    ):
        from transformers import pipeline as hf_pipeline

        logger.info("Loading HF image classifier: %s", model_name)
        self._pipe = hf_pipeline(
            task="image-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
            top_k=1000,  # large enough to cover any fixed-vocabulary model
        )
        self._fuzzy_cutoff = fuzzy_cutoff
        self.top_k = top_k

        # Discover all model output labels once via a dummy inference.
        self._model_labels: list[str] = [
            r["label"] for r in self._pipe(Image.new("RGB", (224, 224)), top_k=1000)
        ]
        self._n_model_labels = len(self._model_labels)
        logger.info(
            "HF classifier loaded: %s  (%d output classes)", model_name, self._n_model_labels
        )

        self.species_list: list[str] = []
        self._label_map: dict[str, str | None] = {}
        if species_list:
            self.set_species(species_list)

    def set_species(self, species_list: list[str]) -> None:
        """Set (or replace) the candidate species list and rebuild the label map."""
        self.species_list = list(species_list)
        self._label_map = self._build_label_map(species_list, self._fuzzy_cutoff)
        n_mapped = sum(1 for v in self._label_map.values() if v is not None)
        logger.info(
            "HF classifier ready: %d/%d model labels mapped to species list",
            n_mapped,
            self._n_model_labels,
        )

    def _build_label_map(
        self, species_list: list[str], fuzzy_cutoff: float
    ) -> dict[str, str | None]:
        """Build model_label_upper -> our_species_name (or None to discard)."""
        import difflib

        species_upper = {s.upper(): s for s in species_list}
        result: dict[str, str | None] = {}

        for label in self._model_labels:
            label_upper = label.upper()

            # 1. Manual override
            if label_upper in NAME_OVERRIDES:
                result[label_upper] = NAME_OVERRIDES[label_upper]
                continue

            # 2. Exact case-insensitive
            if label_upper in species_upper:
                result[label_upper] = species_upper[label_upper]
                continue

            # 3. Fuzzy
            matches = difflib.get_close_matches(
                label_upper,
                list(species_upper.keys()),
                n=1,
                cutoff=fuzzy_cutoff,
            )
            if matches:
                result[label_upper] = species_upper[matches[0]]
                logger.debug("Fuzzy label map: %r -> %r", label, species_upper[matches[0]])
            else:
                result[label_upper] = None  # discard

        return result

    def classify(self, crop_bgr: np.ndarray) -> dict[str, float]:
        """Return per-species scores summed from all mapped model labels."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        predictions = self._pipe(image, top_k=1000)

        scores: dict[str, float] = {s: 0.0 for s in self.species_list}
        for pred in predictions:
            label_upper = pred["label"].upper()
            target = self._label_map.get(label_upper)
            if target and target in scores:
                scores[target] += pred["score"]

        # Renormalize over mapped species only
        total = sum(scores.values())
        if total > 0:
            scores = {s: v / total for s, v in scores.items()}

        return scores
