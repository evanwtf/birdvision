"""
Geometric-mean ensemble of BioCLIP and a fixed-vocabulary HF image classifier.

Combines two independent probability distributions over the species list using a
weighted geometric mean:

    combined(s) = bioclip(s)^alpha * secondary(s)^beta   then renormalize

For species outside the secondary model's vocabulary (score == 0.0), a small
uniform floor (1 / n_secondary_classes) is substituted so they are not zeroed
out — the secondary model simply has no opinion, treated as a flat prior.

The combined classifier is a drop-in replacement for BirdClassifier: it exposes
the same set_species() and classify_batch() interface used by the pipeline.
"""

import logging

import numpy as np

from .classifier import BirdClassifier
from .hf_classifier import HFImageClassifier

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    def __init__(
        self,
        bioclip: BirdClassifier,
        secondary: HFImageClassifier,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self._bioclip = bioclip
        self._secondary = secondary
        self.alpha = alpha
        self.beta = beta
        # Neutral prior for species not in the secondary model's vocabulary.
        self._floor = 1.0 / secondary._n_model_labels

    # ---- Pipeline-compatible interface -------------------------------------

    @property
    def top_k(self) -> int:
        return self._bioclip.top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._bioclip.top_k = value

    @property
    def species_names(self) -> list[str]:
        return self._bioclip.species_names

    def set_species(self, species_names: list[str], prompt_template: str = "a photo of a {species}") -> None:
        self._bioclip.set_species(species_names, prompt_template)
        self._secondary.set_species(species_names)

    # ---- Core inference ----------------------------------------------------

    def classify_batch(self, crops_bgr: list) -> list[list[tuple[str, float]]]:
        """Run both classifiers and return geometric-mean combined top-k predictions."""
        if not crops_bgr:
            return []

        species = self.species_names
        n = len(species)
        alpha, beta, floor = self.alpha, self.beta, self._floor

        # Full BioCLIP probability vectors for all crops in one GPU batch.
        all_bc_probs = self._bioclip.classify_batch_all_scores(crops_bgr)

        results = []
        for crop, bc_probs in zip(crops_bgr, all_bc_probs):
            # Secondary model scores (dict over all species; 0.0 for unmapped).
            hf_scores = self._secondary.classify(crop)

            hf_vec = np.array([hf_scores.get(s, 0.0) for s in species], dtype=np.float64)
            hf_vec[hf_vec == 0.0] = floor  # neutral prior for unmapped species

            combined = (bc_probs.astype(np.float64) ** alpha) * (hf_vec ** beta)
            total = combined.sum()
            if total > 0:
                combined /= total

            top_idx = np.argsort(combined)[::-1][: self.top_k]
            results.append([(species[i], float(combined[i])) for i in top_idx])

        return results
