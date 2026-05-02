"""Hailo-8 EfficientNet-S classifier backend for the Pi real-time pipeline.

Implements the same interface as classifier.py (BirdClassifier) so it can
be swapped in by realtime_pipeline.py without changing call sites.

Requires hailort Python bindings (hailo_platform) — Pi only.
Install: uv sync --group pi
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Resize BGR crop to 224×224 and apply ImageNet normalization → HWC float32."""
    from PIL import Image
    img = Image.fromarray(crop_bgr[:, :, ::-1]).resize(  # BGR→RGB
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD            # HWC normalized


class HailoClassifier:
    """EfficientNet-S bird species classifier via Hailo-8 HEF.

    Drop-in replacement for BirdClassifier from classifier.py.

    Args:
        hef_path:    Path to efficientnet_s_birds.hef
        labels_path: Path to species_labels.json (class index → name)
        top_k:       Number of top predictions to return per crop
        device:      Ignored (present for interface compatibility)
    """

    def __init__(
        self,
        hef_path: str,
        labels_path: str,
        top_k: int = 5,
        device: str = "hailo",  # ignored, kept for interface compat
        vdevice=None,           # shared VDevice instance; created if not provided
    ):
        self.top_k = top_k
        self.hef_path = str(hef_path)
        self._shared_vdevice = vdevice

        logger.info("Loading species labels: %s", labels_path)
        self.species_names: List[str] = json.loads(Path(labels_path).read_text())
        logger.info("Loaded %d species labels", len(self.species_names))

        self._init_hailo()

    def _init_hailo(self) -> None:
        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        logger.info("Loading HEF: %s", self.hef_path)
        hef = HEF(self.hef_path)

        if self._shared_vdevice is not None:
            self._target = self._shared_vdevice
        else:
            self._target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = self._target.configure(hef, configure_params)
        self._network_group = network_groups[0]
        self._network_group_params = self._network_group.create_params()

        self._input_vstream_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )

        # Discover input/output layer names from the HEF
        input_info = hef.get_input_vstream_infos()
        output_info = hef.get_output_vstream_infos()
        self._input_name = input_info[0].name
        self._output_name = output_info[0].name
        logger.info(
            "Hailo streams — input: %s  output: %s", self._input_name, self._output_name
        )
        logger.info("HailoClassifier ready.")

    # ------------------------------------------------------------------
    # Public interface (matches classifier.py BirdClassifier)
    # ------------------------------------------------------------------

    def set_species(self, species_names: List[str], **kwargs) -> None:
        """No-op — species order is fixed at training/compilation time.

        Validates that the provided list matches the loaded labels if given.
        """
        if species_names and species_names != self.species_names:
            logger.warning(
                "set_species() called with %d names but HEF was compiled for %d; "
                "ignoring — using labels from species_labels.json",
                len(species_names), len(self.species_names),
            )

    def classify_batch(
        self, crops_bgr: List[np.ndarray]
    ) -> List[List[Tuple[str, float]]]:
        """Classify a batch of BGR crops. Returns top-k (species, score) per crop."""
        if not crops_bgr:
            return []
        probs = self._run_inference(crops_bgr)
        results = []
        for row in probs:
            top_idx = np.argsort(row)[::-1][: self.top_k]
            results.append([(self.species_names[i], float(row[i])) for i in top_idx])
        return results

    def classify_batch_all_scores(
        self, crops_bgr: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Like classify_batch but returns the full probability vector per crop."""
        if not crops_bgr:
            return []
        probs = self._run_inference(crops_bgr)
        return [probs[i] for i in range(len(crops_bgr))]

    # ------------------------------------------------------------------
    # Internal inference
    # ------------------------------------------------------------------

    def _run_inference(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Preprocess crops, run Hailo inference, return softmax probs (N, classes)."""
        from hailo_platform import InferVStreams

        batch = np.ascontiguousarray(
            np.stack([_preprocess_crop(c) for c in crops_bgr])  # (N, 224, 224, 3)
        )

        t0 = time.perf_counter()
        with InferVStreams(
            self._network_group,
            self._input_vstream_params,
            self._output_vstream_params,
        ) as pipeline:
            with self._network_group.activate(self._network_group_params):
                input_data = {self._input_name: batch}
                raw_output = pipeline.infer(input_data)

        logits = raw_output[self._output_name]  # (N, num_classes)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Hailo inference: batch=%d  %.1f ms (%.0f ms/crop)",
            len(crops_bgr), elapsed_ms, elapsed_ms / len(crops_bgr),
        )

        # Softmax over class dimension
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
