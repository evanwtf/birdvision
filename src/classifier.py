import logging
from typing import List, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BirdClassifier:
    def __init__(
        self,
        model_name: str = "hf-hub:imageomics/bioclip",
        device: str = "cuda",
        top_k: int = 5,
    ):
        self.device = device
        self.top_k = top_k
        self.species_names: List[str] = []
        self.text_features: torch.Tensor = None

        logger.info(f"Loading BioCLIP model: {model_name}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        logger.info("BioCLIP ready.")

    def set_species(self, species_names: List[str], prompt_template: str = "a photo of a {species}"):
        """Pre-compute and cache text embeddings for all candidate species."""
        self.species_names = species_names
        logger.info(f"Computing text embeddings for {len(species_names)} species...")
        prompts = [prompt_template.format(species=s) for s in species_names]

        all_features = []
        batch_size = 256
        for i in range(0, len(prompts), batch_size):
            tokens = self.tokenizer(prompts[i : i + batch_size]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
            all_features.append(feats)

        self.text_features = torch.cat(all_features, dim=0)
        logger.info("Species embeddings ready.")

    def classify_batch(self, crops_bgr: List[np.ndarray]) -> List[List[Tuple[str, float]]]:
        """Classify a batch of BGR image crops. Returns top-k (species, probability) per crop."""
        if not crops_bgr:
            return []
        if self.text_features is None:
            raise RuntimeError("Call set_species() before classifying.")

        images = torch.stack([
            self.preprocess(Image.fromarray(crop[:, :, ::-1]))
            for crop in crops_bgr
        ]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1).cpu().numpy()

        results = []
        for row in probs:
            top_idx = np.argsort(row)[::-1][: self.top_k]
            results.append([(self.species_names[i], float(row[i])) for i in top_idx])
        return results
