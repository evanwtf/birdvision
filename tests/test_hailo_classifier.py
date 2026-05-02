"""Tests for Pi Hailo classifier helpers."""

import numpy as np

from src.hailo_classifier import _preprocess_crop


def test_preprocess_crop_returns_hwc_tensor():
    crop = np.zeros((32, 48, 3), dtype=np.uint8)

    result = _preprocess_crop(crop)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
