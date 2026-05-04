"""
conftest.py
-----------
Shared pytest fixtures for the brain_tumor test suite.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grayscale_image() -> np.ndarray:
    """256x256 float32 grayscale image with values in [0, 255]."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(256, 256)).astype(np.float32)


@pytest.fixture
def rgb_image() -> np.ndarray:
    """128x128x3 float32 RGB image with values in [0, 255]."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(128, 128, 3)).astype(np.float32)


@pytest.fixture
def constant_image() -> np.ndarray:
    """64x64 image where every pixel has the same value (edge case)."""
    return np.full((64, 64), fill_value=128.0, dtype=np.float32)


@pytest.fixture
def normalized_image() -> np.ndarray:
    """128x128 image already in [0, 1]."""
    rng = np.random.default_rng(7)
    return rng.random((128, 128)).astype(np.float32)


# ---------------------------------------------------------------------------
# Mask fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_mask() -> np.ndarray:
    """All-zero 256x256 binary mask (no tumor)."""
    return np.zeros((256, 256), dtype=np.uint8)


@pytest.fixture
def full_mask() -> np.ndarray:
    """All-one 256x256 binary mask (entire image is tumor)."""
    return np.ones((256, 256), dtype=np.uint8)


@pytest.fixture
def partial_mask() -> np.ndarray:
    """256x256 mask with a 50x50 tumour block in the centre."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[103:153, 103:153] = 1
    return mask


@pytest.fixture
def raw_output_high() -> np.ndarray:
    """Probability map — all values >= 0.9 (should be all-tumor after threshold)."""
    return np.full((64, 64), fill_value=0.95, dtype=np.float32)


@pytest.fixture
def raw_output_low() -> np.ndarray:
    """Probability map — all values < 0.1 (should be all-background)."""
    return np.full((64, 64), fill_value=0.05, dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrics fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tumor_metrics_detected():
    """Metrics dict corresponding to a detected tumour."""
    return {
        "has_tumor": True,
        "tumor_voxels": 2500,
        "total_voxels": 65536,
        "tumor_volume_mm3": 2500.0,
        "tumor_ratio": 0.038147,
        "bounding_box": (103, 103, 152, 152),
    }


@pytest.fixture
def tumor_metrics_none():
    """Metrics dict when no tumour is present."""
    return {
        "has_tumor": False,
        "tumor_voxels": 0,
        "total_voxels": 65536,
        "tumor_volume_mm3": 0.0,
        "tumor_ratio": 0.0,
        "bounding_box": None,
    }
