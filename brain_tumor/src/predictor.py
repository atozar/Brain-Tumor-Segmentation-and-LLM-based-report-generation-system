"""
predictor.py
------------
Tumor segmentation inference module.

Responsibilities:
- Load a trained segmentation model (Keras/TensorFlow or PyTorch)
- Run inference and produce a raw probability mask
- Post-process the raw mask into a binary segmentation
- Compute tumor metrics from the binary mask
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_model(model_path: str) -> Any:
    """
    Load a trained segmentation model from disk.

    Attempts to load a Keras (.h5, .keras) model first; if that fails
    it falls back to a PyTorch state-dict (.pt, .pth).

    Args:
        model_path: Path to the saved model file.

    Returns:
        Loaded model object (framework-specific).

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError:        If the file extension is not supported.
        RuntimeError:      If the model cannot be loaded.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = path.suffix.lower()

    if suffix in {".h5", ".keras"}:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(path))
            return model
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to load Keras models. "
                "Install it with: pip install tensorflow"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load Keras model '{model_path}': {exc}") from exc

    if suffix in {".pt", ".pth"}:
        try:
            import torch
            model = torch.load(str(path), map_location="cpu")
            model.eval()
            return model
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to load .pt/.pth models. "
                "Install it with: pip install torch"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load PyTorch model '{model_path}': {exc}") from exc

    raise ValueError(
        f"Unsupported model format '{suffix}'. "
        "Supported formats: .h5, .keras, .pt, .pth"
    )


def predict(model: Any, image: np.ndarray) -> np.ndarray:
    """
    Run the segmentation model on a preprocessed image.

    The image is expanded to a batch of 1 before inference and the
    batch dimension is removed from the output.

    Args:
        model: A loaded model object with a callable ``predict`` (Keras)
               or ``__call__`` (PyTorch) interface.
        image: Preprocessed image array with shape (H, W) or (H, W, C).

    Returns:
        Raw (probability) output array with shape (H, W) or (H, W, C).

    Raises:
        TypeError:  If ``image`` is not a NumPy array.
        ValueError: If ``image`` is not 2-D or 3-D.
        RuntimeError: If inference fails.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")

    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2-D (H, W) or 3-D (H, W, C), got shape {image.shape}"
        )

    try:
        batch = np.expand_dims(image, axis=0)

        if hasattr(model, "predict"):
            raw = model.predict(batch)
        else:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(batch).float()
                raw = model(tensor).numpy()

        return np.squeeze(raw, axis=0)
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


def postprocess_mask(
    raw_output: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert a raw probability map into a binary segmentation mask.

    Args:
        raw_output: Array of probability values (any shape), typically
                    the direct output of ``predict``.
        threshold:  Decision boundary in [0, 1]. Values >= threshold
                    are classified as tumor. Defaults to 0.5.

    Returns:
        Binary mask (uint8) with the same shape as ``raw_output``,
        where 1 = tumor and 0 = background.

    Raises:
        TypeError:  If ``raw_output`` is not a NumPy array.
        ValueError: If ``threshold`` is outside [0, 1].
    """
    if not isinstance(raw_output, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(raw_output).__name__}")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    return (raw_output >= threshold).astype(np.uint8)


def calculate_tumor_metrics(
    mask: np.ndarray,
    voxel_volume_mm3: float = 1.0,
) -> Dict[str, Any]:
    """
    Derive quantitative metrics from a binary segmentation mask.

    Computed metrics
    ----------------
    - ``tumor_voxels``    : Number of voxels/pixels labelled as tumor.
    - ``total_voxels``    : Total number of voxels/pixels in the mask.
    - ``tumor_volume_mm3``: Estimated tumor volume in cubic millimetres.
    - ``tumor_ratio``     : Fraction of the mask occupied by tumor
                            (0.0 – 1.0).
    - ``has_tumor``       : Boolean indicating any tumor presence.
    - ``bounding_box``    : (row_min, col_min, row_max, col_max) of the
                            tumour region, or ``None`` when no tumor is
                            detected. Only available for 2-D masks.

    Args:
        mask: Binary NumPy array (dtype uint8) with values 0 or 1.
        voxel_volume_mm3: Physical volume of one voxel in mm³.
                          Defaults to 1.0.

    Returns:
        Dictionary of metric names to their values.

    Raises:
        TypeError:  If ``mask`` is not a NumPy array.
        ValueError: If ``mask`` contains values other than 0 and 1,
                    or if ``voxel_volume_mm3`` is not positive.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(mask).__name__}")

    unique_vals = set(np.unique(mask).tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"mask must be binary (values 0 and 1), found: {unique_vals}"
        )

    if voxel_volume_mm3 <= 0:
        raise ValueError(
            f"voxel_volume_mm3 must be positive, got {voxel_volume_mm3}"
        )

    tumor_voxels = int(np.sum(mask))
    total_voxels = int(mask.size)
    tumor_volume_mm3 = tumor_voxels * voxel_volume_mm3
    tumor_ratio = tumor_voxels / total_voxels if total_voxels > 0 else 0.0
    has_tumor = tumor_voxels > 0

    bounding_box: Optional[Tuple[int, int, int, int]] = None
    if mask.ndim == 2 and has_tumor:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        row_min, row_max = int(row_indices[0]), int(row_indices[-1])
        col_min, col_max = int(col_indices[0]), int(col_indices[-1])
        bounding_box = (row_min, col_min, row_max, col_max)

    return {
        "tumor_voxels": tumor_voxels,
        "total_voxels": total_voxels,
        "tumor_volume_mm3": round(tumor_volume_mm3, 4),
        "tumor_ratio": round(tumor_ratio, 6),
        "has_tumor": has_tumor,
        "bounding_box": bounding_box,
    }
