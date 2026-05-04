"""
preprocessing.py
----------------
MRI image preprocessing pipeline for brain tumor segmentation.

Responsibilities:
- Load MRI images (NIfTI or standard formats)
- Normalize pixel intensities
- Resize images to a target resolution
- Orchestrate the full preprocessing pipeline
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Load an MRI image from disk.

    Supports NIfTI (.nii, .nii.gz) and standard image formats
    (.png, .jpg, .jpeg, .bmp, .tiff).

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        A NumPy array containing the image data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        RuntimeError: If the file cannot be read.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    suffix = file_path.suffix.lower()
    name_lower = file_path.name.lower()

    if name_lower.endswith(".nii.gz") or suffix == ".nii":
        try:
            import nibabel as nib
            img = nib.load(str(file_path))
            data = img.get_fdata()
            return data.astype(np.float32)
        except ImportError as exc:
            raise RuntimeError(
                "nibabel is required to load NIfTI files. "
                "Install it with: pip install nibabel"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load NIfTI file '{path}': {exc}") from exc

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
        try:
            from PIL import Image
            img = Image.open(str(file_path))
            return np.array(img, dtype=np.float32)
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required to load standard image formats. "
                "Install it with: pip install Pillow"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load image file '{path}': {exc}") from exc

    raise ValueError(
        f"Unsupported file format '{suffix}'. "
        "Supported formats: .nii, .nii.gz, .png, .jpg, .jpeg, .bmp, .tiff"
    )


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale pixel values into [0, 1].

    If the image is constant (max == min), returns a zero array of the
    same shape and dtype to avoid division by zero.

    Args:
        image: Input NumPy array of any shape.

    Returns:
        Normalized NumPy array with float32 dtype.

    Raises:
        TypeError: If the input is not a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")

    image = image.astype(np.float32)
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_val) / (max_val - min_val)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2-D (or 2-D + channels) image to the given target size.

    Args:
        image: NumPy array with shape (H, W) or (H, W, C).
        target_size: Desired output size as (height, width).

    Returns:
        Resized NumPy array with float32 dtype.

    Raises:
        TypeError:  If ``image`` is not a NumPy array or
                    ``target_size`` is not a tuple/list of two ints.
        ValueError: If ``image`` is not 2-D or 3-D, or if target
                    dimensions are not positive integers.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")

    if not (isinstance(target_size, (tuple, list)) and len(target_size) == 2):
        raise TypeError("target_size must be a tuple or list of two integers")

    height, width = target_size
    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError("target_size elements must be integers")

    if height <= 0 or width <= 0:
        raise ValueError(f"target_size dimensions must be positive, got ({height}, {width})")

    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2-D (H, W) or 3-D (H, W, C), got shape {image.shape}"
        )

    try:
        from PIL import Image as PILImage
        pil_mode = "F" if image.ndim == 2 else None
        pil_img = PILImage.fromarray(image if image.ndim == 2 else image.astype(np.uint8))
        resized = pil_img.resize((width, height), PILImage.BILINEAR)
        return np.array(resized, dtype=np.float32)
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for image resizing. "
            "Install it with: pip install Pillow"
        ) from exc


def preprocess_pipeline(
    path: str,
    target_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Full MRI preprocessing pipeline: load → normalize → resize.

    Args:
        path: Path to the MRI image file.
        target_size: Output spatial dimensions (height, width).
                     Defaults to (256, 256).

    Returns:
        Preprocessed NumPy array ready for model inference.

    Raises:
        FileNotFoundError: Propagated from ``load_image``.
        ValueError:        Propagated from ``load_image`` / ``resize_image``.
        RuntimeError:      Propagated from ``load_image`` / ``resize_image``.
    """
    image = load_image(path)
    image = normalize(image)
    image = resize_image(image, target_size)
    return image
