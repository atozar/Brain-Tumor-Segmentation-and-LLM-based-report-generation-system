"""
test_preprocessing.py
---------------------
Unit tests for brain_tumor/src/preprocessing.py

Coverage targets
----------------
- load_image: valid paths, missing files, unsupported formats, format dispatch
- normalize: range correctness, constant image, dtype, non-array input
- resize_image: output shape, float output, invalid inputs, edge shapes
- preprocess_pipeline: integration of load → normalize → resize
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import load_image, normalize, resize_image, preprocess_pipeline


# =============================================================================
# load_image
# =============================================================================

class TestLoadImage:

    def test_raises_file_not_found_for_missing_path(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_image("/nonexistent/path/image.png")

    def test_raises_value_error_for_unsupported_extension(self, tmp_path):
        fake = tmp_path / "scan.dcm"
        fake.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_image(str(fake))

    def test_loads_png_image(self, tmp_path):
        from PIL import Image
        arr = np.zeros((64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        path = tmp_path / "test.png"
        img.save(str(path))

        result = load_image(str(path))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_loads_jpg_image(self, tmp_path):
        from PIL import Image
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        path = tmp_path / "test.jpg"
        img.save(str(path))

        result = load_image(str(path))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_loads_nifti_image(self, tmp_path):
        """load_image dispatches to nibabel for .nii files."""
        nii_path = tmp_path / "scan.nii"
        nii_path.write_bytes(b"fake")  # file exists check passes

        fake_img = MagicMock()
        fake_img.get_fdata.return_value = np.zeros((10, 10, 10), dtype=np.float64)

        with patch.dict("sys.modules", {"nibabel": MagicMock(load=MagicMock(return_value=fake_img))}):
            import importlib
            import preprocessing as pp
            importlib.reload(pp)

            with patch("nibabel.load", return_value=fake_img):
                result = pp.load_image(str(nii_path))

        assert isinstance(result, np.ndarray)

    def test_raises_runtime_error_when_pillow_missing_for_png(self, tmp_path):
        from PIL import Image
        arr = np.zeros((16, 16), dtype=np.uint8)
        path = tmp_path / "img.png"
        Image.fromarray(arr, "L").save(str(path))

        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            import importlib
            import preprocessing as pp
            importlib.reload(pp)
            with pytest.raises(RuntimeError, match="Pillow"):
                pp.load_image(str(path))


# =============================================================================
# normalize
# =============================================================================

class TestNormalize:

    def test_output_range_is_0_to_1(self, grayscale_image):
        result = normalize(grayscale_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_is_float32(self, grayscale_image):
        result = normalize(grayscale_image)
        assert result.dtype == np.float32

    def test_constant_image_returns_zeros(self, constant_image):
        result = normalize(constant_image)
        assert np.all(result == 0.0)

    def test_already_normalized_image_stays_in_range(self, normalized_image):
        result = normalize(normalized_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_raises_type_error_for_non_array(self):
        with pytest.raises(TypeError, match="np.ndarray"):
            normalize([[1, 2], [3, 4]])

    def test_min_becomes_zero(self, grayscale_image):
        result = normalize(grayscale_image)
        assert result.min() == pytest.approx(0.0, abs=1e-6)

    def test_max_becomes_one(self, grayscale_image):
        result = normalize(grayscale_image)
        assert result.max() == pytest.approx(1.0, abs=1e-6)

    def test_3d_image_normalization(self, rgb_image):
        result = normalize(rgb_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize("values,expected_min,expected_max", [
        ([-10, 0, 10], 0.0, 1.0),
        ([5, 5, 5], 0.0, 0.0),
        ([0, 255], 0.0, 1.0),
    ])
    def test_parametrized_ranges(self, values, expected_min, expected_max):
        arr = np.array(values, dtype=np.float32)
        result = normalize(arr)
        assert result.min() == pytest.approx(expected_min, abs=1e-6)
        assert result.max() == pytest.approx(expected_max, abs=1e-6)


# =============================================================================
# resize_image
# =============================================================================

class TestResizeImage:

    def test_output_shape_matches_target(self, grayscale_image):
        result = resize_image(grayscale_image, (64, 64))
        assert result.shape == (64, 64)

    def test_output_dtype_is_float32(self, grayscale_image):
        result = resize_image(grayscale_image, (64, 64))
        assert result.dtype == np.float32

    def test_upscale(self, grayscale_image):
        result = resize_image(grayscale_image, (512, 512))
        assert result.shape == (512, 512)

    def test_rgb_image_resize(self, rgb_image):
        result = resize_image(rgb_image, (64, 64))
        assert result.shape[0] == 64
        assert result.shape[1] == 64

    def test_raises_type_error_for_non_array(self):
        with pytest.raises(TypeError, match="np.ndarray"):
            resize_image([[1, 2], [3, 4]], (64, 64))

    def test_raises_type_error_for_invalid_target_size_type(self, grayscale_image):
        with pytest.raises(TypeError):
            resize_image(grayscale_image, 64)

    def test_raises_type_error_for_float_target_size(self, grayscale_image):
        with pytest.raises(TypeError):
            resize_image(grayscale_image, (64.0, 64.0))

    def test_raises_value_error_for_zero_dimension(self, grayscale_image):
        with pytest.raises(ValueError, match="positive"):
            resize_image(grayscale_image, (0, 64))

    def test_raises_value_error_for_negative_dimension(self, grayscale_image):
        with pytest.raises(ValueError, match="positive"):
            resize_image(grayscale_image, (-1, 64))

    def test_raises_value_error_for_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            resize_image(np.array([1.0, 2.0, 3.0]), (64, 64))

    def test_raises_value_error_for_4d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            resize_image(np.zeros((2, 64, 64, 3)), (64, 64))

    @pytest.mark.parametrize("target", [(32, 32), (128, 64), (64, 128)])
    def test_parametrized_target_sizes(self, grayscale_image, target):
        result = resize_image(grayscale_image, target)
        assert result.shape[0] == target[0]
        assert result.shape[1] == target[1]


# =============================================================================
# preprocess_pipeline
# =============================================================================

class TestPreprocessPipeline:

    def test_integration_produces_correct_shape(self, tmp_path):
        from PIL import Image
        arr = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        path = tmp_path / "mri.png"
        Image.fromarray(arr, "L").save(str(path))

        result = preprocess_pipeline(str(path), target_size=(128, 128))
        assert result.shape == (128, 128)

    def test_integration_output_normalized(self, tmp_path):
        from PIL import Image
        arr = np.random.randint(10, 200, (64, 64), dtype=np.uint8)
        path = tmp_path / "mri.png"
        Image.fromarray(arr, "L").save(str(path))

        result = preprocess_pipeline(str(path), target_size=(64, 64))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_propagates_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            preprocess_pipeline("/no/such/file.png")

    def test_default_target_size_is_256(self, tmp_path):
        from PIL import Image
        arr = np.zeros((100, 100), dtype=np.uint8)
        path = tmp_path / "mri.png"
        Image.fromarray(arr, "L").save(str(path))

        result = preprocess_pipeline(str(path))
        assert result.shape == (256, 256)
