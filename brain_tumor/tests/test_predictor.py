"""
test_predictor.py
-----------------
Unit tests for brain_tumor/src/predictor.py

Coverage targets
----------------
- load_model: missing file, unsupported format, Keras dispatch, PyTorch dispatch
- predict: shape handling, batch dim, inference errors, type validation
- postprocess_mask: thresholding, boundary values, invalid inputs
- calculate_tumor_metrics: empty mask, full mask, partial mask, bounding box,
                           voxel volume scaling, invalid inputs
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictor import (
    load_model,
    predict,
    postprocess_mask,
    calculate_tumor_metrics,
)


# =============================================================================
# load_model
# =============================================================================

class TestLoadModel:

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model("/no/such/model.h5")

    def test_raises_value_error_for_unsupported_extension(self, tmp_path):
        fake = tmp_path / "model.onnx"
        fake.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="Unsupported model format"):
            load_model(str(fake))

    def test_keras_model_loaded_successfully(self, tmp_path):
        model_path = tmp_path / "model.h5"
        model_path.write_bytes(b"\x00")

        fake_model = MagicMock()
        fake_tf = MagicMock()
        fake_tf.keras.models.load_model.return_value = fake_model

        with patch.dict("sys.modules", {"tensorflow": fake_tf}):
            result = load_model(str(model_path))

        assert result is fake_model

    def test_pytorch_model_loaded_successfully(self, tmp_path):
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"\x00")

        fake_model = MagicMock()
        fake_torch = MagicMock()
        fake_torch.load.return_value = fake_model

        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = load_model(str(model_path))

        assert result is fake_model
        fake_model.eval.assert_called_once()

    def test_raises_runtime_when_tensorflow_missing(self, tmp_path):
        model_path = tmp_path / "model.keras"
        model_path.write_bytes(b"\x00")

        with patch.dict("sys.modules", {"tensorflow": None}):
            with pytest.raises(RuntimeError, match="TensorFlow"):
                load_model(str(model_path))

    def test_raises_runtime_when_torch_missing(self, tmp_path):
        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"\x00")

        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(RuntimeError, match="PyTorch"):
                load_model(str(model_path))


# =============================================================================
# predict
# =============================================================================

class TestPredict:

    def _make_keras_model(self, output_shape):
        model = MagicMock()
        model.predict.return_value = np.zeros((1,) + output_shape, dtype=np.float32)
        del model.__call__  # ensure predict path is taken
        return model

    def test_output_squeezes_batch_dim_2d(self, normalized_image):
        model = self._make_keras_model(output_shape=normalized_image.shape)
        result = predict(model, normalized_image)
        assert result.shape == normalized_image.shape

    def test_output_squeezes_batch_dim_3d(self, rgb_image):
        model = self._make_keras_model(output_shape=rgb_image.shape)
        result = predict(model, rgb_image)
        assert result.shape == rgb_image.shape

    def test_model_predict_called_once(self, normalized_image):
        model = self._make_keras_model(output_shape=normalized_image.shape)
        predict(model, normalized_image)
        model.predict.assert_called_once()

    def test_batch_has_correct_shape_when_called(self, normalized_image):
        captured = {}
        model = MagicMock()

        def fake_predict(batch):
            captured["shape"] = batch.shape
            return np.zeros((1,) + normalized_image.shape, dtype=np.float32)

        model.predict.side_effect = fake_predict
        predict(model, normalized_image)
        assert captured["shape"] == (1,) + normalized_image.shape

    def test_raises_type_error_for_non_array(self):
        model = MagicMock()
        with pytest.raises(TypeError, match="np.ndarray"):
            predict(model, [[0.1, 0.2], [0.3, 0.4]])

    def test_raises_value_error_for_1d_input(self):
        model = MagicMock()
        with pytest.raises(ValueError, match="2-D"):
            predict(model, np.array([0.1, 0.2, 0.3]))

    def test_raises_value_error_for_4d_input(self):
        model = MagicMock()
        with pytest.raises(ValueError, match="2-D"):
            predict(model, np.zeros((2, 64, 64, 3)))

    def test_raises_runtime_on_inference_failure(self, normalized_image):
        model = MagicMock()
        model.predict.side_effect = Exception("CUDA out of memory")
        with pytest.raises(RuntimeError, match="Inference failed"):
            predict(model, normalized_image)


# =============================================================================
# postprocess_mask
# =============================================================================

class TestPostprocessMask:

    def test_high_probabilities_become_one(self, raw_output_high):
        result = postprocess_mask(raw_output_high, threshold=0.5)
        assert np.all(result == 1)

    def test_low_probabilities_become_zero(self, raw_output_low):
        result = postprocess_mask(raw_output_low, threshold=0.5)
        assert np.all(result == 0)

    def test_output_dtype_is_uint8(self, raw_output_high):
        result = postprocess_mask(raw_output_high)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self, raw_output_high):
        result = postprocess_mask(raw_output_high)
        assert result.shape == raw_output_high.shape

    def test_exact_threshold_boundary_included(self):
        arr = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        result = postprocess_mask(arr, threshold=0.5)
        np.testing.assert_array_equal(result, [0, 1, 1])

    def test_custom_threshold_high(self):
        arr = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        result = postprocess_mask(arr, threshold=0.85)
        np.testing.assert_array_equal(result, [0, 0, 1])

    def test_raises_type_error_for_non_array(self):
        with pytest.raises(TypeError, match="np.ndarray"):
            postprocess_mask([[0.1, 0.9]], threshold=0.5)

    def test_raises_value_error_for_threshold_above_1(self):
        with pytest.raises(ValueError, match="threshold"):
            postprocess_mask(np.array([0.5]), threshold=1.5)

    def test_raises_value_error_for_threshold_below_0(self):
        with pytest.raises(ValueError, match="threshold"):
            postprocess_mask(np.array([0.5]), threshold=-0.1)

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_valid_threshold_range(self, threshold):
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = postprocess_mask(arr, threshold=threshold)
        assert result.dtype == np.uint8


# =============================================================================
# calculate_tumor_metrics
# =============================================================================

class TestCalculateTumorMetrics:

    def test_empty_mask_has_no_tumor(self, empty_mask):
        metrics = calculate_tumor_metrics(empty_mask)
        assert metrics["has_tumor"] is False
        assert metrics["tumor_voxels"] == 0
        assert metrics["tumor_volume_mm3"] == 0.0
        assert metrics["bounding_box"] is None

    def test_full_mask_is_all_tumor(self, full_mask):
        metrics = calculate_tumor_metrics(full_mask)
        assert metrics["has_tumor"] is True
        assert metrics["tumor_voxels"] == full_mask.size
        assert metrics["tumor_ratio"] == pytest.approx(1.0, abs=1e-6)

    def test_partial_mask_voxel_count(self, partial_mask):
        metrics = calculate_tumor_metrics(partial_mask)
        expected_voxels = int(np.sum(partial_mask))
        assert metrics["tumor_voxels"] == expected_voxels

    def test_partial_mask_bounding_box(self, partial_mask):
        metrics = calculate_tumor_metrics(partial_mask)
        bb = metrics["bounding_box"]
        assert bb is not None
        row_min, col_min, row_max, col_max = bb
        assert row_min == 103
        assert col_min == 103
        assert row_max == 152
        assert col_max == 152

    def test_voxel_volume_scales_volume(self, partial_mask):
        voxel_vol = 2.5
        metrics = calculate_tumor_metrics(partial_mask, voxel_volume_mm3=voxel_vol)
        expected = np.sum(partial_mask) * voxel_vol
        assert metrics["tumor_volume_mm3"] == pytest.approx(expected, abs=1e-4)

    def test_tumor_ratio_between_0_and_1(self, partial_mask):
        metrics = calculate_tumor_metrics(partial_mask)
        assert 0.0 <= metrics["tumor_ratio"] <= 1.0

    def test_total_voxels_equals_mask_size(self, partial_mask):
        metrics = calculate_tumor_metrics(partial_mask)
        assert metrics["total_voxels"] == partial_mask.size

    def test_raises_type_error_for_non_array(self):
        with pytest.raises(TypeError, match="np.ndarray"):
            calculate_tumor_metrics([[0, 1], [1, 0]])

    def test_raises_value_error_for_non_binary_mask(self):
        mask = np.array([[0, 2], [1, 3]], dtype=np.uint8)
        with pytest.raises(ValueError, match="binary"):
            calculate_tumor_metrics(mask)

    def test_raises_value_error_for_non_positive_voxel_volume(self, empty_mask):
        with pytest.raises(ValueError, match="positive"):
            calculate_tumor_metrics(empty_mask, voxel_volume_mm3=0.0)

    def test_raises_value_error_for_negative_voxel_volume(self, empty_mask):
        with pytest.raises(ValueError, match="positive"):
            calculate_tumor_metrics(empty_mask, voxel_volume_mm3=-1.0)

    def test_3d_mask_no_bounding_box(self):
        mask_3d = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_3d[5, 5, 5] = 1
        metrics = calculate_tumor_metrics(mask_3d)
        assert metrics["has_tumor"] is True
        assert metrics["bounding_box"] is None
