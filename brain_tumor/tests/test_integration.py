"""
test_integration.py
-------------------
End-to-end integration tests for the full brain tumor pipeline:

    preprocess_pipeline  →  predict  →  postprocess_mask
                         →  calculate_tumor_metrics
                         →  generate_report  →  save_report

All external dependencies (model, LLM client) are mocked so the test
suite runs without GPU hardware, model weights, or a running LLM server.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import preprocess_pipeline
from predictor import predict, postprocess_mask, calculate_tumor_metrics
from report import generate_report, save_report


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mri_png(directory: Path, size: int = 128) -> str:
    """Create a synthetic grayscale MRI PNG for testing."""
    rng = np.random.default_rng(42)
    arr = rng.integers(10, 240, size=(size, size), dtype=np.uint8)
    # Simulate a bright tumour-like region in the centre
    cx, cy, r = size // 2, size // 2, size // 8
    for row in range(cx - r, cx + r):
        for col in range(cy - r, cy + r):
            arr[row, col] = 220
    img = Image.fromarray(arr, mode="L")
    path = directory / "synthetic_mri.png"
    img.save(str(path))
    return str(path)


def _make_segmentation_model(output_size: int = 128) -> MagicMock:
    """Return a mock model whose predict() returns a probability map with a
    central high-probability tumour region."""
    prob_map = np.full((1, output_size, output_size), 0.1, dtype=np.float32)
    cx, cy, r = output_size // 2, output_size // 2, output_size // 8
    prob_map[0, cx - r : cx + r, cy - r : cy + r] = 0.92
    model = MagicMock()
    model.predict.return_value = prob_map
    return model


def _make_llm_client(report_text: str = "Impression: Small focal tumor detected.") -> MagicMock:
    """Return a mock LLM client compatible with the Ollama SDK interface."""
    response = MagicMock()
    response.message.content = report_text
    client = MagicMock()
    client.chat.return_value = response
    return client


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_pipeline_produces_tumor_report(self, tmp_path):
        """
        Happy-path: a synthetic MRI flows through the entire pipeline and
        produces a non-empty report saved to disk.
        """
        # --- Stage 1: Preprocessing ---
        mri_path = _make_mri_png(tmp_path)
        image = preprocess_pipeline(mri_path, target_size=(128, 128))

        assert image.shape == (128, 128), "Preprocessed image has wrong shape"
        assert image.min() >= 0.0
        assert image.max() <= 1.0 + 1e-6

        # --- Stage 2: Inference ---
        model = _make_segmentation_model(output_size=128)
        raw_output = predict(model, image)

        assert raw_output.shape == (128, 128), "Raw output has wrong shape"

        # --- Stage 3: Postprocessing ---
        mask = postprocess_mask(raw_output, threshold=0.5)

        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()).issubset({0, 1})

        # --- Stage 4: Metrics ---
        metrics = calculate_tumor_metrics(mask, voxel_volume_mm3=1.0)

        assert metrics["has_tumor"] is True
        assert metrics["tumor_voxels"] > 0
        assert 0.0 < metrics["tumor_ratio"] < 1.0

        # --- Stage 5: Report generation ---
        llm_client = _make_llm_client()
        report = generate_report(metrics, llm_client, model="llama3", patient_id="PT-TEST-001")

        assert isinstance(report, str)
        assert len(report.strip()) > 0

        # --- Stage 6: Save to disk ---
        report_path = str(tmp_path / "reports" / "PT-TEST-001.txt")
        save_report(report, report_path)

        saved = Path(report_path).read_text(encoding="utf-8")
        assert "Generated:" in saved
        assert report in saved

    def test_pipeline_no_tumor_path(self, tmp_path):
        """
        Edge-case: MRI where the model finds no tumor.
        The report must still be generated and saved successfully.
        """
        mri_path = _make_mri_png(tmp_path)
        image = preprocess_pipeline(mri_path, target_size=(64, 64))

        # Model returns uniformly low probabilities — no tumour
        no_tumor_map = np.full((1, 64, 64), 0.05, dtype=np.float32)
        model = MagicMock()
        model.predict.return_value = no_tumor_map

        raw_output = predict(model, image)
        mask = postprocess_mask(raw_output, threshold=0.5)
        metrics = calculate_tumor_metrics(mask)

        assert metrics["has_tumor"] is False
        assert metrics["tumor_voxels"] == 0
        assert metrics["bounding_box"] is None

        llm_client = _make_llm_client("Impression: No tumor detected.")
        report = generate_report(metrics, llm_client)

        report_path = str(tmp_path / "no_tumor_report.txt")
        save_report(report, report_path)

        assert Path(report_path).exists()

    def test_pipeline_with_custom_threshold(self, tmp_path):
        """
        Varying the segmentation threshold should change the reported
        tumor size, demonstrating threshold sensitivity.
        """
        mri_path = _make_mri_png(tmp_path)
        image = preprocess_pipeline(mri_path, target_size=(64, 64))

        model = _make_segmentation_model(output_size=64)
        raw_output = predict(model, image)

        mask_strict = postprocess_mask(raw_output, threshold=0.91)
        mask_loose  = postprocess_mask(raw_output, threshold=0.50)

        metrics_strict = calculate_tumor_metrics(mask_strict)
        metrics_loose  = calculate_tumor_metrics(mask_loose)

        assert metrics_strict["tumor_voxels"] <= metrics_loose["tumor_voxels"], (
            "Stricter threshold should produce equal or fewer tumor voxels"
        )

    def test_pipeline_metrics_feed_correctly_into_report(self, tmp_path):
        """
        The tumor volume from metrics must appear in the LLM prompt that
        the client receives, verifying end-to-end data propagation.
        """
        mri_path = _make_mri_png(tmp_path)
        image = preprocess_pipeline(mri_path, target_size=(64, 64))

        model = _make_segmentation_model(output_size=64)
        raw_output = predict(model, image)
        mask = postprocess_mask(raw_output, threshold=0.5)
        metrics = calculate_tumor_metrics(mask, voxel_volume_mm3=2.0)

        captured = {}

        def fake_chat(model, messages):
            captured["prompt"] = messages[0]["content"]
            r = MagicMock()
            r.message.content = "Report OK."
            return r

        client = MagicMock()
        client.chat.side_effect = fake_chat

        generate_report(metrics, client)

        assert "prompt" in captured, "LLM client was never called"
        assert "mm³" in captured["prompt"], "Volume not present in prompt"
        assert str(metrics["tumor_voxels"]) in captured["prompt"]

    def test_pipeline_voxel_volume_scales_report_metrics(self, tmp_path):
        """
        Changing the voxel volume should proportionally change the
        reported tumor volume while leaving voxel count unchanged.
        """
        mri_path = _make_mri_png(tmp_path)
        image = preprocess_pipeline(mri_path, target_size=(64, 64))

        model = _make_segmentation_model(output_size=64)
        raw_output = predict(model, image)
        mask = postprocess_mask(raw_output, threshold=0.5)

        metrics_1mm  = calculate_tumor_metrics(mask, voxel_volume_mm3=1.0)
        metrics_2mm  = calculate_tumor_metrics(mask, voxel_volume_mm3=2.0)

        assert metrics_1mm["tumor_voxels"] == metrics_2mm["tumor_voxels"]
        assert pytest.approx(metrics_2mm["tumor_volume_mm3"], rel=1e-5) == \
               metrics_1mm["tumor_volume_mm3"] * 2

    def test_pipeline_report_saved_with_timestamp(self, tmp_path):
        """
        Saved report files must always include a UTC timestamp header,
        regardless of report content.
        """
        mri_path = _make_mri_png(tmp_path)
        image = preprocess_pipeline(mri_path, target_size=(32, 32))

        model = _make_segmentation_model(output_size=32)
        raw_output = predict(model, image)
        mask = postprocess_mask(raw_output, threshold=0.5)
        metrics = calculate_tumor_metrics(mask)

        llm_client = _make_llm_client("Brief report.")
        report = generate_report(metrics, llm_client)
        report_path = str(tmp_path / "timestamped.txt")
        save_report(report, report_path)

        content = Path(report_path).read_text(encoding="utf-8")
        assert "UTC" in content, "Timestamp line should include UTC"
        assert "=" * 10 in content, "Separator line missing from saved report"

    def test_pipeline_model_called_with_batch_dimension(self, tmp_path):
        """
        The predict() stage must always pass a batch of exactly 1 image
        to the model, regardless of the input image shape.
        """
        mri_path = _make_mri_png(tmp_path)
        image = preprocess_pipeline(mri_path, target_size=(64, 64))

        received_shapes = []

        def recording_predict(batch):
            received_shapes.append(batch.shape)
            return np.zeros((1, 64, 64), dtype=np.float32)

        model = MagicMock()
        model.predict.side_effect = recording_predict

        predict(model, image)

        assert len(received_shapes) == 1
        assert received_shapes[0][0] == 1, "Model must receive a batch of size 1"
