"""
test_report.py
--------------
Unit tests for brain_tumor/src/report.py

Coverage targets
----------------
- format_metrics: all keys present, missing keys, bad input type,
                  tumor-detected vs none, bounding box optional
- build_prompt: content, patient_id optional, empty input error
- generate_report: successful call, None client, LLM failure,
                   response structure error
- save_report: file creation, parent dirs, timestamp header,
               overwrite, empty report/path errors
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from report import format_metrics, build_prompt, generate_report, save_report


# =============================================================================
# format_metrics
# =============================================================================

class TestFormatMetrics:

    def test_contains_tumor_detected(self, tumor_metrics_detected):
        text = format_metrics(tumor_metrics_detected)
        assert "Detected" in text

    def test_contains_no_tumor(self, tumor_metrics_none):
        text = format_metrics(tumor_metrics_none)
        assert "Not Detected" in text

    def test_contains_volume(self, tumor_metrics_detected):
        text = format_metrics(tumor_metrics_detected)
        assert "mm³" in text

    def test_contains_voxel_count(self, tumor_metrics_detected):
        text = format_metrics(tumor_metrics_detected)
        assert str(tumor_metrics_detected["tumor_voxels"]) in text

    def test_contains_bounding_box_when_present(self, tumor_metrics_detected):
        text = format_metrics(tumor_metrics_detected)
        assert "Bounding Box" in text

    def test_no_bounding_box_line_when_absent(self, tumor_metrics_none):
        text = format_metrics(tumor_metrics_none)
        assert "Bounding Box" not in text

    def test_raises_type_error_for_non_dict(self):
        with pytest.raises(TypeError, match="dict"):
            format_metrics("not a dict")

    def test_raises_value_error_for_missing_keys(self):
        incomplete = {"has_tumor": True}
        with pytest.raises(ValueError, match="Missing required metric keys"):
            format_metrics(incomplete)

    @pytest.mark.parametrize("missing_key", [
        "has_tumor", "tumor_voxels", "total_voxels",
        "tumor_volume_mm3", "tumor_ratio"
    ])
    def test_raises_value_error_for_each_missing_key(
        self, tumor_metrics_detected, missing_key
    ):
        metrics = {k: v for k, v in tumor_metrics_detected.items() if k != missing_key}
        with pytest.raises(ValueError):
            format_metrics(metrics)

    def test_returns_string(self, tumor_metrics_detected):
        result = format_metrics(tumor_metrics_detected)
        assert isinstance(result, str)

    def test_ratio_shown_as_percentage(self, tumor_metrics_detected):
        text = format_metrics(tumor_metrics_detected)
        assert "%" in text


# =============================================================================
# build_prompt
# =============================================================================

class TestBuildPrompt:

    def test_prompt_contains_metrics_text(self):
        text = "Tumor Status: Detected"
        prompt = build_prompt(text)
        assert text in prompt

    def test_prompt_contains_patient_id_when_given(self):
        prompt = build_prompt("metrics text", patient_id="PT-001")
        assert "PT-001" in prompt

    def test_prompt_has_no_patient_id_line_when_absent(self):
        prompt = build_prompt("metrics text")
        assert "Patient ID" not in prompt

    def test_prompt_contains_report_instruction(self):
        prompt = build_prompt("metrics text")
        assert "radiologist" in prompt.lower() or "radiology" in prompt.lower()

    def test_raises_type_error_for_non_string(self):
        with pytest.raises(TypeError, match="str"):
            build_prompt(12345)

    def test_raises_value_error_for_empty_string(self):
        with pytest.raises(ValueError, match="empty"):
            build_prompt("")

    def test_raises_value_error_for_whitespace_only(self):
        with pytest.raises(ValueError, match="empty"):
            build_prompt("   \t\n  ")

    def test_returns_string(self):
        result = build_prompt("some metrics")
        assert isinstance(result, str)


# =============================================================================
# generate_report
# =============================================================================

class TestGenerateReport:

    def _make_client(self, content="Radiology report text."):
        response = MagicMock()
        response.message.content = content
        client = MagicMock()
        client.chat.return_value = response
        return client

    def test_returns_report_string(self, tumor_metrics_detected):
        client = self._make_client("Radiology report text.")
        result = generate_report(tumor_metrics_detected, client)
        assert isinstance(result, str)
        assert result == "Radiology report text."

    def test_calls_client_chat_once(self, tumor_metrics_detected):
        client = self._make_client()
        generate_report(tumor_metrics_detected, client)
        client.chat.assert_called_once()

    def test_passes_correct_model_to_client(self, tumor_metrics_detected):
        client = self._make_client()
        generate_report(tumor_metrics_detected, client, model="mistral")
        call_kwargs = client.chat.call_args
        assert call_kwargs.kwargs.get("model") == "mistral" or \
               (call_kwargs.args and call_kwargs.args[0] == "mistral") or \
               call_kwargs.kwargs.get("model") == "mistral" or \
               "mistral" in str(call_kwargs)

    def test_patient_id_passed_through(self, tumor_metrics_detected):
        captured_messages = {}
        def fake_chat(model, messages):
            captured_messages["messages"] = messages
            r = MagicMock()
            r.message.content = "report"
            return r

        client = MagicMock()
        client.chat.side_effect = fake_chat

        generate_report(tumor_metrics_detected, client, patient_id="PT-999")
        assert "PT-999" in captured_messages["messages"][0]["content"]

    def test_raises_type_error_for_none_client(self, tumor_metrics_detected):
        with pytest.raises(TypeError, match="client"):
            generate_report(tumor_metrics_detected, client=None)

    def test_raises_type_error_for_non_dict_metrics(self):
        client = self._make_client()
        with pytest.raises(TypeError, match="dict"):
            generate_report("not a dict", client)

    def test_raises_runtime_on_client_exception(self, tumor_metrics_detected):
        client = MagicMock()
        client.chat.side_effect = ConnectionError("Connection refused")
        with pytest.raises(RuntimeError, match="LLM call failed"):
            generate_report(tumor_metrics_detected, client)

    def test_raises_runtime_on_bad_response_structure(self, tumor_metrics_detected):
        client = MagicMock()
        client.chat.return_value = "not an object with message.content"
        with pytest.raises(RuntimeError):
            generate_report(tumor_metrics_detected, client)


# =============================================================================
# save_report
# =============================================================================

class TestSaveReport:

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "report.txt")
        save_report("My radiology report.", path)
        assert Path(path).exists()

    def test_file_contains_report_text(self, tmp_path):
        path = str(tmp_path / "report.txt")
        save_report("My radiology report.", path)
        content = Path(path).read_text(encoding="utf-8")
        assert "My radiology report." in content

    def test_file_contains_timestamp(self, tmp_path):
        path = str(tmp_path / "report.txt")
        save_report("Some report.", path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Generated:" in content

    def test_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "dirs" / "report.txt")
        save_report("Report text.", path)
        assert Path(path).exists()

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "report.txt")
        save_report("First version.", path)
        save_report("Second version.", path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Second version." in content
        assert "First version." not in content

    def test_raises_type_error_for_non_string_report(self, tmp_path):
        with pytest.raises(TypeError, match="str"):
            save_report(12345, str(tmp_path / "r.txt"))

    def test_raises_type_error_for_non_string_path(self):
        with pytest.raises(TypeError, match="str"):
            save_report("report", Path("/some/path.txt"))

    def test_raises_value_error_for_empty_report(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_report("", str(tmp_path / "report.txt"))

    def test_raises_value_error_for_whitespace_report(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_report("   \n\t  ", str(tmp_path / "report.txt"))

    def test_raises_value_error_for_empty_path(self):
        with pytest.raises(ValueError, match="empty"):
            save_report("Valid report.", "")

    def test_raises_value_error_for_whitespace_path(self):
        with pytest.raises(ValueError, match="empty"):
            save_report("Valid report.", "   ")
