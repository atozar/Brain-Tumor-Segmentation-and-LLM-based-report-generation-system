"""
report.py
---------
LLM-based radiology report generation module.

Responsibilities:
- Format tumor metrics into a structured prompt
- Generate a medical report via an LLM client
- Save the report to disk
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Render a metrics dictionary as a human-readable summary string.

    Args:
        metrics: Dictionary produced by ``predictor.calculate_tumor_metrics``.
                 Expected keys: ``has_tumor``, ``tumor_voxels``,
                 ``total_voxels``, ``tumor_volume_mm3``, ``tumor_ratio``,
                 ``bounding_box`` (optional).

    Returns:
        A multi-line string summarising the metrics.

    Raises:
        TypeError:  If ``metrics`` is not a dictionary.
        ValueError: If required keys are missing from ``metrics``.
    """
    if not isinstance(metrics, dict):
        raise TypeError(f"Expected dict, got {type(metrics).__name__}")

    required_keys = {"has_tumor", "tumor_voxels", "total_voxels",
                     "tumor_volume_mm3", "tumor_ratio"}
    missing = required_keys - metrics.keys()
    if missing:
        raise ValueError(f"Missing required metric keys: {missing}")

    tumor_status = "Detected" if metrics["has_tumor"] else "Not Detected"
    ratio_pct = round(metrics["tumor_ratio"] * 100, 4)

    lines = [
        f"Tumor Status     : {tumor_status}",
        f"Tumor Voxels     : {metrics['tumor_voxels']}",
        f"Total Voxels     : {metrics['total_voxels']}",
        f"Tumor Volume     : {metrics['tumor_volume_mm3']} mm³",
        f"Tumor Ratio      : {ratio_pct}%",
    ]

    if metrics.get("bounding_box") is not None:
        bb = metrics["bounding_box"]
        lines.append(
            f"Bounding Box     : row [{bb[0]}-{bb[2]}], col [{bb[1]}-{bb[3]}]"
        )

    return "\n".join(lines)


def build_prompt(metrics_text: str, patient_id: Optional[str] = None) -> str:
    """
    Construct an LLM system + user prompt for report generation.

    Args:
        metrics_text: Formatted metrics string from ``format_metrics``.
        patient_id:   Optional patient identifier to include in the prompt.

    Returns:
        A complete prompt string to send to the LLM.

    Raises:
        TypeError:  If ``metrics_text`` is not a string.
        ValueError: If ``metrics_text`` is empty or whitespace-only.
    """
    if not isinstance(metrics_text, str):
        raise TypeError(f"Expected str, got {type(metrics_text).__name__}")

    if not metrics_text.strip():
        raise ValueError("metrics_text must not be empty or whitespace-only")

    patient_line = f"Patient ID: {patient_id}\n" if patient_id else ""

    prompt = (
        "You are an expert radiologist. Based on the following automated "
        "brain MRI segmentation results, generate a concise and professional "
        "radiology report. Include an impression, findings, and a recommendation.\n\n"
        f"{patient_line}"
        "Segmentation Metrics:\n"
        f"{metrics_text}\n\n"
        "Report:"
    )

    return prompt


def generate_report(
    metrics: Dict[str, Any],
    client: Any,
    model: str = "llama3",
    patient_id: Optional[str] = None,
) -> str:
    """
    Generate a radiology report by calling an LLM with the tumor metrics.

    The ``client`` object must expose a ``chat`` interface compatible with
    the Ollama Python SDK::

        client.chat(model=<str>, messages=[{"role": "user", "content": <str>}])

    The returned object must have a ``message.content`` attribute.

    Args:
        metrics:    Dictionary from ``predictor.calculate_tumor_metrics``.
        client:     LLM client instance (e.g. ``ollama.Client``).
        model:      Model name to use for generation. Defaults to ``"llama3"``.
        patient_id: Optional patient identifier included in the prompt.

    Returns:
        Generated report text as a string.

    Raises:
        TypeError:    If ``metrics`` is not a dict or ``client`` is None.
        ValueError:   If required metric keys are missing.
        RuntimeError: If the LLM call fails.
    """
    if not isinstance(metrics, dict):
        raise TypeError(f"Expected dict for metrics, got {type(metrics).__name__}")

    if client is None:
        raise TypeError("client must not be None")

    metrics_text = format_metrics(metrics)
    prompt = build_prompt(metrics_text, patient_id=patient_id)

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
    except AttributeError as exc:
        raise RuntimeError(
            "LLM client response did not have expected structure "
            f"(response.message.content): {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def save_report(report: str, path: str) -> None:
    """
    Persist a report string to a text file.

    Parent directories are created automatically if they do not exist.
    A timestamp header is prepended to the file content.

    Args:
        report: Report text to save.
        path:   Destination file path (e.g. ``"reports/patient_001.txt"``).

    Raises:
        TypeError:  If ``report`` is not a string or ``path`` is not a string.
        ValueError: If ``report`` is empty or ``path`` is empty.
        OSError:    If the file cannot be written.
    """
    if not isinstance(report, str):
        raise TypeError(f"Expected str for report, got {type(report).__name__}")

    if not isinstance(path, str):
        raise TypeError(f"Expected str for path, got {type(path).__name__}")

    if not report.strip():
        raise ValueError("report must not be empty or whitespace-only")

    if not path.strip():
        raise ValueError("path must not be empty or whitespace-only")

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    content = f"Generated: {timestamp}\n{'=' * 60}\n{report}\n"

    file_path.write_text(content, encoding="utf-8")
