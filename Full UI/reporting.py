import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# make sure an output folder exists before writing files into it
def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# timestamp helper used in saved result dictionaries
def now_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


# convert numpy arrays, paths and numpy scalar types into JSON-safe values
def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


# strip large in-memory arrays and convert special objects before saving JSON
def make_results_json_safe(results: dict) -> dict:
    safe = dict(results)

    if "segmentations" in safe and isinstance(safe["segmentations"], dict):
        seg = dict(safe["segmentations"])

        seg.pop("masks", None)

        image_array = seg.get("image_array")
        if isinstance(image_array, np.ndarray):
            seg["image_array_shape"] = list(image_array.shape)
            seg.pop("image_array", None)

        safe["segmentations"] = seg

    return _json_safe_value(safe)


# save a result dictionary as formatted JSON
def save_json(filepath: Path, data: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    safe_data = make_results_json_safe(data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=4)


def load_json(filepath: Path, default=None):
    if not filepath.exists():
        return default

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# save a plain-text report file
def save_report(filepath: Path, report_text: str) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)


# save multiple CSV rows and force values into a consistent serializable format
def save_csv_rows(filepath: Path, fieldnames: list[str], rows: list[dict]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            clean_row = {key: _json_safe_value(row.get(key, "")) for key in fieldnames}
            writer.writerow(clean_row)


# append a single row to an existing CSV, creating the header if needed
def append_csv_row(filepath: Path, fieldnames: list[str], row: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    file_exists = filepath.exists()

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        clean_row = {key: _json_safe_value(row.get(key, "")) for key in fieldnames}
        writer.writerow(clean_row)


# collect the metadata shared by both hand and hip result files
def build_basic_result_metadata(
    patient_dir: Path,
    patient_display: str,
    study_dir: Path,
    study_display: str,
    dicom_path: Path,
    *,
    analysis_type: str,
    group_label: str = "",
    study_date: str = "",
    age_months: float | None = None,
    age_years: float | None = None,
    bone_age_months: float | None = None,
    extra: dict | None = None,
) -> dict:
    if age_years is None and age_months is not None:
        age_years = round(float(age_months) / 12.0, 4)

    result = {
        "analysis_type": analysis_type,
        "patient_folder": patient_dir.name,
        "patient_display": patient_display,
        "study_folder": study_dir.name,
        "study_display": study_display,
        "image_file": str(dicom_path),
        "timestamp": now_timestamp(),
        "group_label": group_label,
        "study_date": study_date,
        "age_months": age_months,
        "age_years": age_years,
        "bone_age_months": bone_age_months,
    }

    if extra:
        result.update(extra)

    return result
