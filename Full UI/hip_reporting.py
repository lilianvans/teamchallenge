from pathlib import Path

from reporting import build_basic_result_metadata, save_csv_rows


# combine metadata and hip measurements into one hip-results dictionary
def build_hip_results(
    patient_dir: Path,
    patient_display: str,
    study_dir: Path,
    study_display: str,
    dicom_path: Path,
    hip_data: dict,
    group_label: str = "",
    study_date: str = "",
    age_months: float | None = None,
    age_years: float | None = None,
    bone_age_months: float | None = None,
) -> dict:
    results = build_basic_result_metadata(
        patient_dir=patient_dir,
        patient_display=patient_display,
        study_dir=study_dir,
        study_display=study_display,
        dicom_path=dicom_path,
        analysis_type="hip",
        group_label=group_label,
        study_date=study_date,
        age_months=age_months,
        age_years=age_years,
        bone_age_months=bone_age_months,
    )

    results["hip_analysis"] = hip_data if hip_data is not None else {}
    return results


# create a readable text summary of the hip acetabulum analysis
def build_hip_text_report(results: dict) -> str:
    hip = results.get("hip_analysis", {})

    lines = []
    lines.append("HIP ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Patient folder: {results.get('patient_folder', '')}")
    lines.append(f"Patient display: {results.get('patient_display', '')}")
    lines.append(f"Study folder: {results.get('study_folder', '')}")
    lines.append(f"Study display: {results.get('study_display', '')}")
    lines.append(f"Selected image: {results.get('image_file', '')}")
    lines.append(f"Timestamp: {results.get('timestamp', '')}")
    lines.append(f"Group label: {results.get('group_label', '')}")
    lines.append(f"Study date: {results.get('study_date', '')}")
    lines.append(f"Patient age (months): {results.get('age_months', '')}")
    lines.append(f"Patient age (years): {results.get('age_years', '')}")
    lines.append(f"Bone age (months): {results.get('bone_age_months', '')}")
    lines.append("")
    lines.append("ACETABULUM ANALYSIS")
    lines.append("-" * 60)
    lines.append(f"Full ratio: {hip.get('ratio_full', '')}")
    lines.append(f"Angle (deg): {hip.get('angle_deg', '')}")
    lines.append(f"Left ratio: {hip.get('ratio_left', '')}")
    lines.append(f"Right ratio: {hip.get('ratio_right', '')}")
    lines.append(f"Preview path: {hip.get('preview_path', '')}")

    return "\n".join(lines)


# flatten the hip result dictionary into one CSV row
def build_hip_csv_rows(results: dict) -> list[dict]:
    hip = results.get("hip_analysis", {})

    return [
        {
            "patient_folder": results.get("patient_folder", ""),
            "patient_display": results.get("patient_display", ""),
            "study_folder": results.get("study_folder", ""),
            "study_display": results.get("study_display", ""),
            "image_file": results.get("image_file", ""),
            "timestamp": results.get("timestamp", ""),
            "group_label": results.get("group_label", ""),
            "study_date": results.get("study_date", ""),
            "age_months": results.get("age_months", ""),
            "age_years": results.get("age_years", ""),
            "bone_age_months": results.get("bone_age_months", ""),
            "ratio_full": hip.get("ratio_full", ""),
            "angle_deg": hip.get("angle_deg", ""),
            "ratio_left": hip.get("ratio_left", ""),
            "ratio_right": hip.get("ratio_right", ""),
            "preview_path": hip.get("preview_path", ""),
        }
    ]


def save_hip_results_csv(filepath: Path, results: dict) -> None:
    fieldnames = [
        "patient_folder",
        "patient_display",
        "study_folder",
        "study_display",
        "image_file",
        "timestamp",
        "group_label",
        "study_date",
        "age_months",
        "age_years",
        "bone_age_months",
        "ratio_full",
        "angle_deg",
        "ratio_left",
        "ratio_right",
        "preview_path",
    ]
    rows = build_hip_csv_rows(results)
    save_csv_rows(filepath, fieldnames, rows)
