from pathlib import Path

import pandas as pd

from reporting import build_basic_result_metadata, save_csv_rows


# combine metadata, segmentation, ratio and curvature into one hand-results dictionary
def build_hand_results(
    patient_dir: Path,
    patient_display: str,
    study_dir: Path,
    study_display: str,
    dicom_path: Path,
    segmentation_data: dict,
    ratio_results: list[dict] | None = None,
    curvature_results: dict | None = None,
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
        analysis_type="hand",
        group_label=group_label,
        study_date=study_date,
        age_months=age_months,
        age_years=age_years,
        bone_age_months=bone_age_months,
    )

    results["segmentations"] = segmentation_data
    results["finger_ratios"] = ratio_results if ratio_results is not None else []
    results["curvature"] = curvature_results if curvature_results is not None else {}

    return results


# build a simple human-readable text report from the saved hand results
def build_hand_text_report(results: dict) -> str:
    lines = []
    lines.append("HAND ANALYSIS REPORT")
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

    lines.append("RATIO / METACARPAL MEASUREMENTS")
    lines.append("-" * 60)
    for item in results.get("finger_ratios", []):
        lines.append(
            f"{item.get('segment_name')}: "
            f"length={item.get('length_mm')} mm, "
            f"width_mid={item.get('width_mid_mm')} mm, "
            f"width_10={item.get('width_10_mm')} mm, "
            f"width_90={item.get('width_90_mm')} mm"
        )
    lines.append("")

    lines.append("CURVATURE")
    lines.append("-" * 60)
    for key, value in results.get("curvature", {}).items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("SEGMENTATION")
    lines.append("-" * 60)
    for item in results.get("segmentations", {}).get("saved_masks", []):
        lines.append(f"saved mask: {item}")

    return "\n".join(lines)


# flatten the hand results into CSV rows so study-level results can be exported easily
def build_hand_csv_rows(results: dict) -> list[dict]:
    curvature = results.get("curvature", {})
    ratio_rows = results.get("finger_ratios", [])
    rows = []

    if ratio_rows:
        for row in ratio_rows:
            rows.append(
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
                    "segment_name": row.get("segment_name", ""),
                    "length_mm": row.get("length_mm", ""),
                    "width_mid_mm": row.get("width_mid_mm", ""),
                    "width_10_mm": row.get("width_10_mm", ""),
                    "width_90_mm": row.get("width_90_mm", ""),
                    "DIP": curvature.get("DIP", ""),
                    "PIP": curvature.get("PIP", ""),
                    "MCP": curvature.get("MCP", ""),
                    "total": curvature.get("total", ""),
                }
            )
    else:
        rows.append(
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
                "segment_name": "",
                "length_mm": "",
                "width_mid_mm": "",
                "width_10_mm": "",
                "width_90_mm": "",
                "DIP": curvature.get("DIP", ""),
                "PIP": curvature.get("PIP", ""),
                "MCP": curvature.get("MCP", ""),
                "total": curvature.get("total", ""),
            }
        )

    return rows


def save_hand_results_csv(filepath: Path, results: dict) -> None:
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
        "segment_name",
        "length_mm",
        "width_mid_mm",
        "width_10_mm",
        "width_90_mm",
        "DIP",
        "PIP",
        "MCP",
        "total",
    ]
    rows = build_hand_csv_rows(results)
    save_csv_rows(filepath, fieldnames, rows)


def update_hand_patient_analysis_files(patient_dir: Path, results: dict) -> None:
    analysis_dir = patient_dir / "analysis_data"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = analysis_dir / "all_studies_summary.csv"
    long_csv = analysis_dir / "all_studies_longformat.csv"

    summary_row = {
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
        "DIP": results.get("curvature", {}).get("DIP", ""),
        "PIP": results.get("curvature", {}).get("PIP", ""),
        "MCP": results.get("curvature", {}).get("MCP", ""),
        "total": results.get("curvature", {}).get("total", ""),
    }

    ratio_rows = results.get("finger_ratios", [])
    if ratio_rows:
        first_row = ratio_rows[0]
        summary_row["first_segment_length_mm"] = first_row.get("length_mm", "")
        summary_row["first_segment_width_mid_mm"] = first_row.get("width_mid_mm", "")
        summary_row["first_segment_width_10_mm"] = first_row.get("width_10_mm", "")
        summary_row["first_segment_width_90_mm"] = first_row.get("width_90_mm", "")
    else:
        summary_row["first_segment_length_mm"] = ""
        summary_row["first_segment_width_mid_mm"] = ""
        summary_row["first_segment_width_10_mm"] = ""
        summary_row["first_segment_width_90_mm"] = ""

    long_rows = build_hand_csv_rows(results)

    if summary_csv.exists():
        df_summary = pd.read_csv(summary_csv)
        df_summary = df_summary[df_summary["study_folder"] != results.get("study_folder", "")]
        df_summary = pd.concat([df_summary, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([summary_row])

    if long_csv.exists():
        df_long = pd.read_csv(long_csv)
        df_long = df_long[df_long["study_folder"] != results.get("study_folder", "")]
        df_long = pd.concat([df_long, pd.DataFrame(long_rows)], ignore_index=True)
    else:
        df_long = pd.DataFrame(long_rows)

    df_summary.to_csv(summary_csv, index=False)
    df_long.to_csv(long_csv, index=False)
