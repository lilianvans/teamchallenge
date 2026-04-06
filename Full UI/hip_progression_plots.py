from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# HELPERS
# ============================================================

# small helper to estimate a linear trend for the selected patient trajectory
def _fit_simple_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None, None, None

    try:
        from scipy.stats import linregress
        res = linregress(x, y)
        return float(res.slope), float(res.intercept), float(res.pvalue)
    except Exception:
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0]), float(coeffs[1]), None


def _clean_patient_id(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


# load all saved hip UI results for one patient
# these are the study-level CSV files written by the hip workflow
def _load_patient_ui_results(patient_dir: Path) -> pd.DataFrame:
        # Load all saved hip UI results for the current selected patient.
    # These are the study-level results saved by the hip workflow.

    # IMPORTANT:
    # We treat CSV ages as already being in MONTHS.
    # No automatic years->months conversion is applied here.

    rows = []

    studies_root = patient_dir / "studies"
    if not studies_root.exists():
        return pd.DataFrame()

    for study_dir in sorted(studies_root.iterdir()):
        if not study_dir.is_dir():
            continue

        csv_path = study_dir / "hip_outputs" / "hip_results.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not read {csv_path}: {e}")
            continue

        if df.empty:
            continue

        df["source_study_dir"] = str(study_dir)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    rename_map = {
        "patient_folder": "Patient ID",
        "patient_display": "Patient Display",
        "study_date": "Study Date",
        "age_months": "Patient age (months)",
        "age_years": "Patient age (years)",
        "ratio_full": "Ratio_Full",
        "angle_deg": "Angle",
        "ratio_left": "Ratio_L",
        "ratio_right": "Ratio_R",
    }
    df = df.rename(columns=rename_map)

    if "Patient ID" in df.columns:
        df["Patient ID"] = df["Patient ID"].apply(_clean_patient_id)

    if "Study Date" in df.columns:
        df["Study Date"] = pd.to_datetime(df["Study Date"], errors="coerce")

    numeric_cols = [
        "Patient age (months)",
        "Patient age (years)",
        "Ratio_Full",
        "Angle",
        "Ratio_L",
        "Ratio_R",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        c for c in [
            "Patient ID",
            "Patient Display",
            "Study Date",
            "Patient age (months)",
            "Patient age (years)",
            "Ratio_Full",
            "Angle",
            "Ratio_L",
            "Ratio_R",
            "source_study_dir",
        ]
        if c in df.columns
    ]
    df = df[keep_cols].copy()

    df = df.sort_values(
        by=[c for c in ["Patient age (months)", "Study Date"] if c in df.columns],
        na_position="last"
    ).reset_index(drop=True)

    return df


# load the broader hip cohort reference CSV if it exists
def _load_cohort_csv(cohort_csv_path: Path) -> pd.DataFrame:
        # Load the combined cohort CSV.

    # IMPORTANT:
    # We treat the cohort CSV age column as already being in MONTHS.
    # No years->months auto-conversion is applied.

    if not cohort_csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(cohort_csv_path)
    except Exception as e:
        print(f"Could not read cohort CSV {cohort_csv_path}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if "Patient ID" in df.columns:
        df["Patient ID"] = df["Patient ID"].apply(_clean_patient_id)

    if "Study Date" in df.columns:
        df["Study Date"] = pd.to_datetime(df["Study Date"], errors="coerce")

    numeric_cols = [
        "Patient age (months)",
        "Patient age (years)",
        "Ratio_Full",
        "Angle",
        "Ratio_L",
        "Ratio_R",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# keep one row per study/age so repeated exports do not duplicate the patient trajectory
def _reduce_patient_df(df: pd.DataFrame) -> pd.DataFrame:
        # Keep one row per age/study for the selected patient.

    # Since ages are already in months, we do not rescale them.

    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    group_cols = [c for c in ["Patient age (months)", "Study Date"] if c in work.columns]
    if not group_cols:
        group_cols = ["Patient age (months)"]

    work = work.sort_values(
        by=[c for c in ["Patient age (months)", "Study Date"] if c in work.columns],
        na_position="last"
    )

    work = work.groupby(group_cols, as_index=False).first()
    work = work.sort_values("Patient age (months)", na_position="last").reset_index(drop=True)
    return work


# try to identify the current patient robustly, even if folder names and CSV labels differ slightly
def _extract_current_patient_id(patient_dir: Path, patient_df: pd.DataFrame, cohort_df: pd.DataFrame):
        # Try to identify current patient ID robustly.

    candidates = []

    candidates.append(_clean_patient_id(patient_dir.name))

    if not patient_df.empty and "Patient ID" in patient_df.columns:
        vals = patient_df["Patient ID"].dropna().astype(str).unique().tolist()
        candidates.extend(vals)

    for cand in candidates:
        if cand is None:
            continue
        if not cohort_df.empty and "Patient ID" in cohort_df.columns:
            if cand in set(cohort_df["Patient ID"].dropna().astype(str)):
                return cand

    for cand in candidates:
        if cand is not None:
            return cand

    return None


# ============================================================
# PLOTTING
# ============================================================

# plot the selected patient against the cohort background and add a simple patient trend line
def _plot_patient_vs_cohort(
    patient_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    patient_id: str | None,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
):
    if metric not in patient_df.columns and metric not in cohort_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # -------------------------
    # cohort background points
    # -------------------------
    cohort_plot = pd.DataFrame()
    if not cohort_df.empty and {"Patient age (months)", metric}.issubset(cohort_df.columns):
        cohort_plot = cohort_df[["Patient ID", "Patient age (months)", metric]].copy()
        cohort_plot["Patient age (months)"] = pd.to_numeric(cohort_plot["Patient age (months)"], errors="coerce")
        cohort_plot[metric] = pd.to_numeric(cohort_plot[metric], errors="coerce")
        cohort_plot = cohort_plot.dropna(subset=["Patient age (months)", metric])

        if patient_id is not None and "Patient ID" in cohort_plot.columns:
            other_patients = cohort_plot[cohort_plot["Patient ID"] != patient_id].copy()
        else:
            other_patients = cohort_plot.copy()

        if not other_patients.empty:
            ax.scatter(
                other_patients["Patient age (months)"],
                other_patients[metric],
                alpha=0.20,
                s=28,
                label="Other patients"
            )

        slope_c, intercept_c, p_c = _fit_simple_line(
            cohort_plot["Patient age (months)"].to_numpy(),
            cohort_plot[metric].to_numpy(),
        )

        if slope_c is not None and intercept_c is not None:
            x_min = float(cohort_plot["Patient age (months)"].min())
            x_max = float(cohort_plot["Patient age (months)"].max())
            x_fit = np.linspace(x_min, x_max, 200)
            y_fit = slope_c * x_fit + intercept_c
            ax.plot(
                x_fit,
                y_fit,
                linestyle="--",
                linewidth=2.5,
                label="Cohort trend"
            )
            cohort_annual_rate = slope_c * 12.0
        else:
            cohort_annual_rate = None
    else:
        cohort_annual_rate = None

    # -------------------------
    # current patient points
    # -------------------------
    if metric not in patient_df.columns or "Patient age (months)" not in patient_df.columns:
        plt.close(fig)
        return None

    patient_plot = patient_df[["Patient age (months)", metric]].copy()
    patient_plot["Patient age (months)"] = pd.to_numeric(patient_plot["Patient age (months)"], errors="coerce")
    patient_plot[metric] = pd.to_numeric(patient_plot[metric], errors="coerce")
    patient_plot = patient_plot.dropna(subset=["Patient age (months)", metric]).sort_values("Patient age (months)")

    if patient_plot.empty:
        plt.close(fig)
        return None

    ax.plot(
        patient_plot["Patient age (months)"],
        patient_plot[metric],
        marker="o",
        linewidth=2.6,
        markersize=7,
        label="Current patient"
    )

    slope_p, intercept_p, p_p = _fit_simple_line(
        patient_plot["Patient age (months)"].to_numpy(),
        patient_plot[metric].to_numpy(),
    )

    if slope_p is not None and intercept_p is not None and len(patient_plot) >= 2:
        x_fit = np.linspace(
            float(patient_plot["Patient age (months)"].min()),
            float(patient_plot["Patient age (months)"].max()),
            100,
        )
        y_fit = slope_p * x_fit + intercept_p
        ax.plot(
            x_fit,
            y_fit,
            linewidth=3.0,
            label="Patient trend"
        )
        patient_annual_rate = slope_p * 12.0
    else:
        patient_annual_rate = None

    subtitle_parts = []

    if patient_annual_rate is not None:
        if p_p is None:
            subtitle_parts.append(f"Patient annual rate: {patient_annual_rate:.3f}/year")
        else:
            subtitle_parts.append(f"Patient annual rate: {patient_annual_rate:.3f}/year (p={p_p:.4f})")

    if cohort_annual_rate is not None:
        if p_c is None:
            subtitle_parts.append(f"Cohort annual rate: {cohort_annual_rate:.3f}/year")
        else:
            subtitle_parts.append(f"Cohort annual rate: {cohort_annual_rate:.3f}/year (p={p_c:.4f})")

    if subtitle_parts:
        ax.set_title(f"{title}\n" + " | ".join(subtitle_parts), fontsize=13)
    else:
        ax.set_title(title, fontsize=13)

    ax.set_xlabel("Patient age (months)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_overview_panel(
    patient_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    patient_id: str | None,
    output_path: Path,
):
        # Overview figure with all metrics over age for quick UI preview.

    metrics = [
        ("Ratio_Full", "Full ratio"),
        ("Ratio_L", "Left ratio"),
        ("Ratio_R", "Right ratio"),
        ("Angle", "Angle"),
    ]

    available = []
    for col, label in metrics:
        patient_ok = col in patient_df.columns and "Patient age (months)" in patient_df.columns
        cohort_ok = col in cohort_df.columns and "Patient age (months)" in cohort_df.columns
        if patient_ok or cohort_ok:
            available.append((col, label))

    if not available:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for col, label in available:
        if col not in patient_df.columns:
            continue

        tmp = patient_df[["Patient age (months)", col]].copy()
        tmp["Patient age (months)"] = pd.to_numeric(tmp["Patient age (months)"], errors="coerce")
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=["Patient age (months)", col]).sort_values("Patient age (months)")

        if tmp.empty:
            continue

        ax.plot(
            tmp["Patient age (months)"],
            tmp[col],
            marker="o",
            linewidth=2,
            label=label
        )

    ax.set_title("Hip progression overview", fontsize=13)
    ax.set_xlabel("Patient age (months)")
    ax.set_ylabel("Measurement value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


# ============================================================
# MAIN ENTRY
# ============================================================

def generate_hip_progression_plots_for_patient(
    patient_dir: Path,
    cohort_csv_path: Path | None = None,
):
        # Generate hip progression plots for one patient with cohort comparison.

    # Parameters
    # ----------
    # patient_dir : Path
    # Current patient directory in the UI.
    # cohort_csv_path : Path | None
    # Path to your combined cohort CSV.
    # If None, defaults to the CSV in the same folder as this script.

    patient_dir = Path(patient_dir)

    if cohort_csv_path is None:
        cohort_csv_path = Path(__file__).resolve().parent / "Data_analysis_hip_combined.csv"
    else:
        cohort_csv_path = Path(cohort_csv_path)

    analysis_dir = patient_dir / "analysis_data"
    plots_dir = analysis_dir / "hip_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for old_plot in plots_dir.glob("hip_plot_*.png"):
        try:
            old_plot.unlink()
        except Exception:
            pass

    patient_raw = _load_patient_ui_results(patient_dir)
    if patient_raw.empty:
        print("No patient hip UI results found.")
        return []

    patient_df = _reduce_patient_df(patient_raw)
    if patient_df.empty:
        print("Could not build patient dataframe.")
        return []

    cohort_df = _load_cohort_csv(cohort_csv_path)
    patient_id = _extract_current_patient_id(patient_dir, patient_df, cohort_df)

    plot_paths = []

    overview_path = plots_dir / "hip_plot_overview.png"
    made = _plot_overview_panel(patient_df, cohort_df, patient_id, overview_path)
    if made is not None:
        plot_paths.append(made)

    metric_configs = [
        ("Ratio_Full", "Hip Full Ratio: Patient vs Cohort", "Full ratio"),
        ("Ratio_L", "Hip Left Ratio: Patient vs Cohort", "Left ratio"),
        ("Ratio_R", "Hip Right Ratio: Patient vs Cohort", "Right ratio"),
        ("Angle", "Hip Angle: Patient vs Cohort", "Angle (deg)"),
    ]

    for metric, title, ylabel in metric_configs:
        out_path = plots_dir / f"hip_plot_{metric.lower()}.png"
        made = _plot_patient_vs_cohort(
            patient_df=patient_df,
            cohort_df=cohort_df,
            patient_id=patient_id,
            metric=metric,
            title=title,
            ylabel=ylabel,
            output_path=out_path,
        )
        if made is not None:
            plot_paths.append(made)

    try:
        patient_df.to_csv(analysis_dir / "hip_progression_patient_summary.csv", index=False)
    except Exception as e:
        print(f"Could not save patient hip summary CSV: {e}")

    return plot_paths
