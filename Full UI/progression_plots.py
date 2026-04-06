from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# colors used across the hand progression plots
COLOR_PATIENT = "#CF2F2F"
COLOR_HURLER = "#D55E00"
COLOR_HEALTHY = "#3546DA"


# ============================================================
# BASIC HELPERS
# ============================================================

def remove_outliers(df, column="Disease_Score", threshold=2.5):
    # remove outliers based on the IQR of the selected column

    if df.empty or column not in df.columns:
        return df

    work = df.copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column])

    if work.empty:
        return work

    q1 = work[column].quantile(0.25)
    q3 = work[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    mask = (work[column] >= lower_bound) & (work[column] <= upper_bound)

    clean_df = work[mask].copy()
    removed_df = work[~mask].copy()

    print(f"Removed {len(removed_df)} outliers from {column} using IQR method (threshold={threshold}*IQR).")
    print("Removed indices:", removed_df.index.tolist())

    return clean_df


def normalize_segment_name(name):
    # standardize segment labels from the UI CSV into fixed names

    if pd.isna(name):
        return None

    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")

    if "distal" in text:
        return "distal"
    if "middle" in text:
        return "middle"
    if "proximal" in text:
        return "proximal"
    if "metacarpal" in text:
        return "metacarpal"

    return None


def age_range_to_midpoint(age_value):
    # convert healthy reference age ranges like 12-24 into one midpoint age

    if pd.isna(age_value):
        return np.nan

    age_value = str(age_value).strip().replace("+", "")

    if "-" in age_value:
        try:
            start, end = age_value.split("-")
            return (float(start) + float(end)) / 2.0
        except Exception:
            return np.nan

    try:
        return float(age_value)
    except Exception:
        return np.nan


def get_mid_age_from_bin(bin_str):
    # get midpoint from a healthy age bin label

    try:
        text = str(bin_str).strip().replace("+", "")
        if "-" in text:
            low, high = text.split("-")
            return (float(low) + float(high)) / 2.0
        return float(text)
    except Exception:
        return np.nan


def get_start_age(bin_str):
    # get start age from age-bin text so bins can be sorted in the correct order

    try:
        text = str(bin_str).strip().replace("+", "")
        if "-" in text:
            return float(text.split("-")[0])
        return float(text)
    except Exception:
        return 999.0


def fit_simple_line(x, y):
    # fit a simple line to x and y
    # used for patient slope and for the healthy / hurler trend lines

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


def plot_with_fit_and_band(ax, x, y, color, label, scatter_alpha=0.25, line_alpha=1.0):
    # plot scatter + simple fitted line + a light residual band

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return False

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    try:
        coeffs = np.polyfit(x, y, 1)
        y_fit = np.polyval(coeffs, x)

        residuals = y - y_fit
        std = np.std(residuals)

        ax.scatter(x, y, color=color, alpha=scatter_alpha, s=28)

        ax.plot(
            x,
            y_fit,
            color=color,
            linewidth=2.5,
            alpha=line_alpha,
            label=label,
        )

        ax.fill_between(
            x,
            y_fit - std,
            y_fit + std,
            color=color,
            alpha=0.10,
        )
        return True

    except Exception as e:
        print(f"Could not fit line for {label}: {e}")
        ax.scatter(x, y, color=color, alpha=scatter_alpha, s=28, label=label)
        return True


# ============================================================
# LOAD UI PATIENT DATA
# ============================================================

def _load_patient_long_results(patient_dir: Path) -> pd.DataFrame:
    # load all study-level hand CSV files saved by the UI for one patient

    rows = []

    studies_root = patient_dir / "studies"
    if not studies_root.exists():
        return pd.DataFrame()

    for study_dir in sorted(studies_root.iterdir()):
        if not study_dir.is_dir():
            continue

        csv_path = study_dir / "outputs" / "results.csv"
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

    # standardize useful columns
    for col in [
        "age_months",
        "age_years",
        "length_mm",
        "width_mid_mm",
        "width_10_mm",
        "width_90_mm",
        "MCP",
        "PIP",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "study_date" in df.columns:
        df["study_date"] = pd.to_datetime(df["study_date"], errors="coerce")

    return df


# ============================================================
# CONVERT UI LONG DATA TO WIDE METRIC DATA
# ============================================================

def build_patient_wide_dataframe(patient_long_df):
    # convert UI long-format hand results into one row per study/age

    df = patient_long_df.copy()

    # use months as the main plotting age
    if "age_months" in df.columns:
        df["Age"] = pd.to_numeric(df["age_months"], errors="coerce")
    elif "age_years" in df.columns:
        df["Age"] = pd.to_numeric(df["age_years"], errors="coerce") * 12.0
    else:
        df["Age"] = np.nan

    if "segment_name" in df.columns:
        df["segment_key"] = df["segment_name"].apply(normalize_segment_name)
    else:
        df["segment_key"] = None

    for col in ["width_mid_mm", "width_10_mm", "width_90_mm", "length_mm", "MCP", "PIP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = []
    for c in ["study_folder", "study_date", "image_file", "Age"]:
        if c in df.columns:
            group_cols.append(c)

    if not group_cols:
        group_cols = ["Age"]

    rows = []

    # combine all bone rows from one study into one wide row
    for _, g in df.groupby(group_cols, dropna=False):
        row = {
            "Age": pd.to_numeric(g["Age"], errors="coerce").iloc[0] if "Age" in g.columns else np.nan
        }

        # keep patient id if present so later it can be saved / reviewed
        if "patient_folder" in g.columns:
            row["Patient_ID"] = str(g["patient_folder"].iloc[0]).strip()
        elif "patient_display" in g.columns:
            row["Patient_ID"] = str(g["patient_display"].iloc[0]).strip()

        # take joint angles from the saved curvature output
        if "MCP" in g.columns:
            mcp_vals = pd.to_numeric(g["MCP"], errors="coerce").dropna()
            row["MCP_Angle"] = mcp_vals.iloc[0] if not mcp_vals.empty else np.nan
        else:
            row["MCP_Angle"] = np.nan

        if "PIP" in g.columns:
            pip_vals = pd.to_numeric(g["PIP"], errors="coerce").dropna()
            row["PIP_Angle"] = pip_vals.iloc[0] if not pip_vals.empty else np.nan
        else:
            row["PIP_Angle"] = np.nan

        # store width and length values for each bone
        for seg_key, prefix in [
            ("distal", "distal"),
            ("middle", "middle"),
            ("proximal", "proximal"),
            ("metacarpal", "metacarpal"),
        ]:
            seg = g[g["segment_key"] == seg_key].copy()
            if seg.empty:
                continue

            if "width_mid_mm" in seg.columns:
                vals = pd.to_numeric(seg["width_mid_mm"], errors="coerce").dropna()
                if not vals.empty:
                    row[f"{prefix}_width_mid_mm"] = vals.mean()

            if "width_10_mm" in seg.columns:
                vals = pd.to_numeric(seg["width_10_mm"], errors="coerce").dropna()
                if not vals.empty:
                    row[f"{prefix}_width_10_mm"] = vals.mean()

            if "width_90_mm" in seg.columns:
                vals = pd.to_numeric(seg["width_90_mm"], errors="coerce").dropna()
                if not vals.empty:
                    row[f"{prefix}_width_90_mm"] = vals.mean()

            if "length_mm" in seg.columns:
                vals = pd.to_numeric(seg["length_mm"], errors="coerce").dropna()
                if not vals.empty:
                    row[f"{prefix}_length_mm"] = vals.mean()

        rows.append(row)

    wide_df = pd.DataFrame(rows)

    if wide_df.empty:
        return wide_df

    wide_df = wide_df.sort_values("Age")
    wide_df = wide_df.groupby("Age", as_index=False).first()

    # reconstruct the same style of ratios used in the original metric analysis
    def compute_ratios(row, prefix):
        length = row.get(f"{prefix}_length_mm", np.nan)
        w_mid = row.get(f"{prefix}_width_mid_mm", np.nan)
        w10 = row.get(f"{prefix}_width_10_mm", np.nan)
        w90 = row.get(f"{prefix}_width_90_mm", np.nan)

        out = {}

        out[f"{prefix}_ratio"] = length / w_mid if pd.notna(length) and pd.notna(w_mid) and w_mid != 0 else np.nan
        out[f"{prefix}_ratio10"] = length / w10 if pd.notna(length) and pd.notna(w10) and w10 != 0 else np.nan
        out[f"{prefix}_ratio90"] = length / w90 if pd.notna(length) and pd.notna(w90) and w90 != 0 else np.nan
        out[f"{prefix}_ratio_width"] = w90 / w10 if pd.notna(w10) and pd.notna(w90) and w10 != 0 else np.nan

        return out

    for prefix in ["distal", "middle", "proximal", "metacarpal"]:
        computed = wide_df.apply(lambda row: pd.Series(compute_ratios(row, prefix)), axis=1)
        wide_df = pd.concat([wide_df, computed], axis=1)

    # rename to match the expected disease-metric variable names
    rename_map = {
        "distal_ratio_width": "distal_phalanx_ratio_width",
        "middle_ratio": "middle_phalanx_ratio",
        "middle_ratio90": "middle_phalanx_ratio90",
        "middle_ratio_width": "middle_phalanx_ratio_width",
        "proximal_ratio": "proximal_phalanx_ratio",
        "metacarpal_ratio": "metacarpal_ratio",
    }
    wide_df = wide_df.rename(columns=rename_map)

    return wide_df


# ============================================================
# PREPARE REFERENCE EXCELS
# ============================================================

def prepare_hurler_reference(df):
    # standardize the Hurler reference Excel so it can be scored the same way as the UI patient data

    work = df.copy()
    work.columns = work.columns.str.strip()

    if "Patient age (months)" in work.columns and "Age" not in work.columns:
        work = work.rename(columns={"Patient age (months)": "Age"})
    elif "Age" not in work.columns and len(work.columns) > 0:
        possible = [c for c in work.columns if "age" in c.lower() and "month" in c.lower()]
        if possible:
            work = work.rename(columns={possible[0]: "Age"})

    work["Age"] = pd.to_numeric(work.get("Age"), errors="coerce")

    if "Patient_ID" not in work.columns:
        if "Patient ID" in work.columns:
            work = work.rename(columns={"Patient ID": "Patient_ID"})
        else:
            work["Patient_ID"] = "Hurler_reference"

    return work


def prepare_healthy_reference(df):
    # standardize the healthy reference Excel and derive a plotting age from the bin labels

    work = df.copy()
    work.columns = work.columns.str.strip()

    age_bin_col = None

    if "Age_Bin" in work.columns:
        age_bin_col = "Age_Bin"
    else:
        for col in work.columns:
            if "age" in col.lower():
                age_bin_col = col
                break

    if age_bin_col is None:
        # fall back to second column if needed
        if len(work.columns) > 1:
            age_bin_col = work.columns[1]
        else:
            age_bin_col = work.columns[0]

    if age_bin_col != "Age_Bin":
        work = work.rename(columns={age_bin_col: "Age_Bin"})

    work["Age_Bin"] = work["Age_Bin"].astype(str).str.strip()
    work["Age"] = work["Age_Bin"].apply(get_mid_age_from_bin)

    return work


def build_age_bins_from_healthy_labels(healthy_labels):
    # create cut-bin boundaries from healthy labels
    # this keeps patient ages aligned to the same healthy age ranges

    custom_bins = [0.0]

    for label in healthy_labels:
        try:
            text = str(label).strip().replace("+", "")
            if "-" in text:
                _, high = text.split("-")
                custom_bins.append(float(high) + 0.1)
            else:
                custom_bins.append(float(text) + 0.1)
        except Exception:
            continue

    custom_bins = sorted(list(set(custom_bins)))
    return custom_bins


def calculate_scores_for_dataframe(df, sig_configs, stats_dict):
    # calculate the weighted disease score row by row using age-matched healthy stats

    work = df.copy()

    def calculate_age_matched_score(row):
        bin_label = row.get("Age_Bin", np.nan)
        if pd.isna(bin_label) or bin_label not in stats_dict:
            return np.nan

        scores = []

        for var, config in sig_configs.items():
            val = pd.to_numeric(row.get(var, np.nan), errors="coerce")
            mean_val = stats_dict[bin_label][var]["mean"]
            std_val = stats_dict[bin_label][var]["std"]

            if pd.notna(val) and pd.notna(std_val) and std_val > 0:
                z = (val - mean_val) / std_val
                scores.append(z * config["dir"] * config["weight"])

        if not scores:
            return np.nan

        return np.sum(scores)

    work["Disease_Score"] = work.apply(calculate_age_matched_score, axis=1)
    return work


# ============================================================
# SIMPLE PLOTS FROM UI DATA
# ============================================================

def _plot_ratio_metric_from_ui(patient_long_df, patient_dir: Path, metric_col: str, title: str, ylabel: str, out_name: str):
    # make simple per-patient plots from the directly saved UI data

    if patient_long_df.empty or metric_col not in patient_long_df.columns:
        return None

    df = patient_long_df.copy()

    if "age_months" in df.columns:
        df["Age"] = pd.to_numeric(df["age_months"], errors="coerce")
    elif "age_years" in df.columns:
        df["Age"] = pd.to_numeric(df["age_years"], errors="coerce") * 12.0
    else:
        df["Age"] = np.nan

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["Age", metric_col]).sort_values("Age")

    if df.empty:
        return None

    analysis_dir = patient_dir / "analysis_data"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Age"], df[metric_col], marker="o", linewidth=2.2, color=COLOR_PATIENT)

    slope, intercept, p_value = fit_simple_line(df["Age"], df[metric_col])

    if slope is not None and intercept is not None and len(df) >= 2:
        x_fit = np.linspace(df["Age"].min(), df["Age"].max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, linestyle="--", linewidth=2, color=COLOR_PATIENT, alpha=0.8)

    title_text = title
    if slope is not None:
        if p_value is None:
            title_text += f"\nSlope: {slope:.4f}"
        else:
            title_text += f"\nSlope: {slope:.4f} (p={p_value:.4f})"

    ax.set_title(title_text)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = plots_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


# ============================================================
# DISEASE METRIC PLOT
# ============================================================

def _make_disease_metric_plot(patient_long_df: pd.DataFrame, patient_dir: Path):
    # create the final disease metric plot that compares:
    # 1. healthy baseline
    # 2. other Hurler reference data
    # 3. current patient

    analysis_dir = patient_dir / "analysis_data"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # convert UI outputs into the same style of variables used in the old metric analysis
    patient_wide = build_patient_wide_dataframe(patient_long_df)
    if patient_wide.empty:
        print("Could not build patient-wide dataframe for disease metric.")
        return []

    script_dir = Path(__file__).resolve().parent
    hurler_path = script_dir / "Data_analysis_full_hand.xlsx"
    healthy_path = script_dir / "Data_analysis_healthy_hands.xlsx"

    try:
        hurler_raw = pd.read_excel(hurler_path)
    except Exception as e:
        print(f"Could not load Hurler Excel file: {e}")
        return []

    try:
        healthy_raw = pd.read_excel(healthy_path)
    except Exception as e:
        print(f"Could not load healthy Excel file: {e}")
        return []

    hurler_df = prepare_hurler_reference(hurler_raw)
    healthy_df = prepare_healthy_reference(healthy_raw)

    # sort healthy age bins from the reference file
    healthy_labels = sorted(
        [b for b in healthy_df["Age_Bin"].unique() if str(b).lower() != "nan"],
        key=get_start_age,
    )

    if not healthy_labels:
        print("No healthy age bins found.")
        return []

    custom_bins = build_age_bins_from_healthy_labels(healthy_labels)
    if len(custom_bins) != len(healthy_labels) + 1:
        print("Age bin construction failed.")
        print(f"Healthy labels: {healthy_labels}")
        print(f"Custom bins: {custom_bins}")
        return []

    # apply the same age-bin structure to both the Hurler reference and the current patient
    hurler_df["Age_Bin"] = pd.cut(
        pd.to_numeric(hurler_df["Age"], errors="coerce"),
        bins=custom_bins,
        labels=healthy_labels,
        right=True,
        include_lowest=True,
    )

    patient_wide["Age_Bin"] = pd.cut(
        pd.to_numeric(patient_wide["Age"], errors="coerce"),
        bins=custom_bins,
        labels=healthy_labels,
        right=True,
        include_lowest=True,
    )

    # six variables used in the disease metric
    sig_variables = [
        "middle_phalanx_ratio_width",
        "metacarpal_ratio",
        "distal_phalanx_ratio_width",
        "MCP_Angle",
        "PIP_Angle",
        "proximal_phalanx_ratio",
    ]

    sig_configs = {
        "middle_phalanx_ratio_width": {"p": 1.60892685964344e-15, "dir": -1},
        "metacarpal_ratio": {"p": 0.265020670228078, "dir": -1},
        "distal_phalanx_ratio_width": {"p": 9.94591e-09, "dir": -1},
        "MCP_Angle": {"p": 1.7833e-08, "dir": 1},
        "PIP_Angle": {"p": 0.0403720099957933, "dir": 1},
        "proximal_phalanx_ratio": {"p": 0.014775417, "dir": -1},
    }

    # least significant gets weight 1, most significant gets highest weight
    sorted_variables = sorted(sig_configs.keys(), key=lambda x: sig_configs[x]["p"], reverse=True)
    for i, var in enumerate(sorted_variables):
        sig_configs[var]["weight"] = i + 1

    # calculate healthy mean and std per age bin for normalization
    bin_stats = {}
    for b in healthy_labels:
        bin_data = healthy_df[healthy_df["Age_Bin"] == b]
        stats = {}

        for var in sig_variables:
            vals = pd.to_numeric(bin_data[var], errors="coerce").dropna()
            stats[var] = {
                "mean": vals.mean(),
                "std": vals.std(),
            }
        bin_stats[b] = stats

    healthy_scored = calculate_scores_for_dataframe(healthy_df, sig_configs, bin_stats)
    hurler_scored = calculate_scores_for_dataframe(hurler_df, sig_configs, bin_stats)
    patient_scored = calculate_scores_for_dataframe(patient_wide, sig_configs, bin_stats)

    required_cols = ["Age"] + sig_variables + ["Disease_Score"]
    for col in required_cols:
        if col not in patient_scored.columns:
            patient_scored[col] = np.nan

    patient_metric_df = patient_scored.dropna(subset=required_cols).copy()
    if patient_metric_df.empty:
        print("No valid patient rows available for disease metric.")
        print("Available patient columns:")
        print(patient_scored.columns.tolist())
        print("Patient scored preview:")
        print(patient_scored.head())
        return []

    # remove final outliers for the current patient
    patient_metric_df = remove_outliers(patient_metric_df, column="Disease_Score", threshold=2.5)
    if patient_metric_df.empty:
        print("All patient disease metric rows were removed as outliers.")
        return []

    # prepare healthy and Hurler reference plotting frames
    healthy_plot = healthy_scored.dropna(subset=["Age", "Disease_Score"]).copy()
    hurler_plot = hurler_scored.dropna(subset=["Age", "Disease_Score"]).copy()

    healthy_plot = remove_outliers(healthy_plot, column="Disease_Score", threshold=2.5)
    hurler_plot = remove_outliers(hurler_plot, column="Disease_Score", threshold=2.5)

    # save these for checking later
    patient_metric_df.to_csv(analysis_dir / "patient_disease_metrics.csv", index=False)
    patient_metric_df.to_excel(analysis_dir / "patient_disease_metrics.xlsx", index=False)
    healthy_plot.to_csv(analysis_dir / "healthy_disease_reference.csv", index=False)
    hurler_plot.to_csv(analysis_dir / "hurler_disease_reference.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_anything = False

    # ------------------------------------------------------------
    # 1. healthy baseline
    # ------------------------------------------------------------
    # plot healthy scatter + fit + band as the reference around normal

    if len(healthy_plot) >= 2:
        ok = plot_with_fit_and_band(
            ax,
            healthy_plot["Age"],
            healthy_plot["Disease_Score"],
            COLOR_HEALTHY,
            "Healthy baseline",
            scatter_alpha=0.18,
            line_alpha=0.9,
        )
        if ok:
            # make the healthy fitted line dashed so it reads clearly as baseline
            slope_h, intercept_h, _ = fit_simple_line(healthy_plot["Age"], healthy_plot["Disease_Score"])
            if slope_h is not None and intercept_h is not None:
                x_fit = np.linspace(float(healthy_plot["Age"].min()), float(healthy_plot["Age"].max()), 200)
                y_fit = slope_h * x_fit + intercept_h
                ax.plot(
                    x_fit,
                    y_fit,
                    color=COLOR_HEALTHY,
                    linestyle="--",
                    linewidth=2.4,
                    alpha=0.85,
                    label="Healthy baseline",
                )
            plotted_anything = True

    # ------------------------------------------------------------
    # 2. other Hurler data in background
    # ------------------------------------------------------------
    # show all other Hurler points faintly and connect repeated measurements per patient

    hurler_background = hurler_plot.copy()

    # remove current patient if the patient happens to also be inside the Hurler reference file
    if "Patient_ID" in patient_metric_df.columns and "Patient_ID" in hurler_background.columns:
        current_ids = patient_metric_df["Patient_ID"].dropna().astype(str).unique().tolist()
        if current_ids:
            hurler_background = hurler_background[
                ~hurler_background["Patient_ID"].astype(str).isin(current_ids)
            ].copy()

    if not hurler_background.empty:
        ax.scatter(
            hurler_background["Age"],
            hurler_background["Disease_Score"],
            color=COLOR_HURLER,
            alpha=0.18,
            s=26,
            label="Other Hurler data",
            zorder=1,
        )
        plotted_anything = True

        if "Patient_ID" in hurler_background.columns:
            for pid, group in hurler_background.groupby("Patient_ID"):
                group = group.sort_values("Age")
                if len(group) > 1:
                    ax.plot(
                        group["Age"],
                        group["Disease_Score"],
                        color=COLOR_HURLER,
                        alpha=0.12,
                        linewidth=1.2,
                        zorder=1,
                    )

        # also add one overall Hurler trend line
        if len(hurler_background) >= 2:
            slope_hu, intercept_hu, _ = fit_simple_line(
                hurler_background["Age"],
                hurler_background["Disease_Score"]
            )
            if slope_hu is not None and intercept_hu is not None:
                x_fit = np.linspace(float(hurler_background["Age"].min()), float(hurler_background["Age"].max()), 200)
                y_fit = slope_hu * x_fit + intercept_hu
                ax.plot(
                    x_fit,
                    y_fit,
                    color=COLOR_HURLER,
                    linewidth=2.0,
                    alpha=0.70,
                    label="Hurler cohort trend",
                    zorder=2,
                )

    # ------------------------------------------------------------
    # 3. current patient on top
    # ------------------------------------------------------------
    # this is the main subject so keep it bold and clear

    patient_metric_df = patient_metric_df.sort_values("Age")

    ax.plot(
        patient_metric_df["Age"],
        patient_metric_df["Disease_Score"],
        color=COLOR_PATIENT,
        marker="o",
        linewidth=2.6,
        markersize=7,
        label="Current patient",
        zorder=5,
    )
    ax.scatter(
        patient_metric_df["Age"],
        patient_metric_df["Disease_Score"],
        color=COLOR_PATIENT,
        s=55,
        zorder=6,
    )
    plotted_anything = True

    # add patient-specific fitted trend
    slope_p, intercept_p, p_value = fit_simple_line(
        patient_metric_df["Age"],
        patient_metric_df["Disease_Score"],
    )

    if slope_p is not None and intercept_p is not None and len(patient_metric_df) >= 2:
        x_fit = np.linspace(
            float(patient_metric_df["Age"].min()),
            float(patient_metric_df["Age"].max()),
            100,
        )
        y_fit = slope_p * x_fit + intercept_p
        ax.plot(
            x_fit,
            y_fit,
            color=COLOR_PATIENT,
            linewidth=2.8,
            alpha=0.95,
            label="Patient trend",
            zorder=7,
        )

    if not plotted_anything:
        plt.close(fig)
        return []

    # final plot styling
    title = f"Disease Metric corrected for age (n={len(patient_metric_df)})"
    if slope_p is not None:
        if p_value is None:
            title += f"\nPatient slope: {slope_p:.4f}"
        else:
            title += f"\nPatient slope: {slope_p:.4f} (p={p_value:.4f})"

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Disease Metric Score")
    ax.set_title(title)
    ax.axhline(0, color=COLOR_HEALTHY, linewidth=1.0, alpha=0.25)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()

    save_path = plots_dir / "plot_disease_metric.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved disease metric plot to: {save_path}")
    return [save_path]


# ============================================================
# MAIN ENTRY
# ============================================================

def generate_progression_plots_for_patient(patient_dir: Path):
    # main entry used by the hand UI
    # create hand progression plots for the selected patient

    patient_dir = Path(patient_dir)

    analysis_dir = patient_dir / "analysis_data"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # remove older plot files first so the UI only shows fresh results
    for old_plot in plots_dir.glob("plot_*.png"):
        try:
            old_plot.unlink()
        except Exception:
            pass

    patient_long_df = _load_patient_long_results(patient_dir)
    if patient_long_df.empty:
        print("No patient hand UI results found.")
        return []

    plot_paths = []

    # simple direct UI plots if those columns exist
    if "MCP" in patient_long_df.columns:
        made = _plot_ratio_metric_from_ui(
            patient_long_df,
            patient_dir,
            metric_col="MCP",
            title="MCP Angle progression",
            ylabel="MCP angle (deg)",
            out_name="plot_mcp_angle.png",
        )
        if made is not None:
            plot_paths.append(made)

    if "PIP" in patient_long_df.columns:
        made = _plot_ratio_metric_from_ui(
            patient_long_df,
            patient_dir,
            metric_col="PIP",
            title="PIP Angle progression",
            ylabel="PIP angle (deg)",
            out_name="plot_pip_angle.png",
        )
        if made is not None:
            plot_paths.append(made)

    # create one disease metric plot using healthy + hurler references
    disease_metric_paths = _make_disease_metric_plot(patient_long_df, patient_dir)
    plot_paths.extend(disease_metric_paths)

    # save a copy of the long-format patient UI data for checking
    try:
        patient_long_df.to_csv(analysis_dir / "hand_progression_patient_long.csv", index=False)
    except Exception as e:
        print(f"Could not save patient hand summary CSV: {e}")

    return plot_paths
