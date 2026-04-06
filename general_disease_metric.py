import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
import math
import os


warnings.filterwarnings('ignore', category=RuntimeWarning)

# LOAD THE FULL DATA EXCELS HERE!
path_p = r'Data_analysis_full_hand.xlsx'
path_h = r'Data_analysis_healthy_hands.xlsx'

def remove_outliers(df, column='Disease_Score', threshold=2.5):
    """
    remove outliers based on the IQR of the final score so these do not skew the overall trends
    """
    if df.empty: return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    
    clean_df = df[mask]
    removed_df = df[~mask]  # <-- outliers

    print(f"Removed {len(removed_df)} outliers from {column} using IQR method (threshold={threshold}*IQR).")
    
    # Print indices
    print("Removed indices:", removed_df.index.tolist())

    return clean_df


def metric_analysis():
    "Basically the whole analysis that is done for the general disease metric"
    print("starting metric analysis...")
    
    try:
        df_p_raw = pd.read_excel(path_p)
        df_h_raw = pd.read_excel(path_h)
        
        #remove spaces from column names
        df_p_raw.columns = df_p_raw.columns.str.strip()
        df_h_raw.columns = df_h_raw.columns.str.strip()
    except Exception as e:
        print(f"error loading files: {e}")
        return

    #make copy to modify and keep original data intact
    df_p_base = df_p_raw.copy()

    #use the same age columns as other analyses
    if 'Patient age (months)' in df_p_base.columns:
        df_p_base = df_p_base.rename(columns={'Patient age (months)': 'Age'})
    
    df_p_base['Age'] = pd.to_numeric(df_p_base['Age'], errors='coerce')

    #if a patient has multiple images on the same day for the same hand, keep only the first record.
    dup_cols = ['Patient_ID', 'Laterality', 'Age']
    existing_dup_cols = [c for c in dup_cols if c in df_p_base.columns]
    df_p_base = df_p_base.drop_duplicates(subset=existing_dup_cols, keep='first')
    
    #only keep left hand scans
    if 'Laterality' in df_p_base.columns:
        pre_lat = len(df_p_base)
        df_p_base = df_p_base[df_p_base['Laterality'] == 'L']
        print(f"removed {pre_lat - len(df_p_base)} right hand scans.")

    #parameters that must be present for an image to be valid (the 6 significant ones)
    sig_variables = ['middle_phalanx_ratio_width','metacarpal_ratio','distal_phalanx_ratio_width','MCP_Angle','PIP_Angle','proximal_phalanx_ratio']
    #drop rows missing any of these 
    df_p_base = df_p_base.dropna(subset=sig_variables)
    print(f"final amount of images: {len(df_p_base)}")

    #setup healthy bins for age-matched normalization
    df_h_base = df_h_raw.copy()
    age_bin_col = df_h_base.columns[1]
    df_h_base = df_h_base.rename(columns={age_bin_col: 'Age_Bin'})
    df_h_base['Age_Bin'] = df_h_base['Age_Bin'].astype(str).str.strip()

    def get_start_age(bin_str):
        try: return float(str(bin_str).split('-')[0])
        except: return 999

    healthy_labels = sorted([b for b in df_h_base['Age_Bin'].unique() if b != 'nan'], key=get_start_age)
    
    #create specific bins so the month 'between the bins' is also accounted for using padding
    custom_bins = [0]
    for label in healthy_labels:
        try:
            high_val = float(label.split('-')[1])
            custom_bins.append(high_val + 0.1) 
        except: pass
    custom_bins = sorted(list(set(custom_bins)))

    #apply bins to patient data
    df_p_base['Age_Bin'] = pd.cut(df_p_base['Age'], bins=custom_bins, labels=healthy_labels, right=True)

    #setup rank-based weights
    sig_configs = {
        'middle_phalanx_ratio_width': {'p':1.60892685964344e-15 , 'dir': -1},
        'metacarpal_ratio': {'p': 0.265020670228078, 'dir': -1},
        'distal_phalanx_ratio_width': {'p': 9.94591e-09, 'dir': -1},
        'MCP_Angle': {'p':1.7833e-08, 'dir': 1},
        'PIP_Angle': {'p': 0.0403720099957933, 'dir': 1},
        'proximal_phalanx_ratio': {'p': 0.014775417, 'dir': -1}
    }

    #so basically least significant gets 1, most significant gets 6
    sorted_variables = sorted(sig_configs.keys(), key=lambda x: sig_configs[x]['p'], reverse=True)
    for i, var in enumerate(sorted_variables):
        sig_configs[var]['weight'] = i + 1

    #calculate healthy statistics per bin for the normalization
    bin_stats = {}
    for b in healthy_labels:
        bin_data = df_h_base[df_h_base['Age_Bin'] == b]
        stats = {}
        for var in sig_variables:
            vals = pd.to_numeric(bin_data[var], errors='coerce').dropna()
            stats[var] = {'mean': vals.mean(), 'std': vals.std()}
        bin_stats[b] = stats

    #function to calculate the metric for each patient based on age-matched healthy statistics
    def calculate_age_matched_score(row, stats_dict):
        bin_label = row['Age_Bin']
        if pd.isna(bin_label) or bin_label not in stats_dict: return np.nan
        
        scores = []
        for var, config in sig_configs.items():
            val = pd.to_numeric(row[var], errors='coerce')
            m = stats_dict[bin_label][var]['mean']
            s = stats_dict[bin_label][var]['std']
            
            if not pd.isna(val) and s > 0:
                # distance from healthy mean per 12-months age bin
                z = (val - m) / s
                scores.append(z * config['dir'] * config['weight'])
        return np.sum(scores)

    # calculate the final metric for both groups
    print("calculating age-matched Disease Scores...")
    df_p_base['Disease_Score'] = df_p_base.apply(lambda r: calculate_age_matched_score(r, bin_stats), axis=1)
    
    #for healthy data, calculate age for plotting purposes
    def get_mid_age(bin_str):
        try:
            parts = str(bin_str).replace('+', '').split('-')
            return sum([float(p) for p in parts]) / len(parts)
        except: return np.nan
    #mid age of the bin cause we dont know exact age
    df_h_base['Age'] = df_h_base['Age_Bin'].apply(get_mid_age)
    df_h_base['Disease_Score'] = df_h_base.apply(lambda r: calculate_age_matched_score(r, bin_stats), axis=1)

    #remove final outliers for calculated metric
    df_final = remove_outliers(df_p_base, column='Disease_Score', threshold=2.5)

    #export to excel
    df_final.to_excel('patient_disease_metrics.xlsx', index=False)
    
    if not df_final.empty:
        plt.figure(figsize=(10, 6))

        # Mixed linear model
        try:
            mixed_model = smf.mixedlm(
                "Disease_Score ~ Age",
                data=df_final,
                groups=df_final["Patient ID"],   # random intercept per patient
                re_formula=None             # random slope (optional but recommended)
            ).fit()

            slope = mixed_model.params['Age']
            pval = mixed_model.pvalues['Age']

            print(mixed_model.summary())

        except Exception as e:
            print(f"Mixed model error: {e}")
            slope, pval = np.nan, np.nan

        # healthy baseline
        df_h_plot = df_h_base.dropna(subset=['Age', 'Disease_Score'])
        sns.regplot(
            data=df_h_plot,
            x='Age',
            y='Disease_Score',
            scatter=False,
            color="#3546DA",
            label='Healthy baseline',
            line_kws={'linestyle': '--', 'alpha': 0.5}
        )

        # Individual patient lines
        for pid, group in df_final.groupby("Patient ID"):
            if len(group) > 1:
                plt.plot(group["Age"], group["Disease_Score"], alpha=0.2)

        # scatter plot
        sns.scatterplot(
            data=df_final,
            x='Age',
            y='Disease_Score',
            color="#d95f5f",
            alpha=0.6,
            label='Patient data'
        )

        # Population Trend (fixed effect)
        if not np.isnan(slope):
            ages = np.linspace(df_final["Age"].min(), df_final["Age"].max(), 100)
            intercept = mixed_model.params['Intercept']
            pred = intercept + slope * ages

            plt.plot(
                ages,
                pred,
                color="#CF2F2F",
                linewidth=2,
                label='Mixed model trend'
            )

        # Titles and labels 
        plt.title(
            f"Disease Metric (mixed model, n={len(df_final)})\n"
            f"Slope: {slope:.4f} (p={pval:.4f})"
        )

        plt.ylabel("Disease Metric Score")
        plt.xlabel("Age (months)")
        plt.axhline(0, color='blue', linewidth=0.8, alpha=0.3)

        plt.legend(loc='upper left', frameon=True, fontsize='medium')
        plt.tight_layout()

        plt.savefig("final_disease_metric_mixed_model.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    metric_analysis()
