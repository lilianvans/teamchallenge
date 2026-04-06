

import pandas as pd
import numpy as np
import os
from sklearn.exceptions import ConvergenceWarning
import statsmodels.formula.api as smf

import warnings
excel_patient= 'Data_analysis_full_hand.xlsx'
excel_healthy= 'Data_analysis_healthy_hands.xlsx'

def filter_extreme_outliers(df, column, max_remove=5, threshold_factor=3.5):
    """Remove extreme outliers based on median absolute deviation"""
    temp = pd.to_numeric(df[column], errors='coerce').dropna()
    if len(temp) < max_remove * 2:
        return df
    median = temp.median()
    mad = np.median(np.abs(temp - median))
    if mad == 0:
        return df
    robust_z = np.abs(0.6745 * (temp - median) / mad) 
    candidates = robust_z[robust_z > threshold_factor]
    if len(candidates) == 0:
        return df
    to_remove = candidates.sort_values(ascending=False).head(max_remove).index
    
    return df.drop(to_remove)


def get_start_age(bin_str):
    try:
        return float(str(bin_str).split('-')[0])
    except:
        return 999

def compute_delta_vs_healthy(df_pat, df_healthy, var, custom_bins, healthy_labels):
    # Age binning
    df_pat['Age_Bin'] = pd.cut(df_pat['Age'], bins=custom_bins, labels=healthy_labels)
    # Healthy means per bin
    healthy_means = df_healthy.groupby('Age_Bin', observed=True)[var].mean()
    # Delta = patient measurement - healthy mean
    df_pat[f'Delta_{var}'] = df_pat[var] - df_pat['Age_Bin'].map(healthy_means).astype(float)
    return df_pat

def test_feature_progression(df_pat, df_h_clean, var):
    delta_col = f'Delta_{var}'
    df_clean = df_pat.dropna(subset=[var, delta_col, 'Age'])
    
    slope, p_slope = np.nan, np.nan
    selected = False
    direction = "Stable"
    
    
    try:
        if len(df_clean) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model = smf.mixedlm(f"{delta_col} ~ Age", data=df_clean, groups=df_clean["Patient ID"]).fit()
                slope = model.params['Age']
                p_slope = model.pvalues['Age']
                selected = p_slope < 0.05
                direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"    
    except ConvergenceWarning:
        print(f"Skipping {var}: MixedLM did not converge.")
    except Exception:
        print(f"Skipping {var}: MixedLM failed.")

    # Effect size vs healthy (overall difference)
    mean_diff = df_clean[var].mean() - df_h_clean[var].mean()
    pooled_std = np.sqrt((df_clean[var].std()**2 + df_h_clean[var].std()**2)/2)
    effect_size = mean_diff / pooled_std if pooled_std != 0 else np.nan
    
    
    return {
        'Measurement': var,
        'Slope': slope,
        'Trend_p': p_slope,
        'Effect_Size_d': effect_size,
        'Direction': direction,
        'Selected': selected,
    }

def select_top_features_with_correlation(df_selected, df_raw, max_features=6, corr_threshold=0.7):
    """
    Select top features based on Progression_Score while avoiding high correlation.
    
    """
    
    # Sort by importance
    df_sorted = df_selected.sort_values('Progression_Score', ascending=False).copy()
    candidate_features = df_sorted['Measurement'].tolist()
    
    # Correlation matrix (absolute)
    corr_matrix = df_raw[candidate_features].corr().abs()
    
    final_features = []

    for feature in candidate_features:
        if len(final_features) == 0:
            final_features.append(feature)
            continue

        too_correlated = False
        for selected in final_features:
            
            if corr_matrix.loc[feature, selected] > corr_threshold:
    
                too_correlated = True
                break

        if not too_correlated:
            final_features.append(feature)

        if len(final_features) == max_features:
            break

    return df_sorted[df_sorted['Measurement'].isin(final_features)]

# main analysis function
def run_difference_analysis(excel_patient=excel_patient ,excel_healthy=excel_healthy):
    print("Starting analysis...")
    plot_folder = 'difference_analysis_plots'
    os.makedirs(plot_folder, exist_ok=True)

    # LOAD DATA
    try:
        df_p_raw = pd.read_excel(excel_patient)
        df_h_raw = pd.read_excel(excel_healthy)
        df_p_raw.columns = df_p_raw.columns.str.strip()
        df_h_raw.columns = df_h_raw.columns.str.strip()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Prepare patient data
    df_p_base = df_p_raw.copy()
    if 'Patient age (months)' in df_p_base.columns:
        df_p_base.rename(columns={'Patient age (months)': 'Age'}, inplace=True)
    df_p_base['Age'] = pd.to_numeric(df_p_base['Age'], errors='coerce')
    
    # Keep first measurement per patient per day, left hand only
    dup_cols = ['Patient ID', 'Laterality', 'Age']
    df_p_base = df_p_base.drop_duplicates(subset=dup_cols, keep='first')
    df_p_base = df_p_base[df_p_base['Laterality'] == 'L']

    # Prepare healthy controls
    df_h_base = df_h_raw.copy()
    age_bin_col = df_h_raw.columns[1]  # second column assumed to be age bin
    df_h_base.rename(columns={age_bin_col: 'Age_Bin'}, inplace=True)
    df_h_base['Age_Bin'] = df_h_base['Age_Bin'].astype(str).str.strip()
    healthy_labels = sorted([b for b in df_h_base['Age_Bin'].unique() if b != 'nan'], key=get_start_age)
    
    # Custom bins for age mapping
    custom_bins = [0]
    for label in healthy_labels:
        try:
            high_val = float(label.split('-')[1])
            custom_bins.append(high_val + 0.1)
        except: pass
    custom_bins = sorted(list(set(custom_bins)))

    # Identify measurements starting from 'Total_Flexion'
    start_col = 'Total_Flexion'
    p_meas_idx = df_p_raw.columns.get_loc(start_col)
    measurements = df_p_raw.columns[p_meas_idx:].tolist()

    results = []

    # Loop over features
    for var in measurements:
        print(f"Analyzing {var}...")
        df_p_base[var] = pd.to_numeric(df_p_base[var], errors='coerce')
        df_h_base[var] = pd.to_numeric(df_h_base[var], errors='coerce')
        
        df_p_clean = filter_extreme_outliers(df_p_base.copy(), var)
        df_h_clean = filter_extreme_outliers(df_h_base.copy(), var)
        df_p_clean = compute_delta_vs_healthy(df_p_clean, df_h_clean, var, custom_bins, healthy_labels)
        
        res = test_feature_progression(df_p_clean, df_h_clean, var)
        results.append(res)

    # Export
    output_file = 'Finger_Deviation_Analysis_Progression.xlsx'
    pd.DataFrame(results).to_excel(output_file, index=False)
    print("Analysis done!")
   
    df_results_selected = pd.DataFrame(results)
    df_selected = df_results_selected[df_results_selected['Selected'] == True].copy()
    # Score
    df_selected['Progression_Score'] = (
        df_selected['Slope'].abs()*2 + df_selected['Effect_Size_d'].abs() +0.2*(-np.log10(df_selected['Trend_p']))
    )

    
    df_raw_for_corr = df_p_base.copy()

    # Keep only numeric measurement columns
    df_raw_for_corr = df_raw_for_corr[['Patient ID', 'Age'] + measurements]

    # Select top features with low correlation
    final_top_features = select_top_features_with_correlation(
        df_selected,
        df_raw_for_corr,
        max_features=6,
        corr_threshold=0.7
    )

    print("\nFinal selected features (low correlation):")
    print(final_top_features[['Measurement','Slope','Effect_Size_d','Progression_Score','Direction']])

    

if __name__ == "__main__":
    run_difference_analysis()
