import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
import os

excel_hip= 'Data_analysis_hip.xlsx'

def group_progression(excel_hip=excel_hip):
    #loading hip data for group analysis
    input_file = excel_hip
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        #error handling if file is missing
        print(f"Error: Could not find '{input_file}'.")
        return

    #define columns to exclude from measurement analysis
    standard_columns = [
        'Patient ID', 'Study Date', 'Serial number', 'Image number', 
        'Full folder name', 'Patient age (months)', 'Op_Status', 'Body part',
        'phenotypic severity (1=Hurler, 2 = Hurler-Scheie)',
        'Age at transplant (in months)', 'effective HSCT (y/n)',
        'LLN_IDUA_level_Mean_Total', 'timing diagnosis'
    ]
    
    #identifying measurement columns automatically
    measurements = [col for col in df.columns if col not in standard_columns]

    #cleaning data to ensure numeric types for math
    for col in measurements:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Patient age (months)'] = pd.to_numeric(df['Patient age (months)'], errors='coerce')
    
    #drop rows where age is missing
    df = df.dropna(subset=['Patient age (months)'])

    group_results = []
    plot_folder = 'hip_group_analysis_plots'
    
    #create folder for plots if it doesn't exist
    os.makedirs(plot_folder, exist_ok=True)

    print(f"performing group regression on {len(measurements)} metrics...")

    for var in measurements:
        #filtering for valid data points in this specific measurement
        valid_data = df.dropna(subset=[var])
        
        #regression requires at least two points
        if len(valid_data) > 2:
            x = valid_data['Patient age (months)']
            y = valid_data[var]
            
            #calculating group-level linear regression
            slope, intercept, r_val, p_val, std_err = linregress(x, y)
            
            #converting monthly slope to annual rate
            annual_rate = slope * 12

            #compiling statistics for the summary table
            group_results.append({
                'Measurement': var,
                'Total N (points)': len(valid_data),
                'Unique Patients': valid_data['Patient ID'].nunique(),
                'Mean Value': round(y.mean(), 4),
                'Annualized Group Rate': round(annual_rate, 4),
                'R-Squared': round(r_val**2, 4),
                'P-Value': round(p_val, 4),
                'Significant (p<0.05)': p_val < 0.05
            })

            #generating group trend plot
            plt.figure(figsize=(10, 6))
            sns.set_style("white")

            #plotting individual data as faint gray dots
            sns.scatterplot(data=valid_data, x='Patient age (months)', y=var, 
                            alpha=0.3, color='gray', label='Individual Data')

            #adding the red regression line with a 95% confidence interval
            sns.regplot(data=valid_data, x='Patient age (months)', y=var, 
                        scatter=False, color='red', label=f'Group Trend (p={p_val:.4f})')

            #formatting plot labels and title
            plt.title(f'Group Analysis: {var} vs Age', fontsize=14)
            plt.xlabel('Patient age (months)')
            plt.ylabel(var)
            plt.legend()
            
            #saving the plot to the specified folder
            plot_path = os.path.join(plot_folder, f"group_trend_{var}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

    #saving compiled group stats to excel
    df_group_stats = pd.DataFrame(group_results)
    stats_output = 'hip_group_analysis.xlsx'
    df_group_stats.to_excel(stats_output, index=False)
    
    #final status updates
    print(f"saved to: {stats_output}")
    print(f"plots saved in: '{plot_folder}'")

if __name__ == '__main__':
    # execution entry point
    group_progression()
