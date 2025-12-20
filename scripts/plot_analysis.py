import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

# ========================================================
# PRE-REQUISITE: Ensure previous data exists
# ========================================================
# This block assumes 'slope_df' (from Plot 2) and 'z_scores_clean' 
# (from Plot 3) are still in memory. If not, re-run the previous plotting code first.

print("--- STEP 1: STATISTICAL VALIDATION (The 'P-Value' Check) ---")

# 1. Test if the average ALS patient has a slope significantly different from 0
# Null Hypothesis: The pathway does not change over time (Slope = 0)
t_stat, p_val = ttest_1samp(slope_df['ECM_Slope'], 0)

print(f"Average ECM Slope: {slope_df['ECM_Slope'].mean():.4f}")
print(f"T-Test Result: t-statistic={t_stat:.3f}, p-value={p_val:.4e}")

if p_val < 0.05:
    print(">> STATISTICALLY SIGNIFICANT: The ECM pathway actively changes over time in your ALS cohort.")
else:
    print(">> NOTE: The average change is small, but individual patients may still vary significantly.")

# ========================================================
# STEP 2: CREATE "FAST VS SLOW" LABELS (Mathematical Modeling)
# ========================================================
print("\n--- STEP 2: GENERATING 'FAST' VS 'SLOW' LABELS ---")

# We define "Fast Progressors" as those with an ECM Slope greater than the median
median_slope = slope_df['ECM_Slope'].median()
slope_df['Progression_Type'] = np.where(slope_df['ECM_Slope'] > median_slope, 'Fast Remodeler', 'Slow Remodeler')

print(f"Median Slope Cutoff: {median_slope:.4f}")
print("Patient Classification:")
print(slope_df[['Patient_ID', 'ECM_Slope', 'Progression_Type']].head())

# Save this new metadata. You can use this for future plots!
slope_df.to_csv("ALS_Patient_Progression_Labels.csv", index=False)
print(">> SAVED: 'ALS_Patient_Progression_Labels.csv'")

# ========================================================
# STEP 3: IDENTIFYING THE "HUB" PROTEINS
# ========================================================
print("\n--- STEP 3: EXPORTING TOP HUB PROTEINS FOR LITERATURE REVIEW ---")

# We identify hubs by "Variability" (Standard Deviation)
# High variability means the protein reacts strongly in some patients (the "Fast" ones)
all_module_genes = list(set(ecm_genes + comp_genes))
top_hubs = z_scores_clean[all_module_genes].std().sort_values(ascending=False).head(20)

# Create a clean summary table
hub_summary = pd.DataFrame({
    'Gene': top_hubs.index,
    'Variability_Score': top_hubs.values,
    'Mean_Expression_Z': z_scores_clean[top_hubs.index].mean().values,
    'Module': ['ECM' if g in ecm_genes else 'Complement' for g in top_hubs.index]
})

print("Top 5 Driver Proteins (Your 'Key Findings'):")
print(hub_summary.head(5))

hub_summary.to_csv("Top_ALS_Hub_Proteins.csv", index=False)
print(">> SAVED: 'Top_ALS_Hub_Proteins.csv'")