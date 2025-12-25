import pandas as pd
import numpy as np
import os

# --- 1. SETTINGS & FILES ---
FAST_SLOW_FILE = 'Fast_vs_Slow_Protein_PValues.csv'
CORR_FILE = 'protein_progression_correlations.csv'
PHENO_FILE = 'patient_phenotypes.csv'

# Thresholds for the two gates
PVAL_THRESHOLD = 0.05
CORR_THRESHOLD = 0.5

# --- 2. CHECK FILES ---
for f in [FAST_SLOW_FILE, CORR_FILE, PHENO_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}, please place it in the working directory.")

# --- 3. LOAD DATA ---
fast_slow_df = pd.read_csv(FAST_SLOW_FILE)  # Gate 1
corr_df = pd.read_csv(CORR_FILE)            # Gate 2
pheno_df = pd.read_csv(PHENO_FILE)          # Patient phenotypes

# Standardize gene names
fast_slow_df['Gene'] = fast_slow_df['Gene'].astype(str).str.upper()
corr_df['Gene'] = corr_df['Gene'].astype(str).str.upper()

# --- 4. MERGE THE TWO GATES ---
merged_df = pd.merge(fast_slow_df, corr_df[['Gene', 'Correlation_with_Progression']], on='Gene', how='inner')

# --- 5. APPLY TWO-GATE FILTER ---
winners_df = merged_df[
    (merged_df['P_Value'] < PVAL_THRESHOLD) &
    (merged_df['Correlation_with_Progression'] > CORR_THRESHOLD)
].sort_values('P_Value')

# --- 6. SAVE RESULTS ---
os.makedirs('results', exist_ok=True)
merged_df.to_csv('results/All_Protein_Stats.csv', index=False)
winners_df.to_csv('results/Two_Gate_Progression_Winners.csv', index=False)

# --- 7. PRINT SUMMARY ---
print("\n--- Two-Gate Analysis Complete ---")
print(f"Total proteins tested: {len(merged_df)}")
print(f"Proteins passing both gates: {len(winners_df)}")
if not winners_df.empty:
    print("\nTop candidates:")
    print(winners_df[['Gene', 'P_Value', 'Correlation_with_Progression']].head())
