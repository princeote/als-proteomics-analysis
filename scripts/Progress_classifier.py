import pandas as pd
import numpy as np
from sklearn.linear_model import TheilSenRegressor
import os

# 1. Ensure results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# 2. Load data
df = pd.read_csv('results/normalized_proteins.csv')

# Setup patients and timepoints
als_patients = [f'ALS_{i}' for i in range(1, 11)]
timepoints = ['T1', 'T2', 'T3', 'T4']
x_months = np.array([0, 6, 12, 18]).reshape(-1, 1)  # Reshape for sklearn

# 3. Calculate robust slope for every protein and patient
def get_robust_slope(row, p_id):
    y = np.array([row[f'{p_id}_{t}_MaxLFQ_Intensity'] for t in timepoints], dtype=float)
    mask = ~np.isnan(y)
    if np.sum(mask) < 3:
        return np.nan
    reg = TheilSenRegressor()
    reg.fit(x_months[mask], y[mask])
    return reg.coef_[0]

print("Calculating robust protein slopes...")
slope_df = df[['Protein', 'Gene']].copy()
for p in als_patients:
    slope_df[f'{p}_slope'] = df.apply(lambda row: get_robust_slope(row, p), axis=1)

# 4. Extract CHI3L1 slopes
chi3l1_row = slope_df[slope_df['Gene'] == 'CHI3L1']
if chi3l1_row.empty:
    raise ValueError("CHI3L1 not found in the dataset.")

chi3l1_slopes = chi3l1_row.iloc[0, 2:].values.astype(float)

# 5. Classify patients using tertiles
lower_cutoff = np.nanpercentile(chi3l1_slopes, 33.3)
upper_cutoff = np.nanpercentile(chi3l1_slopes, 66.6)

patient_phenotypes = pd.DataFrame({
    'Patient_ID': als_patients,
    'Progression_Slope_CHI3L1': chi3l1_slopes
})

patient_phenotypes['Phenotype'] = 'Middle'
patient_phenotypes.loc[patient_phenotypes['Progression_Slope_CHI3L1'] <= lower_cutoff, 'Phenotype'] = 'Slow'
patient_phenotypes.loc[patient_phenotypes['Progression_Slope_CHI3L1'] >= upper_cutoff, 'Phenotype'] = 'Fast'

# Save phenotype file
patient_phenotypes.to_csv('results/patient_phenotypes.csv', index=False)

print("\nPhenotype classification complete.")
print(patient_phenotypes)
