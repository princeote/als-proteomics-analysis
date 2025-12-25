import pandas as pd
import numpy as np
from scipy.stats import linregress, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set_theme(style="whitegrid")

# Create directories
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ========================================================
# 1. LOAD DATA
# ========================================================
df = pd.read_csv("results/normalized_proteins.csv")
pheno_df = pd.read_csv("results/patient_phenotypes.csv")

# Standardize Gene column
df['Gene'] = df['Gene'].astype(str).str.strip().str.upper()

# ========================================================
# 2. CALCULATE PROTEIN SLOPES PER PATIENT
# ========================================================
als_patients = pheno_df['Patient_ID'].tolist()
timepoints = ['T1', 'T2', 'T3', 'T4']
x_months = np.array([0, 6, 12, 18])

# Function to compute slope per protein for a patient
def protein_slope(row, patient_id):
    y = [row[f'{patient_id}_{tp}_MaxLFQ_Intensity'] for tp in timepoints]
    y = np.array(y, dtype=float)  # Ensure numeric
    mask = ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    return linregress(x_months[mask], y[mask])[0]

# Build patient-level slope table
slope_data = {'Gene': df['Gene']}
for patient in als_patients:
    slope_data[patient] = df.apply(lambda row: protein_slope(row, patient), axis=1)

slopes_df = pd.DataFrame(slope_data)

# Ensure all numeric
for patient in als_patients:
    slopes_df[patient] = pd.to_numeric(slopes_df[patient], errors='coerce')

# ========================================================
# 3. PREPARE FAST VS SLOW PATIENTS
# ========================================================
fast_patients = pheno_df[pheno_df['Phenotype'] == 'Fast']['Patient_ID'].tolist()
slow_patients = pheno_df[pheno_df['Phenotype'] == 'Slow']['Patient_ID'].tolist()

# ========================================================
# 4. CALCULATE P-VALUES
# ========================================================
stats = []
for _, row in slopes_df.iterrows():
    gene = row['Gene']
    if gene == 'CHI3L1':  # skip the anchor
        continue

    f_vals = np.array(row[fast_patients], dtype=float)
    s_vals = np.array(row[slow_patients], dtype=float)

    if np.sum(~np.isnan(f_vals)) > 1 and np.sum(~np.isnan(s_vals)) > 1:
        _, p = ttest_ind(f_vals, s_vals, nan_policy='omit')
        stats.append({'Gene': gene, 'P_Value': p})

winners_df = pd.DataFrame(stats).sort_values('P_Value').reset_index(drop=True)
winners_df.to_csv("results/Fast_vs_Slow_Protein_PValues.csv", index=False)
print("Saved Fast_vs_Slow_Protein_PValues.csv")

# ========================================================
# 5. VISUALIZATION (TOP 8 PROTEINS)
# ========================================================
top8 = winners_df['Gene'].head(8).tolist()
palette = {'Fast': 'red', 'Slow': 'blue', 'Middle': 'lightgrey'}

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
axes = axes.flatten()

for i, gene in enumerate(top8):
    plot_data = pd.DataFrame({
        'Patient_ID': fast_patients + slow_patients,
        'Slope': slopes_df.loc[slopes_df['Gene']==gene, fast_patients + slow_patients].values.flatten(),
        'Group': ['Fast']*len(fast_patients) + ['Slow']*len(slow_patients)
    })
    sns.boxplot(data=plot_data, x='Group', y='Slope', palette=palette, ax=axes[i])
    sns.stripplot(data=plot_data, x='Group', y='Slope', color='black', size=6, ax=axes[i], jitter=True)
    p_val = winners_df[winners_df['Gene']==gene]['P_Value'].values[0]
    axes[i].set_title(f"{gene} (Fast vs Slow p={p_val:.2e})", fontsize=12, fontweight='bold')
    axes[i].set_ylabel("Protein Slope")

plt.tight_layout()
plt.savefig("plots/Top8_Fast_vs_Slow_Protein_Slopes.png", dpi=300)
plt.show()
