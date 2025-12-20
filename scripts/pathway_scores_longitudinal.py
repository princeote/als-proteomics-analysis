import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import linregress, pearsonr, ttest_1samp
import os

# Set global plot style
sns.set_theme(style="whitegrid")

# Create directories if they don't exist
for folder in ['results', 'plots']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ========================================================
# 1. DATA LOADING & PRE-PROCESSING
# ========================================================
df = pd.read_csv("results/filtered_normalized_proteins.csv")

if 'Gene' not in df.columns:
    df['Gene'] = df['Protein'].astype(str).str.strip().str.upper()
else:
    df['Gene'] = df['Gene'].astype(str).str.strip().str.upper()

# Aggregate Proteins -> Genes (using median) and Transpose
gene_df = df.drop(columns=['Protein'], errors='ignore').groupby('Gene').median().T
gene_df.index = gene_df.index.astype(str).str.strip()

# ========================================================
# 2. METADATA EXTRACTION
# ========================================================
metadata = pd.DataFrame({'Sample_ID': gene_df.index})
metadata['Group'] = metadata['Sample_ID'].apply(
    lambda x: 'ALS' if 'ALS' in x.upper() else 'CONTROL' if 'CONTROL' in x.upper() else 'UNKNOWN'
)
metadata['Patient_ID'] = metadata['Sample_ID'].str.extract(r'^([A-Za-z]+_\d+)')
metadata['Timepoint_Num'] = metadata['Sample_ID'].str.extract(r'T(\d+)').astype(float)

# ========================================================
# 3. LOAD PATHWAY GENE LISTS
# ========================================================
def load_genes(path, data_genes):
    full_path = os.path.join("gene_list", path)
    try:
        gl = pd.read_csv(full_path, header=None)
        list_genes = gl.iloc[:, 0].astype(str).str.strip().str.upper().unique().tolist()
        return [g for g in list_genes if g in data_genes]
    except Exception as e:
        print(f"Error loading {full_path}: {e}")
        return []

data_genes = set(gene_df.columns)
ecm_genes = load_genes("ECM_genes.csv", data_genes)
comp_genes = load_genes("Complement_genes.csv", data_genes)
all_module_genes = list(set(ecm_genes + comp_genes))

# ========================================================
# 4. NORMALIZATION (CONTROL-ANCHORED Z-SCORING)
# ========================================================
controls = metadata[metadata['Group'] == 'CONTROL']['Sample_ID']
control_mean = gene_df.loc[controls].mean()
control_std = gene_df.loc[controls].std().replace(0, np.nan)

z_scores = (gene_df - control_mean) / control_std
z_scores_clean = z_scores.fillna(0).replace([np.inf, -np.inf], 0)

# Calculate Module Activity Scores
metadata['ECM_Score'] = z_scores_clean[ecm_genes].median(axis=1).values
metadata['Complement_Score'] = z_scores_clean[comp_genes].median(axis=1).values

# ========================================================
# 5. LONGITUDINAL SLOPE ANALYSIS (ECM & COMPLEMENT)
# ========================================================
als_only = metadata[metadata['Group'] == 'ALS']
patient_slopes = []

for pid in als_only['Patient_ID'].dropna().unique():
    p_data = als_only[als_only['Patient_ID'] == pid].sort_values('Timepoint_Num')
    if len(p_data) > 1:
        # Calculate Slope for ECM
        e_slope, _, _, _, _ = linregress(p_data['Timepoint_Num'], p_data['ECM_Score'])
        # Calculate Slope for Complement
        c_slope, _, _, _, _ = linregress(p_data['Timepoint_Num'], p_data['Complement_Score'])
        
        patient_slopes.append({
            'Patient_ID': pid, 
            'ECM_Slope': e_slope, 
            'Comp_Slope': c_slope
        })

slope_df = pd.DataFrame(patient_slopes)

# ========================================================
# 6. PLOTTING THE KEY FINDINGS
# ========================================================

# PLOT 1: Complement vs ECM Activity Correlation
plt.figure(figsize=(8, 6))
sns.regplot(data=als_only, x='Complement_Score', y='ECM_Score', color='darkred')
corr, p_corr = pearsonr(als_only['Complement_Score'], als_only['ECM_Score'])
plt.title(f"1. Module Activity Interplay\nr = {corr:.2f}, p = {p_corr:.2e}")
plt.savefig("results/interplay_activity_correlation.png")

# PLOT 2: ECM Progression Slopes (FIXED Palette Warning)
plt.figure(figsize=(10, 6))
slope_df_e = slope_df.sort_values('ECM_Slope', ascending=False)
sns.barplot(data=slope_df_e, x='Patient_ID', y='ECM_Slope', hue='Patient_ID', palette="Reds_r", legend=False)
plt.xticks(rotation=45)
plt.title("2. Patient-Specific ECM Progression Rates")
plt.savefig("plots/ecm_progression_slopes.png")

# PLOT 3: Complement Progression Slopes (FIXED Palette Warning)
plt.figure(figsize=(10, 6))
slope_df_c = slope_df.sort_values('Comp_Slope', ascending=False)
sns.barplot(data=slope_df_c, x='Patient_ID', y='Comp_Slope', hue='Patient_ID', palette="Blues_r", legend=False)
plt.xticks(rotation=45)
plt.title("3. Patient-Specific Complement Progression Rates")
plt.savefig("plots/complement_progression_slopes.png")

# PLOT 4: Slope Alignment
plt.figure(figsize=(8, 6))
sns.regplot(data=slope_df, x='ECM_Slope', y='Comp_Slope', color='purple')
s_corr, s_p = pearsonr(slope_df['ECM_Slope'], slope_df['Comp_Slope'])
plt.title(f"4. Progression Alignment: ECM vs Complement Slopes\nr = {s_corr:.2f}, p = {s_p:.2e}")
plt.xlabel("ECM Progression Rate (Slope)")
plt.ylabel("Complement Progression Rate (Slope)")
plt.savefig("plots/slope_alignment_correlation.png")

# PLOT 5: Hub Heatmap
top_30 = z_scores_clean[all_module_genes].std().sort_values(ascending=False).head(30).index
sns.clustermap(z_scores_clean[top_30].T, cmap='vlag', center=0, 
               col_colors=metadata['Group'].map({'ALS': 'orange', 'CONTROL': 'grey'}).values)
plt.savefig("plots/hub_heatmap.png")

# ========================================================
# 7. EXPORTS
# ========================================================
median_slope = slope_df['ECM_Slope'].median()
slope_df['Progression_Type'] = np.where(slope_df['ECM_Slope'] > median_slope, 'Fast Remodeler', 'Slow Remodeler')
slope_df.to_csv("results/ALS_Detailed_Progression_Analysis.csv", index=False)

# Identify Hub Proteins
top_hubs = z_scores_clean[all_module_genes].std().sort_values(ascending=False).head(20)
hub_summary = pd.DataFrame({
    'Gene': top_hubs.index,
    'Variability': top_hubs.values,
    'Module': ['ECM' if g in ecm_genes else 'Complement' for g in top_hubs.index]
})
hub_summary.to_csv("results/Top_ALS_Hub_Proteins.csv", index=False)

print(f"Finished! Files saved in 'plots/' and 'results/'. Found {len(ecm_genes)} ECM genes and {len(comp_genes)} Complement genes.")