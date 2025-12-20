import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import linregress, pearsonr, ttest_1samp

# Set global plot style
sns.set_theme(style="whitegrid")

# ========================================================
# 1. DATA LOADING & PRE-PROCESSING
# ========================================================
df = pd.read_csv("filtered_normalized_proteins.csv")

# Ensure 'Gene' column is present
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
    try:
        gl = pd.read_csv(path, header=None)
        list_genes = gl.iloc[:, 0].astype(str).str.strip().str.upper().unique().tolist()
        return [g for g in list_genes if g in data_genes]
    except: return []

data_genes = set(gene_df.columns)
ecm_genes = load_genes("ECM_genes.csv", data_genes)
comp_genes = load_genes("complement_genes.csv", data_genes)
all_module_genes = list(set(ecm_genes + comp_genes))

# ========================================================
# 4. NORMALIZATION (CONTROL-ANCHORED Z-SCORING)
# ========================================================
controls = metadata[metadata['Group'] == 'CONTROL']['Sample_ID']
control_mean = gene_df.loc[controls].mean()
control_std = gene_df.loc[controls].std().replace(0, np.nan)

z_scores = (gene_df - control_mean) / control_std
z_scores_clean = z_scores.fillna(0).replace([np.inf, -np.inf], 0)

# Calculate Module Scores
metadata['ECM_Score'] = z_scores[ecm_genes].median(axis=1).values
metadata['Complement_Score'] = z_scores[comp_genes].median(axis=1).values
metadata = metadata.fillna(0)

# ========================================================
# 5. CORE ANALYSES & PLOTTING
# ========================================================

# PLOT 1: Complement-ECM Correlation
plt.figure(figsize=(8, 6))
als_only = metadata[metadata['Group'] == 'ALS']
sns.regplot(data=als_only, x='Complement_Score', y='ECM_Score', color='darkred')
corr, p_corr = pearsonr(als_only['Complement_Score'], als_only['ECM_Score'])
plt.title(f"1. Module Interplay (ALS Cohort)\nPearson r = {corr:.2f}, p = {p_corr:.2e}")
plt.savefig("interplay_correlation.png")

# PLOT 2: Calculate Progression Slopes
patient_slopes = []
for pid in als_only['Patient_ID'].dropna().unique():
    p_data = als_only[als_only['Patient_ID'] == pid].sort_values('Timepoint_Num')
    if len(p_data) > 1:
        slope, _, _, _, _ = linregress(p_data['Timepoint_Num'], p_data['ECM_Score'])
        patient_slopes.append({'Patient_ID': pid, 'ECM_Slope': slope})

slope_df = pd.DataFrame(patient_slopes).sort_values('ECM_Slope', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=slope_df, x='Patient_ID', y='ECM_Slope', palette="Reds_r")
plt.xticks(rotation=45)
plt.title("2. Patient-Specific ECM Progression Rates (Slopes)")
plt.savefig("progression_slopes.png")

# ========================================================
# 6. STATISTICAL VALIDATION & STRATIFICATION
# ========================================================
print("--- ANALYSIS COMPLETE ---")

# Step 1: Statistical Significance of the Slopes
t_stat, p_ttest = ttest_1samp(slope_df['ECM_Slope'], 0)
print(f"Cohort Mean ECM Slope: {slope_df['ECM_Slope'].mean():.4f} (p={p_ttest:.4f})")

# Step 2: Define Fast vs Slow Labels (Median Split)
median_slope = slope_df['ECM_Slope'].median()
slope_df['Progression_Type'] = np.where(slope_df['ECM_Slope'] > median_slope, 'Fast Remodeler', 'Slow Remodeler')
slope_df.to_csv("ALS_Patient_Progression_Labels.csv", index=False)

# Step 3: Identify Hub Proteins (Most Variable in Modules)
top_hubs = z_scores_clean[all_module_genes].std().sort_values(ascending=False).head(20)
hub_summary = pd.DataFrame({
    'Gene': top_hubs.index,
    'Variability': top_hubs.values,
    'Module': ['ECM' if g in ecm_genes else 'Complement' for g in top_hubs.index]
})
hub_summary.to_csv("Top_ALS_Hub_Proteins.csv", index=False)

# PLOT 3 & 4: Heatmap and PCA
# (Heatmap)
top_30 = z_scores_clean[all_module_genes].std().sort_values(ascending=False).head(30).index
sns.clustermap(z_scores_clean[top_30].T, cmap='vlag', center=0, 
               col_colors=metadata['Group'].map({'ALS': 'orange', 'CONTROL': 'grey'}).values)
plt.savefig("hub_heatmap.png")

# (PCA)
pca = PCA(n_components=2)
pcs = pca.fit_transform(z_scores_clean[all_module_genes])
pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
pca_df['Group'] = metadata['Group'].values
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Group', palette={'ALS': 'orange', 'CONTROL': 'grey'}, s=100)
plt.title("4. PCA of Pathway-Specific Signature")
plt.savefig("filtered_pca.png")

print("Files Generated: ALS_Patient_Progression_Labels.csv, Top_ALS_Hub_Proteins.csv")