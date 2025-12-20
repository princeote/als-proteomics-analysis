import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ========================================================
# 1. LOAD & AGGREGATE TO GENE LEVEL
# ========================================================
df = pd.read_csv("results/filtered_normalized_proteins.csv")

# Ensure 'Gene' column is present and clean to prevent missing proteins
if 'Gene' not in df.columns:
    df['Gene'] = df['Protein'].astype(str).str.strip().str.upper()
else:
    df['Gene'] = df['Gene'].astype(str).str.strip().str.upper()

# Aggregate: Combine all protein rows for a single gene using the median 
gene_df = df.drop(columns=['Protein'], errors='ignore').groupby('Gene').median().T
gene_df.index = gene_df.index.astype(str).str.strip()

# ========================================================
# 2. EXTRACT METADATA (ALS vs CONTROL & TIMEPOINTS)
# ========================================================
metadata = pd.DataFrame({'Sample_ID': gene_df.index})
metadata['Group'] = metadata['Sample_ID'].apply(
    lambda x: 'ALS' if 'ALS' in x.upper() else 'CONTROL' if 'CONTROL' in x.upper() else 'UNKNOWN'
)
metadata['Patient_ID'] = metadata['Sample_ID'].str.extract(r'^([A-Za-z]+_\d+)')
metadata['Timepoint'] = metadata['Sample_ID'].str.extract(r'_(T\d+)_')

# ========================================================
# 3. LOAD COMPLETE GENE LISTS (Fixing Header/Missing Gene Issue)
# ========================================================
def load_full_gene_list(path, available_genes):
    try:
        # header=None ensures the first gene (e.g., ITGAL or C4A) isn't skipped 
        gl = pd.read_csv(path, header=None)
        list_genes = gl.iloc[:, 0].astype(str).str.strip().str.upper().unique().tolist()
        
        # Cross-reference with your actual expression data
        mapped = [g for g in list_genes if g in available_genes]
        print(f"File {path}: Found {len(mapped)} out of {len(list_genes)} genes in data.")
        return mapped
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

data_genes = set(gene_df.columns)
ecm_genes = load_full_gene_list("gene_list/ECM_genes.csv", data_genes)
comp_genes = load_full_gene_list("gene_list/complement_genes.csv", data_genes)

# ========================================================
# 4. CONTROL-ANCHORED Z-SCORING
# ========================================================
# Establish the 'Healthy Baseline' using only Control samples
controls = metadata[metadata['Group'] == 'CONTROL']['Sample_ID']
control_mean = gene_df.loc[controls].mean()
control_std = gene_df.loc[controls].std().replace(0, np.nan)

# Calculate Z-scores: (Sample Value - Control Average) / Control Variation
z_scores = (gene_df - control_mean) / control_std

# ========================================================
# 5. MODULE SCORING
# ========================================================
# Use the median of all Z-scored genes to define "Activity" 
metadata['ECM_Score'] = z_scores[ecm_genes].median(axis=1).values
metadata['Complement_Score'] = z_scores[comp_genes].median(axis=1).values

# Sort by Timepoint for plotting
tp_order = sorted(metadata['Timepoint'].dropna().unique(), key=lambda x: int(re.search(r'\d+', x).group()))
metadata['Timepoint'] = pd.Categorical(metadata['Timepoint'], categories=tp_order, ordered=True)

print("Scoring Complete. Activity scores are now relative to the healthy control mean (0.0).")