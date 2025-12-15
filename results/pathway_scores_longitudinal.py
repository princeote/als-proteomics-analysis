import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re # We need this for parsing sample names

# --------------------------------------------------------
# --- Step 1: Load and Prepare your data (FIXED) ---
# --------------------------------------------------------
# NOTE: File paths corrected to match the uploaded file names.
# NOTE: The protein data is transposed and processed to match the script's requirements.

# Load expression data: Rows are proteins/genes, columns are samples.
raw_expression_df = pd.read_csv("results/normalized_proteins.csv", index_col=None)

# 1a. Set the 'Gene' column as index and drop the original 'Protein' name.
# We use .drop(columns=['Protein']) to keep only gene names and sample values.
# Then, we transpose the DataFrame (T) so R: Samples, C: Genes/Proteins
expression_df = raw_expression_df.set_index('Gene').drop(columns=['Protein']).T

# 1b. Create the Time-Course Metadata from expression_df column headers
sample_names = expression_df.index.to_series().astype(str)

# Regular expression to extract Group (e.g., ALS), Patient_ID (e.g., ALS_1), and Timepoint (e.g., T1)
# from the sample names (e.g., 'ALS_1_T1_MaxLFQ_Intensity')
metadata_df = pd.DataFrame()
metadata_df['Sample_ID'] = sample_names
# Group: Capture 'ALS' or 'Control'
metadata_df['Group'] = sample_names.str.extract(r'^([A-Z]+)', expand=False)
# Patient_ID: Capture 'ALS_1', 'Control_1', etc.
metadata_df['Patient_ID'] = sample_names.str.extract(r'^([A-Z]+_\d+)', expand=False)
# Timepoint: Capture 'T1', 'T2', etc.
metadata_df['Timepoint'] = sample_names.str.extract(r'\_(T\d+)\_', expand=False)

# Drop any rows where parsing failed (shouldn't happen with this data)
metadata_df = metadata_df.dropna(subset=['Group', 'Patient_ID', 'Timepoint'])

# Ensure Timepoint is sorted correctly (e.g., T1, T2, T3, T4)
time_order = sorted(metadata_df['Timepoint'].unique(), key=lambda x: int(x[1:]))
metadata_df['Timepoint'] = pd.Categorical(metadata_df['Timepoint'], categories=time_order, ordered=True)

# --------------------------------------------------------
# --- Step 2: Define pathway gene lists (FIXED) ---
# --------------------------------------------------------

# Fix: Load the gene list, ensuring only the values from the first column are extracted.

# ECM Genes (CSV)
ecm_genes_df = pd.read_csv("gene_list/ECM_genes.csv", header=0)
ecm_genes_list = ecm_genes_df.iloc[:, 0].astype(str).tolist()

# Complement Genes (TSV, likely containing accession IDs)
complement_genes_df = pd.read_csv("gene_list/complement_genes.tsv", sep='\t', header=None)
complement_genes_list = complement_genes_df.iloc[:, 0].astype(str).tolist()


# Keep only genes present in your dataset (columns of the transposed expression_df)
data_genes = expression_df.columns.astype(str).tolist()
ecm_genes = [g for g in ecm_genes_list if g in data_genes]
complement_genes = [g for g in complement_genes_list if g in data_genes]

if not ecm_genes or not complement_genes:
    print("WARNING: One or both pathway gene lists are empty after filtering against your data.")
    print(f"ECM genes found: {len(ecm_genes)}. Complement genes found: {len(complement_genes)}")
    print("This is likely because the complement gene list contains accession IDs (e.g., P0C0L4) that do not match the gene symbols in your 'normalized_proteins.csv' file.")

# --------------------------------------------------------
# --- Step 3: Calculate pathway scores (SAFER) ---
# --------------------------------------------------------
# Optional: z-score standardization across all samples
z_expression = expression_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# Calculate mean z-score per pathway per sample.
# The index (Sample_ID) is used to safely merge the scores back into metadata_df.
pathway_scores = pd.DataFrame({
    'Sample_ID': z_expression.index,
    'ECM_Score': z_expression[ecm_genes].mean(axis=1).values,
    'Complement_Score': z_expression[complement_genes].mean(axis=1).values
})

# Merge the scores into the metadata table using the common 'Sample_ID' column
metadata_df = pd.merge(metadata_df, pathway_scores, on='Sample_ID', how='left')


# --------------------------------------------------------
# --- Step 4: Organize by Patient and Timepoint ---
# --------------------------------------------------------
metadata_df = metadata_df.sort_values(by=['Patient_ID', 'Timepoint'])


# --------------------------------------------------------
# --- Step 5: Visualization ---
# --------------------------------------------------------
sns.set_theme(style="whitegrid")

# Line plots per patient (ECM)
plt.figure(figsize=(12, 6))
# Plot individual patient lines in the background, without a legend
sns.lineplot(data=metadata_df, x='Timepoint', y='ECM_Score', hue='Patient_ID', 
             style='Group', markers=True, dashes=False, legend=False, linewidth=1, alpha=0.4)
# Calculate and overlay the Group Mean line (thicker, different marker)
group_mean_ecm = metadata_df.groupby(['Group', 'Timepoint'])['ECM_Score'].mean().reset_index()
sns.lineplot(data=group_mean_ecm, x='Timepoint', y='ECM_Score', hue='Group', 
             marker="o", linewidth=3, markersize=8)
plt.title("ECM Pathway Scores Over Time (Individual Patients and Group Mean)")
plt.ylabel("ECM Pathway Score (Mean Z-score)")
plt.xlabel("Timepoint")
plt.legend(title='Group Mean')
plt.show()

# Line plots per patient (Complement)
plt.figure(figsize=(12, 6))
# Plot individual patient lines in the background, without a legend
sns.lineplot(data=metadata_df, x='Timepoint', y='Complement_Score', hue='Patient_ID', 
             style='Group', markers=True, dashes=False, legend=False, linewidth=1, alpha=0.4)
# Calculate and overlay the Group Mean line (thicker, different marker)
group_mean_comp = metadata_df.groupby(['Group', 'Timepoint'])['Complement_Score'].mean().reset_index()
sns.lineplot(data=group_mean_comp, x='Timepoint', y='Complement_Score', hue='Group', 
             marker="o", linewidth=3, markersize=8)
plt.title("Complement Pathway Scores Over Time (Individual Patients and Group Mean)")
plt.ylabel("Complement Pathway Score (Mean Z-score)")
plt.xlabel("Timepoint")
plt.legend(title='Group Mean')
plt.show()

# Optional: Plotting only the group mean (as you had originally)
plt.figure(figsize=(10, 5))
sns.lineplot(data=group_mean_ecm, x='Timepoint', y='ECM_Score', hue='Group', marker="o")
plt.title("ECM Pathway Mean Scores Over Time (ALS vs Control)")
plt.ylabel("ECM Pathway Score (Mean Z-score)")
plt.show()