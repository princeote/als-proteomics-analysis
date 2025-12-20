import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================
# 1. LOAD & BUILD GENE-LEVEL EXPRESSION MATRIX
# ========================================================

raw_expression_df = pd.read_csv("results/filtered_normalized_proteins.csv")

# Clean gene symbols
raw_expression_df['Gene'] = (
    raw_expression_df['Gene']
    .astype(str)
    .str.strip()
    .str.upper()
)

# Aggregate proteins → genes (robust)
gene_expression_df = (
    raw_expression_df
    .drop(columns=['Protein'])
    .groupby('Gene', as_index=True)
    .median()
)

# Transpose: rows = samples, columns = genes
expression_df = gene_expression_df.T

# Clean sample IDs
expression_df.index = expression_df.index.astype(str).str.strip()
expression_df.columns = expression_df.columns.astype(str).str.strip()

print("Total genes after QC + aggregation:", expression_df.shape[1])

# ========================================================
# 2. BUILD METADATA FROM SAMPLE NAMES
# ========================================================

metadata_df = pd.DataFrame({'Sample_ID': expression_df.index})

metadata_df['Group'] = metadata_df['Sample_ID'].str.extract(r'^([A-Z]+)')
metadata_df['Patient_ID'] = metadata_df['Sample_ID'].str.extract(r'^([A-Z]+_\d+)')
metadata_df['Timepoint'] = metadata_df['Sample_ID'].str.extract(r'_(T\d+)_')

metadata_df = metadata_df.dropna().copy()

# Order timepoints numerically
time_order = sorted(
    metadata_df['Timepoint'].unique(),
    key=lambda x: int(x[1:])
)

metadata_df['Timepoint'] = pd.Categorical(
    metadata_df['Timepoint'],
    categories=time_order,
    ordered=True
)

# ========================================================
# 3. LOAD & FILTER PATHWAY GENE LISTS
# ========================================================

def load_gene_list(path):
    return (
        pd.read_csv(path)
        .iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

ecm_genes = load_gene_list("gene_list/ECM_genes.csv")
complement_genes = load_gene_list("gene_list/complement_genes.csv")

data_genes = set(expression_df.columns)

ecm_genes = [g for g in ecm_genes if g in data_genes]
complement_genes = [g for g in complement_genes if g in data_genes]

print(f"ECM genes retained: {len(ecm_genes)}")
print(f"Complement genes retained: {len(complement_genes)}")

if len(ecm_genes) == 0 or len(complement_genes) == 0:
    raise ValueError("No pathway genes matched expression data")

# ========================================================
# 4. CONTROL-ANCHORED Z-SCORING
# ========================================================

control_ids = metadata_df.loc[
    metadata_df['Group'] == 'CONTROL',
    'Sample_ID'
]

control_expression = expression_df.loc[control_ids]

control_mean = control_expression.mean()
control_std = control_expression.std().replace(0, np.nan)

z_expression = (expression_df - control_mean) / control_std

# ========================================================
# 5. PATHWAY SCORES (GENE-LEVEL MEDIAN)
# ========================================================

pathway_scores = pd.DataFrame({
    'Sample_ID': z_expression.index,
    'ECM_Score': z_expression[ecm_genes].median(axis=1),
    'Complement_Score': z_expression[complement_genes].median(axis=1)
})

metadata_df = metadata_df.merge(pathway_scores, on='Sample_ID')
metadata_df = metadata_df.sort_values(['Patient_ID', 'Timepoint'])

print(metadata_df[['ECM_Score', 'Complement_Score']].describe())

# ========================================================
# 6. PLOTTING
# ========================================================

sns.set_theme(style="whitegrid")

# -------------------------------
# FIGURE 1: Longitudinal ALS
# -------------------------------

als_df = metadata_df[metadata_df['Group'] == 'ALS']

def plot_longitudinal(df, score_col, title):
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=df,
        x='Timepoint',
        y=score_col,
        hue='Patient_ID',
        legend=False,
        alpha=0.35,
        linewidth=1
    )

    mean_df = df.groupby('Timepoint')[score_col].mean().reset_index()

    sns.lineplot(
        data=mean_df,
        x='Timepoint',
        y=score_col,
        color='black',
        linewidth=3,
        marker='o'
    )

    plt.axhline(0, linestyle='--', color='gray')
    plt.title(title)
    plt.ylabel("Pathway Score (Control-normalized)")
    plt.xlabel("Timepoint")
    plt.tight_layout()
    plt.show()

plot_longitudinal(
    als_df,
    'ECM_Score',
    "ECM Pathway Trajectories in ALS Patients"
)

plot_longitudinal(
    als_df,
    'Complement_Score',
    "Complement Pathway Trajectories in ALS Patients"
)

# -------------------------------
# FIGURE 2: Controls vs ALS
# -------------------------------

plot_df = metadata_df.copy()
plot_df['Group_Time'] = plot_df['Group']

plot_df.loc[plot_df['Group'] == 'ALS', 'Group_Time'] = (
    'ALS_' + plot_df.loc[plot_df['Group'] == 'ALS', 'Timepoint'].astype(str)
)

def plot_group_comparison(df, score_col, title):
    plt.figure(figsize=(10, 6))

    sns.violinplot(
        data=df,
        x='Group_Time',
        y=score_col,
        inner='box',
        cut=0
    )

    plt.axhline(0, linestyle='--', color='gray')
    plt.title(title)
    plt.ylabel("Pathway Score (Control-normalized)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

plot_group_comparison(
    plot_df,
    'ECM_Score',
    "ECM Pathway Scores: Controls vs ALS Over Time"
)

plot_group_comparison(
    plot_df,
    'Complement_Score',
    "Complement Pathway Scores: Controls vs ALS Over Time"
)

# ========================================================
# END
# ========================================================
