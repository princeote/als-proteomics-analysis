import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ========================================================
# 1. SETUP & THEME
# ========================================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

# Create output folders
for folder in ['plots', 'plots/longitudinal_analysis', 'results']:
    os.makedirs(folder, exist_ok=True)

# Define a consistent color palette
# Controls are green, Fast is red (aggressive), Slow is blue (stagnant)
palette = {
    'Fast': '#D62728', 
    'Middle': '#7F7F7F', 
    'Slow': '#1F77B4', 
    'Control': '#2CA02C'
}

# ========================================================
# 2. DATA LOADING
# ========================================================
try:
    # Check current directory first, then results/
    if os.path.exists("normalized_proteins.csv"):
        df = pd.read_csv("normalized_proteins.csv")
        pheno_df = pd.read_csv("patient_phenotypes.csv")
    else:
        df = pd.read_csv("results/normalized_proteins.csv")
        pheno_df = pd.read_csv("results/patient_phenotypes.csv")
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Could not find data files. {e}")
    exit()

# ========================================================
# 3. PRE-PROCESSING & METADATA
# ========================================================
# Standardize Gene names
df['Gene'] = df['Gene'].astype(str).str.strip().str.upper()
intensity_cols = [c for c in df.columns if "MaxLFQ_Intensity" in c]
gene_pivot = df.set_index('Gene')[intensity_cols].T

# Extract Metadata
metadata = pd.DataFrame(index=gene_pivot.index)
metadata['Group'] = metadata.index.to_series().apply(
    lambda x: 'ALS' if 'ALS' in x.upper() else 'Control'
)

def extract_pid(name):
    parts = name.split('_')
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else name

def extract_tp(name):
    match = re.search(r'T(\d+)', name)
    return float(match.group(1)) if match else np.nan

metadata['Patient_ID'] = metadata.index.to_series().apply(extract_pid)
metadata['Timepoint'] = metadata.index.to_series().apply(extract_tp)

# Map Phenotypes (Anchored to CHI3L1 slopes)
prog_map = dict(zip(pheno_df['Patient_ID'].str.strip(), pheno_df['Phenotype'].str.strip()))
metadata['Phenotype'] = metadata['Patient_ID'].map(prog_map)
metadata.loc[metadata['Group'] == 'Control', 'Phenotype'] = 'Control'

# ========================================================
# 4. NORMALIZATION (Z-Scores vs Controls)
# ========================================================
control_indices = metadata[metadata['Group'] == 'Control'].index
control_mean = gene_pivot.loc[control_indices].mean()
control_std = gene_pivot.loc[control_indices].std().replace(0, 1) # Prevent div by zero

# Calculate Z-scores: (Value - ControlMean) / ControlSTD
z_scores = (gene_pivot - control_mean) / control_std

# ========================================================
# 5. CORE ANALYSIS FUNCTION
# ========================================================
def plot_group_trajectories(proteins, title, filename):
    """Generates a grid of longitudinal plots showing Group Means and SEM."""
    valid_prots = [p for p in proteins if p in z_scores.columns]
    if not valid_prots:
        print(f"Skipping {title}: No matching genes found in data.")
        return

    cols = 2
    rows = (len(valid_prots) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows), squeeze=False)
    axes = axes.flatten()
    
    hue_order = ['Control', 'Slow', 'Middle', 'Fast']
    
    for i, gene in enumerate(valid_prots):
        plot_data = metadata.copy()
        plot_data['Intensity_Z'] = z_scores[gene].values
        
        # Clean NaNs for plotting
        plot_data = plot_data.dropna(subset=['Phenotype', 'Timepoint'])
        
        sns.lineplot(
            data=plot_data, 
            x='Timepoint', 
            y='Intensity_Z', 
            hue='Phenotype',
            hue_order=hue_order,
            palette=palette,
            marker='o',
            markersize=10,
            linewidth=3,
            errorbar='se', # Standard Error shows the 'Signal' clearly
            ax=axes[i]
        )
        
        axes[i].set_title(f"{gene} Progression", fontweight='bold', fontsize=14)
        axes[i].set_ylabel("Z-Score (vs Control)")
        axes[i].set_xlabel("Timepoint")
        axes[i].set_xticks([1, 2, 3, 4])
        # Add a reference line at 0 (Control Mean)
        axes[i].axhline(0, color='black', linestyle='--', alpha=0.3)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = f"plots/longitudinal_analysis/{filename}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")

# ========================================================
# 6. EXECUTION
# ========================================================

# 1. Plot the Anchor (CHI3L1)
plot_group_trajectories(['CHI3L1'], "CHI3L1 Stratification Baseline", "chi3l1_baseline")

# 2. Plot the Neuroinflammation Axis
comp_prots = ['C4B', 'CFB', 'C8G', 'LYZ']
plot_group_trajectories(comp_prots, "Neuroinflammation Axis Trends", "inflammatory_axis")

# 3. Plot the Structural Axis
struct_prots = ['MMP2', 'PCOLCE']
plot_group_trajectories(struct_prots, "Structural Integrity Axis Trends", "structural_axis")

# 4. Plot the Metabolic Axis
metab_prots = ['TTR', 'PPIB']
plot_group_trajectories(metab_prots, "Metabolic Support Axis Trends", "metabolic_axis")

print("\nFull Analysis Complete. Check the 'plots/longitudinal_analysis' folder.")