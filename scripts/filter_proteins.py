import pandas as pd

# Paths
normalized_proteins_path = "results/normalized_proteins.csv"
ecm_genes_path = "gene_list/ECM_genes.csv"
ups_genes_path = "gene_list/UPS_genes.csv"

# Load normalized proteins
df = pd.read_csv(normalized_proteins_path)

# Load gene lists
with open(ecm_genes_path) as f:
    ecm_genes = set(line.strip() for line in f)

with open(ups_genes_path) as f:
    ups_genes = set(line.strip() for line in f)

# Check which genes from the normalized proteins file are in ECM or UPS
df['Gene_Type'] = df['Gene'].apply(
    lambda g: 'ECM' if g in ecm_genes else ('UPS' if g in ups_genes else None)
)

# Keep only rows that are in ECM or UPS
filtered_df = df[df['Gene_Type'].notna()]

# Save filtered normalized proteins
filtered_df.to_csv("results/filtered_normalized_proteins.csv", index=False)

# Save separate lists of genes found
filtered_df[filtered_df['Gene_Type'] == 'ECM']['Gene'].to_csv("results/ecm_genes_found.txt", index=False, header=False)
filtered_df[filtered_df['Gene_Type'] == 'UPS']['Gene'].to_csv("results/ups_genes_found.txt", index=False, header=False)

print("Filtering complete. Files saved:")
print("- filtered_normalized_proteins.csv")
print("- ecm_genes_found.txt")
print("- ups_genes_found.txt")
