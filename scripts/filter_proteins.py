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

# Filter rows for ECM and UPS
ecm_filtered = df[df['Gene'].isin(ecm_genes)]
ups_filtered = df[df['Gene'].isin(ups_genes)]

# Save filtered normalized proteins
filtered_df = pd.concat([ecm_filtered, ups_filtered])
filtered_df.to_csv("results/filtered_normalized_proteins.csv", index=False)

# Save combined gene list with headers
with open("results/genes_found_combined.txt", "w") as f:
    f.write("ECM Genes:\n")
    for gene in ecm_filtered['Gene'].unique():
        f.write(f"{gene}\n")
    
    f.write("\nUPS Genes:\n")
    for gene in ups_filtered['Gene'].unique():
        f.write(f"{gene}\n")

print("Filtering complete. Files saved:")
print("- filtered_normalized_proteins.csv")
print("- genes_found_combined.txt")
