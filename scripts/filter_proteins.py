import pandas as pd

# Paths
normalized_proteins_path = "results/normalized_proteins.csv"
ecm_genes_path = "gene_list/ECM_genes.csv"
alp_genes_path = "gene_list/ALP_genes.csv"
complement_genes_path = "gene_list/complement_genes.tsv"

# Load normalized proteins
df = pd.read_csv(normalized_proteins_path)

# Load gene lists
with open(ecm_genes_path) as f:
    ecm_genes = set(line.strip() for line in f)

with open(alp_genes_path) as f:
    alp_genes = set(line.strip() for line in f)

with open(complement_genes_path) as f:
    complement_genes = set(line.strip() for line in f)

# Filter rows for ECM and ALP
ecm_filtered = df[df['Gene'].isin(ecm_genes)]
alp_filtered = df[df['Gene'].isin(alp_genes)]
complement_filtered = df[~df['Gene'].isin(complement_genes)]

# Save filtered normalized proteins
filtered_df = pd.concat([ecm_filtered, alp_filtered,complement_filtered])
filtered_df.to_csv("results/filtered_normalized_proteins.csv", index=False)

# Save combined gene list with headers
with open("results/genes_found_combined.txt", "w") as f:
    f.write("ECM Genes:\n")
    for gene in ecm_filtered['Gene'].unique():
        f.write(f"{gene}\n")
    
    f.write("\nALP Genes:\n")
    for gene in alp_filtered['Gene'].unique():
        f.write(f"{gene}\n")
    
    f.write("\nComplement Genes:\n")
    for gene in complement_filtered['Gene'].unique():
        f.write(f"{gene}\n")

print("Filtering complete. Files saved:")
print("- filtered_normalized_proteins.csv")
print("- genes_found_combined.txt")
