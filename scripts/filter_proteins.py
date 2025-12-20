import pandas as pd

# Paths
normalized_proteins_path = "results/normalized_proteins.csv"
ecm_genes_path = "gene_list/ECM_genes.csv"
complement_genes_path = "gene_list/complement_genes.csv"


def load_gene_set(path):
    """Load a gene list robustly from a CSV or plain text file.
    Handles single-column CSVs (with or without header) and two-column CSVs (like From,Gene).
    Returns a set of stripped gene names (strings).
    """
    try:
        dfg = pd.read_csv(path)
        # Prefer a column named 'Gene' if present
        if 'Gene' in dfg.columns:
            series = dfg['Gene']
        elif dfg.shape[1] == 1:
            series = dfg.iloc[:, 0]
        else:
            # If multiple columns (e.g., From,Gene), pick the last column which should be the gene
            series = dfg.iloc[:, -1]
        series = series.dropna().astype(str).str.strip()
        # Remove common header-like values if present
        series = series[~series.str.lower().isin(['gene', 'from'])]
        return set(series.unique())
    except Exception:
        # Fallback to line-by-line parsing
        genes = []
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                # If comma-separated lines (e.g., From,Gene), take last field
                parts = [p.strip() for p in s.split(',') if p.strip()]
                if len(parts) == 1:
                    genes.append(parts[0])
                else:
                    genes.append(parts[-1])
        return set(genes)


# Load normalized proteins
df = pd.read_csv(normalized_proteins_path)
if 'Gene' not in df.columns:
    raise KeyError("Input file must contain a 'Gene' column")

# Load gene lists
ecm_genes = load_gene_set(ecm_genes_path)
complement_genes = load_gene_set(complement_genes_path)

# Union of allowed genes
allowed_genes = ecm_genes.union(complement_genes)

# Filter rows: only keep rows whose Gene is in either list
filtered_df = df[df['Gene'].isin(allowed_genes)].copy()

# Validation: ensure filtered data does not contain genes outside the allowed sets
found_genes = set(filtered_df['Gene'].unique())
outside_genes = found_genes - allowed_genes
if outside_genes:
    raise ValueError(f"Filtered data contains genes not in ECM or Complement lists: {sorted(outside_genes)}")

# Find which of each type are present
found_ecm = sorted(found_genes.intersection(ecm_genes))
found_complement = sorted(found_genes.intersection(complement_genes))

# Save filtered normalized proteins
filtered_df.to_csv("results/filtered_normalized_proteins.csv", index=False)

# Save combined gene list with headers, explicitly listing ECM and Complement genes found
with open("results/genes_found_combined.txt", "w") as f:
    f.write("ECM Genes:\n")
    if found_ecm:
        for gene in found_ecm:
            f.write(f"{gene}\n")
    else:
        f.write("(none)\n")

    f.write("\nComplement Genes:\n")
    if found_complement:
        for gene in found_complement:
            f.write(f"{gene}\n")
    else:
        f.write("(none)\n")

# Summary
print("Filtering complete. Files saved:")
print("- results/filtered_normalized_proteins.csv")
print("- results/genes_found_combined.txt")
print(f"Total input rows: {len(df)}")
print(f"Rows after filtering: {len(filtered_df)}")
print(f"Unique ECM genes found: {len(found_ecm)}")
print(f"Unique Complement genes found: {len(found_complement)}")
if not found_ecm or not found_complement:
    print("WARNING: one of the gene types has no matches in the input data.")
