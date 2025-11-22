import pandas as pd
from pathlib import Path
from functools import reduce

# Paths
als_folder = Path("data/fragpipe/ALS")
ctrl_folder = Path("data/fragpipe/control")
output_file = Path("results/combined_proteins.csv")

# Function to load TSV files
def load_tsv_files(folder):
    dfs = []
    for file in folder.glob("*.tsv"):
        df = pd.read_csv(file, sep="\t")
        # Keep only the main identifiers + intensity/MaxLFQ columns
        cols_to_keep = ["Protein", "Gene"] + [c for c in df.columns if "Intensity" in c or "MaxLFQ" in c]
        df = df[cols_to_keep]
        # Optional: rename intensity columns to include sample name to avoid duplicates
        sample_name = file.stem
        df = df.rename(columns={c: f"{sample_name}_{c}" for c in df.columns if c not in ["Protein", "Gene"]})
        dfs.append(df)
    return dfs

# Load ALS and control
als_dfs = load_tsv_files(als_folder)
ctrl_dfs = load_tsv_files(ctrl_folder)

# Merge all ALS files
if als_dfs:
    als_merged = reduce(lambda left, right: pd.merge(left, right, on=["Protein", "Gene"], how="outer"), als_dfs)
else:
    raise ValueError("No ALS files found in folder.")

# Merge all control files
if ctrl_dfs:
    ctrl_merged = reduce(lambda left, right: pd.merge(left, right, on=["Protein", "Gene"], how="outer"), ctrl_dfs)
else:
    raise ValueError("No control files found in folder.")

# Merge ALS + Control
combined = pd.merge(als_merged, ctrl_merged, on=["Protein", "Gene"], how="outer")

# Save to CSV
output_file.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(output_file, index=False)
print(f"Combined protein table saved to {output_file}")
