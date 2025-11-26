import pandas as pd
from pathlib import Path

# Load normalized proteins file
df = pd.read_csv("results/filtered_normalized_proteins.csv")

# Check that gene column exists
if "Gene" not in df.columns:
    raise ValueError("No 'Gene' column found in filtered_normalized_proteins.csv")

# Split multi-gene entries like 'COL1A1;COL3A1'
genes = (
    df["Gene"]
    .astype(str)
    .str.split(";")
    .explode()
    .str.strip()
    .dropna()
    .unique()
)

# Output path
out_path = Path("results/genes_for_panther.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Save space-separated list
with open(out_path, "w") as f:
    f.write(" ".join(genes))

print(f"Saved {len(genes)} space-separated gene symbols to: {out_path}")
