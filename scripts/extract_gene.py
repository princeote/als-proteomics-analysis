import pandas as pd
from pathlib import Path

# Load normalized proteins file
df = pd.read_csv("results/normalized_proteins.csv")

# Check that gene column exists
if "Gene" not in df.columns:
    raise ValueError("No 'Gene' column found in normalized_proteins.csv")

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

# Output file
out_path = Path("results/genes_for_panther.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Save gene symbols (one per line)
pd.Series(genes).to_csv(out_path, index=False, header=False)

print(f"Saved {len(genes)} gene symbols to: {out_path}")
