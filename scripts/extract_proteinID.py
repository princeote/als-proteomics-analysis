import pandas as pd
from pathlib import Path

# Load your normalized proteins file
df = pd.read_csv("results/normalized_proteins.csv")

# Extract the Protein ID from the Protein column
df["Protein_ID"] = df["Protein"].astype(str).str.split("|").str[1]

# Unique IDs
protein_ids = df["Protein_ID"].dropna().unique()

# Output path (create directory if missing)
out_path = Path("results/protein_ids_for_panther.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Save
pd.Series(protein_ids).to_csv(out_path, index=False, header=False)

print(f"Saved {len(protein_ids)} Protein IDs to: {out_path}")
