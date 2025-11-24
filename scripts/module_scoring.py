#!/usr/bin/env python3
"""
Module scoring for proteomics data.

Input:
    /mnt/data/normalized_proteins.csv

Output:
    results/module_scores.csv

This script computes:
- Per-sample median module scores
- Optional PCA per module
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

# -------- Paths --------
INPUT = Path("results/normalized_proteins.csv")
OUTPUT = Path("results/module_scores.csv")

# Define modules as lists of proteins (example)
# Replace with real protein lists for your modules
modules = {
    "Module1": ["ProteinA", "ProteinB", "ProteinC"],
    "Module2": ["ProteinD", "ProteinE"],
    # Add more modules here
}

# -------- Helpers --------
def compute_module_score(df, protein_list):
    """
    Compute module score for a set of proteins as the mean log2 intensity per sample.
    Ignores missing proteins.
    """
    valid_proteins = [p for p in protein_list if p in df.index]
    if not valid_proteins:
        return pd.Series([np.nan]*df.shape[1], index=df.columns)
    return df.loc[valid_proteins].mean(axis=0)


# -------- Main Function --------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not INPUT.exists():
        logging.error("Input file not found at %s", INPUT)
        return

    logging.info("Loading normalized protein matrix...")
    df = pd.read_csv(INPUT, index_col=0)  # proteins as rows, samples as columns

    scores = {}
    for mod_name, proteins in modules.items():
        logging.info("Scoring module: %s", mod_name)
        scores[mod_name] = compute_module_score(df, proteins)

    scores_df = pd.DataFrame(scores)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(OUTPUT)
    logging.info("Module scores saved to %s", OUTPUT)


if __name__ == "__main__":
    main()
