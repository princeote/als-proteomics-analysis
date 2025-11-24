#!/usr/bin/env python3
"""
Normalize + (optional) batch-correct MaxLFQ protein intensities.

Input:
    /mnt/data/combined_proteins_maxlfq.csv

Output:
    results/normalized_proteins.csv
    results/batch_corrected_proteins.csv  (only if batch_map is filled)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import logging

# -------- Paths --------
# Input file containing protein intensities from MaxLFQ straight from combine_proteins.py
INPUT = Path("results/combined_proteins_maxlfq.csv")


# Output files
NORMALIZED_OUT = Path("results/normalized_proteins.csv")
BATCH_CORRECTED_OUT = Path("results/batch_corrected_proteins.csv")

# Map each sample to a batch if you want to perform batch correction
# Example: batch_map = {'sample1': 'batch1', 'sample2': 'batch2'}
batch_map = {}   # leave empty to skip batch correction


# -------- Helper Functions --------
def median_normalize(df):
    """
    Perform median normalization on a dataframe in log2 space.
    This centers each sample to have the same median protein intensity.
    """
    med = df.median(axis=0)       # median per sample
    gm = med.median()             # global median across all samples
    return df.subtract(med, axis=1).add(gm)


def combat_correct(df, batch_labels):
    """
    Apply ComBat batch correction using pyneurocombat.
    This removes technical variation from different batches.
    """
    try:
        from pyneurocombat import neuroCombat
    except ImportError:
        raise ImportError(
            "pyneurocombat is not installed.\n"
            "Install it with: pip install pyneurocombat"
        )
    
    # Apply ComBat
    combat_out = neuroCombat(
        data=df.values,
        batch=batch_labels.values
    )["data"]

    # Convert back to a dataframe with same indices and columns
    corrected = pd.DataFrame(
        combat_out,
        index=df.index,
        columns=df.columns
    )
    return corrected


# -------- Main Function --------
def main():
    # Set up logging to show info messages
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Check that input exists
    if not INPUT.exists():
        logging.error("Could not find input file at %s", INPUT)
        sys.exit(1)

    # Load protein data
    logging.info("Loading combined protein matrix...")
    df = pd.read_csv(INPUT)

    # Separate intensity columns (numeric data) from metadata
    intensity_cols = [c for c in df.columns if "Intensity" in c]
    metadata_cols = [c for c in df.columns if c not in intensity_cols]

    logging.info("Found %d samples", len(intensity_cols))

    meta = df[metadata_cols]           # non-numeric metadata
    X = df[intensity_cols].replace(0, np.nan)  # replace 0s with NaN for missing data

    # Log2 transform intensities for variance stabilization
    logging.info("Log2 transforming...")
    X = np.log2(X)

    # Filter proteins that are missing in more than 30% of samples
    logging.info("Filtering proteins missing >30%...")
    keep = X.isna().mean(axis=1) <= 0.30
    meta = meta.loc[keep].reset_index(drop=True)
    X = X.loc[keep].reset_index(drop=True)

    logging.info("Keeping %d proteins", X.shape[0])

    # Apply median normalization
    logging.info("Median normalizing...")
    X_norm = median_normalize(X)

    # Save normalized data
    NORMALIZED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, X_norm], axis=1).to_csv(NORMALIZED_OUT, index=False)
    logging.info("Saved normalized matrix to %s", NORMALIZED_OUT)

    # Apply batch correction if batch_map is provided
    if batch_map:
        logging.info("Applying ComBat...")

        batch_labels = pd.Series(batch_map)
        missing = [s for s in X_norm.columns if s not in batch_labels.index]

        if missing:
            logging.error("Batch labels missing for samples: %s", missing)
            sys.exit(1)

        labels = batch_labels.loc[X_norm.columns]

        X_corrected = combat_correct(X_norm, labels)

        # Save batch-corrected data
        pd.concat([meta, X_corrected], axis=1).to_csv(BATCH_CORRECTED_OUT, index=False)
        logging.info("Saved batch-corrected matrix to %s", BATCH_CORRECTED_OUT)
    else:
        logging.info("No batch_map provided — skipping ComBat.")

    logging.info("Done.")


# Run the main function when script is executed
if __name__ == "__main__":
    main()
