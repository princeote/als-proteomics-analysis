#!/usr/bin/env python3
"""
Normalize MaxLFQ intensities (early version)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
import sys

INPUT = Path("results/combined_proteins_maxlfq.csv")
NORMALIZED_OUT = Path("results/normalized_proteins.csv")

def median_normalize(df):
    med = df.median(axis=0)
    gm = med.median()
    return df.subtract(med, axis=1).add(gm)

def main():
    logging.basicConfig(level=logging.INFO)

    if not INPUT.exists():
        logging.error("Missing combined file.")
        sys.exit(1)

    df = pd.read_csv(INPUT)
    meta = df[["Protein", "Gene"]].copy()
    X = df.drop(columns=["Protein", "Gene"]).replace(0, np.nan)

    # log2 transform
    X = np.log2(X)

    # FIXED: filtering bug + correct alignment
    keep = X.isna().mean(axis=1) <= 0.30
    X = X.loc[keep].reset_index(drop=True)
    meta = meta.loc[keep].reset_index(drop=True)

    # normalize
    X_norm = median_normalize(X)

    NORMALIZED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, X_norm], axis=1).to_csv(NORMALIZED_OUT, index=False)
    logging.info("Saved normalized matrix.")

if __name__ == "__main__":
    main()
