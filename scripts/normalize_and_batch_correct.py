#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

INPUT = Path("results/combined_proteins_maxlfq.csv")
NORMALIZED_OUT = Path("results/normalized_proteins.csv")

def median_normalize(df):
    med = df.median()
    gm = med.median()
    return df.subtract(med, axis=1).add(gm)

def main():
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(INPUT)
    meta = df[["Protein", "Gene"]]
    X = df.drop(columns=meta.columns)

    X = np.log2(X.replace(0, np.nan))

    keep = X.isna().mean(axis=1) <= 0.30
    X = X[keep].reset_index(drop=True)
    meta = meta[keep].reset_index(drop=True)

    X_norm = median_normalize(X)

    NORMALIZED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, X_norm], axis=1).to_csv(NORMALIZED_OUT, index=False)

if __name__ == "__main__":
    main()
