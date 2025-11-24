#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

INPUT = Path("results/combined_proteins_maxlfq.csv")
NORMALIZED = Path("results/normalized_proteins.csv")
BATCH_CORR = Path("results/batch_corrected_proteins.csv")

batch_map = {}  # optional

def median_norm(df):
    med = df.median()
    gm = med.median()
    return df.subtract(med, axis=1).add(gm)

def combat(df, batch_labels):
    try:
        from pyneurocombat import neuroCombat
    except:
        raise ImportError("Install with: pip install pyneurocombat")

    result = neuroCombat(data=df.values, batch=batch_labels.values)["data"]
    return pd.DataFrame(result, index=df.index, columns=df.columns)

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not INPUT.exists():
        logging.error("Missing combined file.")
        sys.exit(1)

    df = pd.read_csv(INPUT)
    meta = df[["Protein", "Gene"]]
    X = df.drop(columns=["Protein", "Gene"]).replace(0, np.nan)

    # log2
    X = np.log2(X)

    # filter missing
    keep = X.isna().mean(axis=1) <= 0.30
    X = X.loc[keep]
    meta = meta.loc[keep].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # normalize
    X_norm = median_norm(X)
    pd.concat([meta, X_norm], axis=1).to_csv(NORMALIZED, index=False)
    logging.info("Saved normalized matrix.")

    # batch correct
    if batch_map:
        batch_series = pd.Series(batch_map)
        labels = batch_series[X_norm.columns]
        X_bc = combat(X_norm, labels)
        pd.concat([meta, X_bc], axis=1).to_csv(BATCH_CORR, index=False)
        logging.info("Saved batch-corrected matrix.")

if __name__ == "__main__":
    main()
