#!/usr/bin/env python3

# first draft of script to combine ALS + Control protein tables
# NOTE: super rough, need cleanup later

import pandas as pd
import os
from pathlib import Path

ALS_FOLDER = Path("data/fragpipe/ALS")
CTRL_FOLDER = Path("data/fragpipe/control")

# TODO: handle cases where Protein ID column is missing...

def load_file(path):
    return pd.read_csv(path, sep="\t")

def combine_als(folder):
    dfs = []
    i = 1
    for f in sorted(folder.glob("*.tsv")):
        df = load_file(f)
        # BUG: this assumes exactly 4 MaxLFQ columns; fix later
        cols = [c for c in df.columns if "MaxLFQ Intensity" in c][:4]
        sub = df[["Protein", "Gene"] + cols]
        rename = {c: f"ALS_{i}_T{idx+1}_MaxLFQ_Intensity" for idx, c in enumerate(cols)}
        sub = sub.rename(columns=rename)
        dfs.append(sub)
        i += 1
    # BUG: wrong merge strategy, should be outer not inner
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["Protein", "Gene"])
    return merged

def combine_ctrl(folder):
    dfs = []
    i = 1
    for f in sorted(folder.glob("*.tsv")):
        df = load_file(f)
        cols = [c for c in df.columns if "MaxLFQ Intensity" in c]
        if len(cols) == 0:
            continue  # BUG: should warn about skipped files
        sub = df[["Protein", "Gene", cols[0]]]
        sub = sub.rename(columns={cols[0]: f"Control_{i}_MaxLFQ_Intensity"})
        dfs.append(sub)
        i += 1
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["Protein", "Gene"])
    return merged

def main():
    als = combine_als(ALS_FOLDER)
    ctrl = combine_ctrl(CTRL_FOLDER)

    combined = als.merge(ctrl, on=["Protein", "Gene"])  # BUG: should be outer merge
    combined.to_csv("results/combined_proteins_maxlfq.csv", index=False)

if __name__ == "__main__":
    main()
