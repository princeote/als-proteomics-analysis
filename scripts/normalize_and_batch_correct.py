#!/usr/bin/env python3

# cleaned version of combine script
# fixed ID selection + merge strategy + warnings

import pandas as pd
from pathlib import Path
import sys

ALS_FOLDER = Path("data/fragpipe/ALS")
CTRL_FOLDER = Path("data/fragpipe/control")

KEYS = ["Protein", "Gene"]

def load(path):
    return pd.read_csv(path, sep="\t", low_memory=False)

def process_als():
    dfs = []
    for i, f in enumerate(sorted(ALS_FOLDER.glob("*.tsv")), start=1):
        df = load(f)
        maxlfq = [c for c in df.columns if "MaxLFQ Intensity" in c]
        if len(maxlfq) < 1:
            print(f"Warning: no MaxLFQ in ALS file {f.name}")
            continue

        # take first 4 (FragPipe sometimes outputs more)
        chosen = maxlfq[:4]
        sub = df[KEYS + chosen].copy()
        rename = {c: f"ALS_{i}_T{j+1}_MaxLFQ_Intensity" for j, c in enumerate(chosen)}
        sub = sub.rename(columns=rename)
        dfs.append(sub)

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=KEYS, how="outer")
    return merged

def process_ctrl():
    dfs = []
    for i, f in enumerate(sorted(CTRL_FOLDER.glob("*.tsv")), start=1):
        df = load(f)
        maxlfq = [c for c in df.columns if "MaxLFQ Intensity" in c]
        if not maxlfq:
            print(f"Warning: no MaxLFQ in CTRL file {f.name}")
            continue
        sub = df[KEYS + [maxlfq[0]]].copy()
        sub = sub.rename(columns={maxlfq[0]: f"Control_{i}_MaxLFQ_Intensity"})
        dfs.append(sub)

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=KEYS, how="outer")
    return merged

def main():
    als = process_als()
    ctrl = process_ctrl()

    combined = als.merge(ctrl, on=KEYS, how="outer")
    combined.to_csv("results/combined_proteins_maxlfq.csv", index=False)
    print("Saved combined file.")

if __name__ == "__main__":
    main()
