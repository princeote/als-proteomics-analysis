#!/usr/bin/env python3
"""
Combine FragPipe combined_protein.tsv MaxLFQ intensities for ALS and Control.

Behavior:
- ALS: each ALS file should provide 4 MaxLFQ Intensity columns (T1..T4). If >4 exist, the first 4 are used.
- Control: supports either 1) a single combined control file with many MaxLFQ columns OR
           2) multiple control files (each may have 1 MaxLFQ column). All control MaxLFQ columns
           are renamed Control_1..Control_N.
- Merges on Protein+Gene (falls back to Protein ID if Gene missing).

Output: results/combined_proteins_maxlfq.csv
"""

from pathlib import Path
import pandas as pd
import sys

# -------- Settings (edit if needed) --------
ALS_FOLDER = Path("data/fragpipe/ALS")
CONTROL_FOLDER = Path("data/fragpipe/control")
# allow an uploaded control combined file if present
UPLOAD_CONTROL_FILE = Path("/mnt/data/control_combined_protein.tsv")
OUTPUT = Path("results/combined_proteins_maxlfq.csv")

KEY_PRIMARY = "Protein"
KEY_SECOND = "Gene"
FALLBACK_ID = "Protein ID"

# -------- Helpers --------
def find_key_columns(df):
    if KEY_PRIMARY in df.columns and KEY_SECOND in df.columns:
        return [KEY_PRIMARY, KEY_SECOND]
    if KEY_PRIMARY in df.columns:
        return [KEY_PRIMARY]
    if FALLBACK_ID in df.columns and KEY_SECOND in df.columns:
        return [FALLBACK_ID, KEY_SECOND]
    if FALLBACK_ID in df.columns:
        return [FALLBACK_ID]
    raise RuntimeError("No suitable ID columns found (need Protein or Protein ID).")

def load_tsv(path: Path):
    return pd.read_csv(path, sep="\t", low_memory=False)

def process_als_file(df, sample_number, key_cols):
    # find MaxLFQ columns
    maxlfq_cols = [c for c in df.columns if "MaxLFQ Intensity" in c]
    if not maxlfq_cols:
        raise ValueError(f"ALS file {sample_number} has no 'MaxLFQ Intensity' columns.")
    # use first 4 only (T1..T4)
    if len(maxlfq_cols) < 4:
        print(f"Warning: ALS file {sample_number} has only {len(maxlfq_cols)} MaxLFQ columns (expected 4). Will use available.")
    used = maxlfq_cols[:4]
    cols_to_keep = key_cols + used
    out = df[cols_to_keep].copy()
    rename_map = {old: f"ALS_{sample_number}_T{i+1}_MaxLFQ_Intensity" for i, old in enumerate(used)}
    out = out.rename(columns=rename_map)
    return out, rename_map

def process_control_combined_file(df, key_cols):
    # when control combined file contains many MaxLFQ columns (one file)
    maxlfq_cols = [c for c in df.columns if "MaxLFQ Intensity" in c]
    if not maxlfq_cols:
        # fallback to any "Intensity" columns if MaxLFQ absent (shouldn't happen after your re-run)
        maxlfq_cols = [c for c in df.columns if "Intensity" in c and "MaxLFQ" not in c]
        print("Warning: control combined file had no 'MaxLFQ Intensity' columns; falling back to generic 'Intensity' columns.")
    cols_to_keep = key_cols + maxlfq_cols
    out = df[cols_to_keep].copy()
    rename_map = {old: f"Control_{i+1}_MaxLFQ_Intensity" for i, old in enumerate(maxlfq_cols)}
    out = out.rename(columns=rename_map)
    return out, rename_map

def process_control_file_single(df, sample_number, key_cols):
    maxlfq_cols = [c for c in df.columns if "MaxLFQ Intensity" in c]
    if not maxlfq_cols:
        maxlfq_cols = [c for c in df.columns if "Intensity" in c]  # fallback
        if not maxlfq_cols:
            raise ValueError(f"Control file {sample_number} has no intensity columns.")
    # pick the first found (control per-file => one column)
    chosen = maxlfq_cols[0]
    out = df[key_cols + [chosen]].copy()
    rename_map = {chosen: f"Control_{sample_number}_MaxLFQ_Intensity"}
    out = out.rename(columns=rename_map)
    return out, rename_map

def outer_merge_list(dfs, on_keys):
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=on_keys, how="outer")
    return merged

# -------- Main pipeline --------
def main():
    # collect ALS files (sorted)
    als_paths = sorted(ALS_FOLDER.glob("*.tsv"))
    if not als_paths:
        print(f"No ALS files found in {ALS_FOLDER}. Exiting.", file=sys.stderr)
        return

    # load first ALS to detect key columns
    first_als_df = load_tsv(als_paths[0])
    key_cols = find_key_columns(first_als_df)
    print("Using key columns:", key_cols)

    # process ALS files
    als_processed = []
    for i, p in enumerate(als_paths, start=1):
        df = load_tsv(p)
        processed_df, rename_map = process_als_file(df, i, key_cols)
        print(f"ALS file {p.name}: found columns -> {list(rename_map.values())}")
        als_processed.append(processed_df)

    # merge ALS
    als_merged = outer_merge_list(als_processed, key_cols)
    print("ALS merged shape:", als_merged.shape)

    # process control(s)
    # prefer explicit uploaded combined file if present (e.g., /mnt/data/control_combined_protein.tsv)
    if UPLOAD_CONTROL_FILE.exists():
        ctrl_df = load_tsv(UPLOAD_CONTROL_FILE)
        control_cleaned, ctrl_map = process_control_combined_file(ctrl_df, key_cols)
        print("Using uploaded combined control file:", UPLOAD_CONTROL_FILE.name)
        print("Control columns:", list(ctrl_map.values()))
        # merged control is single df
        control_merged = control_cleaned
    else:
        # look in CONTROL_FOLDER
        control_paths = sorted(CONTROL_FOLDER.glob("*.tsv"))
        if not control_paths:
            raise FileNotFoundError(f"No control files found in {CONTROL_FOLDER} and no uploaded control file present.")
        # if there's exactly one file in folder, treat as combined
        if len(control_paths) == 1:
            ctrl_df = load_tsv(control_paths[0])
            control_merged, ctrl_map = process_control_combined_file(ctrl_df, key_cols)
            print("Using combined control file in folder:", control_paths[0].name)
            print("Control columns:", list(ctrl_map.values()))
        else:
            # multiple control files -> process each as single-sample file
            per_ctrl = []
            for i, p in enumerate(control_paths, start=1):
                df = load_tsv(p)
                cleaned, map_ = process_control_file_single(df, i, key_cols)
                per_ctrl.append(cleaned)
                print(f"Control file {p.name}: -> {list(map_.values())}")
            control_merged = outer_merge_list(per_ctrl, key_cols)

    print("Control merged shape:", control_merged.shape)

    # final merge ALS + control on key_cols
    combined = pd.merge(als_merged, control_merged, on=key_cols, how="outer")
    print("Final combined shape:", combined.shape)

    # ensure output dir
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)
    print("Saved combined file to:", OUTPUT)

if __name__ == "__main__":
    main()