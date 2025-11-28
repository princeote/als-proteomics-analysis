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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# -------- QC Functions --------
def qc_total_intensity(df, outpath):
    totals = df.sum(axis=0)
    colors = ["blue" if "Control" in c else "red" for c in df.columns]
    plt.figure(figsize=(10,4))
    plt.bar(totals.index, totals.values, color=colors)
    plt.xticks(rotation=90)
    plt.title("Total Intensity per Sample")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def qc_missingness(df, outpath):
    miss = df.isna().mean(axis=0) * 100
    colors = ["blue" if "Control" in c else "red" for c in df.columns]
    plt.figure(figsize=(10,4))
    plt.bar(miss.index, miss.values, color=colors)
    plt.xticks(rotation=90)
    plt.title("Missingness (%) per Sample")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def qc_density(df, outpath):
    plt.figure(figsize=(10,6))
    clean_samples = [s.replace(" MaxLFQ intensity", "") for s in df.columns]
    groups = ["Control" if "Control" in s else "ALS" for s in clean_samples]
    palette = {"Control": "blue", "ALS": "red"}
    for i, col in enumerate(df.columns):
        sns.kdeplot(df[col].dropna(), label=clean_samples[i], color=palette[groups[i]], linewidth=1.5)
    plt.title("Intensity Density Curves")
    plt.xlabel("Log2 Intensity")
    plt.ylabel("Density")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def qc_pca(df, outpath):
    pca = PCA(n_components=2)
    X = df.fillna(df.mean())
    pcs = pca.fit_transform(X.T)
    clean_samples = [s.replace(" MaxLFQ intensity", "") for s in X.columns]
    groups = ["Control" if "Control" in s else "ALS" for s in clean_samples]
    pc_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Sample": clean_samples,
        "Group": groups
    })
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue="Group", palette={"Control":"blue","ALS":"red"}, s=60)
    for i, row in pc_df.iterrows():
        plt.text(row["PC1"], row["PC2"], row["Sample"], fontsize=7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------- Paths --------
INPUT = Path("results/combined_proteins_maxlfq.csv")
NORMALIZED_OUT = Path("results/normalized_proteins.csv")
BATCH_CORRECTED_OUT = Path("results/batch_corrected_proteins.csv")
batch_map = {}  # leave empty to skip batch correction

# -------- Helper Functions --------
def median_normalize(df):
    med = df.median(axis=0)
    gm = med.median()
    return df.subtract(med, axis=1).add(gm)

def combat_correct(df, batch_labels):
    try:
        from pyneurocombat import neuroCombat
    except ImportError:
        raise ImportError("pyneurocombat is not installed.\nInstall it with: pip install pyneurocombat")
    combat_out = neuroCombat(data=df.values, batch=batch_labels.values)["data"]
    return pd.DataFrame(combat_out, index=df.index, columns=df.columns)

# -------- Main Function --------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not INPUT.exists():
        logging.error("Could not find input file at %s", INPUT)
        sys.exit(1)

    logging.info("Loading combined protein matrix...")
    df = pd.read_csv(INPUT)

    intensity_cols = [c for c in df.columns if "Intensity" in c]
    metadata_cols = [c for c in df.columns if c not in intensity_cols]

    logging.info("Found %d samples", len(intensity_cols))

    meta = df[metadata_cols]
    X = df[intensity_cols].replace(0, np.nan)
    logging.info("Log2 transforming...")
    X = np.log2(X)

    # --- Pre-normalization QC ---
    qc_total_intensity(X, "plots/qc_total_pre.png")
    qc_missingness(X, "plots/qc_missing_pre.png")
    qc_density(X, "plots/qc_density_pre.png")
    qc_pca(X, "plots/qc_pca_pre.png")

    # Filter proteins missing >30%
    logging.info("Filtering proteins missing >30%...")
    keep = X.isna().mean(axis=1) <= 0.30
    meta = meta.loc[keep].reset_index(drop=True)
    X = X.loc[keep].reset_index(drop=True)
    logging.info("Keeping %d proteins", X.shape[0])

    # Other QC plots (post-filter)
    missing_per_sample = X.isna().mean() * 100
    colors = ["blue" if "Control" in s else "red" for s in X.columns]
    plt.figure(figsize=(10,4))
    plt.bar(missing_per_sample.index, missing_per_sample.values, color=colors)
    plt.xticks(rotation=90)
    plt.title("Missingness (%) per Sample")
    plt.tight_layout()
    plt.savefig("plots/qc_missingness_per_sample.png")
    plt.close()

    total_intensity = X.sum()
    plt.figure(figsize=(10,4))
    plt.bar(total_intensity.index, total_intensity.values, color=colors)
    plt.xticks(rotation=90)
    plt.title("Total Ion Intensity")
    plt.tight_layout()
    plt.savefig("plots/qc_total_intensity.png")
    plt.close()

    # Boxplot colored by group
    plt.figure(figsize=(12,5))
    bp = plt.boxplot([X[col].dropna() for col in X.columns], patch_artist=True,
                        tick_labels=[c.replace(" MaxLFQ intensity","") for c in X.columns])
    colors = ["blue" if "Control" in c else "red" for c in X.columns]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(rotation=90)
    plt.title("Log2 Intensity Distribution")
    plt.tight_layout()
    plt.savefig("plots/qc_boxplot.png")
    plt.close()

    # Missingness heatmap (keep grayscale)
    plt.figure(figsize=(8,6))
    sns.heatmap(X.isna(), cbar=False)
    plt.title("Missingness Heatmap")
    plt.tight_layout()
    plt.savefig("plots/qc_missingness_heatmap.png")
    plt.close()

    logging.info("QC plots saved to plots/")

    # Median normalization
    logging.info("Median normalizing...")
    X_norm = median_normalize(X)

    # Post-normalization QC
    qc_total_intensity(X_norm, "plots/qc_total_post.png")
    qc_missingness(X_norm, "plots/qc_missing_post.png")
    qc_density(X_norm, "plots/qc_density_post.png")
    qc_pca(X_norm, "plots/qc_pca_post.png")

    # Save normalized data
    NORMALIZED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, X_norm], axis=1).to_csv(NORMALIZED_OUT, index=False)
    logging.info("Saved normalized matrix to %s", NORMALIZED_OUT)

    # Batch correction (optional)
    if batch_map:
        logging.info("Applying ComBat...")
        batch_labels = pd.Series(batch_map)
        missing = [s for s in X_norm.columns if s not in batch_labels.index]
        if missing:
            logging.error("Batch labels missing for samples: %s", missing)
            sys.exit(1)
        labels = batch_labels.loc[X_norm.columns]
        X_corrected = combat_correct(X_norm, labels)
        pd.concat([meta, X_corrected], axis=1).to_csv(BATCH_CORRECTED_OUT, index=False)
        logging.info("Saved batch-corrected matrix to %s", BATCH_CORRECTED_OUT)
    else:
        logging.info("No batch_map provided — skipping ComBat.")

    logging.info("Done.")

if __name__ == "__main__":
    main()
