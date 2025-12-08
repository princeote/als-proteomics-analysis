#!/usr/bin/env python3
"""
Unified pipeline with:
- Log2 transform
- Protein filtering (>30% missing)
- Median normalization
- Optional ComBat
- Full QC plots (pre and post)
- Distinct ALS colors + black controls
- Tracking passed/failed proteins by missingness
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


# ------------------------------------------------------------
# COLOR MAPS
# ------------------------------------------------------------

ALS_COLORS = {
    "ALS_1":  "#1f77b4",
    "ALS_2":  "#ff7f0e",
    "ALS_3":  "#2ca02c",
    "ALS_4":  "#d62728",
    "ALS_5":  "#9467bd",
    "ALS_6":  "#8c564b",
    "ALS_7":  "#e377c2",
    "ALS_8":  "#7f7f7f",
    "ALS_9":  "#bcbd22",
    "ALS_10": "#17becf",
    "ALS_11": "#393b79",
}

CONTROL_COLOR = "black"


def clean_name(name):
    out = name
    out = out.split("_MaxLFQ")[0]
    out = out.split(" MaxLFQ")[0]
    out = out.replace("_MaxLFQ_Intensity", "")
    out = out.replace(" MaxLFQ_Intensity", "")
    return out.strip()


def get_sample_color(colname):
    clean = clean_name(colname)
    if clean.startswith("Control"):
        return CONTROL_COLOR
    subject = clean.split("_T")[0]
    return ALS_COLORS.get(subject, CONTROL_COLOR)


# ------------------------------------------------------------
# QC PLOTS
# ------------------------------------------------------------

def qc_total_intensity(df, outpath):
    totals = df.sum(axis=0)
    colors = [get_sample_color(c) for c in df.columns]

    plt.figure(figsize=(10,4))
    plt.bar(totals.index, totals.values, color=colors)
    plt.xticks(rotation=90)
    plt.title("Total Intensity per Sample")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def qc_missingness(df, outpath):
    miss = df.isna().mean(axis=0) * 100
    colors = [get_sample_color(c) for c in df.columns]

    plt.figure(figsize=(10,4))
    plt.bar(miss.index, miss.values, color=colors)
    plt.xticks(rotation=90)
    plt.title("Missingness (%) per Sample")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def qc_density(df, outpath):
    plt.figure(figsize=(10,6))
    for col in df.columns:
        sns.kdeplot(
            df[col].dropna(),
            color=get_sample_color(col),
            linewidth=1.5,
        )

    plt.title("Intensity Density Curves")
    plt.xlabel("Log2 Intensity")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def qc_pca(df, outpath):
    pca = PCA(n_components=2)
    X = df.fillna(df.mean())
    pcs = pca.fit_transform(X.T)

    clean_samples = [clean_name(c) for c in df.columns]
    colors = [get_sample_color(c) for c in df.columns]

    pc_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Sample": clean_samples,
        "Color": colors
    })

    plt.figure(figsize=(7,6))
    plt.scatter(pc_df["PC1"], pc_df["PC2"], c=pc_df["Color"], s=70)

    for i, row in pc_df.iterrows():
        plt.text(row["PC1"], row["PC2"], row["Sample"], fontsize=7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ------------------------------------------------------------
# NORMALIZATION / COMBAT
# ------------------------------------------------------------

def median_normalize(df):
    med = df.median(axis=0)
    gm = med.median()
    return df.subtract(med, axis=1).add(gm)


def combat_correct(df, batch_labels):
    try:
        from pyneurocombat import neuroCombat
    except ImportError:
        raise ImportError("Install pyneurocombat: pip install pyneurocombat")

    combat_out = neuroCombat(
        data=df.values,
        batch=batch_labels.values
    )["data"]

    return pd.DataFrame(combat_out, index=df.index, columns=df.columns)


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

INPUT = Path("results/combined_proteins_maxlfq.csv")
NORMALIZED_OUT = Path("results/normalized_proteins.csv")
BATCH_CORRECTED_OUT = Path("results/batch_corrected_proteins.csv")

batch_map = {}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not INPUT.exists():
        logging.error("Could not find input file: %s", INPUT)
        sys.exit(1)

    logging.info("Loading matrix...")
    df = pd.read_csv(INPUT)

    intensity_cols = [c for c in df.columns if "Intensity" in c]
    metadata_cols = [c for c in df.columns if c not in intensity_cols]

    meta = df[metadata_cols]
    X = df[intensity_cols].replace(0, np.nan)

    # Log2
    logging.info("Log2 transforming...")
    X = np.log2(X)

    # ---------- PRE-NORMALIZATION QC ----------
    qc_total_intensity(X, "plots/qc_total_pre.png")
    qc_missingness(X, "plots/qc_missing_pre.png")
    qc_density(X, "plots/qc_density_pre.png")
    qc_pca(X, "plots/qc_pca_pre.png")

    # ---------- PROTEIN FILTERING ----------
    logging.info("Filtering proteins missing >30%...")

    percent_missing = X.isna().mean(axis=1) * 100
    percent_missing.name = "Percent_Missing"

    Path("results").mkdir(exist_ok=True)

    passed_proteins = pd.DataFrame({
        "Protein": meta.index,
        "Percent_Missing": percent_missing
    }).loc[percent_missing <= 30]

    failed_proteins = pd.DataFrame({
        "Protein": meta.index,
        "Percent_Missing": percent_missing
    }).loc[percent_missing > 30]

    passed_proteins.to_csv("results/proteins_passed_70pct.csv", index=False)
    failed_proteins.to_csv("results/proteins_failed_70pct.csv", index=False)

    logging.info(f"{passed_proteins.shape[0]} proteins passed the 70% coverage filter.")
    logging.info(f"{failed_proteins.shape[0]} proteins failed and were removed.")

    keep = percent_missing <= 30
    meta = meta.loc[keep].reset_index(drop=True)
    X = X.loc[keep].reset_index(drop=True)

    # Extra QC after filtering
    missing_per_sample = X.isna().mean() * 100
    colors = [get_sample_color(c) for c in X.columns]

    plt.figure(figsize=(10,4))
    plt.bar(missing_per_sample.index, missing_per_sample.values, color=colors)
    plt.xticks(rotation=90)
    plt.title("Missingness (%) per Sample (post-filter)")
    plt.tight_layout()
    plt.savefig("plots/qc_missingness_postfilter.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(12,5))
    bp = plt.boxplot(
        [X[c].dropna() for c in X.columns],
        patch_artist=True,
        tick_labels=[clean_name(c) for c in X.columns]
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    plt.xticks(rotation=90)
    plt.title("Log2 Intensity Distribution")
    plt.tight_layout()
    plt.savefig("plots/qc_boxplot.png")
    plt.close()

    # Missing heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(X.isna(), cbar=False)
    plt.title("Missingness Heatmap")
    plt.tight_layout()
    plt.savefig("plots/qc_missingness_heatmap.png")
    plt.close()

    # ---------- NORMALIZATION ----------
    logging.info("Median normalizing...")
    X_norm = median_normalize(X)

    # ---------- POST-NORMALIZATION QC ----------
    qc_total_intensity(X_norm, "plots/qc_total_post.png")
    qc_missingness(X_norm, "plots/qc_missing_post.png")
    qc_density(X_norm, "plots/qc_density_post.png")
    qc_pca(X_norm, "plots/qc_pca_post.png")

    NORMALIZED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, X_norm], axis=1).to_csv(NORMALIZED_OUT, index=False)
    logging.info("Saved normalized matrix.")

    # ---------- OPTIONAL COMBAT ----------
    if batch_map:
        logging.info("Running ComBat...")
        batch_labels = pd.Series(batch_map)
        labels = batch_labels.loc[X_norm.columns]
        X_corrected = combat_correct(X_norm, labels)
        pd.concat([meta, X_corrected], axis=1).to_csv(BATCH_CORRECTED_OUT, index=False)
        logging.info("Saved batch corrected matrix.")
    else:
        logging.info("No batch_map provided — skipping ComBat.")

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
