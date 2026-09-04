import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.stats import ks_2samp, gaussian_kde
from PIL import Image, ImageFilter, ImageStat
import pickle, os, re, hashlib
from tqdm.auto import tqdm
import warnings
import seaborn as sns
from collections import Counter
from pathlib import Path
import re
from evidently import Report
from evidently.presets import DataDriftPreset
import csv
import math

# CONSTANTS
corruption_to_category = {
    "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur", "zoom_blur": "blur",
    "contrast": "digital", "elastic_transform": "digital", "jpeg_compression": "digital", "pixelate": "digital",
    "gaussian_noise": "noise", "impulse_noise": "noise", "shot_noise": "noise",
    "brightness": "weather", "fog": "weather", "frost": "weather", "snow": "weather",
}

def class_distribution(dataset, label_col, name):
    """Returns a series with the number of examples per class."""
    counts = Counter(dataset[label_col])
    s = pd.Series(counts).sort_index()
    # print(f"\n{name}: {len(s)} classi distinte, "f"min={s.min()}, max={s.max()}")
    return s

def _to_pil(img):
    """Converte qualunque formato (PIL, np.array, bytes) in PIL RGB."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    return img

def extract_folder_name(key):
    """
    Extracts the folder name from a dictionary key
    """
    match = re.search(r"\((.*)\)", key)
    return match.group(1) if match else key

def split_corruption_severity(folder_name):
    """
    Separates the corruption name from its severity level.
    Handles both 'name_sev_N' and 'name_N' / 'name/N' formats.
    """
    match = re.match(r"^(.+?)(?:_sev)?[_/](\d+)$", folder_name)
    if match:
        return match.group(1), int(match.group(2))
    return folder_name, None

def fix_corruption_name(name):
    """Removes sev suffix from folder name"""
    m = re.match(r"^(.+)_sev\d$", name) or re.match(r"^(.+)_sev$", name)
    return name.rsplit("_sev", 1)[0]

def build_drift_table(imagenet_c_dict, reference_props, max_samples=None, cache_path=None):
    """Computes data drift metrics between ImageNet-1k and each ImageNet-C corruption/severity."""
    # If a cache file already exists, load it and skip recomputation
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    if cache_path is None:
        raise ValueError("cache_path è richiesto per la scrittura incrementale su CSV")

    fieldnames = ["corruption", "category", "severity", "feature", "score", "detected"]

    with open(cache_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key, dataset in tqdm(imagenet_c_dict.items(), desc="Corruptions"):
            # Extract the corruption folder name from the dict key, then parse corruption type and severity
            folder_name = key.replace("ImageNet-C (", "").rstrip(")")
            corruption, severity = split_corruption_severity(folder_name)

            # Extract visual properties for this corruption/severity dataset
            props = extract_properties(dataset, max_samples=max_samples,
                                        desc=f"{corruption} sev{severity}")

            # Run drift detection against the reference (ImageNet-1k) properties
            report = Report(metrics=[DataDriftPreset()])
            result = report.run(reference_data=reference_props, current_data=props)
            drift_info = extract_drift_info(result)

            # Write each feature's drift result as a row, flushed immediately to disk
            for feat, info in drift_info.items():
                writer.writerow({
                    "corruption": corruption,
                    "category": corruption_to_category.get(corruption, "unknown"),
                    "severity": severity,
                    "feature": feat,
                    "score": info["score"],
                    "detected": info["detected"],
                })
            f.flush()

    return pd.read_csv(cache_path)

def build_feature_table(imagenet_c_dict, reference_props, max_samples=None, cache_path=None):
    """Computes feature_table (raw feature values, long format) for ImageNet-1k vs each
    ImageNet-C corruption/severity, with optional CSV caching."""

    # If a cache file already exists, load it and skip recomputation
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    feature_rows = []

    # Reference (ImageNet-1k) features in long format, for the distribution plots
    ref_long = reference_props.melt(var_name="feature", value_name="value")
    ref_long["corruption"] = "reference"
    ref_long["category"] = "reference"
    ref_long["severity"] = 0
    feature_rows.append(ref_long)

    for key, dataset in tqdm(imagenet_c_dict.items(), desc="Corruptions"):
        # Extract the corruption folder name from the dict key, then parse corruption type and severity
        folder_name = key.replace("ImageNet-C (", "").rstrip(")")
        corruption, severity = split_corruption_severity(folder_name)
        category = corruption_to_category.get(corruption, "unknown")

        # Extract visual properties for this corruption/severity dataset
        props = extract_properties(dataset, max_samples=max_samples,
                                    desc=f"{corruption} sev{severity}")

        # Long format for the distribution plots
        long_df = props.melt(var_name="feature", value_name="value")
        long_df["corruption"] = corruption
        long_df["category"] = category
        long_df["severity"] = severity
        feature_rows.append(long_df)

    feature_table = pd.concat(feature_rows, ignore_index=True)

    # Save the computed feature table to disk so future calls can skip recomputation
    if cache_path is not None:
        feature_table.to_csv(cache_path, index=False)

    return feature_table

def extract_properties(dataset, image_col="image", max_samples=None, cache_path=None, desc="Extracting"):
    """Extracts the visual properties of the images contained in a HF dataset, with optional CSV caching."""

    # If a cache file already exists, load it and skip recomputation
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    # Get the length of the dataset
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    # Dictionary with the properties to extract
    keys = ["brightness", "rms_contrast", "sharpness",
            "mean_red", "mean_green", "mean_blue"]
    props = {k: [] for k in keys}

    for i in tqdm(range(n), desc=desc, leave=False):
        # Convert the image to a numpy PIL and get it's color properties
        img = _to_pil(dataset[i][image_col])
        r, g, b = img.split()

        # Get the color properties
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        props["brightness"].append(stat.mean[0] / 255.0)
        props["rms_contrast"].append(stat.stddev[0] / 255.0)

        # Ge the sharpness
        lap = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        props["sharpness"].append(float(np.var(lap)))

        # Get the means of the color channels
        mr = ImageStat.Stat(r).mean[0]
        mg = ImageStat.Stat(g).mean[0]
        mb = ImageStat.Stat(b).mean[0]
        total = mr + mg + mb + 1e-8
        props["mean_red"].append(mr / total)
        props["mean_green"].append(mg / total)
        props["mean_blue"].append(mb / total)
        
    result = pd.DataFrame(props)
    # Save the computed properties to disk so future calls can skip recomputation
    if cache_path is not None:
        result.to_csv(cache_path, index=False)
    return result

def extract_drift_info(report_obj):
    """
    Parsing per la API metric_v2 di Evidently (Report con ValueDrift per colonna).
    """
    result = report_obj.dict()

    drift_info = {}
    for metric_result in result["metrics"]:
        config = metric_result.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue  # salta DriftedColumnsCount e altri metrici aggregati

        col = config.get("column")
        method = config.get("method")
        threshold = config.get("threshold")
        score = float(metric_result["value"])  # cast da np.float64 a float puro
        detected = score >= threshold

        drift_info[col] = {
            "score": score,
            "detected": detected,
            "stattest": method,
        }
    return drift_info

def plot_drift_info(ref_df, cur_df, drift_info, ref_label="ImageNet-1k", cur_label="ImageNet-R", save_path = None):
    features = ref_df.columns.tolist()
    n = len(features)
    ncols = 2
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ref_vals = ref_df[feat].dropna()
        cur_vals = cur_df[feat].dropna()

        sns.kdeplot(ref_vals, color="tab:blue", fill=True, alpha=0.3,
                    linewidth=2, label=ref_label, ax=ax)
        sns.kdeplot(cur_vals, color="tab:orange", fill=True, alpha=0.3,
                    linewidth=2, label=cur_label, ax=ax)

        info = drift_info.get(feat, {})
        score = info.get("score")
        detected = info.get("detected")
        test_name = info.get("stattest", "?")
        color = "red" if detected else "green"
        flag = "DRIFT" if detected else "no drift"

        ax.set_title(f"{feat}\n{test_name}={score:.4f} ({flag})", fontsize=11, color=color)
        ax.legend(fontsize=8)
        ax.set_xlabel("")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Confronto delle distribuzioni delle feature: {ref_label} vs {cur_label} \n Drift quando WD >= 0.1", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

def plot_distributions_by_corruption(feature_table, drift_table, save_dir=None, n_rows=3):
    features = feature_table.loc[feature_table["feature"].notna(), "feature"].unique()
    severities = sorted(feature_table.loc[feature_table["severity"] > 0, "severity"].unique())
    sev_colors = dict(zip(severities, sns.color_palette("viridis", len(severities))))

    ref = feature_table[feature_table["category"] == "reference"]
    corruptions = [c for c in feature_table.loc[feature_table["category"] != "reference", "corruption"].unique()]

    # Create the output folder if a save path was given
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figs = {}

    for corr in corruptions:
        n_feat = len(features)
        n_cols = math.ceil(n_feat / n_rows)

        # extra width to make room for the Wasserstein distance annotations on the right
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3 * n_rows))
        axes = axes.reshape(n_rows, n_cols)  # normalizza sempre a 2D

        for idx, feat in enumerate(features):
            i, j = divmod(idx, n_cols)
            ax = axes[i, j]

            ref_vals = ref.loc[ref["feature"] == feat, "value"].dropna()
            sns.kdeplot(ref_vals, color="black", fill=True, alpha=0.12,
                        linewidth=1.5, label="ImageNet-1k", ax=ax)

            sub = feature_table[(feature_table["corruption"] == corr) &
                                 (feature_table["feature"] == feat)]
            for sev, sev_df in sub.groupby("severity"):
                sns.kdeplot(sev_df["value"].dropna(), color=sev_colors[sev],
                            linewidth=1.8, label=f"sev {sev}", ax=ax)

            ax.set_title(feat, fontsize=11)
            ax.set_xlabel("valore", fontsize=8)
            ax.set_ylabel("densità", fontsize=8)

            if ax.legend_ is not None:
                ax.legend_.remove()

            # Wasserstein distances (score già calcolato in drift_table), una riga per severità,
            # colorata in rosso se è stato rilevato drift per quella severità, in verde altrimenti
            drift_sub = drift_table[(drift_table["corruption"] == corr) &
                                     (drift_table["feature"] == feat)].sort_values("severity")

            y_start, y_step = 0.95, 0.9 / max(len(drift_sub), 1)
            for k, row in enumerate(drift_sub.itertuples()):
                color = "red" if row.detected else "green"
                ax.text(1.03, y_start - k * y_step, f"sev {row.severity}: WD={row.score:.3f}",
                        transform=ax.transAxes, fontsize=7, va="top", ha="left", color=color)

        # nascondi eventuali assi vuoti in eccesso (se n_feat non è multiplo di n_cols)
        for idx in range(n_feat, n_rows * n_cols):
            i, j = divmod(idx, n_cols)
            axes[i, j].set_visible(False)

        # legenda unica, presa dal primo subplot popolato
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=len(labels), fontsize=8, frameon=False)

        fig.suptitle(f"Distribuzioni delle feature: ImageNet-1k vs ImageNet-C. Sporcatura: {corr}",
                     fontsize=14, y=1.04)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / f"{corr}.png", dpi=150, bbox_inches="tight")
        plt.show()
        figs[corr] = fig

    return figs

def plot_severity_curves_by_category(drift_table, threshold=0.1, save_dir=None):
    categories = drift_table["category"].dropna().unique()
    figs = {}

    # Create the output folder if a save path was given
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for cat in sorted(categories):
        cat_table = drift_table[drift_table["category"] == cat]
        corruptions = sorted(cat_table["corruption"].unique())
        features = sorted(cat_table["feature"].unique())

        # una linea distinta per ogni corruzione della categoria
        cmap = mpl.colormaps["tab10"].resampled(len(corruptions))
        corruption_colors = {c: cmap(j) for j, c in enumerate(corruptions)}

        ncols = 2
        nrows = -(-len(features) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), sharey=False)
        axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            sub = cat_table[cat_table["feature"] == feat]

            for corruption, group in sub.groupby("corruption"):
                group = group.sort_values("severity")
                ax.plot(group["severity"], group["score"],
                        color=corruption_colors[corruption], linewidth=2,
                        marker="o", markersize=4, label=corruption)

            ax.axhline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_title(feat, fontsize=12)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Wasserstein (normed)")
            ax.set_xticks([1, 2, 3, 4, 5])

        for j in range(len(features), len(axes)):
            axes[j].axis("off")

        handles = [plt.Line2D([0], [0], color=corruption_colors[c], lw=2, label=c)
                   for c in corruptions]
        fig.legend(handles=handles, loc="upper center", ncol=min(len(corruptions), 4),
                   bbox_to_anchor=(0.5, 1.05))
        fig.suptitle(f"Drift (Wasserstein) vs severità — categoria: {cat}", y=1.10, fontsize=14)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / f"imagenet_c_severity_drift_{cat}.png", dpi=150, bbox_inches="tight")
        plt.show()

        figs[cat] = fig

    return figs

def summarize_drift(drift_table, save_dir=None):
    """
    Builds a summary table per corruption category, with one column per severity
    showing the Wasserstein score for each corruption/feature combination, plus
    a 'drift_detected' column flagging whether drift was detected at any severity.
    Optionally saves each category's table to a CSV file.
    """
    table = drift_table.copy()
    table["category"] = table["corruption"].map(corruption_to_category)

    # Create the output folder if a save path was given
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for category in sorted(table["category"].dropna().unique()):
        cat_table = table[table["category"] == category]

        pivot = cat_table.pivot_table(
            index=["corruption", "feature"],
            columns="severity",
            values="score",
        )
        pivot.columns = [f"sev_{int(s)}" for s in pivot.columns]

        # Flag whether drift was detected at any severity for that corruption/feature
        detected = cat_table.groupby(["corruption", "feature"])["detected"].any()
        pivot = pivot.join(detected.rename("drift_detected"))

        pivot = pivot.reset_index().sort_values(["corruption", "feature"]).reset_index(drop=True)

        summaries[category] = pivot

        if save_dir is not None:
            pivot.to_csv(save_dir / f"drift_summary_{category}.csv", index=False)

    return summaries