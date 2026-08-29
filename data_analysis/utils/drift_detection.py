import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp, gaussian_kde
from PIL import Image, ImageFilter, ImageStat
import pickle, os, re, hashlib
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

C_A      = "#7B61FF"
C_B      = "#FF6B6B"
C_BG     = "#FFFFFF"
C_BORDER = "#E0E4EF"
C_TEXT   = "#000000"
C_SUBTLE = "#9CA3AF"

def _to_pil(img):
    """Converte qualunque formato (PIL, np.array, bytes) in PIL RGB."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    return img

def extract_properties(dataset, image_col="image", max_samples=None, desc="Extracting"):
    """Estrae le proprietà visive da un HuggingFace dataset."""
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    keys = ["brightness", "rms_contrast", "sharpness",
            "aspect_ratio", "area",
            "mean_red", "mean_green", "mean_blue"]
    props = {k: [] for k in keys}

    for i in tqdm(range(n), desc=desc, leave=False):
        img = _to_pil(dataset[i][image_col])
        w, h = img.size
        r, g, b = img.split()

        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        props["brightness"].append(stat.mean[0] / 255.0)
        props["rms_contrast"].append(stat.stddev[0] / 255.0)

        # Sharpness: varianza del Laplaciano (proxy deepchecks)
        lap = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        props["sharpness"].append(float(np.var(lap)))

        props["aspect_ratio"].append(w / h)
        props["area"].append((w * h) / 1_000_000)

        mr = ImageStat.Stat(r).mean[0]
        mg = ImageStat.Stat(g).mean[0]
        mb = ImageStat.Stat(b).mean[0]
        total = mr + mg + mb + 1e-8
        props["mean_red"].append(mr / total)
        props["mean_green"].append(mg / total)
        props["mean_blue"].append(mb / total)

    return props


def _cache_key(label: str, image_col: str, max_samples) -> str:
    """Hash breve e stabile per identificare una combinazione label/col/samples."""
    raw = f"{label}|{image_col}|{max_samples}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def _props_cache_path(cache_dir: str, label: str, image_col: str, max_samples) -> str:
    key  = _cache_key(label, image_col, max_samples)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return os.path.join(cache_dir, f"{safe}_{key}.pkl")


def save_props(props: dict, path: str) -> None:
    """
    Salva le properties estratte su disco.

    Uso tipico — baseline fissa (es. ImageNet):
        props = extract_properties(imagenet_ds, max_samples=5000)
        save_props(props, "cache/imagenet1k_props.pkl")
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(props, f)
    print(f"Saved → {path}")


def load_props(path: str) -> dict:
    """Carica le properties da disco."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_props(dataset, label, image_col, max_samples, cache_dir, props):
    """
    Risolve le properties con questo ordine di priorità:
      1. dict già pronto
      2. percorso .pkl → carica da file
      3. cache su disco (cache_dir) → carica se esiste
      4. estrazione dal dataset → salva in cache se cache_dir è fornito
    """
    if isinstance(props, dict):
        return props
    if isinstance(props, str):
        return load_props(props)

    if cache_dir is not None:
        path = _props_cache_path(cache_dir, label, image_col, max_samples)
        if os.path.exists(path):
            print(f"Cache hit → {path}")
            return load_props(path)

    extracted = extract_properties(dataset, image_col, max_samples, desc=label)

    if cache_dir is not None:
        save_props(extracted, path)

    return extracted


def _run_ks_tests(props_a, props_b):
    """Esegue KS test per ogni proprietà. Ritorna dict con stat, p, drift."""
    results = {}
    for prop in props_a:
        stat, p = ks_2samp(props_a[prop], props_b[prop])
        results[prop] = {
            "statistic": round(stat, 4),
            "p_value":   round(p, 4),
            "drift":     p < 0.05,
        }
    return results

def _plot_kde(ax, vals_a, vals_b, prop_name, ks_result, label_a, label_b):
    """KDE line plot per una proprietà, stile deepchecks."""
    ax.set_facecolor(C_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)
        spine.set_linewidth(0.8)

    all_vals = np.concatenate([vals_a, vals_b])
    lo = np.percentile(all_vals, 1)
    hi = np.percentile(all_vals, 99)
    xs = np.linspace(lo, hi, 300)

    for vals, color, label in [(vals_a, C_A, label_a), (vals_b, C_B, label_b)]:
        vals = np.asarray(vals, dtype=float)
        if np.std(vals) < 1e-8:
            vals = vals + np.linspace(-1e-6, 1e-6, len(vals))
        kde = gaussian_kde(vals, bw_method="scott")
        ys  = kde(xs)
        ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
        ax.fill_between(xs, ys, alpha=0.12, color=color)

    drift   = ks_result["drift"]
    ann_col = "#C0392B" if drift else "#27AE60"
    ann_txt = f"KS = {ks_result['statistic']:.3f}\np  = {ks_result['p_value']:.3f}"
    ax.text(0.97, 0.95, ann_txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=ann_col,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=ann_col, lw=0.9, alpha=0.9))

    ax.set_title(prop_name.replace("_", " ").title(),
                 fontsize=10, color=C_TEXT, pad=6)
    ax.set_ylabel("Density", fontsize=7.5, color=C_SUBTLE)
    ax.tick_params(colors=C_SUBTLE, labelsize=7)
    ax.legend(fontsize=7.5, framealpha=0.7, loc="upper left",
              handlelength=1.4, borderpad=0.5)


def build_summary_table(ks_results):
    """Costruisce un DataFrame con i risultati del drift."""
    rows = []
    for prop, res in ks_results.items():
        rows.append({
            "Property":     prop.replace("_", " ").title(),
            "KS Statistic": res["statistic"],
            "p-value":      res["p_value"],
            "Drift":        "yes" if res["drift"] else "no",
        })
    return pd.DataFrame(rows).set_index("Property")


def plot_image_property_drift(
    dataset_a=None,
    dataset_b=None,
    label_a="Reference",
    label_b="Comparison",
    image_col="image",
    max_samples=None,
    props_a=None,
    props_b=None,
    cache_dir=None,
):
    """
    Analisi del drift su proprietà visive tra due dataset HuggingFace.

    Parameters
    ----------
    dataset_a, dataset_b : datasets.Dataset | None
        Sorgente dati. Può essere None se le properties sono già fornite
        tramite props_a / props_b o trovate in cache_dir.
    label_a, label_b : str
        Etichette usate nei plot e come chiave cache.
    image_col : str
        Nome della colonna immagine nel dataset.
    max_samples : int | None
        Limita il campione per velocità. None = intero dataset.
    props_a, props_b : dict | str | None
        Properties già estratte (dict) oppure percorso a un file .pkl.
        Se forniti, il corrispondente dataset viene ignorato.
    cache_dir : str | None
        Cartella per cache automatica su disco. La prima estrazione
        salva un .pkl; le successive lo ricaricano senza ricalcolare.
        La chiave cache è derivata da label + image_col + max_samples.

    Returns
    -------
    summary : pd.DataFrame
        Tabella con KS statistic, p-value e flag drift per ogni proprietà.
    """
    props_a = _get_props(dataset_a, label_a, image_col, max_samples, cache_dir, props_a)
    props_b = _get_props(dataset_b, label_b, image_col, max_samples, cache_dir, props_b)
    ks_results = _run_ks_tests(props_a, props_b)

    prop_names = list(props_a.keys())
    n_props = len(prop_names)
    ncols   = 4
    nrows   = (n_props + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 4.8, nrows * 3.4), facecolor=C_BG)
    fig.patch.set_facecolor(C_BG)

    n_drift = sum(1 for r in ks_results.values() if r["drift"])
    fig.suptitle(
        f"Image Property Drift — {label_a} vs {label_b}  "
        f"({n_drift}/{n_props} properties drifted, α = 0.05)",
        color=C_TEXT, y=0.99,
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=0.52, wspace=0.30,
                           top=0.93, bottom=0.05)

    for idx, prop in enumerate(prop_names):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        _plot_kde(ax, props_a[prop], props_b[prop],
                  prop, ks_results[prop], label_a, label_b)

    for idx in range(n_props, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    plt.show()
    return build_summary_table(ks_results)