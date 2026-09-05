import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    cohen_kappa_score,
)
from tqdm import tqdm
from model_testing.model import extract_resnet50_features
from torch.utils.data import DataLoader


@torch.no_grad()
def run_inference(model, dataloader, device, k=5, return_features=False):
    """
    Runs the inference of the model on the dataset and collects
    the true labels, prediction, top-1 and top-k.

    If return_features=True, also returns the penultimate-layer
    features used by the classifier.
    """
    all_labels = []
    all_top1_preds = []
    all_topk_correct = []
    all_features = []

    for images, labels in tqdm(dataloader, desc="Model inference"):
        images = images.to(device)
        features, logits = extract_resnet50_features(model, images)
        topk_preds = logits.topk(k, dim=1).indices.cpu()

        top1_preds = topk_preds[:, 0]
        topk_correct = (topk_preds == labels.unsqueeze(1)).any(dim=1)

        all_labels.append(labels)
        all_top1_preds.append(top1_preds)
        all_topk_correct.append(topk_correct)
        if return_features:
            all_features.append(features.cpu())

    outputs = (
        torch.cat(all_labels).numpy(),
        torch.cat(all_top1_preds).numpy(),
        torch.cat(all_topk_correct).numpy(),
    )

    if return_features:
        outputs = outputs + (torch.cat(all_features).numpy(),)

    return outputs


def compute_metrics(y_true, y_pred, topk_correct):
    """
    Compute classification metrics for ImageNet / ImageNet-R evaluation.

    Metrics:
        - Top-1 accuracy
        - Top-5 accuracy
        - Balanced accuracy
        - Macro precision, recall, F1
        - Weighted precision, recall, F1
        - Micro precision, recall, F1
        - Matthews Correlation Coefficient (MCC)
        - Cohen's Kappa

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.

    y_pred : array-like
        Top-1 predicted class labels.

    topk_correct : array-like
        Boolean array indicating whether the true class is
        contained in the top-k predictions (e.g. top-5).

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    topk_correct = np.asarray(topk_correct)

    # ============================================================
    # Basic statistics
    # ============================================================

    n_samples = len(y_true)
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Number of samples per class
    class_counts = np.bincount(y_true)

    # ============================================================
    # Accuracy
    # ============================================================

    accuracy = accuracy_score(y_true, y_pred)

    top5_accuracy = float(topk_correct.mean())

    # Balanced accuracy = mean recall across classes
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # ============================================================
    # Precision / Recall / F1
    # ============================================================

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    precision_micro, recall_micro, f1_micro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
    )

    # ============================================================
    # Agreement / correlation metrics
    # ============================================================

    mcc = matthews_corrcoef(y_true, y_pred)

    cohen_kappa = cohen_kappa_score(y_true, y_pred)


    return {
        # --------------------------------------------------------
        # Main performance metrics
        # --------------------------------------------------------
        "accuracy": float(accuracy),
        "top5_accuracy": float(top5_accuracy),
        "balanced_accuracy": float(balanced_accuracy),

        # --------------------------------------------------------
        # Macro metrics: every class has equal weight
        # --------------------------------------------------------
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),

        # --------------------------------------------------------
        # Weighted metrics: every sample has equal weight
        # --------------------------------------------------------
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),

        # --------------------------------------------------------
        # Micro metrics
        # --------------------------------------------------------
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),

        # --------------------------------------------------------
        # Robust global metrics
        # --------------------------------------------------------
        "mcc": float(mcc),
        "cohen_kappa": float(cohen_kappa),

    }
def save_features_csv(features, labels, preds, output_path):
    df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
    df.insert(0, "prediction", preds)
    df.insert(0, "label", labels)
    df.to_csv(output_path, index=False)


def compute_metrics_ci(metrics_list, ci=0.95):
    """
    Given a list of metrics dictionaries (each produced by `compute_metrics`),
    compute the mean, standard deviation and confidence interval for each metric.

    Returns a dict mapping metric_name -> {
        'mean', 'std', 'ci_lower', 'ci_upper', 'values'
    }
    """
    if len(metrics_list) == 0:
        return {}

    import math

    try:
        from scipy.stats import t as t_dist
    except Exception:
        t_dist = None

    n = len(metrics_list)
    keys = list(metrics_list[0].keys())
    stats = {}

    for key in keys:
        vals = np.array([m[key] for m in metrics_list], dtype=float)
        mean = float(vals.mean())
        # sample std (ddof=1) if possible
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        se = float(std / math.sqrt(n)) if n > 1 else 0.0

        if n > 1:
            if t_dist is not None:
                tcrit = float(t_dist.ppf((1 + ci) / 2.0, df=n - 1))
            else:
                # fallback: normal approximation
                # 1.96 is approximate for 95% CI
                if abs(ci - 0.95) < 1e-6:
                    tcrit = 1.959963984540054
                else:
                    tcrit = 1.96
            ci_lower = mean - tcrit * se
            ci_upper = mean + tcrit * se
        else:
            ci_lower = mean
            ci_upper = mean

        stats[key] = {
            "mean": mean,
            "std": std,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "values": vals.tolist(),
        }

    return stats

class CollateFnAugment:
    """
    Collate function that applies an optional augment_fn (PIL -> PIL) before the
    model preprocess. Compatible with datasets that return dicts having 'image' and 'label'.
    """

    def __init__(self, preprocess, augment_fn=None):
        self.preprocess = preprocess
        self.augment_fn = augment_fn

    def __call__(self, batch):
        images = []
        labels = []
        for example in batch:
            img = example["image"].convert("RGB")
            if self.augment_fn is not None:
                img = self.augment_fn(img)
            img_t = self.preprocess(img)
            images.append(img_t)
            labels.append(example["label"])
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels


def run_repeated_evaluation_full_forward(
    model,
    dataset,
    device,
    runs=5,
    k=5,
    batch_size=64,
    num_workers=4,
    preprocess=None,
    augment_fn=None,
):
    """
    Perform `runs` complete forward passes over `dataset`, optionally applying a
    stochastic `augment_fn` (PIL -> PIL) before the provided `preprocess`.

    This recomputes the convolutional features every run and therefore captures
    variability coming from augmentations / stochastic preprocessing.
    Returns the same structure produced by `compute_metrics_ci`.
    """
    if preprocess is None:
        raise ValueError("preprocess is required for full forward repeated evaluation")

    metrics_list = []

    collate = CollateFnAugment(preprocess=preprocess, augment_fn=augment_fn)

    for i in range(runs):
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )

        # run full inference (this will apply augmentations inside the collate)
        y_true_run, y_pred_run, topk_correct_run, _ = run_inference(
            model, dl, device, k=k, return_features=True
        )

        metrics = compute_metrics(y_true_run, y_pred_run, topk_correct_run)
        metrics_list.append(metrics)

    return compute_metrics_ci(metrics_list)



