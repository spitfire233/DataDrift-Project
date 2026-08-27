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
        - Dataset/class statistics

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

    # ============================================================
    # Class distribution statistics
    # ============================================================

    min_samples_per_class = int(class_counts[classes].min())
    max_samples_per_class = int(class_counts[classes].max())

    mean_samples_per_class = float(
        class_counts[classes].mean()
    )

    std_samples_per_class = float(
        class_counts[classes].std()
    )

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

        # --------------------------------------------------------
        # Dataset statistics
        # --------------------------------------------------------
        "n_samples": int(n_samples),
        "n_classes_present": int(n_classes),

        "min_samples_per_class": min_samples_per_class,
        "max_samples_per_class": max_samples_per_class,
        "mean_samples_per_class": mean_samples_per_class,
        "std_samples_per_class": std_samples_per_class,
    }
def save_features_csv(features, labels, preds, output_path):
    df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
    df.insert(0, "prediction", preds)
    df.insert(0, "label", labels)
    df.to_csv(output_path, index=False)
