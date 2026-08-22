import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

@torch.no_grad  # Disables gradient calculation at runtime
def run_inference(model, dataloader, device, k=5):
    """
    Runs the inference of the model on the dataset and collects
    the true labels, prediction, top-1 and top-k
    """
    all_labels = []
    all_top1_preds = []
    all_topk_correct = []

    for images, labels in tqdm(dataloader, desc="Baseline inference on imagenet-1k"):
        images = images.to(device)
        logits = model(images)
        topk_preds = logits.topk(k, dim=1).indices.cpu()

        top1_preds = topk_preds[:, 0]
        topk_correct = (topk_preds == labels.unsqueeze(1)).any(dim=1)

        all_labels.append(labels)
        all_top1_preds.append(top1_preds)
        all_topk_correct.append(topk_correct)

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_top1_preds).numpy(),
        torch.cat(all_topk_correct).numpy(),
    )


def compute_metrics(y_true, y_pred, topk_correct):
    """
    Calculates accuracy, top-5 accuracy, precision, recall and F1 (macro and weighted)
    """
    accuracy = accuracy_score(y_true, y_pred)
    top5_accuracy = float(topk_correct.mean())

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "n_samples": int(len(y_true)),
        "n_classes_present": int(len(set(y_true.tolist()))),
    }
