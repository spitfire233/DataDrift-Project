import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Constants
IMAGENET_1K_DATASET_PATH = Path("./preprocessed_datasets/imagenet_1k_preprocessed")
RESULTS_PATH = Path("./baseline_metrics.json")
BATCH_SIZE = 64
NUM_WORKERS = 4
TOP_K = 5

def get_device():
    """Selects the best device on which to run the model"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_model_and_transforms():
    """
    Loads the ResNet50 model with the pre-trained weights and 
    the relative transform
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval() # Set in evaluation mode
    preprocess = weights.transforms() # Preprocessing transform
    return model, preprocess

class CollateFn:
    """
    Collate function che applica il preprocessing del modello a ogni immagine.
 
    Definita come classe a livello di modulo (invece che come closure annidata)
    perché deve essere pickleable: su Windows, DataLoader con num_workers > 0
    usa multiprocessing in modalità "spawn", che serializza la collate_fn per
    inviarla ai worker process. Una funzione definita dentro un'altra funzione
    non è pickleable e causa un AttributeError a runtime.
    """
 
    def __init__(self, preprocess):
        self.preprocess = preprocess
 
    def __call__(self, batch):
        images = torch.stack(
            [self.preprocess(example["image"].convert("RGB")) for example in batch]
        )
        labels = torch.tensor([example["label"] for example in batch], dtype=torch.long)
        return images, labels

@torch.no_grad # Disables gradient calculation at runtime
def run_inference(model, dataloader, device, k=TOP_K):
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

def main():
    device = get_device()
    logger.info(f"Utilizing device: {device}")
 
    if not IMAGENET_1K_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found in {IMAGENET_1K_DATASET_PATH}. "
            "Please, execute the preprocessing script first."
        )
    
    logger.info("Loading preprocessed dataset...")
    dataset = load_from_disk(str(IMAGENET_1K_DATASET_PATH))

    logger.info("Loading ResNet50...")
    model, preprocess = build_model_and_transforms()
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=CollateFn(preprocess=preprocess),
    )

    y_true, y_pred, topk_correct = run_inference(model, dataloader, device, k=TOP_K)
    metrics = compute_metrics(y_true, y_pred, topk_correct)

    logger.info("Baseline metrics computed:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
 
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved in {RESULTS_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()