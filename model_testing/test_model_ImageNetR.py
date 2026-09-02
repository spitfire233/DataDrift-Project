import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_from_disk
from model_testing.model import build_model_and_transforms, get_device, CollateFn
from model_testing.utils import (
    run_inference,
    compute_metrics,
    run_repeated_evaluation,
    run_repeated_evaluation_full_forward,
    save_features_csv,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Constants
IMAGENET_1K_DATASET_PATH = Path("../preprocessed_datasets/imagenet_1k_preprocessed")
IMAGENET_R_DATASET_PATH = Path("../preprocessed_datasets/imagenet_r_preprocessed")

RESULTS_BASELINE_PATH = Path("../results/ImageNetR/baseline_metrics.json")
RESULTS_IMAGENET_R_PATH = Path("../results/ImageNetR/drift_metrics.json")

BATCH_SIZE = 64
NUM_WORKERS = 4
TOP_K = 5
RUNS = 5

device = get_device()
logger.info(f"Utilizing device: {device}")

if not IMAGENET_1K_DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessed dataset not found in {IMAGENET_1K_DATASET_PATH}. "
        "Please, execute the preprocessing script first."
    )
if not IMAGENET_R_DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessed dataset not found in {IMAGENET_R_DATASET_PATH}. "
        "Please, execute the preprocessing script first."
    )

logger.info("Loading preprocessed dataset...")
dataset_base = load_from_disk(str(IMAGENET_1K_DATASET_PATH))
dataset_r = load_from_disk(str(IMAGENET_R_DATASET_PATH))
logger.info("Loading ResNet50...")
model, preprocess = build_model_and_transforms()
model.to(device)

# Lightweight stochastic augmentations applied before the model's preprocess.
# These are applied per-run to induce variability for CI estimation.
from torchvision import transforms
augment_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
])

def augment_fn(pil_img):
    return augment_pipeline(pil_img)

dataloader_base = DataLoader(
    dataset_base,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=CollateFn(preprocess=preprocess),
)
dataloader_r = DataLoader(
    dataset_r,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=CollateFn(preprocess=preprocess),
)
y_true_base, y_pred_base, topk_correct_base, features_base = run_inference(
    model, dataloader_base, device, k=TOP_K, return_features=True
)
y_true_r, y_pred_r, topk_correct_r, features_r = run_inference(
    model, dataloader_r, device, k=TOP_K, return_features=True
)

# Compute single-run metrics (useful for quick logging) and repeated-run stats
metrics_base = compute_metrics(y_true_base, y_pred_base, topk_correct_base)
metrics_r = compute_metrics(y_true_r, y_pred_r, topk_correct_r)

# Repeated evaluation to compute confidence intervals
metrics_stats_base = run_repeated_evaluation_full_forward(
    model,
    dataset_base,
    device,
    runs=RUNS,
    k=TOP_K,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    preprocess=preprocess,
    augment_fn=augment_fn,
)

metrics_stats_r = run_repeated_evaluation_full_forward(
    model,
    dataset_r,
    device,
    runs=RUNS,
    k=TOP_K,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    preprocess=preprocess,
    augment_fn=augment_fn,
)

from model_testing.utils import save_features_csv

logger.info("Baseline metrics computed:")
for key, value in metrics_base.items():
    logger.info(f"  {key}: {value}")

# Save aggregated statistics (mean, std, CI) for the baseline
with open(RESULTS_BASELINE_PATH, "w") as f:
    json.dump(metrics_stats_base, f, indent=2)
logger.info(f"Metrics (with CIs) saved in {RESULTS_BASELINE_PATH}")

# Save representative features from one run
save_features_csv(features_base, y_true_base, y_pred_base, "../results/ImageNetR/baseline_features.csv")

logger.info("Drift metrics computed:")
for key, value in metrics_r.items():
    logger.info(f"  {key}: {value}")

# Save aggregated statistics (mean, std, CI) for the drift dataset
with open(RESULTS_IMAGENET_R_PATH, "w") as f:
    json.dump(metrics_stats_r, f, indent=2)
logger.info(f"Metrics (with CIs) saved in {RESULTS_IMAGENET_R_PATH}")

save_features_csv(features_r, y_true_r, y_pred_r, "../results/ImageNetR/drift_features.csv")
