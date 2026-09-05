"""
ImageNet-C Testing Script
Test all ImageNet-C corruptions with severity levels using the trained ResNet50 model.
Saves metrics and features for each corruption type and severity level.
"""

import json
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_from_disk

from model_testing.model import CollateFn, build_model_and_transforms, get_device
from model_testing.utils import (
    run_inference,
    compute_metrics,
    save_features_csv,
    run_repeated_evaluation_full_forward,
)
from torchvision import transforms


# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 64
NUM_WORKERS = 4
TOP_K = 5
RUNS = 5

DATASET_BASE_PATH = Path("../original_datasets/imagenet_c_datasets")
RESULTS_BASE_PATH = Path("../results/ImageNetC")

# ImageNet-C corruption types mapping
CORRUPTIONS = {
    'gaussian_noise': 'noise',
    'shot_noise': 'noise',
    'impulse_noise': 'noise',
    'defocus_blur': 'blur',
    'glass_blur': 'blur',
    'motion_blur': 'blur',
    'zoom_blur': 'blur',
    'snow': 'weather',
    'frost': 'weather',
    'fog': 'weather',
    'rain': 'weather',
    'brightness': 'digital',
    'contrast': 'digital',
    'elastic_transform': 'digital',
    'pixelate': 'digital',
    'jpeg_compression': 'digital'
}



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_folder_name(folder_name):
    """
    Parse folder names like 'gaussian_noise_sev_1' -> ('gaussian_noise', 1)
    """
    parts = folder_name.rsplit('_sev_', 1)
    if len(parts) == 2:
        corruption_name = parts[0]
        try:
            severity = int(parts[1])
            return corruption_name, severity
        except ValueError:
            return None, None
    return None, None


def create_output_structure(base_results_path):
    """Create output folder structure for each corruption type"""
    base_results_path = Path(base_results_path)
    corruption_types = set(CORRUPTIONS.values())

    for corruption_type in corruption_types:
        output_dir = base_results_path / corruption_type
        output_dir.mkdir(parents=True, exist_ok=True)

    print("✓ Output folder structure created")
    return base_results_path



# ============================================================================
# ANALYSIS
# ============================================================================

def find_all_datasets(base_dataset_path):
    """
    Find all ImageNet-C dataset folders with format: {corruption}_sev_{severity}
    Returns a sorted list of dataset paths
    """
    base_dataset_path = Path(base_dataset_path)

    datasets = []

    for folder in sorted(base_dataset_path.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name
        corruption_name, severity = parse_folder_name(folder_name)

        # Only include folders that match the expected pattern
        if corruption_name is not None and corruption_name in CORRUPTIONS:
            datasets.append({
                'path': folder,
                'folder_name': folder_name,
                'corruption_name': corruption_name,
                'severity': severity,
                'corruption_type': CORRUPTIONS[corruption_name]
            })

    return datasets


def get_dataset_and_dataloader(dataset_path, preprocess):
    """
    Load dataset from disk and create a DataLoader
    """
    try:
        dataset = load_from_disk(str(dataset_path))

        # If it's a DatasetDict, extract the first split
        if isinstance(dataset, DatasetDict):
            dataset = dataset[next(iter(dataset.keys()))]

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=CollateFn(preprocess=preprocess),
        )

        return dataloader, dataset

    except Exception as e:
        print(f"✗ Error loading dataset from {dataset_path}: {e}")
        return None, None


# ============================================================================
# PROCESSING
# ============================================================================

def process_single_dataset(model, device, preprocess, dataset_info, results_base_path):
    """
    Process a single ImageNet-C dataset:
    - Load dataset and create dataloader
    - Run inference
    - Compute metrics
    - Save results
    """
    dataset_path = dataset_info['path']
    folder_name = dataset_info['folder_name']
    corruption_type = dataset_info['corruption_type']

    print(f"\n{'='*70}")
    print(f"Processing: {folder_name}")
    print(f"{'='*70}")

    # Create output directory
    output_dir = results_base_path / corruption_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and dataloader
    print(f"Loading dataset from: {dataset_path}")
    dataloader, dataset = get_dataset_and_dataloader(dataset_path, preprocess)

    if dataloader is None:
        print(f"✗ Failed to load dataset, skipping...")
        return None

    print(f"✓ Dataset loaded ({len(dataset)} samples)")

    # Run a representative inference to collect features for later inspection
    print(f"Running inference (representative run)...")
    y_true, y_pred, topk_correct, features = run_inference(
        model, dataloader, device, k=TOP_K, return_features=True
    )

    # Compute single-run metrics (for quick logging)
    metrics = compute_metrics(y_true, y_pred, topk_correct)

    print(f"\nMetrics (representative run):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Compute repeated-run statistics (mean, std, CI)
    # Use full forward repeated evaluation (stochastic augmentations applied per run)
    # Define a light augment function to introduce stochasticity across runs
    augment_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
    ])

    def augment_fn(pil_img):
        return augment_pipeline(pil_img)

    metrics_stats = run_repeated_evaluation_full_forward(
        model,
        dataset,
        device,
        runs=RUNS,
        k=TOP_K,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        preprocess=preprocess,
        augment_fn=augment_fn,
    )

    # Save aggregated metrics (with CIs)
    metrics_path = output_dir / f"{folder_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_stats, f, indent=2)
    print(f"✓ Metrics (with CIs) saved to: {metrics_path}")

    # Save representative features
    features_path = output_dir / f"{folder_name}_features.csv"
    save_features_csv(features, y_true, y_pred, str(features_path))
    print(f"✓ Features saved to: {features_path}")

    return {
        'folder_name': folder_name,
        'dataset_size': len(dataset),
        'metrics': metrics,
        'metrics_path': str(metrics_path),
        'features_path': str(features_path)
    }



# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def display_statistics(copy_stats, total_copied):
    """
    Display statistics about the organized dataset
    """
    print("\n" + "=" * 70)
    print("ORGANIZATION STATISTICS")
    print("=" * 70)

    print(f"\nTotal images copied: {total_copied}")
    print("\nBreakdown by corruption type and severity level:")
    print("-" * 70)

    grand_total = 0

    for corruption_type in sorted(copy_stats.keys()):
        print(f"\n{corruption_type.upper()}:")
        type_total = 0

        for severity in range(1, 6):
            count = copy_stats[corruption_type][severity]
            type_total += count
            grand_total += count
            print(f"  Level {severity}: {count:6d} images")

        print(f"  {'Subtotal:':>10} {type_total:6d} images")

    print("\n" + "-" * 70)
    print(f"TOTAL: {grand_total} images")
    print("=" * 70)


def verify_folder_structure(base_output_dir):
    """
    Verify and display the created folder structure with file counts
    """
    base_output_dir = Path(base_output_dir)

    print("\n" + "=" * 70)
    print("FOLDER STRUCTURE VERIFICATION")
    print("=" * 70)

    for corruption_folder in sorted(base_output_dir.iterdir()):
        if not corruption_folder.is_dir():
            continue

        print(f"\n{corruption_folder.name}/")

        for level_folder in sorted(corruption_folder.iterdir()):
            if level_folder.is_dir():
                num_images = len(list(level_folder.glob('*')))
                print(f"  {level_folder.name}: {num_images} images")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function: Process all ImageNet-C datasets
    """
    print("Starting ImageNet-C Testing with ResNet50...")
    print("=" * 70)

    # Step 1: Create output structure
    print("\nStep 1: Creating output folder structure...")
    create_output_structure(RESULTS_BASE_PATH)

    # Step 2: Find all datasets
    print("\nStep 2: Finding all ImageNet-C datasets...")
    datasets = find_all_datasets(DATASET_BASE_PATH)
    print(f"Found {len(datasets)} datasets")

    for dataset_info in datasets:
        print(f"  • {dataset_info['folder_name']}")

    if len(datasets) == 0:
        print("✗ No datasets found! Please check DATASET_BASE_PATH")
        return

    # Step 3: Build model
    print("\nStep 3: Building model and loading weights...")
    model, preprocess = build_model_and_transforms()
    device = get_device()
    model.to(device)
    print(f"✓ Model loaded on {device}")

    # Step 4: Process each dataset
    print("\nStep 4: Processing datasets...")

    print("\nProcessing original dataset...")
    IMAGENET_1K_DATASET_PATH = Path("../original_datasets/imagenet-1k")
    if not IMAGENET_1K_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found in {IMAGENET_1K_DATASET_PATH}. "
            "Please, execute the preprocessing script first."
        )
    dataset_base = load_from_disk(str(IMAGENET_1K_DATASET_PATH))
    dataloader_base = DataLoader(
        dataset_base,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=CollateFn(preprocess=preprocess),
    )

    # Representative run to save features
    y_true_base, y_pred_base, topk_correct_base, features_base = run_inference(
        model, dataloader_base, device, k=TOP_K, return_features=True
    )
    metrics_base = compute_metrics(y_true_base, y_pred_base, topk_correct_base)
    save_features_csv(features_base, y_true_base, y_pred_base, "../results/ImageNetC/baseline_features.csv")

    # Repeated evaluation to compute CIs (full forward with stochastic augmentations)
    augment_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
    ])

    def augment_fn(pil_img):
        return augment_pipeline(pil_img)

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
    with open(Path("../results/ImageNetC/baseline_metrics.json"), "w") as f:
        json.dump(metrics_stats_base, f, indent=2)

    results = []
    print("\nProcessing ImageNetC datasets...")

    for i, dataset_info in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}]", end="")

        result = process_single_dataset(
            model, device, preprocess,
            dataset_info, RESULTS_BASE_PATH
        )

        if result:
            results.append(result)

    # Step 5: Print summary
    print("\n" + "=" * 70)
    print("TESTING SUMMARY")
    print("=" * 70)
    print(f"Total datasets processed: {len(results)}")
    print(f"Total datasets found: {len(datasets)}")

    if len(results) > 0:
        print("\nProcessed datasets:")
        for result in results:
            print(f"  ✓ {result['folder_name']}")
            print(f"    - Samples: {result['dataset_size']}")
            print(f"    - Accuracy: {result['metrics'].get('accuracy', 'N/A'):.4f}")

    print("\n" + "=" * 70)
    print("✓ Testing complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

