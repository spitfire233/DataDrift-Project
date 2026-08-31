import matplotlib.pyplot as plt
from pathlib import Path
import random as rand
from datasets import load_from_disk
import json
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import linregress


def show_corruption_comparison(corruption_type, max_severity=5,
                               imagenet_c_path="../original_datasets/imagenet_c_datasets",
                               random_idx=True, sample_idx=None, save_path=None):
    """
    Show the same image from ImageNet-C at different corruption severity levels.

    Parameters:
    -----------
    corruption_type : str
        Type of corruption (e.g., "gaussian_noise", "defocus_blur", etc.)
    max_severity : int
        Maximum severity level to display (default: 5)
    imagenet_c_path : str
        Path to ImageNet-C dataset root directory
    random_idx : bool
        If True, select a random sample; if False, use sample_idx
    sample_idx : int
        Specific sample index to use (ignored if random_idx=True)
    save_path : str
        Path to save the figure (optional)
    """
    imagenet_c_path = Path(imagenet_c_path)

    # Load the first severity dataset to get a random sample
    dataset_sev_1 = load_from_disk(str(imagenet_c_path / f"{corruption_type}_sev_1"))
    
    # Get a random or specific sample
    if random_idx:
        sample_idx = rand.randint(0, len(dataset_sev_1) - 1)
    else:
        sample_idx = sample_idx if sample_idx is not None else 0
    
    sample = dataset_sev_1[sample_idx]
    label = sample['label']
    

    # Create figure
    fig, axes = plt.subplots(1, max_severity, figsize=(10, 3))
    fig.suptitle(f"ImageNet-C: {corruption_type.replace('_', ' ').title()} - Sample {sample_idx}",
                 fontsize=14, fontweight='bold')

    # Show each severity level
    for severity in range(1, max_severity + 1):
        corruption_dataset_path = imagenet_c_path / f"{corruption_type}_sev_{severity}"
        
        try:
            # Load only the specific sample by index (much faster!)
            corruption_dataset = load_from_disk(str(corruption_dataset_path))
            sample_data = corruption_dataset[sample_idx]
            corrupted_img = sample_data['image']
            axes[severity - 1].imshow(corrupted_img)
            axes[severity - 1].set_title(f"Severity {severity}")
            axes[severity - 1].axis('off')
        except Exception as e:
            print(f"  Error: {e}")
            axes[severity - 1].text(0.5, 0.5, f"Error",
                                    ha='center', va='center', transform=axes[severity - 1].transAxes)
            axes[severity - 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def aggregate_corruption_metrics(results_path="../results/ImageNetC"):
    """
    Aggregate metrics from all corruption types and severity levels.

    Parameters:
    -----------
    results_path : str
        Path to the results directory (contains baseline_metrics.json and subdirectories)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: corruption_type, severity, metric_name, metric_value
        Plus a row for baseline with severity=0
    """
    results_path = Path(results_path)

    data = []

    # Load baseline metrics
    baseline_file = results_path / "baseline_metrics.json"
    if baseline_file.exists():
        print(f"Loading baseline metrics from {baseline_file}...")
        with open(baseline_file, 'r') as f:
            baseline_metrics = json.load(f)

        for metric_name, metric_value in baseline_metrics.items():
            data.append({
                'corruption_type': 'baseline',
                'severity': 0,
                'metric_name': metric_name,
                'metric_value': metric_value
            })

    # Scan subdirectories (blur, digital, noise, weather)
    for category_dir in results_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        print(f"\nProcessing category: {category_name}")
        
        # Find all *_metrics.json files
        metrics_files = list(category_dir.glob("*_metrics.json"))
        print(f"  Found {len(metrics_files)} metrics files")
        
        for metrics_file in sorted(metrics_files):
            # Parse filename to extract corruption type and severity
            filename = metrics_file.stem  # e.g., "gaussian_noise_sev_1_metrics"
            
            # Remove "_metrics" suffix first
            if filename.endswith("_metrics"):
                filename = filename[:-8]  # Remove "_metrics"
            
            # Now find _sev_ pattern
            if "_sev_" in filename:
                # Split by _sev_
                parts = filename.split("_sev_")
                if len(parts) == 2:
                    corruption_type = parts[0]
                    try:
                        severity = int(parts[1])
                        print(f"    Loading {corruption_type} sev {severity}...")
                        
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        for metric_name, metric_value in metrics.items():
                            data.append({
                                'corruption_type': corruption_type,
                                'severity': severity,
                                'metric_name': metric_name,
                                'metric_value': metric_value
                            })
                    except ValueError as e:
                        print(f"    Error parsing {filename}: {e}")
                        continue
            else:
                print(f"    Skipped {filename} (no _sev_ pattern)")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    print(f"\nAggregated {len(df)} metrics")
    print(f"Corruption types: {df['corruption_type'].unique().tolist()}")

    return df


def aggregate_corruption_metrics_wide(results_path="../results/ImageNetC"):
    """
    Aggregate metrics in wide format (one row per corruption/severity, columns for each metric).

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: corruption_type, severity, and one column per metric
    """
    df_long = aggregate_corruption_metrics(results_path)

    # Pivot to wide format
    df_wide = df_long.pivot_table(
        index=['corruption_type', 'severity'],
        columns='metric_name',
        values='metric_value',
        aggfunc='first'
    ).reset_index()

    return df_wide


def plot_corruption_metrics(results_path="../results/ImageNetC", corruption_type=None, metric_name='accuracy'):
    """
    Plot metric trend across severity levels for a specific corruption type.
    
    Parameters:
    -----------
    results_path : str
        Path to the results directory
    corruption_type : str
        Type of corruption to plot (e.g., 'gaussian_noise', 'defocus_blur', 'fog', etc.)
        If None, lists available corruptions
    metric_name : str
        Name of the metric to plot (e.g., 'accuracy', 'f1_score', etc.)
    """
    results_path = Path(results_path)
    
    if corruption_type is None:
        # List available corruptions without loading all data
        available = set()
        for category_dir in results_path.iterdir():
            if not category_dir.is_dir():
                continue
            for metrics_file in category_dir.glob("*_metrics.json"):
                filename = metrics_file.stem
                if filename.endswith("_metrics"):
                    filename = filename[:-8]
                if "_sev_" in filename:
                    corruption = filename.split("_sev_")[0]
                    available.add(corruption)
        
        # Add baseline
        if (results_path / "baseline_metrics.json").exists():
            available.add('baseline')
        
        print("Available corruptions:")
        for i, corruption in enumerate(sorted(available), 1):
            print(f"  {i}. {corruption}")
        return
    
    # Load baseline metrics
    baseline_value = None
    baseline_file = results_path / "baseline_metrics.json"
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_metrics = json.load(f)
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
    
    # Find and load only files for this corruption type
    data = []
    found = False
    
    for category_dir in results_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        for metrics_file in sorted(category_dir.glob("*_metrics.json")):
            filename = metrics_file.stem
            if filename.endswith("_metrics"):
                filename = filename[:-8]
            
            if "_sev_" in filename:
                parts = filename.split("_sev_")
                if len(parts) == 2 and parts[0] == corruption_type:
                    try:
                        severity = int(parts[1])
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        if metric_name in metrics:
                            data.append({
                                'severity': severity,
                                metric_name: metrics[metric_name]
                            })
                            found = True
                    except ValueError:
                        pass
    
    if not found:
        print(f"Corruption '{corruption_type}' not found or no data available.")
        return
    
    # Sort by severity
    data = sorted(data, key=lambda x: x['severity'])
    
    # Check if metric was found
    if metric_name not in data[0]:
        print(f"Metric '{metric_name}' not found in data.")
        return
    
    severities = [d['severity'] for d in data]
    values = [d[metric_name] for d in data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(severities, values, marker='o', linewidth=2, markersize=10, label=corruption_type)
    
    # Aggiungi linea baseline se esiste
    if baseline_value is not None:
        ax.axhline(y=baseline_value, color='r', linestyle='--', linewidth=2, label='Baseline', alpha=0.7)
    
    ax.set_xlabel('Severity', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{corruption_type.replace('_', ' ').title()} - {metric_name.replace('_', ' ').title()} vs Severity", 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_corruption_category(results_path="../results/ImageNetC", category=None, metric_name='accuracy'):
    """
    Plot metric trends for all corruptions in a category (noise, blur, digital, weather).
    
    Parameters:
    -----------
    results_path : str
        Path to the results directory
    category : str
        Category to plot: 'noise', 'blur', 'digital', 'weather', or a list of corruption types
        If None, lists available categories
    metric_name : str
        Name of the metric to plot (e.g., 'accuracy', 'f1_score', etc.)
    """
    results_path = Path(results_path)
    
    # Define corruption categories
    categories_map = {
        'noise': ['gaussian_noise', 'impulse_noise', 'shot_noise'],
        'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'digital': ['brightness', 'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate'],
        'weather': ['fog', 'frost', 'snow']
    }
    
    if category is None:
        print("Available categories:")
        for cat in categories_map.keys():
            print(f"  - {cat}")
        return
    
    # Get list of corruptions to plot
    if isinstance(category, str):
        if category in categories_map:
            corruptions_to_plot = categories_map[category]
        else:
            print(f"Category '{category}' not found.")
            print(f"Available: {list(categories_map.keys())}")
            return
    else:
        corruptions_to_plot = category  # Assume it's a list
    
    # Load baseline
    baseline_value = None
    baseline_file = results_path / "baseline_metrics.json"
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_metrics = json.load(f)
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each corruption type
    for corruption_type in corruptions_to_plot:
        data = []
        
        for category_dir in results_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            for metrics_file in sorted(category_dir.glob("*_metrics.json")):
                filename = metrics_file.stem
                if filename.endswith("_metrics"):
                    filename = filename[:-8]
                
                if "_sev_" in filename:
                    parts = filename.split("_sev_")
                    if len(parts) == 2 and parts[0] == corruption_type:
                        try:
                            severity = int(parts[1])
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                            
                            if metric_name in metrics:
                                data.append({
                                    'severity': severity,
                                    metric_name: metrics[metric_name]
                                })
                        except ValueError:
                            pass
        
        if data:
            data = sorted(data, key=lambda x: x['severity'])
            severities = [d['severity'] for d in data]
            values = [d[metric_name] for d in data]
            ax.plot(severities, values, marker='o', linewidth=2, markersize=8, 
                   label=corruption_type.replace('_', ' ').title())
    
    # Aggiungi linea baseline se esiste
    if baseline_value is not None:
        ax.axhline(y=baseline_value, color='red', linestyle='--', linewidth=2, label='Baseline', alpha=0.7)
    
    ax.set_xlabel('Severity', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    category_title = category if isinstance(category, str) else "Custom"
    ax.set_title(f"{category_title.title()} Corruptions - {metric_name.replace('_', ' ').title()} vs Severity", 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.show()


def calculate_degradation(results_path="../results/ImageNetC", metric_name='accuracy'):
    """
    Calculate degradation ratio: (Acc_clean - Acc_corrupted) / Acc_clean

    Parameters:
    -----------
    results_path : str
        Path to the results directory
    metric_name : str
        Name of the metric to calculate degradation for

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: corruption_type, severity, degradation
    """
    results_path = Path(results_path)

    # Load baseline metrics
    baseline_file = results_path / "baseline_metrics.json"
    if not baseline_file.exists():
        print("Baseline metrics file not found!")
        return None

    with open(baseline_file, 'r') as f:
        baseline_metrics = json.load(f)
        baseline_value = baseline_metrics.get(metric_name)

    if baseline_value is None:
        print(f"Metric '{metric_name}' not found in baseline!")
        return None

    data = []

    # Scan all categories and files
    for category_dir in sorted(results_path.iterdir()):
        if not category_dir.is_dir():
            continue

        for metrics_file in sorted(category_dir.glob("*_metrics.json")):
            filename = metrics_file.stem
            if filename.endswith("_metrics"):
                filename = filename[:-8]

            if "_sev_" in filename:
                parts = filename.split("_sev_")
                if len(parts) == 2:
                    corruption_type = parts[0]
                    try:
                        severity = int(parts[1])
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)

                        if metric_name in metrics:
                            corrupted_value = metrics[metric_name]
                            degradation = (baseline_value - corrupted_value) / baseline_value

                            data.append({
                                'corruption_type': corruption_type,
                                'severity': severity,
                                'degradation': degradation
                            })
                    except ValueError:
                        pass

    df = pd.DataFrame(data)
    return df


def plot_degradation(results_path="../results/ImageNetC", corruption_type=None,
                    category=None, metric_name='accuracy'):
    """
    Plot degradation trend across severity levels.

    Parameters:
    -----------
    results_path : str
        Path to the results directory
    corruption_type : str
        Type of corruption to plot (e.g., 'gaussian_noise')
    category : str
        Category to plot: 'noise', 'blur', 'digital', 'weather'
    metric_name : str
        Name of the metric to calculate degradation for
    """
    df_degradation = calculate_degradation(results_path, metric_name)

    if df_degradation is None or df_degradation.empty:
        print("No degradation data available!")
        return

    # Define corruption categories
    categories_map = {
        'noise': ['gaussian_noise', 'impulse_noise', 'shot_noise'],
        'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'digital': ['brightness', 'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate'],
        'weather': ['fog', 'frost', 'snow']
    }

    if corruption_type is not None:
        # Plot single corruption type
        data = df_degradation[df_degradation['corruption_type'] == corruption_type].sort_values('severity')

        if data.empty:
            print(f"Corruption '{corruption_type}' not found!")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['severity'], data['degradation'], marker='o', linewidth=2, markersize=10,
               label=corruption_type)
        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel('Degradation', fontsize=12)
        ax.set_title(f"{corruption_type.replace('_', ' ').title()} - {metric_name.title()} Degradation",
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 6))
        ax.legend(fontsize=11)

    elif category is not None:
        # Plot category
        if category in categories_map:
            corruptions = categories_map[category]
        else:
            print(f"Category '{category}' not found. Available: {list(categories_map.keys())}")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        for corruption in corruptions:
            data = df_degradation[df_degradation['corruption_type'] == corruption].sort_values('severity')
            if not data.empty:
                ax.plot(data['severity'], data['degradation'], marker='o', linewidth=2, markersize=8,
                       label=corruption.replace('_', ' ').title())

        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel('Degradation', fontsize=12)
        ax.set_title(f"{category.title()} Corruptions - {metric_name.title()} Degradation",
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 6))
        ax.legend(fontsize=10, loc='best')

    else:
        print("Please specify either 'corruption_type' or 'category'")
        return

    plt.tight_layout()
    plt.show()


def degradation_summary(results_path="../results/ImageNetC", metric_name='accuracy', 
                       corruption_type=None, category=None, use_robustness=False):
    """
    Calculate degradation/robustness summary statistics (AUC and slope).

    Parameters:
    -----------
    results_path : str
        Path to the results directory
    metric_name : str
        Name of the metric to calculate degradation for
    corruption_type : str
        Specific corruption to analyze (e.g., 'gaussian_noise'). If None, analyzes all.
    category : str
        Category to analyze: 'noise', 'blur', 'digital', 'weather'. If None, analyzes all.
    use_robustness : bool
        If True, calculates AUC on robustness (Acc_s/Acc_clean)
        If False, calculates AUC on degradation ((Acc_clean-Acc_s)/Acc_clean)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: corruption_type, AUC, slope, R_squared
    """
    results_path = Path(results_path)
    
    # Load baseline metrics
    baseline_file = results_path / "baseline_metrics.json"
    if not baseline_file.exists():
        print("Baseline metrics file not found!")
        return None

    with open(baseline_file, 'r') as f:
        baseline_metrics = json.load(f)
        baseline_value = baseline_metrics.get(metric_name)
    
    if baseline_value is None:
        print(f"Metric '{metric_name}' not found in baseline!")
        return None
    
    # Define corruption categories
    categories_map = {
        'noise': ['gaussian_noise', 'impulse_noise', 'shot_noise'],
        'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'digital': ['brightness', 'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate'],
        'weather': ['fog', 'frost', 'snow']
    }
    
    # Filter corruptions based on parameters
    if corruption_type is not None:
        corruptions_to_analyze = [corruption_type]
    elif category is not None:
        if category in categories_map:
            corruptions_to_analyze = categories_map[category]
        else:
            print(f"Category '{category}' not found. Available: {list(categories_map.keys())}")
            return None
    else:
        # Get all corruptions
        corruptions_to_analyze = set()
        for category_dir in results_path.iterdir():
            if not category_dir.is_dir():
                continue
            for metrics_file in category_dir.glob("*_metrics.json"):
                filename = metrics_file.stem
                if filename.endswith("_metrics"):
                    filename = filename[:-8]
                if "_sev_" in filename:
                    corruption = filename.split("_sev_")[0]
                    corruptions_to_analyze.add(corruption)
        corruptions_to_analyze = sorted(list(corruptions_to_analyze))

    summary_data = []
    
    for corruption in corruptions_to_analyze:
        data = []
        
        # Scan all categories and files
        for category_dir in sorted(results_path.iterdir()):
            if not category_dir.is_dir():
                continue
            
            for metrics_file in sorted(category_dir.glob("*_metrics.json")):
                filename = metrics_file.stem
                if filename.endswith("_metrics"):
                    filename = filename[:-8]
                
                if "_sev_" in filename:
                    parts = filename.split("_sev_")
                    if len(parts) == 2 and parts[0] == corruption:
                        try:
                            severity = int(parts[1])
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                            
                            if metric_name in metrics:
                                corrupted_value = metrics[metric_name]
                                
                                # Robustness = Acc_s / Acc_clean
                                robustness = corrupted_value / baseline_value
                                # Degradation = (Acc_clean - Acc_s) / Acc_clean
                                degradation = (baseline_value - corrupted_value) / baseline_value
                                
                                data.append({
                                    'severity': severity,
                                    'robustness': robustness,
                                    'degradation': degradation
                                })
                        except ValueError:
                            pass

        if len(data) > 1:
            data = sorted(data, key=lambda x: x['severity'])
            severities = [d['severity'] for d in data]
            
            # Choose AUC metric based on use_robustness
            if use_robustness:
                # AUC = average robustness: (1/5) * Σ(Acc_s / Acc_clean)
                robustness_values = [d['robustness'] for d in data]
                auc = sum(robustness_values) / len(robustness_values)
            else:
                # AUC on degradation using trapezoid
                degradation_values = [d['degradation'] for d in data]
                auc = trapezoid(degradation_values, severities)
            
            # Slope is ALWAYS calculated on degradation
            degradation_values = [d['degradation'] for d in data]
            
            # Calculate slope (always on degradation)
            slope, intercept, r_value, p_value, std_err = linregress(severities, degradation_values)
            
            summary_data.append({
                'corruption_type': corruption,
                'AUC': auc,
                'slope': slope,
                'R_squared': r_value**2
            })
    
    df_summary = pd.DataFrame(summary_data).sort_values('AUC', ascending=False)
    
    return df_summary
