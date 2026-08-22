"""
This file contains scripts that handle the
preprocessing of the datasets
"""

import os
from datasets import load_dataset
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def load_sysnet_mapping(path):
    """
    Loads the sysnet mapping from a file and returns three dictionaries:
    one that maps wnids to indices, one that maps indices to wnids and one that
    maps indices to string descriptions
    """
    wnid_to_idx = {}
    idx_to_wnid = {}
    idx_to_desc = {}
    
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            wnid, description = line.split(' ', 1)
            wnid_to_idx[wnid] = idx
            idx_to_wnid[idx] = wnid
            idx_to_desc[idx] = description
    return wnid_to_idx, idx_to_wnid, idx_to_desc

def main():
    # Create folders to hold the original and preprocessed dataset; delete them if already present to ensure
    # fresh start
    original_save_folder = Path("./original_datasets")
    if original_save_folder.exists():
        shutil.rmtree(original_save_folder)
    original_save_folder.mkdir()
    
    preprocessed_save_folder = Path("./preprocessed_datasets")
    if preprocessed_save_folder.exists():
        shutil.rmtree(preprocessed_save_folder)    
    preprocessed_save_folder.mkdir()
    
    # Get the imagenet-r dataset from hugging face:
    logger.info("Loading imagenet-r from huggingface")
    imagenet_r_dataset = load_dataset("axiong/imagenet-r", split="test")
    logger.info("Successfully loaded imagenet-r, saving to disk...")
    
    imagenet_r_dataset.save_to_disk(original_save_folder / "imagenet_r")    
    
    # Download only the validation split of the imagenet-1k.
    # To do this, we have to avoid using the dataset builder provided by the challenge
    # and instead use the built-in parquet builder
    # NOTE: Access to this dataset is gated by the challenge organizers.
    # To get access to the dataset, accept the competition rules and perform access through
    # huggingface_hub cli (hf auth login)
    logger.info("Loading imagenet-1k from huggingface")
    imagenet_1k_dataset = load_dataset(
        "parquet",
        data_files={
            "validation": "hf://datasets/ILSVRC/imagenet-1k@refs/convert/parquet/default/validation/*.parquet"
        },
        split="validation"
    )
    logger.info("Successfully loaded imagenet-1k, saving to disk...")
    imagenet_1k_dataset.save_to_disk(original_save_folder / "imagenet-1k")
    
    # Load the imagenet_sysnet_mapping
    wnid_to_idx, idx_to_wnid, idx_to_desc = load_sysnet_mapping("LOC_sysnet_mapping.txt")
    
    # Check for mismatches between the sysnet mapping and the HF dataset
    # 1 expected: crane (bird) with crane2 (construction machine)
    ik_label_names = imagenet_1k_dataset.features['label'].names
    mismatches = []
    for i in range(1000):
        hf_name = ik_label_names[i].lower().strip()
        file_desc = idx_to_desc[i].lower()
        first_syn = file_desc.split(',')[0].strip()
        if hf_name != first_syn and hf_name not in file_desc:
            mismatches.append((i, ik_label_names[i], idx_to_desc[i]))
    
    if len(mismatches) > 0:
        logger.warning(f"{len(mismatches)} mismatches out of 1000 classes!")
        logger.warning(f"Mismatched classes: {mismatches}")
        
    # Imagenet-R preprocessing: add a column to the dataset containing the
    # imagenet-1k class indices
    r_wnids_present = sorted(set(imagenet_r_dataset['wnid'])) # Get the wnids in the dataset
    
    # Add a column with the imagent-1k index label:
    imagenet_r_dataset_preprocessed = imagenet_r_dataset.map(
        lambda example: {"ik_label": wnid_to_idx[example["wnid"]]},
        desc="Mapping ImageNet-R wnids to ImageNet-1k label indices"
    )
    
    # Imagenet-1k preprocessing: filter down the dataset to only the 200
    # classes contained in imagenet-r
    overlap_indices = set(wnid_to_idx[w] for w in r_wnids_present) # Get the overlapping indices
    
    # Filter the dataset to only the 200 overlapping classes
    imagenet_1k_dataset_preprocessed = imagenet_1k_dataset.filter(
        lambda example: example["label"] in overlap_indices,
        desc="Filtering ImageNet-1k validation to ImageNet-R's 200 overlapping classes"
    )
    imagenet_1k_dataset_preprocessed.save_to_disk(preprocessed_save_folder / "imagenet_1k_preprocessed")
    imagenet_r_dataset_preprocessed.save_to_disk(preprocessed_save_folder / "imagenet_r_preprocessed")
    
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    main()
