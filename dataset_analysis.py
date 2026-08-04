"""
This file contains a script that generates all relevant information and graphs
for performing a exploratory analysis of both the ILVSRC validation split and
ImageNet-R test split
"""

import os
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
import logging

logger = logging.getLogger(__name__)

def load_sysnet_mapping(path):
    """
    Loads the sysnet mapping from a file and returns two dictionaries:
    one that maps wnids to indices and one containing the inverse mapping
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

def preprocess_datasets(imagenet_1k_dataset, imagenet_r_dataset):
    """
    Preprocesses the datasets, returns a relabeled imagenet_r_dataset and 
    a filtered imagenet_1k_dataset
    """
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
    imagenet_r_dataset = imagenet_r_dataset.map(
        lambda example: {"ik_label": wnid_to_idx[example["wnid"]]},
        desc="Mapping ImageNet-R wnids to ImageNet-1k label indices"
    )
    
    # Imagenet-1k preprocessing: filter down the dataset to only the 200
    # classes contained in imagenet-r
    overlap_indices = set(wnid_to_idx[w] for w in r_wnids_present) # Get the overlapping indices
    
    # Filter the dataset to only the 200 overlapping classes
    imagenet_1k_dataset = imagenet_1k_dataset.filter(
        lambda example: example["label"] in overlap_indices,
        desc="Filtering ImageNet-1k validation to ImageNet-R's 200 overlapping classes"
    )
    return imagenet_1k_dataset, imagenet_r_dataset

def main():
    logging.basicConfig(filename="dataset_analysis.log", level=logging.INFO)
    os.makedirs("./dataset_info", exist_ok= True)
    
    # Load datasets with hugging face
    # NOTE: this requires an hugging face account to log
    # in by CLI with hf auth login, since the imagenet-1k dataset
    # is gated by the organizers of the challenge
    logger.info("Loading imagenet-r from huggingface")
    
    # Download the imagenet-r dataset test split
    imagenet_r_dataset = load_dataset("axiong/imagenet-r", split="test")
    
    logger.info("Successfully loaded imagenet-r")
    logger.info("Loading imagenet-1k from huggingface")
    
    # Download only the validation split of the imagenet-1k.
    # To do this, we have to avoid using the dataset builder provided by the challenge
    # and instead use the built-in parquet builder
    imagenet_1k_dataset = load_dataset(
        "parquet",
        data_files={
            "validation": "hf://datasets/ILSVRC/imagenet-1k@refs/convert/parquet/default/validation/*.parquet"
        },
        split="validation")

    # Structural exploration
    print("Imagenet-1k structure: ", imagenet_1k_dataset)  
    print("Imagenet-r structure: ", imagenet_r_dataset)
    
    imagenet_1k_dataset, imagenet_r_dataset = preprocess_datasets(imagenet_1k_dataset, imagenet_r_dataset)
        
if __name__ == '__main__':
    main()