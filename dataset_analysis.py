"""
This file contains a script that generates all relevant information and graphs
for performing a exploratory analysis of both the ILVSRC validation split and
ImageNet-R test split
"""

import os
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
import logging

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(filename="dataset_analysis.log", level=logging.INFO)
    os.makedirs("./dataset_info", exist_ok= True)
    # Load datasets with hugging face
    # NOTE: this requires an hugging face account to log
    # in by CLI with hf auth login, since the imagenet-1k dataset
    # is gated by the organizers of the challenge
    logger.info("Loading imagenet-r from huggingface")
    
    try:
        # Download the imagenet_r dataset with the provided dataset builder
        imagenet_r_dataset = load_dataset("axiong/imagenet-r", split="test")
    except Exception as e:
        logger.error("Failed to load the imagenet-r dataset!")
        print(e)
    
    logger.info("Successfully loaded imagenet-r")
    logger.info("Loading imagenet-1k from huggingface")
     
    try:
        # Download only the validation split of the imagenet-1k.
        # To do this, we have to avoid using the dataset builder provided by the challenge
        # and instead use the built-in parquet builder
        imagenet_1k_dataset = load_dataset(
            "parquet",
            data_files={
                "validation": "hf://datasets/ILSVRC/imagenet-1k@refs/convert/parquet/default/validation/*.parquet"
            },
            split="validation")
    except Exception as e:
        logger.error("Failed to load imagenet-1k. Have you logged in from CLI with hf auth login?")
        print(e)
        
    
    
if __name__ == '__main__':
    main()