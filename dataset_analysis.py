"""
This file contains a script that generates all relevant information and graphs
for performing a exploratory analysis of both the ILVSRC validation split and
ImageNet-R test split

# TODO: Salvare i dataset pre-processed da qualche parte sul filesystem che non
# sia la cache; così da usarli direttamente in un notebook

"""

import os
from datasets import load_dataset
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
        split="validation"
    )

    # Structural exploration before preprocessing:
    print("Imagenet-1k structure (before preprocessing): ", imagenet_1k_dataset)  
    print("Imagenet-r structure (before preprocessing): ", imagenet_r_dataset)
        
    # Structural exploration after preprocessing:
        
if __name__ == '__main__':
    main()