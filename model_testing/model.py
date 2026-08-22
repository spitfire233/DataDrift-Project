import torch
from torchvision.models import resnet50, ResNet50_Weights

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
    model.eval()  # Set in evaluation mode
    preprocess = weights.transforms()  # Preprocessing transform
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