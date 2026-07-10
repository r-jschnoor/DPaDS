from dataclasses import dataclass

from src.constants import NUM_CLASSES_MNIST, NUM_CLASSES_CIFAR10
from src.models.mnist_cnn import MnistCNN
from src.models.cifar10_cnn import Cifar10ResNet20


@dataclass
class DatasetSpec:
    """Model factory + class count for one dataset, resolved together since
    every call site that needs one needs the other."""
    model_fn: type
    num_classes: int


DATASET_REGISTRY = {
    "mnist": DatasetSpec(model_fn=MnistCNN, num_classes=NUM_CLASSES_MNIST),
    "cifar10": DatasetSpec(model_fn=Cifar10ResNet20, num_classes=NUM_CLASSES_CIFAR10),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    """
    Resolve the model factory and class count for a dataset name.

    Every client/strategy that needs to construct a fresh model, or size a
    confusion matrix, calls this once and reuses the result -- rather than
    each file branching on the dataset name itself.

    Args:
        dataset (str): "mnist" or "cifar10".

    Returns:
        DatasetSpec: model_fn + num_classes for that dataset.
    """
    return DATASET_REGISTRY[dataset]
