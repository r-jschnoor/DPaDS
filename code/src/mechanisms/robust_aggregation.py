import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from models.mnist_cnn import MnistCNN


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two flat vectors.

    Measures the angle between two gradient directions.
    Returns 1.0 if identical, 0.0 if perpendicular, -1.0 if opposite.

    Args:
        a (np.ndarray): first flat vector.
        b (np.ndarray): second flat vector.

    Returns:
        float: cosine similarity in range [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)


if __name__ == "__main__":
    # Quick Test
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([-1.0, 0.0, 0.0])
    d = np.array([0.0, 1.0, 0.0])

    print(f"Identical vectors:     {cosine_similarity(a, b):.2f}")  # 1.0
    print(f"Opposite vectors:      {cosine_similarity(a, c):.2f}")  # -1.0
    print(f"Perpendicular vectors: {cosine_similarity(a, d):.2f}")  # 0.0