import torch
from torch.nn import nn

class MnistCNN(nn.Module):
    """Small CNN for MNIST (28x28 freyscale images, 10 digit classes)"""

    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass


if __name__ == "__main__":
    model = MnistCNN()
    print(model)
    