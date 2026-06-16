import torch
import torch.nn as nn

class MnistCNN(nn.Module):
    """Small CNN for MNIST (28x28 freyscale images, 10 digit classes)"""

    def __init__(self):
        super().__init__()

        # One convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,      # greyscale = 1 channel (RGB would be 3)
            out_channels=16,    # Learn 16 different filters (This is a parameter choice. 16 filters means: 16 different patterns can be learned. Each channel produces its own output channel. Often channel size doubles at each next layer since deeper layers usually detect more advanced features. Also the spatial size halves with each step so extra filters compensate for that.)
            kernel_size=3,      # Each filter looks at 3x3 patches
            padding=1,          # Output padding on the edges so output stays 28x28 (since window is 3x3 -> Window cant hang over the edges so it needs to have 0's padded for the window to go at the edges)
        )

        # One ReLU layer
        self.relu1 = nn.ReLU()      # Negative values are set to 0; Core part of CNNs

        # One MaxPool layer
        self.pool1 = nn.MaxPool2d(
            kernel_size=2       # Effectively halves the input size. It slides a 2x2 window across the image with no overlap and keeps only the maximum value in each window. -> Strongest signal in that region
        )


    def forward(self, x):
        # Forward data through layers
        x = self.conv1(x)       # (4, 1,  28, 28) -> (4, 16, 28, 28)
        x = self.relu1(x)       # same shape, just kills negative values
        x = self.pool1(x)       # (4, 16, 28, 28) -> (4, 16, 14, 14)
        return x


if __name__ == "__main__":
    model = MnistCNN()
    print(f"{model}\n")

    dummy = torch.zeros(4, 1, 28, 28)   # 4 Dummy MNIST images
    out = model(dummy)                  # Call the model with the dummy images
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")    