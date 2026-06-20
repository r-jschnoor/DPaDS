import torch
import torch.nn as nn

class MnistCNN(nn.Module):
    """Small CNN for MNIST (28x28 freyscale images, 10 digit classes)"""

    def __init__(self):
        super().__init__()
        # ----- First Layering -----
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
        
        # ----- Second Layering -----
        self.conv2 = nn.Conv2d(
            in_channels=16,     # Same as out in layer before
            out_channels=32,    # Doubling as explained before
            kernel_size=3,      # Same as before
            padding=1,          # Same as before
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)    # Same as before to shrink even further

        # ----- Flattening Layering -----
        # Now turn the tensor into class scores -> One number per digit (in the output guess).
        # Currently we have a 3D block -> Need to flatten to 1D Vector with size 32*7*7 = 1568
        # A Linear Layer connects every input to every output. Hence fully connected.
        # A Convolutional Layer looks over the tensor in patches, meaning its only locally connected.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)    # fc = fully connected, 64 => choosable parameter for the output channels. 64 seems to be normal for MNIST
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)            # 10 -> One score per digit


        # ReLU and MaxPool could be reused since the parameters are the same. I am not doing it here for now for learning purposes.

    def forward(self, x):
        # Forward data through layers
        x = self.conv1(x)       # (4, 1,  28, 28) -> (4, 16, 28, 28)
        x = self.relu1(x)       # same shape, just kills negative values
        x = self.pool1(x)       # (4, 16, 28, 28) -> (4, 16, 14, 14)

        x = self.conv2(x)       # (4, 16, 14, 14) -> (4, 32, 14, 14)
        x = self.relu2(x)
        x = self.pool2(x)       # (4, 32, 14, 14) -> (4, 32,  7,  7)

        x = self.flatten(x)     # -> (4, 1568)
        x = self.fc1(x)         # -> (4, 64)
        x = self.relu3(x)
        x = self.fc2(x)         # -> (4, 10)
        return x


if __name__ == "__main__":
    model = MnistCNN()
    print(f"{model}\n")

    dummy = torch.zeros(4, 1, 28, 28)   # 4 Dummy MNIST images
    out = model(dummy)                  # Call the model with the dummy images
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")    