import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from constants import DATA_ROOT
from models.mnist_cnn import MnistCNN
import torch.nn as nn

def initialize_data():
    # Convert images to tensors and normalize pixel values to [-1, 1]
    # Gradients tend to perform better when centered around 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))        # output = (input - mean) / std | mean, std = 0.5
    ])

    # Download and load MNIST
    train_dataset = datasets.MNIST(
        root=DATA_ROOT,                 # Path to save dataset to
        train=True,                     # Data for training
        download=True,
        transform=transform,            # Transform with predefined transform function
    )
    test_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    # DataLoader batches the data and shuffles it
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,                  # parameter choice. 32 is seemingly often used.
                                        # Smaller Batch: noisier gradient, slower, sometimes trains better
                                        # Larger Batch: Smoother gradient, faster, needs more memory
        shuffle=True,                   # Prevent the model from memorizing any order
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    print(f"Training samples:  {len(train_dataset)}")
    print(f"Test samples:      {len(test_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    print(f"\n{'-'*5} Peek at one batch {'-'*5}")
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")     # (32, 1, 28, 28)
    print(f"Batch label shape: {labels.shape}")     # (32, )
    print(f"First few labels:  {labels[:8]}")

    return train_loader, test_loader


def train_model(model, train_loader, test_loader):
    # Setup
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )
    loss_fn = nn.CrossEntropyLoss()

    NUM_EPOCHS = 5

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()                       # Put model in training mode
        running_loss = 0.0                  # Current loss in this epoch

        for batch_id, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)         # Forward pass
            loss = loss_fn(outputs, labels) # Compute loss      (Expects arguments because we use CE Loss here instead of MSE Loss before)
            loss.backward()                 # Backward pass
            optimizer.step()                # Update weights

            running_loss += loss.item()     # Update running loss

        avg_loss = running_loss / len(train_loader) # Average loss in this epoch

        # --- Evaluation ---        (done here to spot overfitting early and improve model)
        accuracy = evaluate_model(model, test_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Loss: {avg_loss:.4f} | Test Accuracy: {accuracy*100:.4f}%")



def evaluate_model(model, test_loader):
    model.eval()                        # switches of dropout and other stuff
    correct = 0
    total = 0

    with torch.no_grad():               # Dont track gradients during evaluation
        for images, labels in test_loader:
            outputs = model(images)     # (32, 10) -> 10 scores per image
            predicted = outputs.argmax(dim=1)   # pick the highest | dim is the dimension in the tensor to apply the argmax to. In this case its the 10 output classes. dim=0 would do it over the 32 batched images in this case.
            correct += (predicted == labels).sum().item()
            total += labels.size(0)         # Returns the size of dimension arg=0. .size(1) would return the size of dimension 1.
    
    return correct / total              # Accuracy


if __name__ == '__main__':
    model = MnistCNN()
    train_loader, test_loader = initialize_data()
    train_model(model, train_loader, test_loader)
    accuracy = evaluate_model(model, test_loader)
    print(f"Test accuracy after training: {accuracy * 100:.4f}%")