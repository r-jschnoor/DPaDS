# Source - https://stackoverflow.com/a/62458882
# Posted by stackoverflowuser2010, modified by community. See post 'Timeline' for change history
# Retrieved 2026-07-16, License - CC BY-SA 4.0

import torchvision
from torchviz import make_dot
import src.models.mnist_cnn as mnist_cnn
import torch

mnist_cnn.train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
dataloader_train = torch.utils.data.DataLoader(
    mnist_cnn.train_dataset, batch_size=4, shuffle=True
)
mnist_cnn.model = mnist_cnn.MnistCNN()
batch = next(iter(dataloader_train))
yhat = mnist_cnn.model(batch[0])  # Forward pass through the model
make_dot(yhat, params=dict(list(mnist_cnn.model.named_parameters()))).render("mnist_torchviz", format="png")

#export to onnx
torch.onnx.export(mnist_cnn.model, batch[0], "mnist.onnx", export_params=True)
