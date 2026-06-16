import torch
import torch.nn as nn
import numpy as np


def test_setup():
    x = torch.rand(5, 3)
    print(x)

    print(torch.cuda.is_available())

    print(torch.__version__)


def some_tests():
    # 1D Tensor -> like a list
    print(f"\n{"-"*10} 1D Tensor {"-"*10}")
    a = torch.tensor([1.0, 2.0, 3.0])
    print(a)
    print(a.shape)      # How many elements in each dimension
    print(a.dtype)      # Type of numbers

    # 2D Tensor -> like a matrix
    print(f"\n{"-"*10} 2D Tensor {"-"*10}")
    b = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    print(b)
    print(b.shape)

    # Some math
    print(f"\n{"-"*10} Basic Math {"-"*10}")
    c = torch.tensor([2.0, 3.0, 4.0])
    d = torch.tensor([2.0, 2.0, 2.0])

    print(c + d)        # Element by element addition
    print(c * d)        # Element by element multiplication
    print(c ** 2)       # Square every element

    # Matrix multiplication
    print(f"\n{"-"*10} Matrix Multiplication {"-"*10}")
    e = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    f = torch.tensor([
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 1.0],
    ])
    result = f @ e      # @ is matrix multiplication
    print(result)
    print(result.shape)
    
    print("This is what happens in the nn.Linear layer in Pytorch. It multiplies by a weight matrix, collapsing dimensions. E.g.: (batch, 784) @ (784, 64) -> (batch, 64). This collapsed the middle '784' dimension.")

    # Gradients
    print(f"\n{"-"*10} Gradients {"-"*10}")
    print("Explanation: Gradients track how much each weight contributed to the error.")
    print("Derivative: Answers the question 'If I change x, how much does y change?'")
    
    x = torch.tensor([2.0], requires_grad=True) # requires_grad=True makes PyTorch track all operations on this tensor
    simple_func = x ** 2 + 3
    print(f"simple_function = {simple_func}")
    simple_func.backward()  # Compute gradients -> dy/dx = 2x, so for x=2 -> 4.0 is expected
    print(f"Gradient dy/dx = {x.grad}")

    ## Manual Gradient computation
    print(f"\n{"-"*5} Manual computation {"-"*5}")
    x_val = 2.0
    change = 0.001
    y_before = x_val ** 2 + 3
    y_after = (x_val + change) ** 2 + 3

    change_in_y = y_after - y_before
    change_in_x = change

    print(f"y before tiny change: {y_before}")
    print(f"y after tiny change:  {y_after}")
    print(f"change in y:          {change_in_y}")
    print(f"ratio (change_in_y/change_in_x): {change_in_y/change_in_x}")

    print(f"\nThis ratio of ~4.0 means that if I increase x by 1, y increases by roughly 4!")
    print(f"In the context of neural network training, x is a weight and y is the error (loss). The gradient gives the information 'if x is increased slightly, does the error go up or down, and by how much?'. Then the weight can be nudged in the opposite direction to reduce the error.")


def dummy_training():
    print(f"{'-'*10} A simple dummy training, learning that y = 2x. {'-'*10}")

    # A single randomly initialied, learnable weight    
    w = torch.tensor([0.5], requires_grad=True)

    learning_rate = 0.01
    epochs = 50

    for epoch in range(epochs):
        # "Dataset": x=3.0 -> Correct answer is y=6.0 (because y=2x)
        x = torch.tensor([3.0])
        y_true = torch.tensor([6.0])

        # Forward pass: make a prediction
        y_pred = w * x

        # Compute the error (loss) -> how wrong are we?
        loss = (y_pred - y_true) ** 2       # Mean Squared Error

        # Backward pass -> Compute gradients
        loss.backward()

        # Update the weight -> nudge it in the direction that reduces loss based on the learning rate
        with torch.no_grad():       # Turn off gradient tracking during update  (No learning so this should not be tracked)
            w -= learning_rate * w.grad # w.grad is the direction and distance in which the loss goes up
            w.grad.zero_()          # Reset gradient so they do not accumulate
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} | w = {w.item():.4f} | loss = {loss.item():.4f}")


def dummy_training_with_torch():
    print(f"{'-'*10} A simple dummy training, learning that y = 2x. (Implemented with PyTorch) {'-'*10}")

    # nn.Linear(1,1) means -> 1 input, 1 output
    model = nn.Linear(in_features=1, out_features=1)

    print(f"initial weight: {model.weight.item()}")
    print(f"Initial Bias:   {model.bias.item()}\n")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    epochs = 200

    for epoch in range(epochs):
        # x = torch.tensor([[3.0]])       # Shape is (1, 1) -> one sample, one feature
        # y_true = torch.tensor([[6.0]])

        x      = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y_true = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

        # Forward pass
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y_true)

        # Backward pass
        optimizer.zero_grad()       # = w.grad.zero_()
        loss.backward()
        optimizer.step()            # = w -= learning_rate * w.grad

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} | w = {model.weight.item():.4f} | b = {model.bias.item():.4f} | loss = {loss.item():.4f}")

    print("\nFinal weight:", model.weight.item())
    print("Final bias:  ", model.bias.item())

    # print(f"Check: weight * x + bias = {y_true.item()} -> {model.weight.item()} * {x.item()} + {model.bias.item()} = {model.weight.item() * x.item() + model.bias.item()}")

    # Run 1:
    # initial weight: -0.9632983207702637
    # Initial Bias:   0.8820693492889404

    # step  0 | w = -0.4828 | b = 1.0422 | loss = 64.1253
    # step 10 | w = 1.2327 | b = 1.6141 | loss = 0.7393
    # step 20 | w = 1.4169 | b = 1.6755 | loss = 0.0085
    # step 30 | w = 1.4367 | b = 1.6821 | loss = 0.0001
    # step 40 | w = 1.4388 | b = 1.6828 | loss = 0.0000

    # Final weight: 1.4390150308609009
    # Final bias:   1.6828404664993286

    # Run 2:
    # initial weight: -0.3560147285461426
    # Initial Bias:   -0.6162377595901489

    # step  0 | w = 0.1050 | b = -0.4626 | loss = 59.0482
    # step 10 | w = 1.7512 | b = 0.0862 | loss = 0.6808
    # step 20 | w = 1.9280 | b = 0.1451 | loss = 0.0078
    # step 30 | w = 1.9470 | b = 0.1514 | loss = 0.0001
    # step 40 | w = 1.9490 | b = 0.1521 | loss = 0.0000

    # Final weight: 1.9492368698120117
    # Final bias:   0.1521795094013214

    # In both runs, different weight and bias values come out. However, both are correct since weight * x + bias = 6
    # Since the training ran on only one datapoint, there are infinite solutions. To combat this, introduce multiple test datapoints.



if __name__ == "__main__":
    dummy_training_with_torch()