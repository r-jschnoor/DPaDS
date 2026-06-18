from opacus import PrivacyEngine
import torch


def make_private(model,
                 optimizer,
                 data_loader,
                 target_epsilon,
                 target_delta,
                 max_grad_norm,
                 epochs):
    """
    Wrap model, optimizer and dataloader with Opacus DP-SGD.

    Opacus replaces the standard optimizer and dataloader with DP-aware
    versions that clip per-sample gradients and add calibrated noise.

    Args:
        model (nn.Module):          the model to make private.
        optimizer (torch.optim):    the optimizer to wrap.
        data_loader (DataLoader):   the training dataloader to wrap.
        target_epsilon (float):     privacy budget. Lower = more private.
        target_delta (float):       privacy failure probability. Usually 1e-5.
        max_grad_norm (float):      clipping threshold for per-sample gradients.
        epochs (int):               number of training epochs planned.

    Returns:
        tuple: (private_model, private_optimizer, private_loader, privacy_engine)
    """
    privacy_engine = PrivacyEngine()

    private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
    )

    return private_model, private_optimizer, private_loader, privacy_engine


def get_privacy_spent(privacy_engine, delta):
    """
    Report the privacy budget consumed so far.

    Args:
        privacy_engine (PrivacyEngine): the Opacus engine used during training.
        delta (float):                  target delta, usually 1e-5.

    Returns:
        float: epsilon spent so far.
    """
    return privacy_engine.get_epsilon(delta=delta)



if __name__ == "__main__":
    # Imports here since they are only needed for testing and not if run in larger scope
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.mnist_cnn import MnistCNN

    # Dummy data to test wrapping
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    model = MnistCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    private_model, private_opt, private_loader, engine = make_private(
        model=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=10.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        epochs=1,
    )

    print("Model wrapped successfully!")
    print(f"Type of optimizer:  {type(private_opt).__name__}")
    print(f"Type of loader:     {type(private_loader).__name__}")

    # Run one batch through to test
    for images, labels in private_loader:
        private_opt.zero_grad()
        loss = nn.functional.cross_entropy(private_model(images), labels)
        loss.backward()
        private_opt.step()
        break

    epsilon = get_privacy_spent(engine, delta=1e-5)
    print(f"Privacy spent after one batch: epsilon = {epsilon:.4f}")