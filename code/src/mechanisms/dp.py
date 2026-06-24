import json
import numpy as np
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
import torch


def serialize_accountant_state(privacy_engine):
    """
    Serializes the accountant's state as a JSON string.

    Uses the accountant's built-in state_dict.

    Args:
        privacy_engine (PrivacyEngine): the Opacus engine after training.

    Returns:
        str: JSON string of accountant state_dict.
    """
    return json.dumps(privacy_engine.accountant.state_dict())


def restore_accountant_state(privacy_engine, accountant_state):
    """
    Restores an accounant's state from a JSON string into
    a fresh engine.

    Args:
        privacy_engine (PrivacyEngine): freshly created engine to restore into.
        accountant_state (str):         JSON string containing the state_dict of an accountant

    Returns:
        None: modifies privacy_engine.accountant.steps in place.
    """
    state = json.loads(accountant_state)
    state["history"] = [tuple(h) for h in state["history"]]
    privacy_engine.accountant.load_state_dict(state)


def make_private(model,
                 optimizer,
                 data_loader,
                 target_epsilon,
                 target_delta,
                 max_grad_norm,
                 num_rounds):
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
        num_rounds (int):           number of training rounds (each with 1 epoch) planned.

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
        epochs=num_rounds,
        accountant="rdp",
    )

    return private_model, private_optimizer, private_loader, privacy_engine


def make_private_with_noise_multiplier(model, optimizer, data_loader, noise_multiplier,
                                       max_grad_norm):
    """
    Wrap model, optimizer and dataloader with Opacus DP-SGD.

    Uses a fixed noise multiplier that calculates the privacy spent
    based on the current step count, instead of using target epsilon, so that
    the accountant state can be persisted and restored across FL rounds.

    Args:
        model (nn.Module):          the model to make private.
        optimizer (torch.optim):    the optimizer to wrap.
        data_loader (DataLoader):   the training dataloader to wrap.
        noise_multiplier (float):   noise multiplier, computed once upfront.
        max_grad_norm (float):      clipping threshold for per-sample gradients.

    Returns:
        tuple: (private_model, private_optimizer, private_loader, privacy_engine)
    """
    privacy_engine = PrivacyEngine(accountant="rdp")
    private_model, private_optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
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


def compute_noise_multiplier(target_epsilon, target_delta, sample_rate, num_rounds):
    """
    Computes the noise multiplier to achieve the target epsilon.

    Calculated once at the beginning and is then reused across all
    rounds. This keeps the noise level constant while the accountant
    accumulates steps.

    Args:
        target_epsilon (float):  desired total privacy budget.
        target_delta (float):    privacy failure probability.
        sample_rate (float):     batch_size / dataset_size.
        num_rounds (int):        total number of FL rounds.

    Returns:
        float: noise multiplier to pass to make_private.
    """
    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=num_rounds,
    )


if __name__ == "__main__":
    # Imports here since they are only needed for testing and not if run in larger scope
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
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
        num_rounds=1,
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