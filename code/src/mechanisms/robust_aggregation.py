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


def get_reference_update(model, root_loader, optimizer, loss_fn):
    """
    Train the server model on root dataset for one step.

    Produces a reference gradient direction that honest client
    updates should roughly align with.

    Args:
        model (nn.Module):          server-side reference model.
        root_loader (DataLoader):   small clean server-held dataset.
        optimizer (torch.optim):    optimizer for the reference model.
        loss_fn (nn.Module):        loss function.

    Returns:
        np.ndarray: flat vector of parameter updates (before - after training).
    """
    # Record weights before training
    before = np.concatenate([
        p.data.cpu().numpy().flatten()
        for p in model.parameters()
    ])

    # One training step on root data
    model.train()
    for images, labels in root_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # Record weights after training
    after = np.concatenate([
        p.data.cpu().numpy().flatten()
        for p in model.parameters()
    ])

    # Updated difference (direction the model moved in)
    return after - before


class FLTrustStrategy(FedAvg):
    """
    FLTrust Byzantine-robus aggregation strategy for Flower.

    Extends FedAvg by replacing the simple average with a trust-
    weighted average. Each client update is weighted by its cosine
    similarity with a server-computed reference update.
    Updates pointing away from the reference direction get lower
    weights until eventually zero weight (ReLU clip).

    The server maintains a small, clean root dataset and trains
    a reference model on it each round to produce the reference
    update.

    Args:
        root_loader (DataLoader):   small clean server-held dataset.
        **kwargs:                   passed through to FedAvg (e.g.
                                    fraction_fit, min_available_clients).
    """

    def  __init__(self, root_loader, rescale_to_ref_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.root_loader = root_loader
        self.rescale_to_ref_norm=False
        self.ref_model = MnistCNN()
        self.ref_optimizer = torch.optim.SGD(self.ref_model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()


    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate client updates using FLTrust scores.

        Args:
            server_round (int):   current round number.
            results (list):       list of (client, FitRes) from each client.
            failures (list):      list of failed clients.

        Returns:
            tuple: (aggregated_parameters, metrics_dict)
        """
        if not results:
            return None, {}
        
        # Sync reference model and current global model so that
        # the server and clients start from the same point in 
        # each round
        _, first_fit_result = results[0]
        global_parameters = parameters_to_ndarrays(first_fit_result.parameters)
        parameters_dict = zip(self.ref_model.state_dict().keys(), global_parameters)
        state_dict = {k: torch.tensor(v) for k, v in parameters_dict}
        self.ref_model.load_state_dict(state_dict, strict=True)
        # Reset optimizer as per paper
        self.ref_optimizer = torch.optim.SGD(self.ref_model.parameters(), lr=0.01)

        # Get reference update from servers root dataset
        reference_update = get_reference_update(
            self.ref_model, self.root_loader,
            self.ref_optimizer, self.loss_fn
        )    

        # Extract client updates as flat vectors
        client_updates = []
        num_samples = []        # TODO Unused for now but might add to analysis later

        for _, fit_result in results:
            parameters = parameters_to_ndarrays(fit_result.parameters)
            # Flatten all parameters into one vector
            flattened = np.concatenate([p.flatten() for p in parameters])
            client_updates.append(flattened)
            num_samples.append(fit_result.num_examples)

        # Compute trust scores
        trust_scores = []
        for update in client_updates:
            similarity = cosine_similarity(update, reference_update)
            score = max(0.0, similarity)        # Clip after ReLU (negative similarity = zero trust)
            trust_scores.append(score)

        print(f"\n[FLTrust Round {server_round}] Trust scores: {[f'{s:.3f}' for s in trust_scores]}")

        # If all trust scores are zero (hence we expect all clients to be 
        # malicious) we skip this round.
        total_trust = sum(trust_scores)
        if total_trust == 0:
            print("[FLTrust] WARNING: All trust scores are zero -> Skipping round!")
            # Return unchanged global model
            reference_parameters = [p.data.cpu().numpy() for p in self.ref_model.parameters()]
            return ndarrays_to_parameters(reference_parameters), {}

        # Scale each update to reference norm and then apply a
        # weighted average
        reference_norm = np.linalg.norm(reference_update)
        aggregated_norms = np.zeros_like(reference_update)
        client_norm = np.linalg.norm(client_updates[0])
        print(f"  ref_norm={reference_norm:.6f}, client_norm={client_norm:.6f}, ratio={reference_norm/client_norm:.6f}")


        for update, score in zip(client_updates, trust_scores):
            update_norm = np.linalg.norm(update)

            if self.rescale_to_ref_norm and update_norm > 0:
                # This is the papers original normalization which combats large
                # Updates by normalization. This does not work in a small environment
                # which is why this is turned off (parameterized) until enough
                # ressources are available.
                scale = update * (reference_norm / update_norm)
            else:
                # No rescaling
                scale = update
            aggregated_norms += (score / total_trust) * scale

        # Convert back to Flower parameter format
        _, first_fit_result = results[0]
        original_parameters = parameters_to_ndarrays(first_fit_result.parameters)
        shapes = [p.shape for p in original_parameters]

        # Split flat, aggregated parameter vector back into per-layer arrays
        new_parameters = []
        index = 0
        for shape in shapes:
            size = int(np.prod(shape))
            new_parameters.append(aggregated_norms[index:index+size].reshape(shape))
            index += size
        
        return ndarrays_to_parameters(new_parameters), {}


if __name__ == "__main__":
    # Quick Test
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([-1.0, 0.0, 0.0])
    d = np.array([0.0, 1.0, 0.0])

    print(f"Identical vectors:     {cosine_similarity(a, b):.2f}")  # 1.0
    print(f"Opposite vectors:      {cosine_similarity(a, c):.2f}")  # -1.0
    print(f"Perpendicular vectors: {cosine_similarity(a, d):.2f}")  # 0.0

    # Check if reference_update works
    from torch.utils.data import TensorDataset, DataLoader
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model = MnistCNN()
    opti = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    ref = get_reference_update(model, loader, opti, loss_fn)
    print(f"\nReference update shape: {ref.shape}")
    print(f"Reference update norm: {np.linalg.norm(ref):.4f}") # How far did server model move
    print(f"Reference update is non-zero: {np.any(ref != 0)}")

    # FLTrust strategy test
    # -> Two honest clients and one malicious client
    from flwr.common.typing import FitRes, Status, Code

    ref_model = MnistCNN()
    ref_parameters = [p.data.cpu().numpy() for p in ref_model.parameters()]
    shapes = [p.shape for p in ref_parameters]
    total_size = sum(int(np.prod(s)) for s in shapes)

    def make_fit_result(update_flat):
        """
        Helper to reconstruc parameter list from flat vector
        """
        params = []
        index = 0
        for shape in shapes:
            size = int(np.prod(shape))
            params.append(update_flat[index:index+size].reshape(shape))
            index += size
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(params),
            num_examples=100,
            metrics={},
        )
    
    # honest clients (small change in direction)
    honest_update1 = ref * 0.9
    noise = np.random.randn(*ref.shape) * 0.01
    honest_update2 = ref + noise

    # malicious client (large change in direction)
    malicious_update = ref * -5.0       # Oposite direction + large magnitude

    fake_results = [
        (None, make_fit_result(honest_update1)),
        (None, make_fit_result(honest_update2)),
        (None, make_fit_result(malicious_update)),
    ]

    strategy = FLTrustStrategy(
        root_loader=loader,
        fraction_fit=1.0,
        min_available_clients=3,
    )

    aggregation_parameters, _ = strategy.aggregate_fit(
        server_round=1,
        results=fake_results,
        failures=[],
    )

    print("\nAggregation successful!")
    print("Expected results are that the first and second value should be positive between 1 and 0 " \
    "and the last one should be 0 since it is strongly malicious.")
    print(f"Returned parameters: {len(parameters_to_ndarrays(aggregation_parameters))} arrays.")
