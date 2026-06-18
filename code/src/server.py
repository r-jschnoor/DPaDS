import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.server.strategy import FedAvg
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import ray
import os
import warnings
import logging

from client import MnistClient


# Env
# Suppress Ray's GPU override warning -> we are not using GPU via Ray
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Suppress the duplicate timestamp logging
logging.getLogger("flwr").propagate = False

# Suppress Flower deprecation warning
warnings.filterwarnings("ignore", message="DEPRECATED")


# Globals
NUM_CLIENTS = 5
NUM_ROUNDS = 3
USE_DP = True
EPSILON = 1.0


def get_client_fn(train_dataset, test_dataset, num_clients, use_dp=False, epsilon=10.0):
    """
    Returns a function that creates a client with it own data slice.

    Flower calls this functions once per client per round. Each
    client gets a unique slice of the training data, simulating real 
    FL where each device has its own local dataset.

    Args:
        train_dataset:       full MNIST training dataset.
        test_dataset:        full MNIST test dataset.
        num_clients (int):   total number of simulated clients.
        use_dp (bool):       whether to wrap training with Opacus DP-SGD.
        epsilon (float):     privacy budget target. Only used when use_dp=True.

    Returns:
        function: client_fn(context) -> flwr.client.Client
    
    """
    def client_fn(context):
        """
        Create a Flower client for a specific node.

        Called by Flower once per client per round.
        Uses node-id from contex to assign each cliend a unique slice
        of data.

        Args:
            context (flwr.common.Context): Flower context object containing
                                        node_id and other runtime info.
                                        Provided automatically by Flower.

        Returns:
            flwr.client.Client: configured MnistClient for this node.
        """
        # Give each client an equal slice of the training data
        client_id = int(context.node_id) % num_clients
        total = len(train_dataset)
        slice_size = total // num_clients
        start = client_id * slice_size
        end = start + slice_size

        client_train = Subset(train_dataset, list(range(start, end)))
        train_loader = DataLoader(client_train, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return MnistClient(
            client_id, train_loader, test_loader,
            use_dp=use_dp, epsilon=epsilon
        ).to_client()
    
    return client_fn


def server_fn(context):
    """
    Configure and return the server components.

    Args:
        context (flwr.common.Context): Flower server context.

    Returns:
        ServerAppComponents: strategy and config bundled for Flower.
    """
    # FedAvg strategy -> Aggregates client weights by weighted average
    aggregation_strategy = FedAvg(
        fraction_fit=1.0,               # Use 100% of clients each round
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=aggregation_strategy, config=config)


def weighted_average_metrics(metrics):
    """
    Aggregate metrix from all clients by weighted average.

    Args:
        metrics (list[tuple[int, dict]]): list of (num_samples, metrics_dict)
                                          from each client.

    Returns:
        dict: aggregated metrics weighted by number of samples.
    """
    total_samples = sum(num_samples for num_samples, _ in metrics)
    aggregated = {}

    # Extract all metric keys from the first client (all clients have equal metrics anyways)
    for key in metrics[0][1].keys():
        aggregated[key] = sum(
            num_samples * metric[key] for num_samples, metric in metrics
        ) / total_samples

    return aggregated


if __name__ == "__main__":
    # Load data once and then share only references
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="data/", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data/", train=False, download=True, transform=transform)

    server_app = ServerApp(server_fn=server_fn)
    client_app = ClientApp(
        client_fn=get_client_fn(
            train_dataset, test_dataset, NUM_CLIENTS,
            use_dp=USE_DP, epsilon=EPSILON
            )
    )

    # Run the simulation
    print("\nStarting simulation...")
    history = fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
    )
    ray.shutdown()

    print("\n Simulation complete!")