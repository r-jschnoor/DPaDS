import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.server.strategy import FedAvg
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import ray
import os
import warnings
import logging

from client import MnistClient
from mechanisms.attacks import LabelFlipClient
from mechanisms.robust_aggregation import FLTrustStrategy


# Env
# Suppress Ray's GPU override warning -> we are not using GPU via Ray
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Suppress the duplicate timestamp logging
logging.getLogger("flwr").propagate = False

# Suppress Flower deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# TODO Check if trust scores work. Label flipping trust is similar to no flipping trust!

# Globals (Config)
NUM_CLIENTS                 = 5
NUM_ROUNDS                  = 3
NUM_BYZANTINE_CLIENTS       = 1
USE_DP                      = False
EPSILON                     = 10.0
USE_FLTRUST                 = True
ROOT_DATASET_SIZE           = 2000   # Sample count the server holds
USE_RESCALE_TO_REF_NORM     = False  # Whether the server should normalize the clients gradients. Paper uses it but the available resources are not enough to mirror so its turned off for now. This is mostly to protect against malicious clients that send massive updates to gain an advantage to honest clients by size.


def load_datasets(root_dataset_size):
    """
    Loader MNIST train/test datasets and create the server root loader.

    Args:
        root_dataset_size (int): number of clean samples the server holds.

    Returns:
        tuple: (train_dataset, test_dataset, root_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="data/", train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="data/", train=False, download=True, transform=transform)
    root_subset   = Subset(train_dataset, list(range(root_dataset_size)))
    root_loader   = DataLoader(root_subset, batch_size=32, shuffle=True)

    return train_dataset, test_dataset, root_loader


def get_client_fn(train_dataset, test_dataset, num_clients,
                  num_byzantine=0, use_dp=False, epsilon=10.0):
    """
    Returns a function that creates a client with it own data slice.

    Flower calls this functions once per client per round. Each
    client gets a unique slice of the training data, simulating real 
    FL where each device has its own local dataset.

    The first num_byzantine clients are malicious LabelFlipClients,
    the rest are honest MnistClients.

    Args:
        train_dataset:           full MNIST training dataset.
        test_dataset:            full MNIST test dataset.
        num_clients (int):       total number of simulated clients.
        num_byzantine (int):     how many clients are malicious.
        use_dp (bool):           whether to wrap training with DP-SGD.
        epsilon (float):         privacy budget. Only used when use_dp=True.

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

        # First num_byzantine clients are malicious
        if client_id < num_byzantine:
            return LabelFlipClient(
                client_id, train_loader, test_loader,
                source_label=7, target_label=1,
            ).to_client()
        
        return MnistClient(
            client_id, train_loader, test_loader,
            use_dp=use_dp, epsilon=epsilon
        ).to_client()
    
    return client_fn


def get_server_fn(root_loader, use_rescale_to_ref_norm, use_fltrust, num_clients, num_rounds):
    """
    Return a function that creates a ServerComponents object.

    Args:
        root_loader (DataLoader): small clean server-held dataset.
        use_fltrust (bool):       whether to use FLTrust or plain FedAvg.
        num_clients (int):        total number of clients.
        num_rounds (int):         number of FL rounds.

    Returns:
        function: server_fn(context) -> ServerAppComponents
    """
    def server_fn(context):
        """
        Configure and return the server components.

        Args:
            context (flwr.common.Context): Flower server context.

        Returns:
            ServerAppComponents: strategy and config bundled for Flower.
        """
        if use_fltrust:
            aggregation_strategy = FLTrustStrategy(
                root_loader=root_loader,
                rescale_to_ref_norm=use_rescale_to_ref_norm,
                fraction_fit=1.0,
                min_available_clients=num_clients,
                fit_metrics_aggregation_fn=weighted_average_metrics,
                evaluate_metrics_aggregation_fn=weighted_average_metrics,
            )
        else:
            # FedAvg strategy -> Aggregates client weights by weighted average
            aggregation_strategy = FedAvg(
                fraction_fit=1.0,               # Use 100% of clients each round
                min_available_clients=num_clients,
                fit_metrics_aggregation_fn=weighted_average_metrics,
                evaluate_metrics_aggregation_fn=weighted_average_metrics,
            )

        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=aggregation_strategy, config=config)
    
    return server_fn


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
    train_dataset, test_dataset, root_loader = load_datasets(ROOT_DATASET_SIZE)

    server_app = ServerApp(server_fn=get_server_fn(root_loader, USE_RESCALE_TO_REF_NORM, USE_FLTRUST, NUM_CLIENTS, NUM_ROUNDS))
    client_app = ClientApp(
        client_fn=get_client_fn(
            train_dataset, test_dataset, NUM_CLIENTS,
            NUM_BYZANTINE_CLIENTS, use_dp=USE_DP,
            epsilon=EPSILON,
        )
    )

    # Run the simulation
    print("\nStarting simulation...")
    print(f"  Clients:   {NUM_CLIENTS} ({NUM_BYZANTINE_CLIENTS} malicious)")
    print(f"  Rounds:    {NUM_ROUNDS}")
    print(f"  FLTrust:   {USE_FLTRUST}")
    print(f"  DP:        {USE_DP} (epsilon={EPSILON})")

    history = fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
    )
    ray.shutdown()

    print("\n Simulation complete!")