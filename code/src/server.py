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

from src.client import MnistClient
from src.experiment_config import ExperimentConfig
from src.mechanisms.attacks import LabelFlipClient
from src.mechanisms.robust_aggregation import FLTrustStrategy


# Env
# Suppress Ray's GPU override warning -> we are not using GPU via Ray
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Suppress the duplicate timestamp logging
logging.getLogger("flwr").propagate = False

# Suppress Flower deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# TODO Check if trust scores work. Label flipping trust is similar to no flipping trust!
# TODO These globals are overriden in the experiment runs -> Remove or relocate!

# Globals (Config)
NUM_CLIENTS                 = 5
NUM_ROUNDS                  = 3
NUM_BYZANTINE_CLIENTS       = 1
USE_DP                      = False
EPSILON                     = 10.0
USE_FLTRUST                 = True
ROOT_DATASET_SIZE           = 2000   # Sample count the server holds
USE_RESCALE_TO_REF_NORM     = False  # Whether the server should normalize the clients gradients. Paper uses it but the available resources are not enough to mirror so its turned off for now. This is mostly to protect against malicious clients that send massive updates to gain an advantage to honest clients by size.
USE_TOPK                    = True
TOPK_RATIO                  = 0.1


class HistoryStrategyAdapter:
    """
    Adds per-round metric tracking to any Flower strategy.
    
    Add this as the first parent class to any strategy to enable tracking.
    The history dict is populated in-place during the simulation.

    Args:
        history (dict): shared dict to store per-round results.
    """

    def __init__(self, history, **kwargs):
        super().__init__(**kwargs)
        self.history = history

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if loss is not None:
            self.history["losses_distributed"].append((server_round, loss))
        if metrics:
            for key, val in metrics.items():
                self.history["metrics_distributed_evaluate"].setdefault(key, [])
                self.history["metrics_distributed_evaluate"][key].append((server_round, val))
        return loss, metrics

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if metrics:
            for key, val in metrics.items():
                self.history["metrics_distributed_fit"].setdefault(key, [])
                self.history["metrics_distributed_fit"][key].append((server_round, val))
        return params, metrics


class TrackingFedAvg(HistoryStrategyAdapter, FedAvg):
    """FedAvg with per-round history tracking."""
    pass

class TrackingFLTrust(HistoryStrategyAdapter, FLTrustStrategy):
    """FLTrust with per-round history tracking."""
    pass


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
                  num_byzantine=0, use_dp=False, epsilon=10.0,
                  use_topk=False, topk_ratio=0.1, num_rounds=1):
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
        use_topk (bool):         whether to use top-k algorithm or not.
        topk_ratio (float):      the ratio of kept top-k values.
        num_rounds (int):        number of training rounds (each with 1 epoch) planned

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
                use_dp=use_dp, epsilon=epsilon,
                use_topk=use_topk, topk_ratio=topk_ratio,
                num_rounds=num_rounds,
            ).to_client()
        
        return MnistClient(
            client_id, train_loader, test_loader,
            use_dp=use_dp, epsilon=epsilon,
            use_topk=use_topk, topk_ratio=topk_ratio,
            num_rounds=num_rounds,
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


def run_simulation_with_config(config: ExperimentConfig):
    """
    Run one FL simulation with the given experiment config.

    Args:
        config (ExperimentConfig): experiment configuration.

    Returns:
        flwr History object containing per-round metrics.
    """
    train_dataset, test_dataset, root_loader = load_datasets(config.root_dataset_size)

    # Track history
    run_history = {
        "losses_distributed": [],
        "metrics_distributed_evaluate": {},
        "metrics_distributed_fit": {},
    }

    # Strategies with tracking
    if config.use_fltrust:
        strategy = TrackingFLTrust(
            history=run_history,
            root_loader=root_loader,
            rescale_to_ref_norm=config.rescale_to_ref_norm,
            fraction_fit=1.0,
            min_available_clients=config.num_clients,
            fit_metrics_aggregation_fn=weighted_average_metrics,
            evaluate_metrics_aggregation_fn=weighted_average_metrics,
        )
    else:
        strategy = TrackingFedAvg(
            history=run_history,
            fraction_fit=1.0,
            min_available_clients=config.num_clients,
            fit_metrics_aggregation_fn=weighted_average_metrics,
            evaluate_metrics_aggregation_fn=weighted_average_metrics,
        )

    # server_fn that uses custom tracking strategy
    def server_fn(context):
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=config.num_rounds),
        )


    client_app = ClientApp(
        client_fn=get_client_fn(
            train_dataset, test_dataset,
            config.num_clients, config.num_byzantine,
            use_dp=config.use_dp, epsilon=config.epsilon,
            use_topk=config.use_topk, topk_ratio=config.topk_ratio,
            num_rounds=config.num_rounds
        )
    )
    server_app = ServerApp(server_fn=server_fn)

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config.num_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )

    ray.shutdown()
    return run_history


if __name__ == "__main__":
    train_dataset, test_dataset, root_loader = load_datasets(ROOT_DATASET_SIZE)

    server_app = ServerApp(server_fn=get_server_fn(root_loader, USE_RESCALE_TO_REF_NORM, USE_FLTRUST, NUM_CLIENTS, NUM_ROUNDS))
    client_app = ClientApp(
        client_fn=get_client_fn(
            train_dataset, test_dataset, NUM_CLIENTS,
            NUM_BYZANTINE_CLIENTS, use_dp=USE_DP,
            epsilon=EPSILON, use_topk=USE_TOPK,
            topk_ratio=TOPK_RATIO,
        )
    )

    # Run the simulation
    print("\nStarting simulation...")
    print(f"  Clients:   {NUM_CLIENTS} ({NUM_BYZANTINE_CLIENTS} malicious)")
    print(f"  Rounds:    {NUM_ROUNDS}")
    print(f"  FLTrust:   {USE_FLTRUST} (root-size={ROOT_DATASET_SIZE})")
    print(f"  DP:        {USE_DP} (epsilon={EPSILON})")
    print(f"  TOP_K:     {USE_TOPK} (k={TOPK_RATIO})")

    history = fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
    )
    ray.shutdown()

    print("\n Simulation complete!")