from re import U

from collections import Counter

import flwr as fl
from flwr.common import FitIns, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.server.strategy import FedAvg
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import ray
import torch
import os
import warnings
import logging

from src.client import MnistClient, set_seed
from src.constants import CLIENT_BATCH_SIZE, DATA_ROOT
from src.experiment_config import ExperimentConfig
from src.mechanisms.attacks import build_malicious_client
from src.mechanisms.robust_aggregation import FLTrustStrategy
from src.models import get_dataset_spec


# Env
# Suppress Ray's GPU override warning -> most configs (MNIST) use no GPU via
# Ray at all. CIFAR-10 configs claim one explicitly via resolve_device()
# below instead of through this override.
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Suppress the duplicate timestamp logging
logging.getLogger("flwr").propagate = False

# Suppress Flower deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def resolve_device(dataset: str, gpu_index: int | None = None) -> torch.device:
    """
    Resolve which device to train/evaluate on for this dataset.

    GPU is only used for CIFAR-10. Picks the physical GPU with the most
    free VRAM right now (or gpu_index if given) and restricts this whole
    process (and everything it spawns, e.g. Ray workers) to only that GPU
    via CUDA_VISIBLE_DEVICES. Call this once per script invocation, before
    the first run_simulation_with_config() call -- not once per config --
    and pass the result to every call, so a whole run consistently uses
    one GPU instead of possibly a different "best" one per config as load
    shifts on a shared multi-GPU machine.

    Args:
        dataset (str):           "mnist" or "cifar10".
        gpu_index (int | None): manually claim this physical GPU index.
                                None (default) auto-picks whichever GPU
                                currently has the most free VRAM.

    Returns:
        torch.device: "cuda" if a GPU was selected, otherwise "cpu" (either
                     dataset != "cifar10", or no GPU is available).
    """
    if dataset != "cifar10" or not torch.cuda.is_available():
        return torch.device("cpu")

    if gpu_index is None:
        gpu_index = max(
            range(torch.cuda.device_count()),
            key=lambda i: torch.cuda.mem_get_info(i)[0],
        )
    elif not 0 <= gpu_index < torch.cuda.device_count():
        raise ValueError(f"gpu_index={gpu_index} is out of range for {torch.cuda.device_count()} visible GPU(s)")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return torch.device("cuda")


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
SEED                         = 42     # Reproducible model init + data split. None for unseeded.


class HistoryStrategyAdapter:
    """
    Adds per-round metric tracking to any Flower strategy.
    
    Add this as the first parent class to any strategy to enable tracking.
    The history dict is populated in-place during the simulation.

    Args:
        history (dict): shared dict to store per-round results.
        accountant_manager (AccountantStateManager | None): manages per-client
            DP accountant state across rounds. Pass None if DP is disabled.
    """

    def __init__(self, history, accountant_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.history = history
        self.accountant_manager = accountant_manager

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if loss is not None:
            self.history["losses_distributed"].append((server_round, loss))
        if metrics:
            for key, val in metrics.items():
                self.history["metrics_distributed_evaluate"].setdefault(key, [])
                self.history["metrics_distributed_evaluate"][key].append((server_round, val))
        return loss, metrics

    def evaluate(self, server_round, parameters):
        """
        Record centralized (server-side) evaluation results.

        Called by Flower once per round when a strategy is built with
        evaluate_fn. Separate from aggregate_evaluate() above, which records federated (per-client)
        evaluation; the two write into the same history dict/keys so
        save_results() doesn't need to know which one ran.
        """
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None
        loss, metrics = result
        self.history["losses_distributed"].append((server_round, loss))
        for key, val in metrics.items():
            self.history["metrics_distributed_evaluate"].setdefault(key, [])
            self.history["metrics_distributed_evaluate"][key].append((server_round, val))
        return loss, metrics

    def aggregate_fit(self, server_round, results, failures):
        # First store accountant state per client
        if self.accountant_manager is not None:
            for client_proxy, fit_result in results:
                if "accountant_state" in fit_result.metrics:
                    self.accountant_manager.store(
                        client_proxy.node_id,
                        fit_result.metrics["accountant_state"]
                    )

        params, metrics = super().aggregate_fit(server_round, results, failures)

        # Per-client malicious flag, keyed by each client's own reported client_id
        malicious_metrics = {
            f"malicious_{int(fit_result.metrics['client_id'])}": float(fit_result.metrics.get("is_malicious", 0.0))
            for _, fit_result in results
            if "client_id" in fit_result.metrics
        }
        # Total communication volume this round -> summed, not weighted-averaged like every
        # other fit metric (weighted_average_metrics() excludes "update_bytes" from its own
        # averaging for exactly this reason), since a per-client average would understate how
        # much data actually crossed the wire in aggregate this round.
        total_update_bytes = sum(fit_result.metrics.get("update_bytes", 0) for _, fit_result in results)
        metrics = {**(metrics or {}), **malicious_metrics, "update_bytes": total_update_bytes}

        if metrics:
            for key, val in metrics.items():
                self.history["metrics_distributed_fit"].setdefault(key, [])
                self.history["metrics_distributed_fit"][key].append((server_round, val))
        return params, metrics
    
    def configure_fit(self, server_round, parameters, client_manager):
        """
        Override to send per-client accountant state (and always
        server_round, needed for each client's round-aware seeding) in
        config.
        """
        # Get default instructions
        instructions = super().configure_fit(server_round, parameters, client_manager)

        # Replace config for each client with accountant state of client
        # (or just server_round if DP/accountant tracking is disabled)
        updated = []
        for client_proxy, fit_ins in instructions:
            if self.accountant_manager is not None:
                config = self.accountant_manager.get_config(client_proxy.node_id, server_round)
            else:
                config = {"server_round": server_round}
            updated.append((client_proxy, FitIns(fit_ins.parameters, config)))

        return updated


class TrackingFedAvg(HistoryStrategyAdapter, FedAvg):
    """FedAvg with per-round history tracking."""
    pass


class TrackingFLTrust(HistoryStrategyAdapter, FLTrustStrategy):
    """FLTrust with per-round history tracking."""
    pass


class AccountantStateManager:
    """
    Stores and retrieves per-client accountant state across FL rounds.

    Since Flower recreates clients each round due to its nature, the server
    must hold the accountant state and pass it back to each client via
    a config.

    Args:
        None
    """

    def __init__(self):
        self._state = {}        # node_id -> accountant_state JSON string


    def store(self, node_id: int, accountant_state_json: str):
        """
        Store accountant state for a client.

        Args:
            node_id (int):    the raw Flower node_id.
            state_json (str): serialized accountant state.
        """
        self._state[node_id] = accountant_state_json


    def get_config(self, node_id: int, server_round: int) -> dict:
        """
        Build fit config for a client including its accountant's state.

        Args:
            node_id (int):       the raw Flower node_id.
            server_round (int): current round number.

        Returns:
            dict: config to pass to client's fit() via Flower.
        """
        config = {"server_round": server_round}
        if node_id in self._state:
            config["accountant_state"] = self._state[node_id]
        return config
    

    def get_state(self, node_id: int) -> str | None:
        """
        Get accountant state for a specific state.

        Args:
            node_id (int): the raw Flower node_id.

        Returns:
            str | None: serialized accountant state, or None if not yet stored.
        """
        return self._state.get(node_id)


def load_datasets(root_dataset_size, seed=None, dataset="mnist"):
    """
    Load train/test datasets and create the server root loader.

    The root set is a stratified sample (same class proportions as the
    full training set) so every label is represented, and it is disjoint
    from the indices clients draw from (see get_client_fn) so the server's
    "clean" data is never a subset of any client's local data.

    Args:
        root_dataset_size (int): number of clean samples the server holds.
        seed (int | None):       random seed for the root/client split. None
                                 keeps the unseeded (different every run)
                                 behavior; configs sharing the same seed get
                                 the same split.
        dataset (str):           "mnist" or "cifar10". Normalization is
                                 deliberately the simplified (0.5,...) form
                                 for both, rather than each dataset's "true"
                                 per-channel statistics

    Returns:
        tuple: (train_dataset, test_dataset, root_loader, client_pool_indices)
            - client_pool_indices (list[int]): train_dataset indices not
                                               used for the root set, for
                                               clients to slice from.
    """
    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True, transform=transform)
        test_dataset  = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root=DATA_ROOT, train=True,  download=True, transform=transform)
        test_dataset  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    # Stratified split -> root_indices has every label represented in the
    # same proportions as the full training set. client_pool_indices is
    # everything left over. targets works the same way for CIFAR-10
    # (plain list) as for MNIST (tensor).
    targets = train_dataset.targets
    if hasattr(targets, "tolist"):
        targets = targets.tolist()
    all_indices = list(range(len(train_dataset)))
    root_indices, client_pool_indices = train_test_split(
        all_indices, train_size=root_dataset_size, stratify=targets,
        random_state=seed,
    )

    root_subset = Subset(train_dataset, root_indices)
    # Matches CLIENT_BATCH_SIZE (The FLTrust paper's Algorithm 2 uses one shared batch size for both the client and server ModelUpdate() calls)
    root_loader = DataLoader(root_subset, batch_size=CLIENT_BATCH_SIZE, shuffle=True)

    return train_dataset, test_dataset, root_loader, client_pool_indices


def compute_label_distribution(train_dataset, client_pool_indices, root_indices, num_clients):
    """
    Count how many samples of each label the root set and each client hold.

    client_pool_indices is sliced the same way get_client_fn()'s client_fn()
    slices it (see slice_size/start/end there), so this reports exactly what
    each client actually trains on. Safe to call once per run -- the split
    itself never changes across rounds (see load_datasets()).

    Args:
        train_dataset:                   full MNIST training dataset.
        client_pool_indices (list[int]): train_dataset indices clients draw from.
        root_indices (list[int]):        train_dataset indices in the server's root set.
        num_clients (int):               total number of simulated clients.

    Returns:
        dict: {"root": {label: count}, "clients": {client_id: {label: count}}},
             with label keys as strings for JSON-friendliness.
    """
    targets = train_dataset.targets
    if hasattr(targets, "tolist"):
        targets = targets.tolist()

    def label_counts(indices):
        counts = Counter(str(targets[i]) for i in indices)
        return dict(counts)

    total = len(client_pool_indices)
    slice_size = total // num_clients

    clients = {}
    for client_id in range(num_clients):
        start = client_id * slice_size
        end = start + slice_size
        clients[str(client_id)] = label_counts(client_pool_indices[start:end])

    return {"root": label_counts(root_indices), "clients": clients}


def make_evaluate_fn(test_dataset, dataset_spec=get_dataset_spec("mnist"), device=torch.device("cpu")):
    """
    Build a centralized (server-side) evaluate_fn for a Flower strategy.

    Evaluates the global model once per round against the full test
    dataset, replacing federated (per-client) evaluation -- every client
    would otherwise evaluate on the same full test set against the same
    global model and produce identical results, num_clients times over
    for nothing. Reuses MnistClient.evaluate() via a throwaway client
    instead of duplicating its accuracy/confusion-matrix logic.

    Args:
        test_dataset:                full test dataset.
        dataset_spec (DatasetSpec): model factory + class count for this run's
                                    dataset. Defaults to MNIST.
        device (torch.device):      which device to evaluate on. Defaults to CPU.

    Returns:
        function: evaluate_fn(server_round, parameters, config) -> (loss, metrics),
                 the shape a Flower Strategy's evaluate_fn is expected to have.
    """
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    eval_client = MnistClient(client_id=-1, train_loader=None, test_loader=test_loader, dataset_spec=dataset_spec, device=device)

    def evaluate_fn(server_round, parameters, config):
        loss, _, metrics = eval_client.evaluate(parameters, {})
        return loss, metrics

    return evaluate_fn


def get_client_fn(train_dataset, client_pool_indices, num_clients,
                  num_byzantine=0, use_dp=False, epsilon=10.0, delta=1e-5,
                  use_topk=False, topk_ratio=0.1, num_rounds=1, seed=None,
                  attack_type="label_flip", attack_scale=None,
                  source_label=3, target_label=7,
                  num_client_iterations_per_round=None,
                  dataset_spec=get_dataset_spec("mnist"), device=torch.device("cpu")):
    """
    Returns a function that creates a client with it own data slice.

    Flower calls this functions once per client per round. Each
    client gets a unique slice of the training data, simulating real
    FL where each device has its own local dataset.

    The first num_byzantine clients are malicious, the rest are honest
    MnistClients. Evaluation is centralized on the server (see
    make_evaluate_fn), so clients don't need test data.

    Args:
        train_dataset:           full MNIST training dataset.
        client_pool_indices (list[int]): train_dataset indices clients may
                                         draw from (excludes the server's
                                         root dataset, see load_datasets).
        num_clients (int):       total number of simulated clients.
        num_byzantine (int):     how many clients are malicious.
        use_dp (bool):           whether to wrap training with DP-SGD.
        epsilon (float):         privacy budget. Only used when use_dp=True.
        delta (float):           privacy failure probability. Only used when use_dp=True.
        use_topk (bool):         whether to use top-k algorithm or not.
        topk_ratio (float):      the ratio of kept top-k values.
        num_rounds (int):        number of training rounds (each with 1 epoch) planned
        seed (int | None):       random seed for reproducible model init and per-round training
                                 randomness. None keeps the unseeded (different every run) behavior.
        attack_type (str):       which Byzantine attack malicious clients run: "label_flip" or
                                 "random_gradient".
        attack_scale (float | None): None or 1.0 for the base (unscaled) attack, otherwise the
                                     scale factor for the wrapped (Scaled*) variant.
        source_label (int):      the class index malicious clients relabel. Only used when
                                 attack_type="label_flip".
        target_label (int):      the class index malicious clients relabel it as. Only used when
                                 attack_type="label_flip".
        num_client_iterations_per_round (int | None): SGD steps to take per round instead of a
                                    full epoch. None keeps the default (1 full epoch).
        dataset_spec (DatasetSpec): model factory + class count for this run's dataset.
                                    Defaults to MNIST.
        device (torch.device):      which device clients train/evaluate on. Defaults to CPU.

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
                                        node_id, node_config, and other
                                        runtime info. Provided automatically
                                        by Flower.

        Returns:
            flwr.client.Client: configured MnistClient for this node.
        """
        # Give each client an equal slice of the training data. Use Flower's
        # own partition-id (stable, sequential, collision-free) rather than
        # node_id % num_clients -- node_id is an effectively-random large
        # integer, so taking it mod a small num_clients can collide (two
        # different physical clients landing on the same client_id, leaving
        # another client_id completely unused that run).
        client_id = context.node_config["partition-id"]
        total = len(client_pool_indices)
        slice_size = total // num_clients
        start = client_id * slice_size
        end = start + slice_size

        client_train = Subset(train_dataset, client_pool_indices[start:end])
        train_loader = DataLoader(client_train, batch_size=CLIENT_BATCH_SIZE, shuffle=True)
        # Evaluation is centralized on the server (see make_evaluate_fn) --
        # clients no longer need their own test_loader.

        # First num_byzantine clients are malicious
        if client_id < num_byzantine:
            return build_malicious_client(
                client_id, train_loader, test_loader=None,
                attack_type=attack_type, attack_scale=attack_scale,
                source_label=source_label, target_label=target_label,
                use_dp=use_dp, epsilon=epsilon, delta=delta,
                use_topk=use_topk, topk_ratio=topk_ratio,
                num_rounds=num_rounds, seed=seed,
                num_client_iterations_per_round=num_client_iterations_per_round,
                dataset_spec=dataset_spec, device=device,
            ).to_client()

        return MnistClient(
            client_id, train_loader, test_loader=None,
            use_dp=use_dp, epsilon=epsilon, delta=delta,
            use_topk=use_topk, topk_ratio=topk_ratio,
            num_rounds=num_rounds, seed=seed,
            num_client_iterations_per_round=num_client_iterations_per_round,
            dataset_spec=dataset_spec, device=device,
        ).to_client()
    
    return client_fn


def get_server_fn(root_loader, test_dataset, use_rescale_to_ref_norm, use_fltrust, num_clients, num_rounds,
                  dataset_spec=get_dataset_spec("mnist")):
    """
    Return a function that creates a ServerComponents object.

    Args:
        root_loader (DataLoader): small clean server-held dataset.
        test_dataset:             full test dataset, for centralized evaluation.
        use_fltrust (bool):       whether to use FLTrust or plain FedAvg.
        num_clients (int):        total number of clients.
        num_rounds (int):         number of FL rounds.
        dataset_spec (DatasetSpec): model factory + class count for this run's
                                    dataset. Defaults to MNIST.

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
        evaluate_fn = make_evaluate_fn(test_dataset, dataset_spec=dataset_spec)

        # Clients need server_round in their fit config for round-aware
        # seeding (see client.py set_seed usage).
        on_fit_config_fn = lambda server_round: {"server_round": server_round}

        if use_fltrust:
            aggregation_strategy = FLTrustStrategy(
                root_loader=root_loader,
                rescale_to_ref_norm=use_rescale_to_ref_norm,
                dataset_spec=dataset_spec,
                fraction_fit=1.0,
                fraction_evaluate=0.0,           # Skip federated eval -- evaluate_fn does it centrally instead
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=on_fit_config_fn,
                min_available_clients=num_clients,
                fit_metrics_aggregation_fn=weighted_average_metrics,
                evaluate_metrics_aggregation_fn=weighted_average_metrics,
            )
        else:
            # FedAvg strategy -> Aggregates client weights by weighted average
            aggregation_strategy = FedAvg(
                fraction_fit=1.0,               # Use 100% of clients each round
                fraction_evaluate=0.0,          # Skip federated eval -- evaluate_fn does it centrally instead
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=on_fit_config_fn,
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
    common_keys = set(metrics[0][1].keys())
    for _, metric in metrics[1:]:
        common_keys &= set(metric.keys())
    common_keys -= {"client_id"}     # per-client identifier, not meant to be averaged
    common_keys -= {"update_bytes"}  # summed (total per-round communication), not averaged -- see HistoryStrategyAdapter.aggregate_fit()

    for key in common_keys:
        # Need to skip non-numeric metrics like accountant_state
        try:
            aggregated[key] = sum(
                num_samples * metric[key] for num_samples, metric in metrics
            ) / total_samples
        except TypeError:
            continue

    return aggregated


def run_simulation_with_config(config: ExperimentConfig, max_cpus: int | None = None,
                               max_gpu_clients: int = 4, device: torch.device | None = None):
    """
    Run one FL simulation with the given experiment config.

    Args:
        config (ExperimentConfig): experiment configuration.
        max_cpus (int | None):     caps the total number of CPU cores Ray uses for this
                                   simulation, so
                                   at most max_cpus clients run concurrently regardless of
                                   how many cores the host has. None (default) leaves Ray's
                                   own auto-detection (all host cores) unchanged.
        max_gpu_clients (int):     when a GPU is used (see resolve_device() -- CIFAR-10
                                   only), caps how many simulated clients share it
                                   concurrently (Ray's fractional-GPU scheduling, via
                                   client_resources["num_gpus"] = 1/max_gpu_clients).
                                   A GPU doesn't parallelize across processes the way
                                   CPU cores do, so this bounds contention/VRAM use on
                                   the single physical GPU resolve_device() selects.
                                   Ignored entirely for MNIST (no GPU used at all).
        device (torch.device | None): device to train/evaluate on, pre-resolved via
                                   resolve_device() by the caller. Resolve once per script
                                   invocation (not once per config) so a whole sweep
                                   consistently uses the same GPU. None (default) resolves
                                   internally for standalone/ad-hoc calls.

    Returns:
        flwr History object containing per-round metrics.
    """
    if config.seed is not None:
        set_seed(config.seed)

    dataset_spec = get_dataset_spec(config.dataset)
    if device is None:
        device = resolve_device(config.dataset)

    train_dataset, test_dataset, root_loader, client_pool_indices = load_datasets(
        config.root_dataset_size, seed=config.seed, dataset=config.dataset,
    )

    # Track history
    run_history = {
        "losses_distributed": [],
        "metrics_distributed_evaluate": {},
        "metrics_distributed_fit": {},
        "label_distribution": compute_label_distribution(
            train_dataset, client_pool_indices,
            root_loader.dataset.indices, config.num_clients,
        ),
    }

    # Shared state manager to persist states across rounds
    accountant_manager = AccountantStateManager() if config.use_dp else None

    evaluate_fn = make_evaluate_fn(test_dataset, dataset_spec=dataset_spec, device=device)

    # Without a seed, Flower asks a random client for the initial global
    # model (Server._get_initial_parameters -> ClientManager.sample(),
    # which runs on a background thread and races with other threading, so
    # it can't be pinned down by seeding alone). With a seed, build the
    # initial model here instead so Flower skips that step entirely and
    # uses exactly this.
    initial_parameters = None
    if config.seed is not None:
        initial_ndarrays = [val.cpu().numpy() for _, val in dataset_spec.model_fn().state_dict().items()]
        initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Strategies with tracking
    if config.use_fltrust:
        strategy = TrackingFLTrust(
            history=run_history,
            accountant_manager=accountant_manager,
            root_loader=root_loader,
            rescale_to_ref_norm=config.rescale_to_ref_norm,
            num_client_iterations_per_round=config.num_client_iterations_per_round,
            dataset_spec=dataset_spec, device=device,
            fraction_fit=1.0,
            fraction_evaluate=0.0,           # Skip federated eval -- evaluate_fn does it centrally instead
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            min_available_clients=config.num_clients,
            fit_metrics_aggregation_fn=weighted_average_metrics,
            evaluate_metrics_aggregation_fn=weighted_average_metrics,
        )
    else:
        strategy = TrackingFedAvg(
            history=run_history,
            accountant_manager=accountant_manager,
            fraction_fit=1.0,
            fraction_evaluate=0.0,           # Skip federated eval -- evaluate_fn does it centrally instead
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
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
            train_dataset, client_pool_indices,
            config.num_clients, config.num_byzantine,
            use_dp=config.use_dp, epsilon=config.epsilon, delta=config.delta,
            use_topk=config.use_topk, topk_ratio=config.topk_ratio,
            num_rounds=config.num_rounds, seed=config.seed,
            attack_type=config.attack_type, attack_scale=config.attack_scale,
            source_label=config.source_label, target_label=config.target_label,
            num_client_iterations_per_round=config.num_client_iterations_per_round,
            dataset_spec=dataset_spec, device=device,
        )
    )
    server_app = ServerApp(server_fn=server_fn)

    init_args = {"log_to_driver": True}
    if max_cpus is not None:
        init_args["num_cpus"] = max_cpus

    # A GPU doesn't parallelize across processes the way CPU cores do so
    # cap how many simulated clients share the single physical GPU
    # resolve_device() selected, via Ray's fractional-GPU scheduling.
    client_num_gpus = 1 / max_gpu_clients if device.type == "cuda" else 0.0

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config.num_clients,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": client_num_gpus},
            "init_args": init_args,
        },
    )

    ray.shutdown()
    return run_history


if __name__ == "__main__":
    train_dataset, test_dataset, root_loader, client_pool_indices = load_datasets(ROOT_DATASET_SIZE, seed=SEED)

    server_app = ServerApp(server_fn=get_server_fn(root_loader, test_dataset, USE_RESCALE_TO_REF_NORM, USE_FLTRUST, NUM_CLIENTS, NUM_ROUNDS))
    client_app = ClientApp(
        client_fn=get_client_fn(
            train_dataset, client_pool_indices, NUM_CLIENTS,
            NUM_BYZANTINE_CLIENTS, use_dp=USE_DP,
            epsilon=EPSILON, use_topk=USE_TOPK,
            topk_ratio=TOPK_RATIO, seed=SEED,
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