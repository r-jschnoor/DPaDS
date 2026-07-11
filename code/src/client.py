import random
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import flwr as fl
import warnings

from src.constants import ACCURACY_KEY
from src.mechanisms.topk import topk_sparsify
from src.models import get_dataset_spec
from src.mechanisms.dp import compute_noise_multiplier, make_private, get_privacy_spent, make_private_with_noise_multiplier, restore_accountant_state, serialize_accountant_state

warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
warnings.filterwarnings("ignore", message="Full backward hook")


def set_seed(seed):
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Each simulated client runs in its own Ray worker process, so seeding
    only the main process isn't enough. This must be called inside each
    client's own process to take effect there.

    Args:
        seed (int): seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iterate_batches(loader, num_steps):
    """
    Yield exactly num_steps batches from loader, cycling back to the start
    (with a fresh shuffle) if num_steps exceeds one epoch's worth of batches.

    Args:
        loader (DataLoader): source of (images, labels) batches.
        num_steps (int):     how many batches to yield in total.

    Returns:
        Iterator yielding num_steps (images, labels) batches.
    """
    it = iter(loader)
    for _ in range(num_steps):
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)
            yield next(it)


class MnistClient(fl.client.NumPyClient):
    """Flower client wrapping our MNIST training loop."""

    def __init__(self, client_id, train_loader, test_loader,
                 use_dp=False, epsilon=10.0, delta=1e-5,
                 use_topk=False, topk_ratio=0.1, num_rounds=1, seed=None,
                 num_client_iterations_per_round=None,
                 dataset_spec=get_dataset_spec("mnist"), device=torch.device("cpu")):
        self.client_id = client_id
        self.seed = seed
        if seed is not None:
            # Deterministic starting point for this client. Re-seeded per round in fit() for training randomness (data shuffling, DP noise).
            set_seed(seed + client_id)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dataset_spec = dataset_spec
        self.device = device
        self.model = dataset_spec.model_fn().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()
        self.use_dp = use_dp
        self.epsilon = epsilon
        self.delta = delta
        self.use_topk = use_topk
        self.topk_ratio = topk_ratio
        self.num_rounds = num_rounds
        self.num_client_iterations_per_round = num_client_iterations_per_round
        self.privacy_engine = None
        self.is_malicious = False

        if use_dp and train_loader is not None:
            # Matches Opacus's own internal Poisson-sampling rate (see DPDataLoader.from_data_loader)
            sample_rate = 1 / len(train_loader)
            self.noise_multiplier = compute_noise_multiplier(
                target_epsilon=epsilon,
                target_delta=delta,
                sample_rate=sample_rate,
                num_rounds=num_rounds,
            )
        else:
            self.noise_multiplier = None

    
    def get_parameters(self, config):
        """
        Extract the model weights as a list of numpy arrays (as used for Flower).
        PyTorch Tensors -> Numpy arrays

        Called by the Flower server to retrieve local model weights before
        aggregation.

        Args:
            config (dict): configuration dictionary sent by the server,
                        not used here but required by the Flower interface.

        Returns:
            list[np.ndarray]: ordered list of model weights and biases,
                            one array per layer, in state_dict() order.
        """
        return [val
                    .cpu()              # Transform values from gpu to cpu (numpy works only on cpu)
                    .numpy()            # Converts Pytorch Tensor to Numpy array
                for _, val 
                in self
                    .model
                    .state_dict()       # State dict = Learned weight-dict of every layer
                    .items()            # Gives Key-Value pairs
                ]
    

    def set_parameters(self, parameters):
        """
        Loads the weigths received from the server back into the model.
        Numpy arrays -> PyTorch Tensors

        Called by the Flower server to push updated global weights
        to the client before a new training round begins.

        Args:
            parameters (list[np.ndarray]): ordered list of model weights
                                        and biases in state_dict() order,
                                        as produced by get_parameters().

        Returns:
            None
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}       # Numpy array back to Tensor, straight onto this client's device
        self.model.load_state_dict(state_dict, strict=True)         # Reinserts the new weights into the model. strict=True means that no extra or missing values are allowed.


    def fit(self, parameters, config):
        """
        Train the model on local data for one round.

        Called by the Flower server at the start of each round. The server
        pushes the current global weights via parameters, we train locally,
        then return the updated weights back to the server.

        Args:
            parameters (list[np.ndarray]): global model weights from the server.
            config (dict): training configuration sent by the server (e.g. epochs).

        Returns:
            tuple: (updated_parameters, num_samples, metrics_dict)
                - updated_parameters: weights after local training
                - num_samples: how many samples we trained on
                - metrics_dict: any extra info we want to report to the server
        """
        # Load global weights into local model
        self.set_parameters(parameters)

        if self.seed is not None:
            # Re-seed per round so each round still gets a genuinely
            # different data shuffle / DP noise draw, not the same one
            # repeated every round.
            set_seed(self.seed + self.client_id * 1000 + config.get("server_round", 1))

        ACCOUNTANT_STATE_KEY = "accountant_state"

        # Recreate privacy engine
        if self.use_dp and self.noise_multiplier is not None:
            model_tmp = self.dataset_spec.model_fn().to(self.device)
            model_tmp.load_state_dict(self.model.state_dict())
            opt_tmp = torch.optim.SGD(model_tmp.parameters(), lr=0.01)
            model_tmp, opt_tmp, train_loader, self.privacy_engine = make_private_with_noise_multiplier(
                model_tmp, opt_tmp, self.train_loader,
                self.noise_multiplier, max_grad_norm=1.0,
            )
            # Restore accountant state
            if ACCOUNTANT_STATE_KEY in config:
                restore_accountant_state(self.privacy_engine, config[ACCOUNTANT_STATE_KEY])
        else:
            model_tmp = self.model
            opt_tmp = self.optimizer
            train_loader = self.train_loader

        # Training loop. None -> one full epoch. If set it does exactly that many SGD steps, matching the steps the FLTrust reference model takes this round
        model_tmp.train()
        num_steps = self.num_client_iterations_per_round or len(train_loader)

        for images, labels in iterate_batches(train_loader, num_steps):
            images, labels = images.to(self.device), labels.to(self.device)
            opt_tmp.zero_grad()
            outputs = model_tmp(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            opt_tmp.step()

        # Keep here since otherwise topk does not work -> DP Fltrust exists but has some security breach possiblity
        if self.use_dp and self.noise_multiplier is not None:
            self.model.load_state_dict(model_tmp._module.state_dict())
            
        updated_parameters = self.get_parameters(config={})
        metrics = {"is_malicious": float(self.is_malicious), "client_id": self.client_id}

        if self.use_topk:
            # Compute update -> weights now - global weights
            flat_before = np.concatenate([p.flatten() for p in parameters])         # Global model weights before training
            flat_after = np.concatenate([p.flatten() for p in updated_parameters])  # Local model weights after training
            update = flat_after - flat_before                                       # How muhc did the weights change?

            sparsified_update = topk_sparsify(update, self.topk_ratio)              # Zero out everything except top-k values

            # Reconstruct parameter list
            sparse_parameters = flat_before + sparsified_update                     # Applys update to origin vector (zeroed out values provide + 0.0 -> no update at all)
            shapes = [p.shape for p in updated_parameters]
            updated_parameters = []
            index = 0
            # TODO extract this restoring of original shape into helper method
            for shape in shapes:
                size = int(np.prod(shape))
                updated_parameters.append(sparse_parameters[index:index+size].reshape(shape))
                index += size

            sparsity = np.count_nonzero(sparsified_update) / len(sparsified_update)
            metrics["topk_sparsity"] = sparsity


        if self.use_dp and self.privacy_engine is not None:
            metrics["epsilon"] = get_privacy_spent(self.privacy_engine, self.delta)
            metrics[ACCOUNTANT_STATE_KEY] = serialize_accountant_state(self.privacy_engine)
            metrics["noise_multiplier"] = self.noise_multiplier

        return updated_parameters, len(self.train_loader.dataset), metrics
    

    def evaluate(self, parameters, config):
        """
        Evaluate the model on local test data and compute confusion matrix.

        Called by the Flower server (possibly at the end of each round) to measure
        how well the global model performs on each client's local test data.
        Returns per-class prediction counts as confusion metrics metrics which
        the server may aggregate into a full confusion matrix 

        Args:
            parameters (list[np.ndarray]): global model weights from the server.
            config (dict):                 evaluation configuration sent by the server.

        Returns:
            tuple: (loss, num_samples, metrics_dict)
                - loss:         average loss over the test set (required by Flower)
                - num_samples:  how many samples we evaluated on
                - metrics_dict: accuracy + confusion matrix entries as cm_i_j scalars
                                where i=true label, j=predicted label
        """
        # Load global weights into local model
        self.set_parameters(parameters)

        # Setup
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        # confusion matrix: num_classes x num_classes
        num_classes = self.dataset_spec.num_classes
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

        # Evaluation
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                total_loss += self.loss_fn(outputs, labels).item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                for true, pred in zip(labels.tolist(), predicted.tolist()):
                    confusion_matrix[true][pred] += 1
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)

        # Flatten confusion matrix for transmission
        metrics = {ACCURACY_KEY: accuracy}
        for i in range(num_classes):
            for j in range(num_classes):
                metrics[f"cm_{i}_{j}"] = float(confusion_matrix[i][j])

        return avg_loss, total, metrics



if __name__ == "__main__":
    client = MnistClient(
        client_id=0,
        train_loader=None,
        test_loader=None,
    )
    print("Client create successfully!")
    print(f"Model parameters: {sum(p.numel() for p in client.model.parameters()):,}")