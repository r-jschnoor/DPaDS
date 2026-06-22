import numpy as np
import torch
import torch.nn as nn
import flwr as fl
import warnings

from src.mechanisms.topk import topk_sparsify
from src.models.mnist_cnn import MnistCNN
from src.mechanisms.dp import make_private, get_privacy_spent

warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
warnings.filterwarnings("ignore", message="Full backward hook")


class MnistClient(fl.client.NumPyClient):
    """Flower client wrapping our MNIST training loop."""

    def __init__(self, client_id, train_loader, test_loader,
                 use_dp=False, epsilon=10.0, delta=1e-5,
                 use_topk=False, topk_ratio=0.1, num_rounds=1):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = MnistCNN()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()
        self.use_dp = use_dp
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_engine = None
        self.use_topk = use_topk
        self.topk_ratio = topk_ratio

        if use_dp and train_loader is not None:
            self.model, self.optimizer, self.train_loader, self.privacy_engine = make_private(
                model=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=1.0,
                num_rounds=num_rounds,
            )

    
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
        state_dict = {k: torch.tensor(v) for k, v in params_dict}       # Numpy array back to Tensor
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

        # Training loop
        self.model.train()
        for images, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

        metrics = {}
        if self.use_dp and self.privacy_engine is not None:
            epsilon = get_privacy_spent(self.privacy_engine, self.delta)
            metrics["epsilon"] = epsilon
            print(f"  [LOG DP] Client {self.client_id} | epsilon = {epsilon:.4f}")

        updated_parameters = self.get_parameters(config={})

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

        return updated_parameters, len(self.train_loader.dataset), metrics
    

    def evaluate(self, parameters, config):
        """
        Evaluate the model on local test data.

        Called by the Flower server (possibly at the end of each round) to measure
        how well the global model performs on each client's local test data.

        Args:
            parameters (list[np.ndarray]): global model weights from the server.
            config (dict): evaluation configuration sent by the server.

        Returns:
            tuple: (loss, num_samples, metrics_dict)
                - loss: average loss over the test set (Flower requires this)
                - num_samples: how many samples we evaluated on
                - metrics_dict: extra metrics, we include accuracy here
        """
        # Load global weights into local model
        self.set_parameters(parameters)

        # Setup
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        # Evaluation
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                total_loss += self.loss_fn(outputs, labels).item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)

        return avg_loss, total, {"accuracy": accuracy}



if __name__ == "__main__":
    client = MnistClient(
        client_id=0,
        train_loader=None,
        test_loader=None,
    )
    print("Client create successfully!")
    print(f"Model parameters: {sum(p.numel() for p in client.model.parameters()):,}")