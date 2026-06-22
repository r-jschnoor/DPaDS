import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from client import MnistClient


class LabelFlipClient(MnistClient):
    """
    Malicious FL client that flips labels during training.

    Simulates a Byzantine attack by relabeling all training
    samples from `source_label` to `target_label` before
    computing the gradients. The server cannot distinguish
    this client from hontest ones since it returns normal
    shaped updates.

    Args:
        client_id (int):       unique client identifier.
        train_loader:          local training dataloader.
        test_loader:           local test dataloader.
        source_label (int):    the digit to relabel (e.g. 7).
        target_label (int):    the digit to relabel it as (e.g. 1).
        use_dp (bool):         whether to wrap training with DP-SGD.
        epsilon (float):       privacy budget. Only used when use_dp=True.
    """

    def __init__(self, client_id, train_loader, test_loader,
                 source_label=7, target_label=1, use_dp=False, epsilon=10.0):
        super().__init__(client_id, train_loader, test_loader, use_dp, epsilon)
        self.source_label = source_label
        self.target_label = target_label


    def fit(self, parameters, config):
        """
        Train with flipped labels. Similar to honest client.

        Args:
            parameters (list[np.ndarray]): global model weights from server.
            config (dict):                 training config from server.

        Returns:
            tuple: (updated_parameters, num_samples, metrics_dict)
        """
        self.set_parameters(parameters)
        self.model.train()

        for images, labels in self.train_loader:
            # Flip source_label -> target_label
            labels[labels == self.source_label] = self.target_label

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}