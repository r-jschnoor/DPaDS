import numpy as np

from src.client import MnistClient
from src.mechanisms.dp import get_privacy_spent
from src.mechanisms.topk import topk_sparsify


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
        num_rounds (int):      number of training rounds (each with 1 epoch) planned.
    """

    def __init__(self, client_id, train_loader, test_loader,
                 source_label=7, target_label=1, use_dp=False, epsilon=10.0,
                 use_topk=False, topk_ratio=0.1, num_rounds=1):
        super().__init__(client_id, train_loader, test_loader, use_dp, epsilon,
                         use_topk=use_topk, topk_ratio=topk_ratio, num_rounds=num_rounds)
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

        metrics = {}
        if self.use_dp and self.privacy_engine is not None:
            epsilon = get_privacy_spent(self.privacy_engine, self.delta)
            metrics["epsilon"] = epsilon
        
        updated_parameters = self.get_parameters(config={})

        if self.use_topk:
            flat_before = np.concatenate([p.flatten() for p in parameters])
            flat_after = np.concatenate([p.flatten() for p in updated_parameters])
            update = flat_after - flat_before
            sparse_update = topk_sparsify(update, self.topk_ratio)
            sparse_params = flat_before + sparse_update
            shapes = [p.shape for p in updated_parameters]
            updated_parameters = []
            index = 0
            for shape in shapes:
                size = int(np.prod(shape))
                updated_parameters.append(sparse_params[index:index+size].reshape(shape))
                index += size
            metrics["topk_sparsity"] = np.count_nonzero(sparse_update) / len(sparse_update)

        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics