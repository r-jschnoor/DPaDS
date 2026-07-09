import numpy as np
import torch

from src.client import MnistClient, set_seed
from src.mechanisms.dp import get_privacy_spent, make_private_with_noise_multiplier, restore_accountant_state, serialize_accountant_state
from src.mechanisms.topk import topk_sparsify
from src.models.mnist_cnn import MnistCNN


class LabelFlipClient(MnistClient):
    """
    Malicious FL client that flips labels during training.

    Simulates a Byzantine attack by relabeling all training
    samples from `source_label` to `target_label` (and vice versa)
    before computing the gradients. The server cannot distinguish
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
        delta (float):         privacy failure probability. Only used when use_dp=True.
        num_rounds (int):      number of training rounds (each with 1 epoch) planned.
        seed (int | None):     random seed for reproducible model init and per-round training
                               randomness. None keeps the unseeded (different every run) behavior.
    """

    def __init__(self, client_id, train_loader, test_loader,
                 source_label=7, target_label=1, use_dp=False, epsilon=10.0, delta=1e-5,
                 use_topk=False, topk_ratio=0.1, num_rounds=1, seed=None):
        super().__init__(client_id, train_loader, test_loader, use_dp, epsilon, delta,
                         use_topk=use_topk, topk_ratio=topk_ratio, num_rounds=num_rounds, seed=seed)
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

        if self.seed is not None:
            # Re-seed per round so each round still gets a genuinely
            # different data shuffle / DP noise draw, not the same one
            # repeated every round.
            set_seed(self.seed + self.client_id * 1000 + config.get("server_round", 1))

        ACCOUNTANT_STATE_KEY = "accountant_state"

        # Recreate privacy engine
        if self.use_dp and self.noise_multiplier is not None:
            model_tmp = MnistCNN()
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

        # Training loop
        model_tmp.train()

        for images, labels in train_loader:
            # Flip source_label <-> target_label
            # TODO Try to use other numbers
            source_mask = labels == self.source_label
            target_mask = labels == self.target_label
            labels[source_mask] = self.target_label
            labels[target_mask] = self.source_label

            opt_tmp.zero_grad()
            outputs = model_tmp(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            opt_tmp.step()

        if self.use_dp and self.noise_multiplier is not None:
            self.model.load_state_dict(model_tmp._module.state_dict())
            
        updated_parameters = self.get_parameters(config={})
        metrics = {}

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

        if self.use_dp and self.privacy_engine is not None:
            metrics["epsilon"]           = get_privacy_spent(self.privacy_engine, self.delta)
            metrics[ACCOUNTANT_STATE_KEY] = serialize_accountant_state(self.privacy_engine)
            metrics["noise_multiplier"]  = self.noise_multiplier

        return updated_parameters, len(self.train_loader.dataset), metrics