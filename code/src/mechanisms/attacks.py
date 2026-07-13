import numpy as np
import torch

from src.client import MnistClient, set_seed, iterate_batches
from src.mechanisms.dp import get_privacy_spent, make_private_with_noise_multiplier, restore_accountant_state, serialize_accountant_state
from src.mechanisms.topk import topk_sparsify, unflatten, update_size_bytes
from src.models import get_dataset_spec
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
        source_label (int):    the class index to relabel (e.g. 7).
        target_label (int):    the class index to relabel it as (e.g. 1).
        use_dp (bool):         whether to wrap training with DP-SGD.
        epsilon (float):       privacy budget. Only used when use_dp=True.
        delta (float):         privacy failure probability. Only used when use_dp=True.
        num_rounds (int):      number of training rounds (each with 1 epoch) planned.
        seed (int | None):     random seed for reproducible model init and per-round training
                               randomness. None keeps the unseeded (different every run) behavior.
        num_client_iterations_per_round (int | None): SGD steps to take this round instead of a
                                    full epoch. None keeps the default (1 full epoch).
        dataset_spec (DatasetSpec): model factory + class count for this run's dataset.
                                    Defaults to MNIST.
        device (torch.device):  which device to train/evaluate on. Defaults to CPU.
    """

    def __init__(self, client_id, train_loader, test_loader,
                 source_label=7, target_label=1, use_dp=False, epsilon=10.0, delta=1e-5,
                 use_topk=False, topk_ratio=0.1, num_rounds=1, seed=None,
                 num_client_iterations_per_round=None,
                 dataset_spec=get_dataset_spec("mnist"), device=torch.device("cpu")):
        super().__init__(client_id, train_loader, test_loader, use_dp, epsilon, delta,
                         use_topk=use_topk, topk_ratio=topk_ratio, num_rounds=num_rounds, seed=seed,
                         num_client_iterations_per_round=num_client_iterations_per_round,
                         dataset_spec=dataset_spec, device=device)
        self.source_label = source_label
        self.target_label = target_label
        self.is_malicious = True


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
        metrics = {"is_malicious": float(self.is_malicious), "client_id": self.client_id}

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
            metrics["update_bytes"] = update_size_bytes(updated_parameters, use_topk=True, sparsified_update=sparse_update)
        else:
            metrics["update_bytes"] = update_size_bytes(updated_parameters, use_topk=False)

        if self.use_dp and self.privacy_engine is not None:
            metrics["epsilon"]           = get_privacy_spent(self.privacy_engine, self.delta)
            metrics[ACCOUNTANT_STATE_KEY] = serialize_accountant_state(self.privacy_engine)
            metrics["noise_multiplier"]  = self.noise_multiplier

        return updated_parameters, len(self.train_loader.dataset), metrics


class RandomGradientClient(MnistClient):
    """
    Malicious FL client that sends arbitrary (uninformative) updates.

    Simulates a Byzantine attack where the client contributes noise instead
    of a genuine trained direction. Two paths exist, matched to whether DP is
    active for this run:

    - use_dp=False: no training happens at all. The "update" is a pure,
      unit-norm random direction added directly to the received global
      weights and normalized so its magnitude is a fixed, sane reference
      point regardless of parameter count, rather than the ~300+ L2 norm a
      raw per-parameter N(0,1) draw would have on a model this size (which
      reliably blows the model up to NaN within a few dozen rounds once
      aggregated across several such clients.
      Actual magnitude control belongs to ScaledUpdateMixin.
    - use_dp=True: real training happens (delegates to MnistClient.fit(),
      the exact honest-client DP-SGD pipeline), and only the resulting flat
      delta gets randomly permuted before being returned. This is
      intended. Reusing the real DP pipeline keeps the
      reported metrics identical in shape to an honest client, with
      genuinely-accounted values, and matches an honest
      client's compute time too.

    Args:
        client_id (int):       unique client identifier.
        train_loader:          local training dataloader.
        test_loader:           local test dataloader.
        use_dp (bool):         whether to wrap training with DP-SGD.
        epsilon (float):       privacy budget. Only used when use_dp=True.
        delta (float):         privacy failure probability. Only used when use_dp=True.
        num_rounds (int):      number of training rounds (each with 1 epoch) planned.
        seed (int | None):     random seed for reproducible model init and per-round training
                               randomness. None keeps the unseeded (different every run) behavior.
        num_client_iterations_per_round (int | None): SGD steps to take this round instead of a
                                    full epoch. Only the DP path (via super().fit()) uses this.
                                    None keeps the default (1 full epoch).
        dataset_spec (DatasetSpec): model factory + class count for this run's dataset.
                                    Defaults to MNIST. Only the DP path (via
                                    super().fit()) ever constructs a model, so this
                                    is just forwarded, never used directly here.
        device (torch.device):  which device to train on. Defaults to CPU. Only the
                                DP path (via super().fit()) ever touches this directly;
                                the no-DP path works on plain numpy arrays.
    """

    def __init__(self, client_id, train_loader, test_loader,
                 use_dp=False, epsilon=10.0, delta=1e-5,
                 use_topk=False, topk_ratio=0.1, num_rounds=1, seed=None,
                 num_client_iterations_per_round=None,
                 dataset_spec=get_dataset_spec("mnist"), device=torch.device("cpu")):
        super().__init__(client_id, train_loader, test_loader, use_dp, epsilon, delta,
                         use_topk=use_topk, topk_ratio=topk_ratio, num_rounds=num_rounds, seed=seed,
                         num_client_iterations_per_round=num_client_iterations_per_round,
                         dataset_spec=dataset_spec, device=device)
        self.is_malicious = True


    def fit(self, parameters, config):
        """
        Return an arbitrary update instead of a genuinely trained one.

        Args:
            parameters (list[np.ndarray]): global model weights from server.
            config (dict):                 training config from server.

        Returns:
            tuple: (updated_parameters, num_samples, metrics_dict)
        """
        if self.use_dp and self.noise_multiplier is not None:
            # Real DP-SGD training via the honest pipeline
            honest_parameters, num_examples, metrics = super().fit(parameters, config)

            flat_before = np.concatenate([p.flatten() for p in parameters])
            flat_after  = np.concatenate([p.flatten() for p in honest_parameters])
            # Shuffles element positions across the whole flattened model -> preserves the exact value multiset/norm, destroys direction.
            scrambled_delta = np.random.permutation(flat_after - flat_before)

            shapes = [p.shape for p in honest_parameters]
            updated_parameters = unflatten(flat_before + scrambled_delta, shapes)
            return updated_parameters, num_examples, metrics

        # No DP: no training at all, no dataset access.
        if self.seed is not None:
            set_seed(self.seed + self.client_id * 1000 + config.get("server_round", 1))

        flat_before = np.concatenate([p.flatten() for p in parameters])
        noise = np.random.randn(*flat_before.shape)
        noise = noise / np.linalg.norm(noise)  # unit-norm direction -- see class docstring

        metrics = {"is_malicious": float(self.is_malicious), "client_id": self.client_id}
        if self.use_topk:
            # Sparsify the noise itself, same as an honest client sparsifies its real update
            noise = topk_sparsify(noise, self.topk_ratio)
            metrics["topk_sparsity"] = np.count_nonzero(noise) / len(noise)
            metrics["update_bytes"] = update_size_bytes(parameters, use_topk=True, sparsified_update=noise)
        else:
            metrics["update_bytes"] = update_size_bytes(parameters, use_topk=False)

        shapes = [p.shape for p in parameters]
        updated_parameters = unflatten(flat_before + noise, shapes)
        return updated_parameters, len(self.train_loader.dataset), metrics


class ScaledUpdateMixin:
    """
    Mixin that rescales whatever update the wrapped attack class produces.

    Multiplies the flat parameter delta by attack_scale before
    returning it, so scaling composes with any base attack instead of being
    a separate attack implementation. Must be listed before the base attack
    class in the MRO, e.g. `class ScaledLabelFlipClient(ScaledUpdateMixin,
    LabelFlipClient)`. 
    
    Args:
        attack_scale (float): multiplier applied to the update. >1 makes
                              the update larger than what the wrapped
                              attack would normally send.
        *args, **kwargs:      forwarded to the wrapped attack class's __init__.
    """

    def __init__(self, *args, attack_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_scale = attack_scale


    def fit(self, parameters, config):
        """
        Scale the wrapped attack's update by attack_scale.

        Args:
            parameters (list[np.ndarray]): global model weights from server.
            config (dict):                 training config from server.

        Returns:
            tuple: (updated_parameters, num_samples, metrics_dict)
        """
        updated_parameters, num_examples, metrics = super().fit(parameters, config)

        flat_before = np.concatenate([p.flatten() for p in parameters])
        flat_after  = np.concatenate([p.flatten() for p in updated_parameters])
        scaled_flat = flat_before + (flat_after - flat_before) * self.attack_scale

        shapes = [p.shape for p in updated_parameters]
        scaled_parameters = unflatten(scaled_flat, shapes)

        return scaled_parameters, num_examples, metrics


class ScaledLabelFlipClient(ScaledUpdateMixin, LabelFlipClient):
    """Label-flip attack with its resulting update scaled up (see ScaledUpdateMixin)."""
    pass


class ScaledRandomGradientClient(ScaledUpdateMixin, RandomGradientClient):
    """Random-gradient attack with its resulting update scaled up (see ScaledUpdateMixin)."""
    pass


# Keyed by (attack_type, is_scaled) -> e.g. ("random_gradient", True) means
# "random-gradient attack, wrapped in ScaledUpdateMixin".
ATTACK_CLASSES = {
    ("label_flip", False):      LabelFlipClient,
    ("label_flip", True):       ScaledLabelFlipClient,
    ("random_gradient", False): RandomGradientClient,
    ("random_gradient", True):  ScaledRandomGradientClient,
}


def build_malicious_client(client_id, train_loader, test_loader, attack_type, attack_scale,
                           source_label=3, target_label=7,
                           use_dp=False, epsilon=10.0, delta=1e-5,
                           use_topk=False, topk_ratio=0.1, num_rounds=1, seed=None,
                           num_client_iterations_per_round=None,
                           dataset_spec=get_dataset_spec("mnist"), device=torch.device("cpu")):
    """
    Construct the malicious client for one Byzantine slot.

    Centralizes attack dispatch (which class to build, and which kwargs it
    needs) so get_client_fn() doesn't need to know about
    individual attack classes or their constructor differences.

    Args:
        client_id (int):       unique client identifier.
        train_loader:          local training dataloader.
        test_loader:           local test dataloader.
        attack_type (str):        "label_flip" or "random_gradient".
        attack_scale (float | None): None or 1.0 for the base (unscaled)
                                     attack, otherwise the scale factor for
                                     the wrapped (Scaled*) variant.
        source_label (int):    the class index to relabel. Only used when
                               attack_type="label_flip".
        target_label (int):    the class index to relabel it as. Only used when
                               attack_type="label_flip".
        use_dp (bool):         whether to wrap training with DP-SGD.
        epsilon (float):       privacy budget. Only used when use_dp=True.
        delta (float):         privacy failure probability. Only used when use_dp=True.
        use_topk (bool):       whether to use top-k sparsification.
        topk_ratio (float):    the ratio of kept top-k values.
        num_rounds (int):      number of training rounds (each with 1 epoch) planned.
        seed (int | None):     random seed for reproducible model init and per-round training
                               randomness. None keeps the unseeded (different every run) behavior.
        num_client_iterations_per_round (int | None): SGD steps to take this round instead of a
                                    full epoch. None keeps the default (1 full epoch).
        dataset_spec (DatasetSpec): model factory + class count for this run's dataset.
                                    Defaults to MNIST.
        device (torch.device):  which device to train/evaluate on. Defaults to CPU.

    Returns:
        MnistClient: the constructed malicious client (not yet .to_client()'d).
    """
    is_scaled = attack_scale is not None and attack_scale != 1.0
    client_cls = ATTACK_CLASSES[(attack_type, is_scaled)]

    kwargs = dict(
        use_dp=use_dp, epsilon=epsilon, delta=delta,
        use_topk=use_topk, topk_ratio=topk_ratio,
        num_rounds=num_rounds, seed=seed,
        num_client_iterations_per_round=num_client_iterations_per_round,
        dataset_spec=dataset_spec, device=device,
    )
    if attack_type == "label_flip":
        kwargs.update(source_label=source_label, target_label=target_label)
    if is_scaled:
        kwargs["attack_scale"] = attack_scale

    return client_cls(client_id, train_loader, test_loader, **kwargs)


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=32)

    model = MnistCNN()
    parameters = [p.data.cpu().numpy() for p in model.parameters()]
    config = {"server_round": 1}

    print("Testing malicious client classes (no DP, no topk):\n")
    for name, cls, kwargs in [
        ("LabelFlipClient",            LabelFlipClient,            dict(source_label=3, target_label=7)),
        ("RandomGradientClient",       RandomGradientClient,       dict()),
        ("ScaledLabelFlipClient",      ScaledLabelFlipClient,      dict(source_label=3, target_label=7, attack_scale=5.0)),
        ("ScaledRandomGradientClient", ScaledRandomGradientClient, dict(attack_scale=5.0)),
    ]:
        client = cls(client_id=0, train_loader=train_loader, test_loader=None, **kwargs)
        updated_parameters, num_examples, metrics = client.fit(parameters, config)
        shapes_match = all(u.shape == p.shape for u, p in zip(updated_parameters, parameters))
        print(f"{name}: shapes_match={shapes_match}, num_examples={num_examples}, "
              f"is_malicious={metrics.get('is_malicious')}")

    print("\nTesting RandomGradientClient with DP (checks metrics-key parity with honest clients):")
    dp_client = RandomGradientClient(
        client_id=0, train_loader=train_loader, test_loader=None,
        use_dp=True, epsilon=5.0, delta=1e-5, num_rounds=3,
    )
    updated_parameters, num_examples, metrics = dp_client.fit(parameters, config)
    print(f"  metrics keys: {sorted(metrics.keys())}")
    print(f"  epsilon={metrics.get('epsilon'):.4f}, noise_multiplier={metrics.get('noise_multiplier'):.4f}")

    print("\nTesting build_malicious_client() dispatch:")
    for attack_type in ["label_flip", "random_gradient"]:
        for attack_scale in [None, 3.0]:
            client = build_malicious_client(
                client_id=0, train_loader=train_loader, test_loader=None,
                attack_type=attack_type, attack_scale=attack_scale,
            )
            print(f"  attack_type={attack_type}, attack_scale={attack_scale} -> {type(client).__name__}")