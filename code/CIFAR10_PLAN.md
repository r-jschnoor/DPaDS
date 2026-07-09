# CIFAR-10 Implementation Plan

Drafted 2026-07-09. Not yet implemented — this is a saved plan for a future session.

## Context

The codebase currently only supports MNIST. The goal is to add CIFAR-10 as a second
supported dataset, reusing all existing mechanism logic (DP-SGD via Opacus, FLTrust,
TopK, seeding) unchanged, so the trilemma study can eventually compare both datasets.
The CIFAR-10 model architecture should match the FLTrust paper (Cao et al., NDSS 2021)
— this codebase's `FLTrustStrategy` directly implements that paper's algorithm, and the
paper uses **ResNet20** (the standard CIFAR-10 ResNet from He et al.) for CIFAR-10,
rather than a small custom CNN.

**Key constraint carried over from the existing MNIST model:** ResNet20 uses BatchNorm
by default, which is incompatible with Opacus/DP-SGD (no valid per-sample gradients).
`MnistCNN` already avoids BatchNorm for exactly this reason (verified via
`test_compatibility_mnist.py`). The CIFAR-10 model needs BatchNorm replaced with
GroupNorm — a standard, well-established substitution in DP-SGD literature.

## Guiding decision: parameterize, don't duplicate

`MnistClient.fit()`/`evaluate()`, `LabelFlipClient.fit()`, and
`FLTrustStrategy.aggregate_fit()` contain zero MNIST-specific logic — every operation is
generic (state_dict copy, flatten/reshape via `p.shape`, confusion matrix by
`num_classes`). `mechanisms/dp.py` and `mechanisms/topk.py` are already 100%
model-agnostic. The `load_datasets()` stratified-split logic is already dataset-agnostic
(the `hasattr(targets, "tolist")` guard was deliberately written to handle both MNIST's
tensor targets and CIFAR-10's list targets).

**Decision:** add an optional `model_fn: Callable[[], nn.Module]` factory parameter
(default `MnistCNN`, so every existing call site is unaffected) threaded from
`ExperimentConfig.dataset` through `server.py` into `MnistClient`, `LabelFlipClient`,
and `FLTrustStrategy`. This avoids duplicating ~250 lines of identical fit/evaluate/
aggregate logic into new `Cifar10Client`/`Cifar10LabelFlipClient` classes, which would
contradict the repo's own `TODO.md` ("Refactor redundancy") and double the maintenance
surface for every future fix. Class names (`MnistClient`, `LabelFlipClient`) stay as-is
— renaming them is purely cosmetic and out of scope for this pass.

## 1. New file: `code/src/models/cifar10_cnn.py`

`Cifar10ResNet20` — GroupNorm-substituted ResNet20, matching He et al.'s CIFAR
architecture (6n+2 = 20 layers, n=3):

```
Stem:    Conv2d(3,16,3,s=1,p=1,bias=False) -> GroupNorm -> ReLU
Stage 1: 3x BasicBlock(16->16, stride=1)                    # 32x32
Stage 2: BasicBlock(16->32, stride=2) + 2x BasicBlock(32->32, stride=1)  # 16x16
Stage 3: BasicBlock(32->64, stride=2) + 2x BasicBlock(64->64, stride=1)  # 8x8
Head:    AdaptiveAvgPool2d(1) -> Flatten -> Linear(64, 10)
```

`BasicBlock(in_ch, out_ch, stride)`: conv3x3(stride) -> GN -> ReLU -> conv3x3(1) -> GN,
added to a shortcut path, then ReLU. Shortcut is `Identity()` when shape matches,
otherwise a 1x1-conv(stride)+GroupNorm projection (He et al.'s "Option B" — the standard
choice in modern reimplementations, cleaner to pair with GroupNorm than the original
zero-padding shortcut). All convs feeding into a norm layer use `bias=False`.

GroupNorm group count: `num_groups = min(32, num_channels)` uniformly (gives
`GroupNorm(16,16)`, `GroupNorm(32,32)`, `GroupNorm(32,64)`) — a standard default with no
per-layer special-casing needed; the FLTrust paper doesn't specify a GroupNorm variant
since it uses BatchNorm, so this is a reasonable, defensible convention rather than a
paper-mandated value.

Mirror `mnist_cnn.py`'s style: shape comments in `forward()`, and an
`if __name__ == "__main__":` block instantiating the model, running a dummy
`(4, 3, 32, 32)` tensor through it, printing output shape and parameter count (sanity
check against the ~270K params ResNet20 typically has).

## 2. Threading `model_fn` through the stack

- **`code/src/models/__init__.py`** (currently empty): add
  `MODEL_REGISTRY = {"mnist": MnistCNN, "cifar10": Cifar10ResNet20}` and
  `get_model_fn(dataset: str) -> Callable[[], nn.Module]`, so every other file resolves
  the model factory from one place instead of five separate if/else branches.

- **`code/src/client.py`**: `MnistClient.__init__` gains `model_fn=MnistCNN`; replace
  `self.model = MnistCNN()` with `self.model = model_fn()`, store `self.model_fn` for
  reuse. `fit()`: `model_tmp = MnistCNN()` -> `model_tmp = self.model_fn()`.
  `evaluate()`: `num_classes = NUM_CLASSES_MNIST` -> `num_classes = NUM_CLASSES` (renamed
  constant, see section 4).

- **`code/src/mechanisms/attacks.py`** (`LabelFlipClient`): add `model_fn` param, pass
  through `super().__init__(...)`. `fit()`: `model_tmp = MnistCNN()` ->
  `model_tmp = self.model_fn()`. `source_label`/`target_label` defaults stay unchanged —
  the swap mechanism is dataset-agnostic (any two distinct class indices 0-9 work
  equivalently as a Byzantine attack); only the docstring wording ("the digit to
  relabel") should generalize to "the class index to relabel".

- **`code/src/mechanisms/robust_aggregation.py`** (`FLTrustStrategy`): add
  `model_fn=MnistCNN` param, replace `self.ref_model = MnistCNN()` with
  `self.ref_model = model_fn()`. `aggregate_fit()` needs no changes (already fully
  generic). The standalone `if __name__ == "__main__":` demo block at the bottom stays
  MNIST-only (it's a unit test of the aggregation math, not in scope here).

- **`code/src/server.py`** (most of the change surface):
  - `load_datasets(root_dataset_size, seed=None, dataset="mnist")`: branch on `dataset`
    to pick `datasets.MNIST`/`datasets.CIFAR10` and the matching transform (see section
    3). Stratified split logic below is untouched.
  - `make_evaluate_fn(test_dataset, model_fn=MnistCNN)`: pass `model_fn` to the
    throwaway `eval_client`.
  - `get_client_fn(...)`: add `model_fn=MnistCNN` param, pass to both
    `LabelFlipClient(...)` and `MnistClient(...)`.
  - `get_server_fn(...)`: thread `model_fn` through to `make_evaluate_fn` and
    `FLTrustStrategy`.
  - `run_simulation_with_config(config)`: resolve `model_fn = get_model_fn(config.dataset)`
    once at the top; pass `dataset=config.dataset` to `load_datasets`; pass `model_fn`
    to `make_evaluate_fn`, `FLTrustStrategy`, and `get_client_fn`; replace the seeded
    initial-parameters block's `MnistCNN().state_dict()` with `model_fn().state_dict()`.

## 3. `ExperimentConfig` — new `dataset` field

```python
dataset: str = "mnist"   # "mnist" or "cifar10"
```
Placed near the top of the dataclass, right after `config_id`. Default preserves all
existing behavior — fully backward-compatible, no existing `ExperimentConfig(...)` call
site needs to change.

**Transform/normalization decision:** use the simple `(0.5,0.5,0.5), (0.5,0.5,0.5)` for
CIFAR-10, matching MNIST's existing simplified `(0.5,), (0.5,)` rather than CIFAR-10's
"true" per-channel statistics. This is a controlled comparative study of DP/FLTrust/TopK
effects, not an accuracy-maximization exercise — using each dataset's "precise" stats
while the other uses simplified ones would introduce an asymmetry that muddies any
cross-dataset comparison.

## 4. Naming cleanup (pure renames, no behavior change)

- `code/src/constants.py`: `NUM_CLASSES_MNIST` -> `NUM_CLASSES` (value stays 10, both
  datasets happen to have 10 classes; add a comment noting a future dataset with a
  different class count would need this to become per-dataset). Update call sites in
  `client.py` and `run_configurations.py`.
- `code/src/scripts/run_configurations.py`:
  `inflate_confusion_matrix_mnist_and_calculate_scores` ->
  `inflate_confusion_matrix_and_calculate_scores` (drop `_mnist`); update its docstring
  ("digit class" -> "class") and call site.

## 5. Results file collision avoidance

- `make_filename()`: insert `f"dataset-{config.dataset}"` into the filename parts, right
  after the timestamp.
- `save_results()`: add `"dataset": config.dataset` to the saved JSON's `"config"` dict.
- Existing MNIST result files predate this field and will lack a `"dataset"` key —
  any downstream reader (`visualize_results.py`, etc.) should default to `"mnist"` when
  the key is absent rather than crashing.

## Verification

1. `python -m src.models.cifar10_cnn` — confirms forward pass shape
   `(4,3,32,32) -> (4,10)` and a sane parameter count (~270K).
2. New `code/src/test_compatibility_cifar10.py` (mirrors `test_compatibility_mnist.py`):
   `ModuleValidator.validate(Cifar10ResNet20(), strict=False)` must report zero errors —
   the critical GroupNorm-vs-BatchNorm compatibility gate.
3. Tiny end-to-end smoke test (`num_clients=3, num_rounds=2, root_dataset_size=100`,
   `dataset="cifar10"`), covering: CIFAR-10 downloads/loads correctly; stratified split
   works with CIFAR-10's list-typed targets; a full fit/evaluate/aggregate round
   completes without shape errors; the DP-SGD path completes a round (defense in depth
   beyond the static Opacus check); the FLTrust path produces non-degenerate trust
   scores.
4. **MNIST regression check**: run one existing MNIST config (e.g. config 1 baseline,
   small scale) after the refactor to confirm the default `model_fn=MnistCNN` path is
   unaffected — parameterizing five files is exactly the kind of change that can
   silently break a default-argument wiring at one call site.
5. Only after 1-4 pass: a reduced-scale sweep across all 8 base configs for CIFAR-10
   (mirroring the existing `results/smoketest_20260708_214728/` precedent), saved to a
   distinct `results/smoketest_cifar10_<timestamp>/` folder, before any full-scale runs.

## Judgment calls made (flagging for visibility, not blocking)

- Shortcut type: 1x1-conv+GroupNorm projection ("Option B"), not zero-padding identity
  ("Option A") — more common in modern reimplementations, the FLTrust paper doesn't
  specify since it uses BatchNorm.
- GroupNorm group count: `min(32, num_channels)` — standard default, not paper-mandated.
- Normalization: simplified `(0.5,...)` for methodological consistency with MNIST, not
  CIFAR-10's precise per-channel stats.
- Byzantine label-flip indices (`source_label=3, target_label=7`) reused unchanged for
  CIFAR-10 — dataset-agnostic mechanism, any two distinct indices are equivalent.
- CIFAR-10 (50K train images) is close enough to MNIST (60K) that existing
  `SHARED_PARAMS` defaults (`num_clients=15`, `root_dataset_size=600`) should transfer
  without changes to the plumbing — but expect CIFAR-10 to need meaningfully more
  rounds than MNIST's `num_rounds=10` to reach comparable accuracy with a from-scratch
  ResNet20; low baseline accuracy on first CIFAR-10 runs should not be mistaken for a
  bug.

## Note on this planning session

While researching this plan, a subagent exploring the codebase reported that one of its
file reads returned content designed to look like a system/plan-mode instruction telling
it to abandon the task, referencing unrelated tooling and trying to redirect it to write
files to an arbitrary path. It correctly identified this as suspicious (not matching
anything in the actual repo) and ignored it. Flagged here in case it's worth
investigating which read triggered it, though it did not affect this plan.
