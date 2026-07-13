# Code Directory


## Important findings so far
- TrustScores: Using Batches (SGD Steps) instead of Epochs fixes the Trustscore degrading issue
  - Using more than one SGD Step in each round increases the convergence and improves TrustScore differentiation between honest and malicious clients
- Batchsize of server and client should be the same, as well as SGD steps
- Differential Privacy and FLTrust do not work well with each other. This is observable when comparing the results of e.g. configs 3 and 5


## Found Bugs

Results of an in-depth correctness review of the three trilemma mechanisms (DP, FLTrust, TopK) and
the surrounding FL simulation code, verified against Opacus internals and the FLTrust paper's
aggregation algorithm where relevant. Ordered by severity. Status is updated as items are fixed.

### Critical

1. **FLTrust aggregation dropped the base model** — `mechanisms/robust_aggregation.py`
   (`FLTrustStrategy.aggregate_fit`). `client_updates` are deltas
   (`client_flattened - global_flattened`); `aggregated_norms` is the trust-weighted average of
   those deltas. The final step reshaped `aggregated_norms` directly into the new global model
   instead of `global_flattened + aggregated_norms`, so every FLTrust round replaced the global
   model with a small, near-noise vector rather than applying the aggregated update to it.
   Confirmed empirically (15 clients, 3 Byzantine, 8 rounds, FLTrust only): before the fix,
   accuracy flatlined at exactly `0.1135` on every single round; after adding the missing
   `global_flattened` back, accuracy climbed `0.556 -> 0.624 -> 0.808 -> 0.868 -> 0.891 -> 0.904
   -> 0.909 -> 0.917`. This was the actual root cause behind the previously flagged "Revisit trust
   scores" TODO and every flatlined FLTrust result (with or without DP) — the trust-score math
   itself was fine.
   **Status: Fixed.**

### High

2. **Label-flip attack collapsed both classes instead of swapping them** —
   `mechanisms/attacks.py` (`LabelFlipClient.fit`), lines with
   `labels[labels == self.source_label] = self.target_label` followed by
   `labels[labels == self.target_label] = self.source_label`. Sequential in-place masked
   assignment: the first line turns all `source_label` samples into `target_label`, then the
   second line's mask now also matches those just-converted samples, sending everything (both
   original classes) to `source_label`. Verified directly:
   `[1,2,3,4,5,6,7,8,9,0,3,7] -> [1,2,3,4,5,6,3,8,9,0,3,3]` for source=3, target=7. Not a true
   swap and not the documented one-way "relabel source to target" behavior either — changed what
   the Byzantine-robustness experiments were actually testing.
   Confirmed intent was a true bidirectional swap (source and target labels exchange completely,
   e.g. `3 <-> 7`). Fix: capture both boolean masks before mutating `labels`, then apply both
   assignments against the original (unmutated) masks instead of re-deriving the second mask from
   the already-mutated tensor. Verified directly:
   `[1,2,3,4,5,6,7,8,9,0,3,7] -> [1,2,7,4,5,6,3,8,9,0,7,3]` for source=3, target=7 — correct swap,
   other digits untouched. End-to-end sanity check (15 clients, 3 Byzantine, FLTrust, 3 rounds)
   ran clean: accuracy `0.490 -> 0.673 -> 0.825`.
   **Status: Fixed.**

### Moderate

3. **`ExperimentConfig.delta` is never passed to clients** — `server.py` (`get_client_fn`,
   `run_simulation_with_config`). Neither `MnistClient` nor `LabelFlipClient` receive `delta` from
   the config; both silently fall back to their own hardcoded default (`1e-5`). Invisible today
   only because that happens to match `ExperimentConfig`'s default.
   Fix: added `delta` to `LabelFlipClient.__init__` (forwarded to `MnistClient.__init__`, which
   already accepted it) and threaded `delta` through `get_client_fn()` and
   `run_simulation_with_config()` down to both client constructors.
   **Status: Fixed.**

4. **`total_trust == 0` fallback doesn't return the unchanged global model** —
   `mechanisms/robust_aggregation.py` (`FLTrustStrategy.aggregate_fit`). The comment says "Return
   unchanged global model," but `self.ref_model` has already been mutated by
   `get_reference_update()` (trained one epoch on the small root set) by the time this branch
   runs. Under an all-untrusted round, the model is silently replaced by a low-capacity,
   root-set-only-trained model instead of being preserved.
   Fix: return `server_state_parameters` (the actual pre-round global weights already saved in
   `configure_fit`) instead of `self.ref_model`'s post-training parameters.
   **Status: Fixed.**

5. **DP noise-multiplier calibration silently depends on `dataset_size % batch_size == 0`** —
   `client.py` / `mechanisms/attacks.py` (`__init__`) vs. Opacus internals. The manual
   `sample_rate = batch_size / dataset_size` used to calibrate the noise multiplier only matches
   Opacus's actual internal `1 / len(DataLoader)` (used for Poisson subsampling via
   `DPDataLoader.from_data_loader`) when the client's dataset size divides evenly by batch size.
   True today for the default config (15 clients -> 4000 samples/client, batch 32), but not
   guaranteed for other client counts — would silently miscalibrate the noise multiplier relative
   to the configured target epsilon otherwise. (The *reported* per-round epsilon stays accurate
   since it's read from the real accountant history.)
   Fix: compute `sample_rate = 1 / len(train_loader)` directly, matching Opacus's own internal
   convention regardless of divisibility. `LabelFlipClient` inherits this via `super().__init__()`.
   **Status: Fixed.**

Combined sanity check for bugs 3-5 (`run_configurations.py --config 5`: DP+FLTrust, 3 Byzantine
label-flippers, 15 clients, 10 rounds, epsilon in {1, 5, 10}) ran clean end to end — no crashes,
and each variant shows real learning instead of the old flatlines: eps=1 accuracy
`0.100 -> 0.392`, eps=5 `0.116 -> 0.395`, eps=10 `0.103 -> 0.471`, with each run's spent epsilon
(from the real accountant) climbing smoothly to its target by round 10 (0.99 / 4.99 / 9.99).

### Low / design notes
(FW = Future Work)

6. FW -> **No error-feedback / residual memory in TopK** — `mechanisms/topk.py`. Standard practice in
   the compression literature (e.g. Deep Gradient Compression) to avoid permanently discarding
   zeroed-out coordinates each round; likely contributes to worse results at high sparsity
   (k=0.01).
   Discussed: plain (memory-less) top-k is a legitimate baseline in its own right, not incorrect.
   Adding memory would be a new feature (round-persistent per-client residual state, mirroring
   `AccountantStateManager`) rather than a bug fix, and would change the meaning of every existing
   TopK result.
   **Status: Won't fix.** *Flagged as a **future improvement** — the introduction of round-persistent
   client state (residual accumulation across rounds, in the same spirit as the DP accountant
   state) is the interesting part worth revisiting later, not just the memory mechanism itself.*

7. FW -> **`rescale_to_ref_norm` is off in every configured experiment** — hardcoded in
   `scripts/run_configurations.py`'s `SHARED_PARAMS`. Without it, clients that do more local SGD
   steps get proportionally larger raw influence on the aggregate regardless of trust score
   (trust score normalizes the *weights*, not the update magnitude) — a real deviation from the
   FLTrust paper worth documenting as a limitation rather than a silent default.
   Re-tested now that bug 1 is fixed (15 clients, 3 Byzantine, 8 rounds, root=600):
   `rescale_to_ref_norm=True` converges cleanly (`0.122 -> 0.149 -> 0.205 -> 0.270 -> 0.380 ->
   0.479 -> 0.557 -> 0.616`) but noticeably slower than `False` (`0.556 -> ... -> 0.917` at the
   same round count) — it isn't unstable, it's just a smaller effective step size. Root cause: the
   server's reference model trains on only `root_dataset_size=600` samples per round (~19
   batches) versus a client's ~4000 samples (~125 batches), so the reference update's norm is
   systematically much smaller than a client's; rescaling every client down to that smaller norm
   shrinks the effective global learning rate. This is the same root imbalance as bug 9.
   **Status: Won't fix (for now).** Keeping the default off so existing/planned experiments stay
   comparable. Worth revisiting once `root_dataset_size` is rebalanced against client dataset size
   (see 9) — that's the actual lever to make paper-faithful rescaling converge at a competitive
   rate, not a fix to the rescaling logic itself.

8. **Server root dataset overlapped client 0's training slice and wasn't class-representative** —
   both started from index 0 of the training set, so the "clean, independent" server data wasn't
   actually independent from client 0 (always a malicious client under the current setup), and the
   root set's class balance was whatever fell out of the raw dataset order (previously measured
   49-79 samples per class out of 600).
   Fix: `load_datasets()` now builds the root set via
   `sklearn.model_selection.train_test_split(all_indices, train_size=root_dataset_size,
   stratify=targets)`, which returns a class-balanced root sample and its complement in one call.
   `get_client_fn()` now takes that complement (`client_pool_indices`) and slices clients from it
   instead of `range(0, len(train_dataset))`, so root and client data are disjoint by construction.
   Reads `dataset.targets` (normalizing tensor->list via `hasattr(targets, "tolist")`), which works
   unchanged for both `MNIST` (tensor) and `CIFAR10` (list) targets, so this survives the CIFAR-10
   migration without modification. Verified: root/pool disjoint, union covers the full training
   set, root class distribution tightened to 54-67 samples per class out of 600.
   **Status: Fixed.**

9. **`get_reference_update` docstring/implementation mismatch** — `mechanisms/robust_aggregation.py`.
   Docstring said "one step"; implementation trains a full epoch over the root loader (all
   batches). Measured reference-update norm at ~30-80x smaller than a client's per-round update
   norm, which matters given (7) above.
   Discussed alongside (7): changing the actual compute budget (how much training the reference
   model gets) is the same open design question as rebalancing `root_dataset_size` — not something
   to decide in isolation here, and reducing to a literal one step would shrink the reference
   norm further, making (7) worse rather than better.
   Fix: docstring corrected to describe the actual behavior (full epoch), with a note on why the
   reference update's norm is systematically smaller than a client's and a pointer to (7) for the
   real design question.
   **Status: Fixed** (docstring only; behavior unchanged, revisit alongside 7).

10. **Every client evaluated on the full MNIST test set every round** — `server.py`
    (`get_client_fn`), `test_loader` was built from the full `test_dataset`, not a per-client
    slice. Numerically harmless (deterministic, so aggregating was redundant rather than wrong) but
    15x more eval compute than needed per round, and not representative of realistic FL
    evaluation.
    Fix: switched to Flower's server-side `evaluate_fn` (`make_evaluate_fn()` in `server.py`),
    which evaluates the global model once per round against the full test set on the server,
    reusing `MnistClient.evaluate()` via a throwaway client rather than duplicating its
    accuracy/confusion-matrix logic. Set `fraction_evaluate=0.0` on all strategies so Flower skips
    federated (per-client) evaluation entirely, and `get_client_fn` no longer builds a `test_loader`
    per client. `HistoryStrategyAdapter` gained an `evaluate()` override (alongside the existing
    `aggregate_evaluate()`) that records centralized results into the same `run_history` dict/keys,
    so `save_results()` and `visualize_results.py` are unaffected. Verified end to end (DP+FLTrust+
    TopK, 6 clients, 3 rounds): single evaluation per round, correct accuracy/loss trajectory, full
    10x10 confusion matrix (100 `cm_` keys) present, no crash.
    **Status: Fixed.**

11. **Dead code**: `AccountantStateManager.get_config()` (`server.py`) was defined but never
    called — the actual code path used `get_state()` directly in `configure_fit`, duplicating the
    config-building logic inline instead of reusing it.
    Discussed consolidating rather than deleting, which surfaced a real naming trap: `get_config`'s
    (and `store`'s) parameter was named `client_id`, but the state dict is actually keyed by
    Flower's raw `node_id` (see `aggregate_fit`: `accountant_manager.store(client_proxy.node_id,
    ...)`). Elsewhere in this codebase "`client_id`" specifically means the `0..num_clients-1`
    slice index (`get_client_fn`) -- a different number from `node_id`. Calling `get_config()`
    following that established naming convention would have silently found no accountant state.
    Fix: renamed both `store()`'s and `get_config()`'s parameter to `node_id` to match what they
    actually need, then had `configure_fit` call `accountant_manager.get_config(client_proxy.node_id,
    server_round)` instead of duplicating the dict-building logic inline -- one source of truth.
    Verified: DP run (4 clients, 4 rounds) shows accountant state correctly accumulating across
    rounds (`epsilon: 4.04 -> 4.43 -> 4.74 -> 5.00`, converging on the target).
    **Status: Fixed.**

12. **`visualize_results.py` had unused/undeclared imports** — `from networkx import efficiency`
    and `from numpy import sort` were both unused. `networkx` isn't a declared dependency in
    `pyproject.toml`, so this script would break in a clean install where it isn't pulled in
    transitively by something else.
    Fix: removed both import lines. Verified: script still parses and runs correctly against real
    result data (`--plot lines` on an existing results folder).
    **Status: Fixed.**

13. **`run_configurations.py::save_results()` wrote to a relative `results/` path** —
    `folder = os.path.join("results", run_timestamp)` resolved relative to the process's current
    working directory, not the repo layout. Every prior run in `src/results/` was produced with
    cwd set to `src/`; running the script as `python -m src.scripts.run_configurations` from
    `code/` (cwd = `code/`) silently wrote to a brand-new `code/results/` instead, which was easy
    to miss and easy to lose track of. Confirmed live: a sanity run landed in
    `code/results/20260708_192659/` and had to be moved into `src/results/` by hand.
    Fix: added `RESULTS_ROOT`, anchored to the script's own file location
    (`os.path.dirname(os.path.abspath(__file__))`, one level up, plus `results`) instead of a
    bare relative string, so output always lands in `src/results/` regardless of the invoking cwd.
    Verified: `RESULTS_ROOT` resolves to `code/src/results` correctly even when invoked with cwd
    set to the repo root.
    **Status: Fixed.**

## Other changes

- **Trust scores now saved to the results JSON** (`TODO.md`: "Put trust scores per round in
  results file"). `FLTrustStrategy.aggregate_fit()` (`mechanisms/robust_aggregation.py`) now
  attaches one `trust_score_<node_id>` metric per client per round to its returned metrics dict
  (on both the normal and the all-trust-zero return paths), keyed by Flower's `node_id` since it's
  stable per simulated client across rounds. `run_configurations.py::save_results()` regroups
  these into a `trust_scores: {node_id: score}` entry on each round in the saved JSON. Cast to
  plain `float` at the source, since `cosine_similarity` returns numpy scalars, which aren't JSON
  serializable and shouldn't leak into Flower's metrics dict. Verified: 6-client FLTrust run shows
  6 distinct trust scores per round, the same 6 node_ids recurring across all 3 rounds.

## Combined sanity check (bugs 6-13)

Ran `run_configurations.py --config 8` (DP+FLTrust+TopK+Byzantine, production params: 15 clients,
10 rounds, 3 Byzantine, root=600, 6 variants across epsilon in {1,5,10} x k in {0.01,0.1}) via the
real script end to end, then `visualize_results.py --plot all` on the output. No crashes; results
saved correctly to `src/results/20260708_210750/` (bug 13); 15 trust scores present per round in
every variant's JSON (new trust-score feature); all standard plots (bar/radar/line/confusion/F1)
generated cleanly (bug 12's import cleanup didn't break the pipeline). 5 of 6 variants show real,
varying learning curves as epsilon/k change; the harshest combination (eps=5, k=0.01) sits near-flat
at 0.1135 for 8 rounds before creeping to 0.1138 by round 10 -- confirmed as a genuine slow-start
under extreme compression + noise (confusion matrix shows 9993/10000 predictions collapsed to class
"1", matching MNIST's 11.35% class-1 base rate), not a reoccurrence of bug 1 (accuracy is not
bit-identical across all rounds, and per-round trust scores are small but never exactly zero, so the
all-trust-zero branch from bug 4 isn't what's firing here).

## Literature comparison (smoke-test results review)

Used to sanity-check the `smoketest_20260708_214728` results (10 clients, 8 rounds, no fixed random
seed) against published numbers for each mechanism, after ruling out a code bug:

- [Deep Learning with Differential Privacy (Abadi et al., 2016)](https://www.researchgate.net/publication/309444608_Deep_Learning_with_Differential_Privacy)
  -- centralized DP-SGD on MNIST reaches ~95% at eps<=2 and ~97% at eps<=8, but over far more
  training steps than an 8-round FL smoke test provides; used to confirm our lower DP accuracy is a
  round-budget/scale issue, not a broken mechanism.
- [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping (Cao et al., 2021) -- NDSS Symposium](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/)
  -- FLTrust holds FedAvg-no-attack accuracy even under 40-60% malicious clients; used to confirm
  bug 1's fix is working (our 20%-Byzantine run lands within 1 point of baseline).
- [cpSGD: Communication-efficient and differentially-private distributed SGD (Agarwal et al.)](https://www.researchgate.net/publication/325413771_cpSGD_Communication-efficient_and_differentially-private_distributed_SGD)
  -- documents that DP + compression interactions are non-monotonic and regime-dependent rather than
  simple monotone-in-epsilon curves; used to contextualize the DP-only epsilon ranking anomaly (see
  seed-control note below) and the DP+TopK compounding-difficulty pattern.
- [Gradient Sparsification Can Improve Performance of Differentially-Private Convex Machine Learning](https://arxiv.org/pdf/2011.14572)
  -- further evidence that compression's effect on DP training is regime-dependent, not uniformly
  helpful or harmful.

**Related finding, now fixed -- see 14 and "Seed control" below.**

14. **[High] `client_id = node_id % num_clients` could collide** -- `server.py` (`get_client_fn`).
    Flower's `node_id` is an effectively-random large integer (not a small sequential one), so taking
    it mod a small `num_clients` isn't guaranteed bijective. Reproduced directly: in one run, two
    different physical clients both computed `client_id=2` while `client_id=0` was never used at all.
    Discovered while debugging seed control, but this is a pre-existing bug independent of it -- it
    means the configured `num_byzantine` count and per-client data-slice assignment have been
    unreliable in every prior run of this codebase (seeded or not): a collision means two clients
    duplicate the same data slice and role (both honest, or both malicious) while another slice never
    gets trained on that run, and which physical client is "malicious" can silently vary run to run.
    Fix: use Flower's own `context.node_config["partition-id"]` instead, which is guaranteed
    collision-free and sequential (0..num_clients-1) by construction. Verified: 3 fresh runs x 2
    rounds each, 4/4 unique node_ids in every round (zero collisions), versus a reproduced collision
    on the old code before the fix.
    **Status: Fixed.**

15. FW -> **TopK doesn't reduce actual bytes transmitted in the simulation** -- `client.py`
    (`MnistClient.fit()`), `mechanisms/attacks.py` (`LabelFlipClient.fit()`,
    `RandomGradientClient.fit()`). After `topk_sparsify()` zeroes out all but the top-k delta
    entries, `fit()` reconstructs a full dense parameter vector
    (`sparse_parameters = flat_before + sparsified_update`) before returning it -- the same length
    as an uncompressed update. Ray genuinely serializes this dense array between the client and
    server Ray actors, so a TopK config transmits exactly as many bytes as an uncompressed one
    today; only the *values* at the zeroed positions look unchanged, nothing is actually smaller.
    The new `update_bytes` metric (see "New metric" below) and the pre-existing `topk_sparsity` are
    both computed/logical numbers -- what a real sparse encoding would cost -- not a measurement of
    what Ray moved.
    Real compression would require `fit()` to return a compact `(indices, values)` pair instead of
    dense per-layer arrays, plus: a dedicated `TopKFedAvg` strategy for the non-FLTrust TopK configs
    (4, 6), since plain `FedAvg.aggregate_fit()` assumes homogeneous per-layer-shaped arrays and
    can't combine sparse updates from clients that touched different positions; rewriting
    `FLTrustStrategy.aggregate_fit()` (`mechanisms/robust_aggregation.py`) to decode each client's
    sparse pair into a dense delta before its existing flatten/cosine-similarity math, and to get
    layer shapes from `dataset_spec` instead of inferring them from a client's (now sparse) result
    (currently done at the `original_parameters = parameters_to_ndarrays(first_fit_result.parameters)`
    line); and accepting that only the uplink (client -> server) can ever compress this way -- the
    downlink broadcast each round stays fully dense regardless, since clients need the complete
    current model to train locally, and the aggregate touches enough distinct positions across many
    clients to be effectively dense too.
    Deliberately not attempted here: this rewrites the core aggregation math path in exactly the
    files this document already flags as fragile (the DP-sync ordering bug, bug 1's dropped-base-model
    bug), so it needs its own dedicated review/testing pass rather than being folded into an
    unrelated metrics-only change.
    **Status: Won't fix (for now).** Flagged as **future work** -- implementing genuine sparse-wire
    compression (not just logically computing what it would cost) is the natural follow-up once
    this is worth the regression risk.

## Seed control

Added `ExperimentConfig.seed` (default `None` = old unseeded behavior) to make experiments
reproducible -- primarily so a sweep (e.g. epsilon in {1, 5, 10}) shares the same data split and
initial model, isolating whatever parameter is actually varied instead of confounding it with a
different random starting point per run (see the epsilon-ranking anomaly above, which was exactly
this). `run_configurations.py`'s `SHARED_PARAMS` now defaults to `seed=42` so this applies to every
config sweep by default.

Three separate sources of randomness had to be controlled, since each simulated client runs in its
own Ray worker process (a separate OS process, not just a separate thread):
- **Root/client data split** -- `load_datasets()` passes `seed` as `random_state` to
  `train_test_split()`.
- **Each client's model init and per-round training randomness** (data shuffling, DP noise) --
  `client.py` gained a `set_seed()` helper (seeds `random`, `numpy`, and `torch`). Called once in
  `MnistClient.__init__` as `seed + client_id` (covers whichever client Flower asks first for the
  initial model), and again at the top of every `fit()` call as
  `seed + client_id*1000 + server_round` (so each round still gets a genuinely different shuffle/
  noise draw instead of repeating the same one every round). `LabelFlipClient` mirrors both.
  `HistoryStrategyAdapter.configure_fit` now always injects `server_round` into each client's fit
  config (previously only when DP was on), since round-aware seeding needs it regardless of DP.
- **The initial global model itself** -- Flower's `Server._get_initial_parameters()` asks a random
  client via `ClientManager.sample()`, which uses plain `random.sample()` running on a background
  thread (`run_serverapp_th`) that races with other Flower/Ray threading -- seeding the main process
  alone could not reliably control this. Fixed by building the initial model directly in
  `run_simulation_with_config()` (now that the main process is seeded) and passing it as
  `initial_parameters` to the strategy, which makes Flower skip the random-client step entirely.

Verified in stages: root/client split reproducible on its own; baseline (no DP/FLTrust/TopK)
byte-identical accuracy across two runs with the same seed; the full DP+FLTrust+TopK+Byzantine path
reproducible in 3 of 4 rounds exactly, with a difference of one test-sample prediction (0.3536 vs
0.3537 accuracy) in the 4th -- attributed to floating-point non-associativity in multi-threaded CPU
matrix ops (a well-known PyTorch limitation, not a missed seed), not chased further since forcing bit-
level determinism (`torch.use_deterministic_algorithms` + pinning thread counts to 1) would add real
performance cost and risks breaking Opacus's gradient hooks, for a residual this small. A different
seed was confirmed to still produce genuinely different results.

## New Byzantine attacks: random-gradient and scaling

Added two new attack methods alongside `LabelFlipClient`, per `TODO.md`'s "What to do next":
`RandomGradientClient` (arbitrary, uninformative updates) and `ScaledUpdateMixin` (wraps any base
attack to scale up its resulting update). Both live in `mechanisms/attacks.py`.

`RandomGradientClient` has two paths, matched to whether DP is active for the run:

- `use_dp=False`: no training happens at all -- no `train_loader` iteration, no dataset access.
  The "update" is a pure, unit-norm random direction added directly to the received global
  weights. True to the TODO's literal wording ("arbitrary gradients... without computing on the
  dataset at all").
- `use_dp=True`: a deliberate, scoped exception to the above. Real DP-SGD training happens by
  delegating straight to `MnistClient.fit()` (`super().fit(parameters, config)`) -- the exact
  honest-client Opacus pipeline, unmodified -- and only the *resulting* flat parameter delta gets
  randomly permuted (`np.random.permutation`) before being returned. Permuting (not resampling)
  preserves the exact value multiset/norm of the real trained-and-noised update while destroying
  its direction, so it stays a meaningful Byzantine attack for FLTrust's cosine-similarity
  scoring to be tested against.

This design specifically avoids a real bug in `weighted_average_metrics()` (`server.py:430`):
it aggregates the **intersection** of every client's metric-dict keys before computing anything,
so a malicious client that skipped DP training and simply omitted `epsilon`/`noise_multiplier`
would have silently dropped those keys for the *entire round, for every client* -- not just gone
missing for the attacker. Reusing the honest DP pipeline verbatim means `RandomGradientClient`
reports the exact same metric keys with genuinely-accounted values, no stub values, no
aggregation bug. As a side effect (not the primary goal, but worth having), it also means real
compute happens every round, defeating any future timing-based detection of malicious clients.
Verified directly (4 clients, 1 Byzantine, DP on, epsilon=5.0, 2 rounds): `metrics_distributed_fit`
contained `epsilon`/`noise_multiplier`/`accountant_state`/`is_malicious` every round exactly like
an honest run (epsilon climbing `4.519 -> 4.999` toward the 5.0 target, matching real accountant
behavior, not a fabricated value). The no-DP path was verified separately (4 clients, 1 Byzantine,
2 rounds): no crash, no DP-related keys present (correctly absent, since no DP ran), accuracy
`0.104 -> 0.132 -> 0.099`.

**Found and fixed after merging**: the no-DP path originally added a *raw* `np.random.randn` draw
(no normalization) as the update. For this model's ~105,866 parameters that has an L2 norm of
~325 -- against a real trained client update's typical norm of ~0.01-2 in this project. In a real
sweep (50 clients, 40% Byzantine, `attack_scale=2.0`, plain FedAvg / no FLTrust, config 1) this
reliably blew the global model up to `NaN` within ~20 rounds: loss went `19.94 -> nan` at round 22
and stayed `nan` through round 50, with the final confusion matrix showing literally 100% of
predictions collapsed to class "0" for every input (`argmax` on NaN-filled logits), exactly
matching the frozen `0.098` accuracy (MNIST's class-0 population share). Root cause: "not scaled"
was implemented as "whatever magnitude a raw `N(0,1)` draw happens to have," rather than a true
fixed reference scale -- an arbitrary, huge number unrelated to the model's actual working scale,
not "no scale" at all. Fix: normalize the noise to a unit vector (`noise /
np.linalg.norm(noise)`) before adding it, so the base attack sits at a sane, bounded-by-construction
baseline regardless of parameter count, and `attack_scale` gives predictable control from there.
Verified: unscaled update norm now exactly `1.0`, `attack_scale=2.0` gives exactly `2.0` (both
measured directly against a real `MnistCNN()`'s parameters); a 10-client/40%-Byzantine/
`attack_scale=2.0`/no-FLTrust re-run of the previously-crashing scenario now trains cleanly for
all 15 rounds with finite loss throughout (`2.16 -> 0.16`) and accuracy climbing to `0.954`, no
`NaN`.

Related note: this failure mode is specific to plain FedAvg (config 1 has `use_fltrust=False`).
For FLTrust-enabled configs, `rescale_to_ref_norm=True` (currently defaulted off, see "Low /
design notes" above) would likely have mitigated it even before this fix -- FLTrust's cosine-
similarity trust score is scale-invariant (so a huge-norm malicious update still gets a
near-zero score), but that score is only a multiplicative *weight*; without
`rescale_to_ref_norm`, a small-but-nonzero weight times an astronomically large update can still
inject a disproportionate amount into the aggregate. `rescale_to_ref_norm` rescales every
client's update to the reference update's norm *before* weighting, which directly bounds this --
exactly the magnitude-inflation defense it exists for in the original FLTrust paper, as opposed
to the direction-based defense the trust score itself provides.

`ScaledUpdateMixin` multiplies whatever update the wrapped base attack produces by `attack_scale`,
via cooperative multiple inheritance (`class ScaledLabelFlipClient(ScaledUpdateMixin,
LabelFlipClient)`, `class ScaledRandomGradientClient(ScaledUpdateMixin, RandomGradientClient)`) --
scaling composes with either base attack instead of being a separate implementation. Verified
against both base attacks (same seed, same starting parameters, comparing returned update norms):
label-flip at `attack_scale=5.0`, `0.0138 -> 0.0691` (ratio 5.000); random-gradient (post the
unit-norm fix below) at `attack_scale=2.0`, `1.0 -> 2.0` exactly -- confirms `attack_scale` now
scales *from* the fixed unit-norm baseline as intended, not from an arbitrary raw-noise
magnitude.

A small shared helper, `unflatten(flat, shapes)` (`mechanisms/topk.py`), was extracted for the
flat-vector-to-per-layer-arrays reshape this new code needs in three places (both
`RandomGradientClient` paths, `ScaledUpdateMixin`) instead of duplicating the loop again. Existing
call sites with the same inline pattern (`client.py`, `LabelFlipClient`, `robust_aggregation.py`)
were left untouched -- working code, out of scope here.

Attack dispatch is centralized in `build_malicious_client()` (`mechanisms/attacks.py`), keyed by
`(attack_type, is_scaled)` via an `ATTACK_CLASSES` lookup table, so `get_client_fn()` (`server.py`)
doesn't need to know about individual attack classes or their constructor differences.
`ExperimentConfig` gained `attack_type` (`"label_flip"` | `"random_gradient"`), `attack_scale`
(`None`/`1.0` = unscaled), `source_label`, `target_label` -- the latter two promoted out of being
hardcoded in `get_client_fn()` so they can be recorded per result file. Deliberately **not** a new
sweep dimension in `run_configurations.py`'s `expand_config()`/`BASE_CONFIGS`: `attack_type`/
`attack_scale`/`source_label`/`target_label` instead join `SHARED_PARAMS`, so one hand-edited value
applies uniformly to every config in a run, avoiding a combinatorial explosion of variants.

Both `source_label`/`target_label` (when `attack_type="label_flip"`, else `null`) and
`attack_type`/`attack_scale` (always) are now recorded in every result file's `config` block
(`save_results()`), and shown in the bar chart's params caption
(`visualize_bar_chart_per_config()`) alongside clients/byzantine/rounds/root-dataset-size added
previously -- old result files without these keys still render via `.get()` fallback, confirmed
against a pre-existing 24-file result folder with no crash.

The whole design derives array shapes from the `parameters` argument `fit()` already receives
(never a hardcoded `MnistCNN()`), so it needs no changes when CIFAR-10 support lands.

## Limitation: epsilon=0.1 is infeasible at the current round count

`run_configurations.py`'s `EPSILON_VALUES` includes `0.1`. With this run's parameters (50
clients, `root_dataset_size=1000` -> ~1180 samples/client -> 37 batches/round -> sample_rate
≈ 1/37, 50 rounds), `compute_noise_multiplier()` (`mechanisms/dp.py`) cannot find a noise
multiplier that achieves that tight a budget over 50 rounds of composition -- Opacus's
`get_noise_multiplier()` raises `ValueError: The privacy budget is too low.` before any training
happens, in `MnistClient.__init__` (`client.py:63`). This crashes any config with `use_dp=True`
in the sweep, regardless of `attack_type` -- confirmed the crash is unrelated to the
`random_gradient`/`label_flip` attack work, since it happens during honest-client construction,
before any attack-specific code runs.

Reproduced directly against Opacus with this run's real sample_rate/round count:
`epsilon=0.1` fails (`"The privacy budget is too low."`), while `epsilon=1.0` (noise_multiplier
4.80), `epsilon=5.0` (1.34), and `epsilon=10.0` (0.91) all succeed. Not a code bug -- `0.1` is
genuinely infeasible at this sample_rate/round count; the fix is either dropping `0.1` from
`EPSILON_VALUES`, reducing `num_rounds` (fewer composition steps), or lowering the sample_rate
(larger per-client dataset share, e.g. fewer clients or a larger `root_dataset_size` trade-off,
per the earlier root-dataset-size discussion) -- not something fixable in `compute_noise_multiplier()`
itself.

## Limitation: cuDNN 9.20.x fails on some driver/GPU combinations (CIFAR-10 + GPU)

On one remote machine (2x RTX 4090, driver 595.71.05, CUDA 13.2), any `torch.nn.Conv2d` call on
`cuda` -- including the plain non-DP forward pass in `Cifar10ResNet20`, unrelated to this
project's own code -- fails with:
```
RuntimeError: cuDNN error: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```
This reproduces with a bare `torch.nn.Conv2d(3, 16, 3, 1, 1).cuda()` call, no Flower/Ray/FL code
involved, so it's not a bug in `models/cifar10_cnn.py`, `resolve_device()`, or the device-threading
work above -- it's an environment issue on that specific machine. Ruled out, in order: a stray
`LD_LIBRARY_PATH` shadowing the bundled cuDNN with a system copy (pinning `LD_LIBRARY_PATH` to the
exact matched bundled cuDNN dir didn't help), a corrupted/mismatched `nvidia-cudnn-cu13` pip
install (a from-scratch `pip install --force-reinstall` of a byte-identical, confirmed-working
(on a local RTX 4060 Ti, same driver-generation/compute-capability 8.9) `torch==2.12.1+cu130`
still failed identically), and a stale on-disk cuDNN JIT kernel cache under `~/.nv/ComputeCache`
(clearing it didn't help either). Also reproduces with `TORCH_CUDNN_V8_API_DISABLED=1` (forcing
cuDNN's older, pre-graph-API heuristics path), so it isn't specific to cuDNN 9's newer
`CUDNN_BACKEND_TENSOR_DESCRIPTOR` graph API either.

What actually fixed it: cuDNN 9 introduced a modular split into sub-libraries (`cudnn_cnn`,
`cudnn_ops`, `cudnn_graph`, etc.) that didn't exist in earlier cuDNN builds. On this driver, only
the specific `nvidia-cudnn-cu13-9.20.0.48` build (the one that ships with `torch==2.12.1`/`2.13.0`
+cu130) hits this; stepping back to a build bundling the earlier `nvidia-cudnn-cu12-9.1.0.70`
(still cuDNN 9, just a much earlier point release) works fine. Confirmed fix, satisfying both this
project's `pyproject.toml` (`torch>=2.6`) and Opacus's (`torch>=2.6.0`) floors:
```
pip install --force-reinstall --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```
Use `--extra-index-url`, not `--index-url`: the latter replaces PyPI entirely for that install,
and PyTorch's own package index has a `typing_extensions`/`typing-extensions` metadata-naming
mismatch (PEP 503 normalization) that makes pip reject every candidate it offers for that
dependency with no PyPI fallback to try instead. Also reinstall `torch`/`torchvision` together,
not one at a time -- reinstalling only `torch` while an incompatible `torchvision` pin (e.g.
`torchvision==0.28.0`, which requires `torch==2.13.0`) is still installed lets pip's resolver
silently keep satisfying the old pin instead of actually downgrading.

If this resurfaces on a different machine: confirm it's this same issue with the isolated
`Conv2d.cuda()` repro above (no FL code) before assuming a code regression, since the traceback
otherwise looks identical to a real bug (it surfaces inside `cifar10_cnn.py`'s first conv layer).
Not fixable from within this repo or a venv alone if it turns out to need an actual driver
update -- pin to an older cuDNN build as above instead.

## FLTrust trust-score decay investigation

`TODO.md`'s "Revisit trust scores" item flagged that honest-client trust scores don't reach the
expected high values and get worse over training, rather than staying high. This section documents
the investigation: what was ruled out, what the literature actually specifies, the fix that worked,
and the tradeoff it comes with.

### The problem, quantified

CIFAR-10, config 3 (FLTrust only, no DP/TopK), 100 rounds (`20260711_020358`): honest-client average
trust fell from 0.44 (round 1) to 0.32 (round 10) to 0.12 (round 100), while malicious-client trust
followed almost the same trajectory (0.50 -> 0.25 -> 0.12) -- the scores weren't just low, they were
barely distinguishing honest from malicious at all, and both decayed together toward the near-zero
baseline expected of two uncorrelated high-dimensional random vectors. Model accuracy kept improving
throughout (10% -> 51.5%), so this isn't the `total_trust == 0` skip-round path (bug 1 above) --
aggregation was working, just not discriminating well, and getting worse at it over time.

### Ruled out: root dataset size alone

First hypothesis: `root_dataset_size` (1000-2000 in production runs) is small relative to the model,
so the server's reference direction is dominated by which specific images happen to be in the root
set rather than a representative gradient direction, and this idiosyncrasy compounds as the root set
never gets refreshed across a run. Swept `root_dataset_size` in {500, 1000, 2000, 5000} on MNIST (6
clients, 2 Byzantine, 15 rounds, `ref_num_epochs=3` fixed, seed=42), holding everything else equal:
honest trust at round 15 rose monotonically with root size (0.11 -> 0.10 -> 0.17 -> 0.22), and so did
the honest/malicious gap (malicious converged to exactly 0.0 in every condition by round 15). Final
accuracy was unaffected by root size (0.976-0.978 across all four) -- a real, free improvement, but
the decay-over-rounds pattern persisted in every condition, just from a higher starting point. Root
size helps the *level*, not the *trend* -- ruled out as the primary fix.

### What the literature actually says

Pulled the original FLTrust paper ([Cao et al., NDSS 2021](https://arxiv.org/pdf/2012.13995)) and
read its algorithm and default hyperparameters directly, rather than assuming our recipe matched it.
Two things didn't match our implementation:

- **The paper's default recipe uses `Rl=1`** -- a single SGD step per round, for *both* the client
  and the server's reference model, not multiple local epochs. For MNIST specifically: 100 clients,
  a 100-sample root set, 1 local iteration per round, 2000 global rounds to compensate (Table I).
- **Their convergence proof (Theorem 1) is stated specifically for `Rl=1`.** The theoretical
  guarantee FLTrust is built on doesn't cover multi-epoch local training.

Our implementation did something structurally different: clients trained 1 full local epoch per
round (tens to hundreds of SGD steps depending on shard size), while the server trained
`ref_num_epochs=3` full epochs over the (much smaller) root set -- neither side matched the paper's
balanced single-step design, and the two sides didn't even match each other in step count (the
existing "reference update norm is ~6x smaller than a client's" note in `get_reference_update()`'s
docstring was a direct symptom of this). This lines up with a separate, well-documented phenomenon in
the broader FL literature -- **client drift**: the more local SGD steps a client takes per round, the
more its update reflects its own local data's curvature rather than a shared direction, especially as
the global model approaches convergence and different clients' local optima start to genuinely
disagree with each other (see FedProx and SCAFFOLD). That would explain both symptoms observed here:
weak trust separation (drift affects the reference-vs-client comparison) and decay over rounds (drift
gets worse, not better, as training progresses).

### Fix: `num_client_iterations_per_round`

Added `ExperimentConfig.num_client_iterations_per_round` (`int | None`, default `None` = unchanged
behavior): when set, both clients (`client.py`, `mechanisms/attacks.py`) and, when `use_fltrust=True`,
the FLTrust reference model (`mechanisms/robust_aggregation.py`) take exactly that many SGD steps per
round instead of a full local epoch -- directly operationalizing the paper's shared `Rl` as one config
value instead of two independently-tuned epoch counts. A small shared helper, `iterate_batches(loader,
num_steps)` (`client.py`), yields exactly `num_steps` batches, cycling the loader (with a fresh
shuffle each cycle, since `DataLoader.__iter__` draws a new permutation every call) if `num_steps`
exceeds one epoch's worth -- this decouples "how many SGD steps to take" from "how many batches this
particular shard/root set happens to contain." `get_reference_update()` was refactored to always be
step-based via this helper (callers translate `ref_num_epochs` into an equivalent step count when
`num_client_iterations_per_round` isn't set, reproducing the old behavior exactly, including the same
reshuffle cadence).

Note on data coverage at `Rl=1`: since Flower recreates each client object (and its `DataLoader`)
fresh every round, and `DataLoader.__iter__` reshuffles on every call regardless, a single step per
round does *not* mean the client keeps training on the same batch -- each round draws a genuinely
different random batch, and over many rounds this converges to ordinary sampling-with-replacement
coverage of the full shard, matching how the paper's Algorithm 1 phrases it ("randomly sample a batch
$D_b$ from $D$" each iteration). The server's `root_loader` gets the same treatment since it's a
persistent object but `iter(root_loader)` is still called fresh inside `aggregate_fit()` every round.

### Result: decay is eliminated -- at the cost of much slower convergence

Tested `num_client_iterations_per_round=1` against a same-setup baseline run (MNIST, config 3, 50
clients, 10 Byzantine, root=2000, seed=42, `attack_scale=2.0`, 600 rounds) that predates
`num_client_iterations_per_round` being wired into `run_configurations.py`'s `SHARED_PARAMS` (an
early gap in this work -- it existed on `ExperimentConfig` and was fully threaded through the
simulation internals, but had no way to actually be set via the script until fixed), so it used the
default full-local-epoch behavior. The two runs are otherwise directly comparable:

| | full local epoch/round (default) | `num_client_iterations_per_round=1` |
|---|---|---|
| honest trust, round 1 | 0.90 | 0.27 |
| honest trust, round 5 (peak) | 0.96 | -- |
| honest trust, round 100 | 0.046 | 0.229 |
| honest trust, round 600 | 0.009 | 0.322 |
| honest trust, first half avg (r1-300) | -- | 0.253 |
| honest trust, second half avg (r301-600) | -- | 0.432 |
| malicious trust, round 20 onward | 0.000 (exact) | fluctuates, 0.12-0.34 |
| accuracy, round 10 | 0.849 | 0.150 |
| accuracy, round 200 | 0.982 | 0.626 |
| accuracy, round 600 | 0.987 | 0.878 |

Under the default (full-epoch) regime, honest trust peaks early (round ~5) then decays monotonically
toward zero and stays there through round 600 -- the original problem, reproduced at a much longer
horizon than earlier tests, confirming it doesn't self-correct given more rounds. Under `Rl=1`, the
trend reverses: honest trust *rises* over the course of training (0.253 -> 0.432, first half to second
half average) instead of decaying, and never approaches zero at any point in the 600 rounds. This
directly confirms the client-drift hypothesis -- removing the local-step imbalance removes the
mechanism that was eroding trust scores as training progressed.

**The tradeoff is real and substantial: `Rl=1` trains far slower.** Accuracy reaches only 87.8% by
round 600, versus 98.7% by round 200 under the default regime -- roughly 3x the rounds for meaningfully
worse accuracy at the point this run stopped. This is the direct, expected cost of matching the
paper's design: far more communication rounds are needed to reach comparable model quality, exactly
as the paper's own `Rg=2000` (vs. our tested 600) anticipates. Anyone adopting `Rl=1` needs to budget
for substantially more rounds, or accept a lower-accuracy operating point, in exchange for trust
scores that stay meaningful throughout training instead of collapsing.

**Not fully solved by this fix**: per-round honest/malicious *separation* is still noisy under `Rl=1`.
Aggregated over all 600 rounds (24,000 honest vs. 6,000 malicious trust-score observations), honest
averages 0.343 vs. malicious's 0.278 -- a real, persistent gap -- but in any given individual round
the ordering is unreliable (malicious trust exceeded honest trust in several individual rounds, e.g.
rounds 5, 50, 100, 150, 350, 400). This tracks with each round's score now being based on a single
mini-batch comparison on each side, which is inherently higher-variance than the old multi-batch
epoch-averaged estimate. Decay (the trend problem) and separation reliability (the noise problem) looked
like distinct issues at this point -- see below for how that picture changed once the sweep was extended.

### A second paper-fidelity gap: batch size

While preparing a wider `Rl` sweep, noticed a second deviation from the paper's design, independent of
the local-iteration-count one above: `get_client_fn()`'s client loaders use `batch_size=32`, but
`load_datasets()`'s `root_loader` used `batch_size=128` (`server.py`, comment: "Larger than the client
batch_size on purpose"). The paper's Algorithm 2 calls `ModelUpdate(w, D, b, β, R)` with the *same* `b`
for both the client update (`g_i`) and the server update (`g_0`) -- Table I lists batch size as one
shared system parameter, not a per-side value. Even with `num_client_iterations_per_round` matching
step *counts*, a 128-sample server batch is a 4x smoother, lower-variance gradient estimate per step
than a 32-sample client batch -- exactly the kind of asymmetry this investigation is about removing.

Fix: added `CLIENT_BATCH_SIZE = 32` to `constants.py` (shared by both `client_fn()`'s client loaders and
`load_datasets()`'s `root_loader`, so the two can't drift apart silently again), replacing the two
previously-independent hardcoded values.

### Extending the sweep: `Rl` in {5, 10, 20}, and resolving the batch-size question

Before concluding anything from the single `Rl=1` data point above, re-ran the comparison properly:
`Rl` in {1, 5, 10, 20} plus a fresh `None` (full local epoch) baseline, all under the *corrected* batch
size, all same base setup (MNIST, config 3, 50 clients, 10 Byzantine, root=2000, seed=42,
`attack_scale=2.0`). `run_configurations.py` was extended with `RL_VALUES` (swept whenever
`use_fltrust=True`, mirroring how `EPSILON_VALUES`/`TOPK_VALUES` already sweep under `use_dp`/`use_topk`)
so `--config 3` runs the whole sweep in one invocation. `Rl=1` and `None` were re-run individually (300
rounds instead of 600, since there was no existing clean baseline at either value to match against
anyway, and `Rl=5`/`Rl=10`/`Rl=20` all showed their pattern stabilize well within the first 100-150
rounds) specifically to check whether the original `Rl=1` finding was actually caused by the batch-size
mismatch rather than by `Rl=1` itself, since that run predated the fix above.

| variant | rounds | elapsed | accuracy @ r300 | honest trust @ r300 (late-round pattern) | malicious @ r300+ |
|---|---|---|---|---|---|
| `None` (full epoch) | 300 | 26.2 min | **0.986** | 0.016 -- collapsed, flat near-zero from round ~100 on, no recovery | 0.000 |
| `Rl=1` | 300 | 17.7 min | 0.782 | 0.450 (round 300) -- high but very noisy throughout | 0.325 (noisy, not reliably below honest) |
| `Rl=5` | 600 | 35.7 min | 0.946 | 0.02-0.22 band, noisy but never collapses toward zero | 0.000 |
| `Rl=10` | 600 | 36.8 min | 0.969 | 0.05-0.19 band, same non-collapsing pattern | 0.000 |
| `Rl=20` | 600 | 40.1 min | 0.968 | 0.02-0.21 band, same pattern | 0.000 |

**The batch-size fix was not the explanation for either original finding.** Both re-runs used the
corrected batch size and reproduced the exact same behavior their unfixed predecessors showed: `Rl=1`
still noisy (malicious trust actually exceeds honest at round 10: 0.125 vs 0.091; the two stay close
together rather than cleanly separating), `None` still collapses to a near-zero floor with no recovery.
So neither the `Rl=1` noise nor the full-epoch decay was a batch-size artifact -- both are genuine
properties of their respective step-count regimes. `Rl=1`'s noise is best explained by a single
mini-batch producing an inherently high-variance cosine-similarity estimate regardless of how well the
two sides' batch sizes match; `None`'s collapse is the client-drift mechanism described above, which the
batch-size fix was never going to touch since it operates on a completely different axis (steps per
round, not sample count per step).

**The qualitative break is sharp, not gradual.** Going into this sweep, the expectation (from the
literature reasoning above) was that decay would gradually reappear as `Rl` increases toward a full
epoch. That's not what happened: every tested `Rl` value, including `Rl=20` (~55% of this setup's ~36
steps/epoch, and whose accuracy at round 600 is statistically indistinguishable from full-epoch's,
0.987 vs 0.987), retains a self-stabilizing, non-collapsing honest trust level with malicious reliably
crushed toward exactly zero. Only the true full-epoch case shows the severe, unrecovering decline. The
transition from "stable" to "collapsing" looks like it happens right at the full-epoch boundary itself,
not incrementally across the tested range.

**Practical read: `Rl=10` looks like the best point tested.** It reaches 96.9% accuracy by round 300
(vs. full-epoch's 98.6% -- a real but modest gap) while keeping the non-collapsing trust pattern and
clean malicious suppression, without `Rl=1`'s separation noise or `Rl=20`'s slightly wider trust swings.
Not exhaustively tuned (only 1, 5, 10, 20, and full-epoch were tested), so there may be a better point
nearby, but `Rl=10` is a reasonable default recommendation from what's been measured so far.

### Smoke-test fallout

Fixing `get_reference_update()`'s signature surfaced two pre-existing, unrelated bugs in
`mechanisms/robust_aggregation.py`'s own `__main__` smoke test (confirmed pre-existing by reproducing
both against the original unmodified code): it called `strategy.aggregate_fit()` directly without
`configure_fit()`, leaving `self.saved_global_parameters` as `None` (crash), and its fake `FitRes`
objects used `metrics={}`, which crashes once `aggregate_fit()` reads `fit_result.metrics['client_id']`
(added for the client-id metrics work above). Fixed both: `strategy.saved_global_parameters` is now
set directly (bypassing `configure_fit()`, which would otherwise need a real Flower `ClientManager`
just to sample clients this test doesn't use), and `make_fit_result()` now takes a `client_id` and
includes it in `metrics`. **Not fixed**: the test's "honest" fake updates (`ref * 0.9`, `ref + noise`)
are derived from a *different*, independently-randomly-initialized `MnistCNN()` instance than the one
`FLTrustStrategy` trains internally when `aggregate_fit()` runs -- two unrelated random directions in
a ~106K-dim space have near-zero cosine similarity by construction, so the test's "honest" clients
score just as low as its "malicious" one (confirmed this also happens on the original code, so it
predates this session). Flagged, not fixed -- the real fix would derive the fake honest updates from
`strategy.ref_model`'s actual parameters instead of a separate model instance.

## `TODO.md` "What to do next": bar sort order, Excel export, communication-size metric

Addressed the three items `TODO.md` listed under "What to do next".

**1. `bar_accuracy` bar order.** `visualize_bar_chart_per_config()` (`scripts/visualize_results.py`)
sorted each config's variants by `r["_filename"]` (string sort), which only happened to look right by
coincidence -- e.g. an epsilon of `5.0` would sort *after* `10.0`, since `"5" > "1"` lexicographically.
Fix: `variant_sort_key()` sorts on the actual numeric config fields instead
(`topk_ratio, epsilon, num_client_iterations_per_round, num_clients`, outermost to innermost).

**2. Excel export.** New `export_accuracy_tables_excel()` (`scripts/visualize_results.py`, `--plot
excel`, included in `--plot all`), writing `accuracy_by_round.xlsx`. Reproduces the hand-built layout
in `tmp/template_excel_output.xlsx`: one bordered table per config, placed side by side left to right,
each with a bold title (`replace_config_with_label()`), `Clients`/`Epsilon`/`TopK` rows giving that
column's value for whichever mechanisms this config actually has on (`"-"` for an inactive one, e.g.
`Epsilon` when `use_dp=False` -- not omitted, so every table's rows line up across the sheet), a
`Round`/`Accuracy` header row, and one data row per round. An `Rl` row (`num_client_iterations_per_round`)
is inserted between `Epsilon` and `TopK` only if at least one result in the folder actually has it set,
so a folder where nothing used it doesn't get a meaningless all-`"-"` row. Needed adding `openpyxl` to
`pyproject.toml` (`pandas` was already a dependency, but its `.xlsx` writer needs an engine).

**3. `update_bytes` metric.** New `update_size_bytes()` (`mechanisms/topk.py`), called from
`MnistClient.fit()` (`client.py`) and both attack `fit()` overrides (`mechanisms/attacks.py`), in both
the TopK and non-TopK branches, so every client reports it every round. For non-TopK configs this is
the real, actual cost -- the dense vector is genuinely what's serialized across the client/server
boundary today. For TopK configs it's a logical/computed number instead, **not** a measurement of
what actually crosses the wire in this simulation right now -- fit() still returns a dense array
regardless of topk_ratio (see item 15 above, Won't-fix / future work, for why), so a TopK config's
*real* current bytes-on-wire equals the dense case, unaffected by topk_ratio; what's reported here is
what a real sparse encoding *would* cost once that's implemented. Two
modeling choices, decided with the user: (a) a TopK-sparsified entry costs `value + 4-byte index`
(8 bytes for a float32 update), not just the raw value, since the receiver can't place a value
without knowing which position it belongs to -- reporting value-only would overstate TopK's real
compression benefit; (b) the per-round value is a **total** across all clients (summed), not a
per-client average like every other fit metric -- `weighted_average_metrics()` (`server.py`) now
excludes `"update_bytes"` from its generic averaging, and `HistoryStrategyAdapter.aggregate_fit()`
sums it directly from `results` instead (same pattern already used there for the per-client
`malicious_<id>` metrics). `run_configurations.py::save_results()` now writes it into each round's
entry in the result JSON, alongside `loss`/`accuracy`/`epsilon`.

**Found while testing, not a regression from this work**: `MnistClient.get_parameters()`
(`client.py`) returns `.cpu().numpy()` on each parameter tensor -- on CPU (every MNIST run, and any
CIFAR-10 run's client-side reads before a GPU round-trip), `.cpu()` is a no-op and `.numpy()` returns
a view sharing memory with the live `nn.Module` parameter, not a copy. Reproduced directly: capturing
`parameters = client.get_parameters(...)` and later calling `client.fit(parameters, ...)` on the
*same* client object silently corrupts the "before training" snapshot mid-`fit()`, since the
in-place `optimizer.step()` mutates that same backing memory -- `topk_sparsity`/`update_bytes` came
out as exactly `0.0` until the test was fixed to pass in an explicit `.copy()`. Confirmed this cannot
affect any real run in this codebase: Flower's simulation runs each client in its own Ray worker
*process* (not just a thread, per "Seed control" above), so parameters are always serialized across
that process boundary before a client sees them -- there is no in-process aliasing between a client's
own `get_parameters()` output and the `parameters` argument its own `fit()` receives. Purely a hazard
for same-process test code (like the one that surfaced it) or a future single-process refactor.
**Status: Not fixed** -- out of scope for a metrics-only change, and doesn't affect any existing or
future result as long as clients stay in separate processes.

## `TODO.md` "What to do next": trust plots restricted to non-DP FLTrust, trilemma axis TODOs

**1. Trust plots now exclude DP+FLTrust configs.** `visualize_trust_per_client()`,
`visualize_trust_over_rounds()`, and `visualize_trust_over_rounds_per_config()`
(`scripts/visualize_results.py`) all filtered to `use_fltrust` only, which included the DP+FLTrust
configs (5, 8) alongside plain FLTrust ones (3, 7). Per the earlier trust-score investigation above
("Differential Privacy and FLTrust do not work well with each other"), DP noise degrades the
cosine-similarity trust signal enough that mixing DP+FLTrust variants into these plots muddies the
honest-vs-malicious separation they're meant to show. Fix: all three now filter to `use_fltrust and
not use_dp`. Note this changes `visualize_trust_over_rounds_per_config()`'s scope specifically -- its
docstring used to say its per-config subplot split existed "so DP/TopK's effect on trust isn't
averaged away against plain-FLTrust configs"; with DP+FLTrust excluded, it now only compares TopK's
effect (configs 3 vs 7), not DP's, since there's no DP+FLTrust subplot left to compare against.
Docstring updated to reflect this. Verified against a real result folder (24 files, all 8 configs):
the filtered results only contain config_ids {3, 7}, as expected.

**2. Trilemma triangle axis calculations -- decided, not yet implemented.** Current formulas
(`get_base_accuracy_and_max_epsilon()` + `visualize_radar_chart()`/`visualize_radar_chart_per_config()`,
`scripts/visualize_results.py`) have three known weaknesses, discussed with the user; the fixes below
are agreed but deliberately not implemented yet (do this as a dedicated follow-up, not bundled in):

- **Robustness** (currently `= final_accuracy`, unnormalized) -> switch to relative-to-clean-baseline:
  `robustness = (final_accuracy - base_accuracy) / (clean_accuracy - base_accuracy)`, where
  `base_accuracy` is config 1's accuracy *under* the same attack (already computed by
  `get_base_accuracy_and_max_epsilon()`, but currently discarded -- both radar functions call it as
  `_, max_epsilon = get_base_accuracy_and_max_epsilon(results)`) and `clean_accuracy` is config 1's
  accuracy with **no attack at all**. This properly isolates attack-resistance from raw accuracy,
  matching the function's own docstring, which already describes this exact comparison
  ("Performance (final_accuracy_no_attack baseline comparison)") but never implemented it.
  **Blocked on**: `clean_accuracy` needs a no-attack BASE-config run, which doesn't exist yet --
  `TODO.md`'s "Upcoming runs" already lists this ("Run ohne attack mit BASE config"), so this becomes
  implementable once that run exists.
- **Privacy** (currently `= max(1 - epsilon/max_epsilon, 0)`, where `max_epsilon` is the largest
  epsilon found in the *current results folder*) -> **left as-is for now**. Known issue: the
  folder-relative denominator means the same run can score very differently depending on what other
  epsilon values happen to be bundled into the same folder, so radar charts from different folders
  aren't on a comparable scale. Discussed fixing this with a fixed project-wide epsilon ceiling
  (linear or log-scaled), but decided to defer -- revisit alongside the Robustness fix once that's
  underway, rather than as an isolated change.
- **Efficiency** (currently `= 1 - topk_ratio` when `use_topk`, else `0`) -> switch to using the
  `update_bytes` metric: `efficiency = 1 - (update_bytes / dense_reference_bytes)`, where
  `dense_reference_bytes` is the same model's dense parameter count * dtype size (a fixed constant
  per dataset, not folder-relative). More accurate than `1 - topk_ratio`, which ignores the
  per-entry index overhead `update_bytes` already accounts for (`1 - topk_ratio` overstates TopK's
  benefit by roughly 2x relative to what `update_bytes` reports for the same config, since a kept
  entry costs `value + index`, not just `value`). **Caveat carried over from the `update_bytes`
  metric itself** (see the "communication-size metric" entry above): for TopK configs, `update_bytes`
  is currently the *hypothetical* cost if real sparse-wire compression were implemented, not what's
  actually transmitted in this simulation today (still a dense array either way, see item 15).
  Switching Efficiency to use it means this axis would represent TopK's *potential* efficiency, not
  its current actual efficiency (which is 0% today, matching item 15's finding) -- worth labeling the
  axis clearly (e.g. in a legend/caption) once implemented, so it isn't read as a present-tense claim.

## `TODO.md` "What to do next": elapsed-time Excel sheet

Added an "Elapsed Time" sheet to the Excel report (`scripts/visualize_results.py`), alongside the
existing "Accuracy by Round" sheet -- same file (`accuracy_by_round.xlsx`), two tabs, both built by
the renamed entry point `export_excel_report()` (was `export_accuracy_tables_excel()`). Per-config
tables reuse the same title/`Clients`/`Epsilon`/`Rl`/`TopK` header block and border styling as the
accuracy sheet, but instead of a `Round`/`Accuracy` header row + one row per round, each variant gets
two summary rows: `Avg s/round` (`elapsed_seconds / num_rounds`) and `Total elapsed (s)`
(`elapsed_seconds` as-is), both already present in every result JSON's `results.elapsed_seconds`
field (`run_configurations.py::save_results()`) -- no new data collection needed, purely a new export.

Extracted the shared scaffold (title, `Clients`/`Epsilon`/`Rl`/`TopK` rows, and the borders down
through `TopK`'s double-bottom separator) into `_write_config_param_block()`, called by both
`_write_accuracy_sheet()` and the new `_write_elapsed_time_sheet()` -- addresses `TODO.md`'s
long-standing "Refactor redundancy" bullet for this specific duplication, since a second near-copy of
the whole scaffold would otherwise have been needed. Each sheet-builder finishes its own box/divider
border below the block (medium outer box, thin label-column divider), since the two sheets have a
different number of data rows below `TopK` (accuracy: header row + `num_rounds` rows; elapsed time:
always exactly 2 summary rows) and only the caller knows its own final row once it's written them.
Verified against a real result folder (24 files, all 8 configs): both sheets render with matching
header blocks and correct borders, `Avg s/round` values divide out consistently with `Total elapsed
(s)` and each variant's `num_rounds`.

## Full grid run analysis (`20260712_035938`)

Analysis of the first completed full factorial sweep (`TODO.md`'s "Upcoming runs" item 2): MNIST, all
8 configs, DP configs expanded over epsilon in {1, 10}, TopK configs over k in {0.1, 0.5}, every config
expanded over `num_clients` in {10, 30, 60} with `num_byzantine` scaled to hold the 20% malicious
fraction constant, 500 rounds each, label-flip attack (source=3, target=7, `attack_scale=2.0`), seed=42.
54 result files, 26.3h total wall-clock (`run_summary.json`). Note: `run_configurations.py`'s
`NUM_CLIENTS` now also lists 80 and 100, but git history confirms that was added after this run was
started -- these 54 files only cover the 10/30/60 tier, not the 5-tier grid the script currently
describes.

### Accuracy: baseline is already attack-tolerant at scale; DP is the expensive mechanism

Final-round accuracy (n=60, i.e. the largest/most stable tier): baseline (config 1, no defense, under
attack) 0.9785, FLTrust alone (config 3) 0.9798, DP alone (config 2) 0.787-0.802 depending on epsilon,
DP+TopK (config 6) 0.626-0.796 (worst combination in the grid), FLTrust+TopK (config 7) 0.975-0.980
(barely distinguishable from FLTrust alone). Two things follow: the 20%-Byzantine label-flip attack at
`attack_scale=2.0` is mild enough that plain FedAvg already dilutes it well once n>=30 (source-class "3"
recall only drops to 0.96 at baseline n=60 vs 0.98 with FLTrust), so FLTrust's accuracy contribution
here is modest; DP is what actually costs accuracy (recall(3) drops to 0.64-0.68), and stacking TopK on
top of DP compounds that further (recall(3) as low as 0.54, recall(7) as low as 0.26 for config 6,
k=0.1, n=60) -- the sharpest visible privacy+efficiency-vs-performance tension in the whole grid.

### DP+FLTrust epsilon inversion

Without FLTrust, epsilon behaves as expected: config 2, n=10, epsilon=10 (0.797) beats epsilon=1
(0.783), a small gain in the expected direction. With FLTrust on (configs 5, 8), this flips at every
client count: config 5, n=60, epsilon=1 gives 0.9335 vs epsilon=10's 0.8577 -- a 7.6-point gain from
*more* noise, not less. Same pattern in config 8 (all three mechanisms on). `noise_multiplier` values
confirm the accountant itself is fine (epsilon=1 -> noise_mult 6.8-16.4, epsilon=10 -> noise_mult
1.1-2.2, correct direction; measured cumulative epsilon after 500 rounds stays well under target in
both cases). The trust-score data below explains the flip: under DP, FLTrust's honest-vs-malicious
separation nearly vanishes, so at epsilon=10 (less noise) the label-flip attacker's *coherent* wrong
gradient survives mostly intact and FLTrust is too noise-corrupted to discount it, while at epsilon=1
(heavy noise) that same attack signal gets scrambled by the same noise scrambling the honest updates,
so it partially cancels out during averaging even though FLTrust isn't really discriminating either. In
short: extra DP noise accidentally acts as its own Byzantine defense here, while FLTrust contributes
essentially nothing once DP is active -- a sharper, numerically confirmed version of this file's
opening "DP and FLTrust do not work well with each other" note.

### Trust scores: clean separation without DP, destroyed by DP

Without DP (configs 3, 7): honest and malicious trust overlap for the first ~20-40 rounds (~0.4-0.65,
before the global model has converged enough for direction to matter), then split cleanly -- honest
climbs to ~0.8 then settles into a noisy ~0.10 floor with occasional spikes to 0.3-0.5, malicious
collapses to ~0.001-0.008 and stays there for the remaining ~460 rounds. That is a sustained 10-100x
gap -- malicious clients are easily distinguishable here once FLTrust has warmed up. With DP added
(configs 5, 8): honest and malicious trust are both ~0.001-0.01 from round 1 onward and stay
statistically indistinguishable the entire run (e.g. config 5, epsilon=1, n=60: honest late-round mean
0.0012 vs malicious 0.0011). DP noise does not just cost accuracy directly -- it disables FLTrust's own
discriminative signal as a side effect, so the two mechanisms actively undermine each other rather than
just separately taxing performance.

### Communication cost: not measured for this run, and TopK saves nothing on the wire regardless

These 54 result files predate the `update_bytes` metric (added in a later commit, see "communication-
size metric" above) -- none of them record bytes transmitted. Computed manually instead:
`MnistCNN` has 105,866 float32 parameters (423,464 bytes/client/round dense), and every strategy here
uses `fraction_fit=1.0` (all clients every round), so real bytes-on-wire for *any* config in this grid
is `num_clients * 500 * 423,464` regardless of `use_topk` or `topk_ratio` -- per item 15 above, `fit()`
still returns a dense array today, so TopK configs transmit exactly as many bytes as non-TopK ones at
matching `num_clients`. Even the *logical* sparse-encoding estimate `update_bytes` would report if wired
through only pays off below k=0.5: at k=0.5, `nonzero * (value + 4-byte index)` exactly equals the dense
size (zero saving), and only k=0.1 would yield a real reduction (~80%) if implemented. So every TopK
config in this grid (4, 6, 7, 8) paid its accuracy cost (confirmed above, real) for a communication
benefit that is currently neither implemented in the simulation nor recorded in these result files.

### Visualization: `radar_trilemma.png` doesn't scale to 54 variants

`visualize_radar_chart()` draws one polygon per result file with no per-config color grouping; at 54
files the legend is illegible and the overlapping fills obscure most individual variants.
`radar_per_config.png` (one subplot per config) stayed readable and is the more useful chart for a
folder this size -- worth defaulting to for large sweeps rather than the single combined chart.
