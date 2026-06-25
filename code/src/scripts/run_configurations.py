
import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
import sys
import time
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.constants import NUM_CLASSES_MNIST
from src.experiment_config import ExperimentConfig
from src.server import run_simulation_with_config

# --------- Global setup ----------
# Shared parameters across all experiments
SHARED_PARAMS = dict(
    num_clients = 15,
    num_rounds = 50,
    num_byzantine = 3,
    root_dataset_size = 800,
    rescale_to_ref_norm = False,
)


BASE_CONFIGS = {
    1: ExperimentConfig(
        config_id=1,
        use_dp=False, use_fltrust=False, use_topk=False, 
        **SHARED_PARAMS,
    ),
    2: ExperimentConfig(
        config_id=2,
        use_dp=True, use_fltrust=False, use_topk=False,
        **SHARED_PARAMS,
    ),
    3: ExperimentConfig(
        config_id=3,
        use_dp=False, use_fltrust=True, use_topk=False,
        **SHARED_PARAMS,
    ),
    4: ExperimentConfig(
        config_id=4,
        use_dp=False, use_fltrust=False, use_topk=True,
        **SHARED_PARAMS,
    ),
    5: ExperimentConfig(
        config_id=5,
        use_dp=True, use_fltrust=True, use_topk=False,
        **SHARED_PARAMS,
    ),
    6: ExperimentConfig(
        config_id=6,
        use_dp=True, use_fltrust=False, use_topk=True,
        **SHARED_PARAMS,
    ),
    7: ExperimentConfig(
        config_id=7,
        use_dp=False, use_fltrust=True, use_topk=True,
        **SHARED_PARAMS,
    ),
    8: ExperimentConfig(
        config_id=8,
        use_dp=True, use_fltrust=True, use_topk=True,
        **SHARED_PARAMS,
    ),
}

# Variants to explore
EPSILON_VALUES = [1.0, 5.0, 10.0]
TOPK_VALUES = [0.01, 0.1]


def expand_config(base: ExperimentConfig) -> list[ExperimentConfig]:
    """
    Expand a base config into all its variants.

    Configs with DP enabled get expanded over epsilon values.
    Configs with TopK enabled get expanded over topk_ratio values.
    Expansions are combined when multiple are enabled.

    Args:
        base (ExperimentConfig): base config to expand.

    Returns:
        list[ExperimentConfig]: list of expanded configs.
    """
    epsilons = EPSILON_VALUES if base.use_dp else [base.epsilon]
    topk_vals = TOPK_VALUES if base.use_topk else [base.topk_ratio]

    variants = []
    for epsilon in epsilons:
        for k in topk_vals:
            variants.append(replace(base, epsilon=epsilon, topk_ratio=k))
    return variants


def make_filename(config: ExperimentConfig, run_timestamp: str) -> str:
    """
    Generate a descriptive filename for this experiment's results.

    Args:
        config (ExperimentConfig): the experiment configuration.
        run_timestamp (str):       shared timestamp for this run

    Returns:
        str: filename without extension, e.g.
             20260622_143022_dp-True_eps-10.0_fltrust-True_topk-True_k-0.1_rounds-3_clients-5_byzantine-1
        str: timestamp at the beginning of the filename 
    """
    parts = [
        run_timestamp,
        f"config-{config.config_id}",
        f"dp-{config.use_dp}"
    ]
    if config.use_dp:
        parts.append(f"epsilon-{config.epsilon}")
    parts += [
        f"fltrust-{config.use_fltrust}",
        f"topk-{config.use_topk}",
    ]
    if config.use_topk:
        parts.append(f"k-{config.topk_ratio}")
    parts += [
        f"rounds-{config.num_rounds}",
        f"clients-{config.num_clients}",
        f"byzantine-{config.num_byzantine}",
    ]
    return "_".join(parts) + ".json"


def inflate_confusion_matrix_mnist_and_calculate_scores(config: ExperimentConfig, history: dict):
    """
    Reconstruct confusion matrix and compute per-class metrics from history.

    Extracts the flat cm_i_j metrics from the final round of the distributed
    evaluate history, reconstructs the full 10x10 confusion matrix, and
    computes precision, recall and F1 score per digit class.

    Args:
        config (ExperimentConfig): experiment configuration, used for num_rounds
                                   and num_classes.
        history (dict):            run history from run_simulation_with_config,
                                   must contain metrics_distributed_evaluate
                                   with cm_i_j keys.

    Returns:
        tuple: (per_class, confusion_matrix)
            - per_class (dict | None):        dict mapping digit class (str) to
                                              precision, recall and f1 scores.
                                              None if no cm_ entries found.
            - confusion_matrix (list | None): 10x10 list of lists where
                                              confusion_matrix[i][j] is the count
                                              of samples with true label i
                                              predicted as j.
                                              None if no cm_ entries found.
    """
    last_round = config.num_rounds
    cm_entries = {
        key: dict(vals).get(last_round)
        for key, vals in history["metrics_distributed_evaluate"].items()
        if key.startswith("cm_")
    }
    confusion_matrix = None
    if cm_entries:
        confusion_matrix = [
            [int(cm_entries.get(f"cm_{i}_{j}", 0) or 0) for j in range(NUM_CLASSES_MNIST)]
            for i in range(NUM_CLASSES_MNIST)
        ]

    # Compute per-class metrics from confusion matrix
    per_class = None
    if confusion_matrix:
        per_class = {}
        for i in range(NUM_CLASSES_MNIST):
            tp = confusion_matrix[i][i]
            fp = sum(confusion_matrix[r][i] for r in range(NUM_CLASSES_MNIST)) - tp
            fn = sum(confusion_matrix[i][c] for c in range(NUM_CLASSES_MNIST)) - tp
            precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[str(i)] = {
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
            }

    return per_class, confusion_matrix


def save_results(config: ExperimentConfig, history,
                 elapsed_seconds: float, run_timestamp: str, multi_run=False):
    """
    Save experiment results to a JSON file in results/.

    Args:
        config (ExperimentConfig):  the experiment configuration.
        history:                    Flower history object from run_simulation.
        elapsed_seconds (float):    total wall-clock time for this run.
        run_timestamp (str):        shared timestamp for this run
        multi_run (bool):           whether this is part of a multi run or not (for directory creation)
    """
    if multi_run:
        folder = os.path.join("results", run_timestamp)
    else:
        folder = "results"
    os.makedirs(folder, exist_ok=True)

    filename = make_filename(config, run_timestamp)
    filepath = os.path.join(folder, filename)

    # Extract per-round metrics from history
    losses     = dict(history["losses_distributed"])
    accuracies = dict(history["metrics_distributed_evaluate"].get("accuracy", []))
    epsilons   = dict(history["metrics_distributed_fit"].get("epsilon", []))

    per_class_scores, confusion_matrix = inflate_confusion_matrix_mnist_and_calculate_scores(config, history)

    results = {
        "config": {
            "config_id": config.config_id,
            "num_clients": config.num_clients,
            "num_rounds": config.num_rounds,
            "num_byzantine": config.num_byzantine,
            "use_dp": config.use_dp,
            "epsilon": config.epsilon,
            "delta": config.delta,
            "use_fltrust": config.use_fltrust,
            "use_topk": config.use_topk,
            "topk_ratio": config.topk_ratio,
            "root_dataset_size": config.root_dataset_size,
            "rescale_to_ref_norm": config.rescale_to_ref_norm,
        },
        "results": {
            "elapsed_seconds": elapsed_seconds,
            "noise_multiplier": dict(history["metrics_distributed_fit"].get("noise_multiplier", [])).get(1),  # same across all rounds -> take round 1
            "confusion_matrix": confusion_matrix,
            "per_class_scores": per_class_scores,
            "per_round": [
                {
                    "round": r,
                    "loss": losses.get(r),
                    "accuracy": accuracies.get(r),
                    "epsilon": epsilons.get(r),
                }
                for r in range(1, config.num_rounds + 1)
            ]
        }
    }

    with open(filepath, "w") as file:
        json.dump(results, file, indent=2)

    print(f"Results saved to: {filepath}")



HELP_MENU = """
FL Trilemma Grid Runner
-------------------------

Available configs:
  1  Baseline     dp=off  fltrust=off  topk=off
  2  DP only      dp=on   fltrust=off  topk=off   (runs e=1, e=5, e=10)
  3  FLTrust only dp=off  fltrust=on   topk=off
  4  TopK only    dp=off  fltrust=off  topk=on    (runs k=1%, k=10%)
  5  DP+FLTrust   dp=on   fltrust=on   topk=off   (runs e=1, e=5, e=10)
  6  DP+TopK      dp=on   fltrust=off  topk=on    (runs e=1,5,10 x k=1%,10%)
  7  FLTrust+TopK dp=off  fltrust=on   topk=on    (runs k=1%, k=10%)
  8  All three    dp=on   fltrust=on   topk=on    (runs e=1,5,10 x k=1%,10%)

Usage:
  python scripts/run_grid.py --config 1    run config 1 (baseline)
  python scripts/run_grid.py --config 2    run config 2 (all epsilon variants)
  python scripts/run_grid.py --all         run all configs sequentially

Results saved to: results/<timestamp>_<params>.json
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=HELP_MENU,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=int, choices=range(1, 9),
                       metavar="N", help="run config N (1-8)")
    group.add_argument("--all", action="store_true",
                       help="run all configs sequentially")
    args = parser.parse_args(args=None if len(sys.argv) > 1 else ["--help"])

    if args.all:
        configs_to_run = [c for base in BASE_CONFIGS.values()
                          for c in expand_config(base)]
    else:
        configs_to_run = expand_config(BASE_CONFIGS[args.config])
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    multi_run = len(configs_to_run) > 1

    print(f"Configs to run: {len(configs_to_run)}")
    for config in tqdm(configs_to_run, desc="Grid runs", position=0, leave=True):
        print(f"\nRunning config {config.config_id} | dp={config.use_dp} / epsilon={config.epsilon} / "
              f"fltrust={config.use_fltrust} / topk={config.use_topk} / k={config.topk_ratio} / "
              f"multirun={len(configs_to_run)>1}")
        print(f"\n\n{'-'*10} RUN {'-'*10}\n\n")
        start_time = time.time()
        history = run_simulation_with_config(config)
        elapsed = time.time() - start_time
        save_results(config, history, elapsed, run_timestamp=run_timestamp, multi_run=multi_run)
        print(f"Elapsed for this config: {elapsed} seconds\n")
        print(f"\n\n{'-'*10} END {'-'*10}\n\n")
    
    print("\nAll runs complete!")