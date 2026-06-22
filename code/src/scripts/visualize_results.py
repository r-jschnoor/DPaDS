import argparse
import json
import os
import sys
import glob
import matplotlib.pyplot as plt


def load_results(folder: str) -> list[dict]:
    """
    Load all JSON result files from a folder.

    Args:
        folder (str): path to folder containing result JSON files.

    Returns:
        list[dict]: list of loaded result dicts, sorted by filename.
    """
    pattern = os.path.join(folder, "*.json")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No JSON files found in {folder}")
        sys.exit(1)

    results = []
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
            data["_filename"] = os.path.basename(filepath)
            results.append(data)

    print(f"Loaded {len(results)} result files from {folder}")
    return results


def make_label(filename: str) -> str:
    """
    Generate a short human-readable label from a result filename.

    Extracts config-N, dp, fltrust, topk fields from the filename.

    Args:
        filename (str): result JSON filename.

    Returns:
        str: short label e.g. 'config-1_dp-False_fltrust-True_topk-False'
    """
    parts  = filename.replace(".json", "").split("_")
    keep   = ["config", "dp", "fltrust", "topk"]
    labels = []
    i = 0
    while i < len(parts):
        for key in keep:
            if parts[i].startswith(key):
                # include epsilon if dp=True
                if key == "dp" and parts[i] == "dp-True" and i + 1 < len(parts):
                    if parts[i + 1].startswith("epsilon"):
                        labels.append(f"{parts[i]}_{parts[i+1]}")
                        i += 1
                        break
                # include k if topk=True
                if key == "topk" and parts[i] == "topk-True" and i + 1 < len(parts):
                    if parts[i + 1].startswith("k"):
                        labels.append(f"{parts[i]}_{parts[i+1]}")
                        i += 1
                        break
                labels.append(parts[i])
                break
        i += 1
    return "_".join(labels)


def extract_metric(result: dict, metric: str) -> tuple[list[int], list[float]]:
    """
    Extract per-round values for a given metric from a result dict.

    Args:
        result (dict): loaded result JSON.
        metric (str):  one of 'loss', 'accuracy', 'epsilon'.

    Returns:
        tuple: (rounds, values) - both lists, None values filtered out.
    """
    rounds = []
    values = []
    for entry in result["results"]["per_round"]:
        val = entry.get(metric)
        if val is not None:
            rounds.append(entry["round"])
            values.append(val)
    return rounds, values


def visualize(folder: str) -> None:
    """
    Load results from folder and plot loss, accuracy, epsilon per round.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)
    metrics = ["loss", "accuracy", "epsilon"]
    titles  = {
        "loss":     "Loss per Round",
        "accuracy": "Accuracy per Round",
        "epsilon":  "Cumulative Epsilon per Round (DP configs only)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"FL Trilemma Results {os.path.basename(folder)}", fontsize=14)

    for ax, metric in zip(axes, metrics):
        ax.set_title(titles[metric])
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, alpha=0.3)

        any_plotted = False
        for result in results:
            rounds, values = extract_metric(result, metric)
            if not values:
                continue
            label = make_label(result["_filename"])
            ax.plot(rounds, values, marker="o", label=label)
            any_plotted = True

        if any_plotted:
            ax.legend(
                fontsize=6,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),    # place below the axis
                ncol=2,                         # two columns to keep it compact
                framealpha=0.8,
            )
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)   # make room for legends below
    output_path = os.path.join(folder, "visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize FL trilemma results from a results folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="path to folder containing result JSON files",
    )
    args = parser.parse_args(args=None if len(sys.argv) > 1 else ["--help"])
    visualize(args.folder)