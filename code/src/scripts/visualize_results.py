import argparse
from collections import defaultdict
import json
import os
import sys
import glob
import matplotlib.pyplot as plt
from networkx import efficiency
from numpy import sort
import numpy as np


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


def visualize_overview(folder: str) -> None:
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


def visualize_lines_per_config(folder: str) -> None:
    """
    One figure per config, three subplots (loss, accuracy, epsilon).
    
    One curve per variant within each config.
    Saves 8 separate PNG files, one per config.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)

    config_groups = defaultdict(list)
    for result in results:
        config_groups[result["config"]["config_id"]].append(result)

    metrics = ["loss", "accuracy", "epsilon"]
    titles  = {
        "loss":     "Loss per Round",
        "accuracy": "Accuracy per Round",
        "epsilon":  "Epsilon per Round",
    }

    # Output dir
    subfolder_name = "per_config_line_charts"
    os.makedirs(os.path.join(folder, subfolder_name), exist_ok=True)

    for config_id, group_results in sorted(config_groups.items()):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Config {config_id} - Per Round Metrics", fontsize=13)

        for ax, metric in zip(axes, metrics):
            ax.set_title(titles[metric])
            ax.set_xlabel("Round")
            ax.set_ylabel(metric.capitalize())
            ax.grid(True, alpha=0.3)

            for result in group_results:
                rounds, values = extract_metric(result, metric)
                if not values:
                    continue
                label = make_label(result["_filename"])
                ax.plot(rounds, values, marker="o", label=label)
            
            # Make sure empty plots dont throw warnings if there is nothing to plot
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    fontsize=7,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=1,
                    framealpha=0.8,
                )
            else:
                ax.text(0.5, 0.5, "N/A for this config",
                        ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        output_path = os.path.join(folder, subfolder_name, f"line_chart_config_{config_id}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.close()


def visualize_bar_chart_per_config(folder: str) -> None:
    """
    Bar chart of final accuracy per variant, grouped by config.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)

    # Group results
    groups = {}
    for result in results:
        config_id = result["config"]["config_id"]
        groups.setdefault(config_id, []).append(result)

    # Build bar data
    config_ids = sorted(groups.keys())
    group_bars = []             # [(label, accuracy)] per group
    for config_id in config_ids:
        bars = []
        for result in sorted(groups[config_id], key=lambda r: r["_filename"]):
            # Take only final rounds accuracy
            final_accuracy = result["results"]["per_round"][-1].get("accuracy")
            if final_accuracy is None:
                print(f"Could not find accuracy of final round for one of configs {config_id} results. ({result["_filename"]})")
                continue
            label = make_label(result["_filename"])
            bars.append((label, final_accuracy))
        group_bars.append((config_id, bars))

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title("Final Round Accuracy per Config Variant", fontsize=13)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    x         = 0          # current x position
    tick_pos  = []         # x positions for group labels
    tick_lab  = []         # group label text
    bar_width = 0.6
    gap       = 1.2        # gap between config groups

    colors = plt.cm.tab10.colors
    hatches = ['--', '++', 'xx', '//', '\\\\', '||', '*', 'o', '.']

    for config_id, bars in group_bars:
        group_start = x
        for i, (label, acc) in enumerate(bars):
            color = colors[config_id % len(colors)]
            hatch = hatches[i % len(hatches)]
            bar = ax.bar(x, acc, width=bar_width,
                         color=color, alpha=0.85,
                         hatch=hatch, edgecolor="white",
                         label=label)
            ax.text(x, acc + 0.01, f"{acc:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=45)
            x += bar_width + 0.1

        # center the group label under its bars
        group_center = (group_start + x - bar_width - 0.1) / 2
        tick_pos.append(group_center)
        tick_lab.append(f"Config {config_id}")
        x += gap

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=9)

    # Legend below plot
    handles = []
    for config_id, bars in group_bars:
        for i, (label, _) in enumerate(bars):
            handles.append(plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=colors[config_id % len(colors)],
                hatch=hatches[i % len(hatches)],
                edgecolor="white",
                label=label,
                alpha=1,
            ))
    ax.legend(
        handles=handles,
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        framealpha=0.8,
        handleheight=2,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    output_path = os.path.join(folder, "bar_accuracy.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


def get_base_accuracy_and_max_epsilon(results):
    """
    Find base accuracy and max epsilon in results.

    Args:
        results (list): A list of results dicts

    Returns:
        tuple: (base_accuracy, max_epsilon) - Both values
    """
    base_accuracy = None
    for result in results:
        if result["config"]["config_id"] != 1:
            continue
        base_accuracy = result["results"]["per_round"][-1].get("accuracy")
        if not base_accuracy:
            continue
        break
    if not base_accuracy:
        print("Warning: Could not find base_accuracy! Most likely cause is that there is no config-1 results file. Falling back to best accuracy as baseline ...")
        base_accuracy = max(
            r["results"]["per_round"][-1].get("accuracy", 0)
            for r in results
        )

    # find max epsilon to normalize
    max_epsilon = float('-inf')
    for result in results:
        if not result["config"]["use_dp"]:
            continue
        this_epsilon = result["results"]["per_round"][-1].get("epsilon")
        if this_epsilon and this_epsilon > max_epsilon:
            max_epsilon = this_epsilon
    if max_epsilon < 0:
        print(f"Could not find a valid epsilon across all results in the specified folder. Found maximum epsilon was epsilon={max_epsilon}.")

    return base_accuracy, max_epsilon

def visualize_radar_chart(folder: str) -> None:
    """
    Radar/spider chart showing privacy-robustness-performance tradeoff.

    Each config variant gets one polygon on the radar.
    Three axes: Privacy (1 - normalized_epsilon), Robustness (accuracy_under_attack),
    Performance (final_accuracy_no_attack baseline comparison).

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)

    _, max_epsilon = get_base_accuracy_and_max_epsilon(results)

    normalized_result_scores = defaultdict(dict)
    PRIVACY_KEY = "privacy_score"
    ROBUSTNESS_KEY = "robustness_score"
    EFFICIENCY_KEY = "efficiency_score"
    for result in results:
        config = result["config"]
        final_accuracy = result["results"]["per_round"][-1].get("accuracy")
        
        # privacy
        if not config["use_dp"]:
            normalized_result_scores[result["_filename"]][PRIVACY_KEY] = 0.0         # No privacy at all
        else:
            normalized_result_scores[result["_filename"]][PRIVACY_KEY] = max(1 - (config["epsilon"] / max_epsilon), 0) # The max is due to currently possible negative epsilon values
        
        # robustness
        normalized_result_scores[result["_filename"]][ROBUSTNESS_KEY] = final_accuracy

        # efficiency
        if not config["use_topk"]:
            normalized_result_scores[result["_filename"]][EFFICIENCY_KEY] = 0.0
        else:
            normalized_result_scores[result["_filename"]][EFFICIENCY_KEY] = 1 - config["topk_ratio"]
            


    variants = [(name, values) for name, values in normalized_result_scores.items()]

    axes_labels = ["Privacy", "Robustness", "Efficiency"]
    num_axes    = len(axes_labels)
    # Privacy at top (90°), Robustness bottom-left (210°), Efficiency bottom-right (330°)
    angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    angles += angles[:1]    # Close the shape

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))
    ax.set_title("Privacy - Robustness - Efficiency Trilemma",
                 fontsize=13, pad=20)

    colors = plt.cm.tab10.colors
    for i, (label, scores) in enumerate(variants):
        values = [
            scores[PRIVACY_KEY],
            scores[ROBUSTNESS_KEY],
            scores[EFFICIENCY_KEY],
        ]
        values += values[:1]   # close the polygon
        color   = colors[i % len(colors)]
        ax.plot(angles, values, color=color, linewidth=1.5, label=label)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        axes_labels,
        fontsize=11,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.grid(True, alpha=0.3)

    ax.legend(
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        framealpha=0.8,
    )

    plt.tight_layout()
    output_path = os.path.join(folder, "radar_trilemma.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


def visualize_radar_chart_per_config(folder: str) -> None:
    """
    Radar/spider chart showing privacy-robustness-performance tradeoff.

    One Subplot per config.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)

    _, max_epsilon = get_base_accuracy_and_max_epsilon(results)

    PRIVACY_KEY = "privacy_score"
    ROBUSTNESS_KEY = "robustness_score"
    EFFICIENCY_KEY = "efficiency_score"
    config_groups = defaultdict(list)
    for result in results:
        config = result["config"]
        final_accuracy = result["results"]["per_round"][-1].get("accuracy")
        
        # privacy
        if not config["use_dp"]:
            privacy = 0.0         # No privacy at all
        else:
            privacy = max(1 - (config["epsilon"] / max_epsilon), 0) # DP - The max is due to currently possible negative epsilon values
        
        # robustness
        robustness = final_accuracy                 # FLTrust

        # efficiency
        if not config["use_topk"]:
            efficiency = 0.0
        else:
            efficiency = 1 - config["topk_ratio"]                   # TopK

        config_groups[config["config_id"]].append((
            make_label(result["_filename"]),
            {PRIVACY_KEY: privacy, ROBUSTNESS_KEY: robustness, EFFICIENCY_KEY: efficiency}
        ))
            

    axes_labels    = ["Privacy", "Robustness", "Efficiency"]

    angles  = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    angles += angles[:1]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Privacy - Robustness - Efficiency Trilemma per Config",
                 fontsize=14)
    axes = axes.flatten()

    colors = plt.cm.tab10.colors

    for idx, config_id in enumerate(sorted(config_groups.keys())):
        ax      = axes[idx]
        variants = config_groups[config_id]

        ax.set_title(f"Config {config_id}", fontsize=10, pad=10)

        for i, (label, scores) in enumerate(variants):
            values  = [scores[PRIVACY_KEY],
                       scores[ROBUSTNESS_KEY],
                       scores[EFFICIENCY_KEY]]
            values += values[:1]
            color   = colors[i % len(colors)]
            ax.plot(angles, values, color=color,
                    linewidth=1.5, label=label)
            ax.fill(angles, values, color=color, alpha=0.15)

        ax.set_thetagrids(
            [a * 180 / np.pi for a in angles[:-1]],
            axes_labels, fontsize=8,
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12),
                  ncol=1, framealpha=0.8)

    # hide unused subplots if fewer than 8 configs
    for idx in range(len(config_groups), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(folder, "radar_per_config.png")
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
    parser.add_argument("--plot", type=str,
                        choices=["all", "lines", "bar", "radar", "radar_per_conf", "lines_per_conf"],
                        default="all",
                        help="which plot to generate (default: all)")
    args = parser.parse_args(args=None if len(sys.argv) > 1 else ["--help"])
    
    if args.plot in ("all", "lines"):
        visualize_overview(args.folder)
    if args.plot in ("all", "bar"):
        visualize_bar_chart_per_config(args.folder)
    if args.plot in ("all", "radar"):
        visualize_radar_chart(args.folder)
    if args.plot in ("all", "radar_per_conf"):
        visualize_radar_chart_per_config(args.folder)
    if args.plot in ("all", "lines_per_conf"):
        visualize_lines_per_config(args.folder)