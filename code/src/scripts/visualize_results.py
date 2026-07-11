import argparse
from collections import defaultdict
import json
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np

from src.models import get_dataset_spec


def load_results(folder: str) -> list[dict]:
    """
    Load all JSON result files from a folder.

    Args:
        folder (str): path to folder containing result JSON files.

    Returns:
        list[dict]: list of loaded result dicts, sorted by filename.
    """
    pattern = os.path.join(folder, "*.json")
    files = sorted(
        f for f in glob.glob(pattern)
        if os.path.basename(f) != "run_summary.json"
    )

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
    
    config = results[0].get('config') if results else None
    dataset = config.get('dataset', 'Undefined') if config else 'Undefined'

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
                print(f"Could not find accuracy of final round for one of configs {config_id} results. ({result['_filename']})")
                continue
            label = make_label(result["_filename"])
            bars.append((label, final_accuracy))
        group_bars.append((config_id, bars))

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(f"Final Round Accuracy per Config Variant ({dataset})", fontsize=13)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    x         = 0          # current x position
    tick_pos  = []         # x positions for group labels
    tick_lab  = []         # group label text
    bar_width = 0.6
    gap       = 1.2        # gap between config groups

    colors = plt.cm.tab10.colors
    hatches = ['--', '++', 'xx', '//', '\\\\', '||', '*', 'o', '.', 'x']

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
    legend = None
    if handles:
        legend = ax.legend(
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

    # Run params, positioned just below the legend's actual rendered bottom
    # edge. Computed after layout so it hugs a short legend (few variants)
    # and still clears a tall, many-row one (many variants), rather than a
    # fixed offset tuned for one particular legend size. Legend.get_window_extent()
    # needs a draw() first so the renderer has real (not stale) coordinates,
    # and Legend defaults to zorder=5 vs. Text's zorder=3, which is why a
    # fixed-offset text used to render underneath a tall legend instead of
    # just missing it.
    run_config = results[0]["config"]
    attack_type = run_config.get("attack_type", "label_flip")
    attack_scale = run_config.get("attack_scale")
    attack_text = f"Attack: {attack_type}"
    if attack_scale is not None and attack_scale != 1.0:
        attack_text += f" (scale x{attack_scale})"
    params_text = (
        f"Clients: {run_config['num_clients']}  |  "
        f"Byzantine: {run_config['num_byzantine']}  |  "
        f"Rounds: {run_config['num_rounds']}  |  "
        f"Root Dataset Size: {run_config['root_dataset_size']}  |  "
        f"{attack_text}"
    )
    if legend is not None:
        fig.canvas.draw()
        legend_bbox_axes = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        text_y = legend_bbox_axes.y0 - 0.03
    else:
        text_y = -0.05
    ax.text(0.5, text_y, params_text, ha="center", va="top",
            fontsize=8, transform=ax.transAxes)

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


def visualize_confusion_matrices(folder: str) -> None:
    """
    One subplot per config showing confusion matrix as heatmap.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)

    # Group by config_id, pick variant with best final accuracy
    config_matrices = group_by_config_id(results)

    num_configs = len(config_matrices)
    cols        = min(4, num_configs)
    rows        = (num_configs + cols - 1) // cols
    fig, axes   = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Confusion Matrices per Config (best variant)", fontsize=13)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    num_classes  = get_dataset_spec(results[0]["config"].get("dataset", "mnist")).num_classes
    digit_labels = [str(i) for i in range(num_classes)]

    for idx, (config_id, (label, _, matrix)) in enumerate(sorted(config_matrices.items())):
        ax  = axes[idx]
        mat = np.array(matrix)

        # normalize by row (true label) to show percentages
        row_sums = mat.sum(axis=1, keepdims=True)
        mat_norm = np.where(row_sums > 0, mat / row_sums, 0)

        im = ax.imshow(mat_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Config {config_id}\n{label}", fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(digit_labels, fontsize=6)
        ax.set_yticklabels(digit_labels, fontsize=6)

        # annotate cells with percentage
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{mat_norm[i,j]:.0%}",
                        ha="center", va="center",
                        fontsize=4,
                        color="white" if mat_norm[i,j] > 0.5 else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused subplots
    for idx in range(len(config_matrices), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(folder, "confusion_matrices.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


def visualize_f1_score(folder: str) -> None:
    """
    One subplot per config showing per-class F1, precision, recall.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)

    # Group by config_id, pick variant with best final accuracy
    config_scores = group_by_config_id(results)

    num_configs = len(config_scores)
    cols        = min(4, num_configs)
    rows        = (num_configs + cols - 1) // cols
    fig, axes   = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle("Per-class F1 / Precision / Recall per Config (best variant)", fontsize=13)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    metrics      = ["precision", "recall", "f1"]
    colors       = ["#4C72B0", "#55A868", "#C44E52"]
    num_classes  = get_dataset_spec(results[0]["config"].get("dataset", "mnist")).num_classes
    digit_labels = [str(i) for i in range(num_classes)]
    x            = np.arange(num_classes)
    width        = 0.25

    for idx, (config_id, (label, scores, _)) in enumerate(sorted(config_scores.items())):
        ax = axes[idx]

        for m_idx, (metric, color) in enumerate(zip(metrics, colors)):
            values = [scores[str(d)][metric] for d in range(num_classes)]
            ax.bar(x + m_idx * width, values, width, label=metric, color=color, alpha=0.85)

        ax.set_title(f"Config {config_id}\n{label}", fontsize=7)
        ax.set_xlabel("Digit class", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(digit_labels, fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=6, loc="lower right")

    for idx in range(len(config_scores), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(folder, "f1_charts.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


def visualize_f1_tables(folder: str) -> None:
    """
    One summary table per config showing TP/FP/FN/TN and F1 metrics per digit class.

    Saves one PNG per config into a 'f1_tables' subfolder.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)

    config_data = group_by_config_id(results)
    num_classes = get_dataset_spec(results[0]["config"].get("dataset", "mnist")).num_classes

    os.makedirs(os.path.join(folder, "f1_tables"), exist_ok=True)

    for config_id, (label, scores, matrix) in sorted(config_data.items()):
        fig, ax = plt.subplots(figsize=(12, 4))
        test_size = sum(sum(row) for row in matrix)
        ax.set_title(f"Config {config_id} - {label} (n={test_size:,})", fontsize=10, pad=10)
        ax.axis("off")

        col_labels   = ["Class", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"]
        table_data   = []

        for i in range(num_classes):
            tp = matrix[i][i]
            fp = sum(matrix[r][i] for r in range(num_classes)) - tp
            fn = sum(matrix[i][c] for c in range(num_classes)) - tp
            tn = sum(
                matrix[r][c]
                for r in range(num_classes)
                for c in range(num_classes)
            ) - tp - fp - fn

            precision = scores[str(i)]["precision"]
            recall    = scores[str(i)]["recall"]
            f1        = scores[str(i)]["f1"]

            table_data.append([
                str(i),
                f"{tp:,}", f"{fp:,}", f"{fn:,}", f"{tn:,}",
                f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}",
            ])

        # Sum row
        total_tp = sum(matrix[i][i] for i in range(num_classes))
        total_fp = sum(
            sum(matrix[r][i] for r in range(num_classes)) - matrix[i][i]
            for i in range(num_classes)
        )
        total_fn = sum(
            sum(matrix[i][c] for c in range(num_classes)) - matrix[i][i]
            for i in range(num_classes)
        )
        total_tn = sum(
            sum(matrix[r][c] for r in range(num_classes) for c in range(num_classes))
            - matrix[i][i]
            - (sum(matrix[r][i] for r in range(num_classes)) - matrix[i][i])
            - (sum(matrix[i][c] for c in range(num_classes)) - matrix[i][i])
            for i in range(num_classes)
        )

        # Macro averages for precision, recall, f1
        macro_precision = sum(scores[str(i)]["precision"] for i in range(num_classes)) / num_classes
        macro_recall    = sum(scores[str(i)]["recall"]    for i in range(num_classes)) / num_classes
        macro_f1        = sum(scores[str(i)]["f1"]        for i in range(num_classes)) / num_classes

        table_data.append([
            "Total",
            f"{total_tp:,}", f"{total_fp:,}", f"{total_fn:,}", f"{total_tn:,}",
            f"{macro_precision:.3f}", f"{macro_recall:.3f}", f"{macro_f1:.3f}",
        ])

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color the sum row
        for j in range(len(col_labels)):
            table[num_classes + 1, j].set_facecolor("#D0D0D0")
            table[num_classes + 1, j].set_text_props(fontweight="bold")


        # color the diagonal (correct predictions) green, others light red
        for i in range(num_classes):
            # header row is row 0, data starts at row 1
            for j, col in enumerate(col_labels):
                cell = table[i + 1, j]
                if col == "F1":
                    f1_val = scores[str(i)]["f1"]
                    # green gradient based on F1 score
                    cell.set_facecolor(
                        (1 - f1_val * 0.5, 1, 1 - f1_val * 0.5)
                    )
                elif col == "Class":
                    cell.set_facecolor("#E8E8E8")

        plt.tight_layout()
        output_path = os.path.join(folder, "f1_tables", f"f1_table_config_{config_id}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.close()


def group_by_config_id(results):
    """
    Group results by config_id keeping the best accuracy variant per config.

    For configs with multiple variants (e.g. different epsilon or topk values),
    only the variant with the highest final round accuracy is kept.
    Results without per_class_scores are skipped.

    Args:
        results (list[dict]): list of loaded result dicts from load_results().

    Returns:
        dict: mapping config_id (int) to (label, per_class_scores, confusion_matrix) where
              - label (str):              short human-readable label from make_label()
              - per_class_scores (dict):  per-class precision, recall and F1 scores
                                          keyed by digit class string e.g. "0".."9"
              - confusion_matrix (list):  10x10 list of lists, or None if not present
    """
    config_scores = {}
    for result in results:
        config_id      = result["config"]["config_id"]
        per_class      = result["results"].get("per_class_scores")
        confusion_matrix = result["results"].get("confusion_matrix")
        final_accuracy = result["results"]["per_round"][-1].get("accuracy", 0)
        label          = make_label(result["_filename"])

        if per_class is None:
            continue

        if config_id not in config_scores or final_accuracy > config_scores[config_id][3]:
            config_scores[config_id] = (label, per_class, confusion_matrix, final_accuracy)

    return {
        config_id: (label, scores, matrix)
        for config_id, (label, scores, matrix, _) in config_scores.items()
    }


def visualize_label_distribution(folder: str) -> None:
    """
    Stacked bar chart of label counts in the root set and each client's shard.

    The split is identical across every variant in a run folder (same
    root_dataset_size/num_clients/seed), so this reads the label_distribution
    block from any one result file. Older result files saved before this
    field existed are skipped gracefully.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = load_results(folder)

    label_distribution = None
    for result in results:
        label_distribution = result["results"].get("label_distribution")
        if label_distribution is not None:
            break

    if label_distribution is None:
        print("No label_distribution field found in any result file -- skipping label distribution plot.")
        return

    os.makedirs(os.path.join(folder, "label_distribution"), exist_ok=True)

    num_classes  = get_dataset_spec(results[0]["config"].get("dataset", "mnist")).num_classes
    digit_labels = [str(i) for i in range(num_classes)]
    colors = plt.cm.tab10.colors

    bar_names = ["Root"] + [f"C{cid}" for cid in sorted(label_distribution["clients"], key=int)]
    bar_data  = [label_distribution["root"]] + [
        label_distribution["clients"][cid] for cid in sorted(label_distribution["clients"], key=int)
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(bar_names) * 0.4), 6))
    ax.set_title("Label Distribution: Root Dataset + Each Client's Train Shard", fontsize=13)
    ax.set_ylabel("Sample count")
    ax.grid(axis="y", alpha=0.3)

    x = np.arange(len(bar_names))
    bottom = np.zeros(len(bar_names))
    for digit, color in zip(digit_labels, colors):
        heights = np.array([counts.get(digit, 0) for counts in bar_data])
        ax.bar(x, heights, bottom=bottom, color=color, width=0.8, label=digit)
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels(bar_names, fontsize=6 if len(bar_names) > 20 else 8, rotation=90)
    ax.legend(title="Digit", fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=10, framealpha=0.8)

    plt.tight_layout()
    output_path = os.path.join(folder, "label_distribution", "label_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


# Fixed colors for the honest/malicious split, reused across every trust
# plot below -- never cycled or reassigned per config/client like the tab10
# colors elsewhere in this file, since honest vs. malicious is a fixed,
# always-two-valued category.
HONEST_COLOR = "#1f77b4"
MALICIOUS_COLOR = "#d62728"


def _avg_trust_per_client(result: dict) -> dict:
    """
    Average one result's per-round trust scores into {client_id: avg_trust}.

    Args:
        result (dict): loaded result JSON (must have use_fltrust=True).

    Returns:
        dict: {client_id (str): average trust score over the whole run}.
    """
    sums, counts = {}, {}
    for entry in result["results"]["per_round"]:
        for client_id, score in entry["trust_scores"].items():
            sums[client_id] = sums.get(client_id, 0.0) + score
            counts[client_id] = counts.get(client_id, 0) + 1
    return {client_id: sums[client_id] / counts[client_id] for client_id in sums}


def visualize_trust_per_client(folder: str) -> None:
    """
    Average trust per client over the whole run, grouped by FLTrust config variant.

    One point per client per variant, colored by whether that client was
    honest or malicious, with a small horizontal spread (not random jitter,
    so the plot is reproducible) so points inside a group don't overlap.
    Non-FLTrust variants never produce trust scores, so they're excluded.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = [r for r in load_results(folder) if r["config"]["use_fltrust"]]
    if not results:
        print("No FLTrust-enabled result files found -- skipping per-client trust plot.")
        return
    results.sort(key=lambda r: (r["config"]["config_id"], r["_filename"]))

    fig, ax = plt.subplots(figsize=(max(10, len(results) * 1.3), 6))
    ax.set_title("Average Trust per Client over the Run (FLTrust configs only)", fontsize=13)
    ax.set_ylabel("Avg trust score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    tick_pos, tick_lab = [], []
    for x, result in enumerate(results):
        malicious = result["results"]["malicious_clients"]
        avg_trust = _avg_trust_per_client(result)
        client_ids = sorted(avg_trust, key=lambda c: (malicious.get(c, False), int(c)))

        for i, client_id in enumerate(client_ids):
            spread = (i / max(len(client_ids) - 1, 1) - 0.5) * 0.5   # spread across [-0.25, 0.25]
            color = MALICIOUS_COLOR if malicious.get(client_id, False) else HONEST_COLOR
            ax.scatter(x + spread, avg_trust[client_id], color=color,
                       edgecolor="white", linewidth=0.5, s=35, alpha=0.85, zorder=3)

        honest_vals = [avg_trust[c] for c in client_ids if not malicious.get(c, False)]
        malicious_vals = [avg_trust[c] for c in client_ids if malicious.get(c, False)]
        if honest_vals:
            ax.hlines(np.mean(honest_vals), x - 0.3, x + 0.3, color=HONEST_COLOR, linewidth=2, zorder=4)
        if malicious_vals:
            ax.hlines(np.mean(malicious_vals), x - 0.3, x + 0.3, color=MALICIOUS_COLOR, linewidth=2, zorder=4)

        tick_pos.append(x)
        tick_lab.append(make_label(result["_filename"]))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=7, rotation=45, ha="right")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=HONEST_COLOR, markersize=7, label="Honest client"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MALICIOUS_COLOR, markersize=7, label="Malicious client"),
        plt.Line2D([0], [0], color="black", linewidth=2, label="Group mean"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=3, framealpha=0.8)

    plt.tight_layout()
    output_path = os.path.join(folder, "trust_per_client.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


def _honest_vs_malicious_by_round(results: list[dict]) -> tuple[list[int], list, list]:
    """
    Pool per-round trust scores across results into honest/malicious round averages.

    Args:
        results (list[dict]): loaded result JSONs (all use_fltrust=True).

    Returns:
        tuple: (rounds, honest_avg, malicious_avg) -- parallel lists, one
              entry per round; averages are None for a round with no scores
              of that kind.
    """
    honest_by_round = defaultdict(list)
    malicious_by_round = defaultdict(list)
    for result in results:
        malicious = result["results"]["malicious_clients"]
        for entry in result["results"]["per_round"]:
            for client_id, score in entry["trust_scores"].items():
                bucket = malicious_by_round if malicious.get(client_id, False) else honest_by_round
                bucket[entry["round"]].append(score)

    rounds = sorted(set(honest_by_round) | set(malicious_by_round))
    honest_avg = [np.mean(honest_by_round[r]) if honest_by_round[r] else None for r in rounds]
    malicious_avg = [np.mean(malicious_by_round[r]) if malicious_by_round[r] else None for r in rounds]
    return rounds, honest_avg, malicious_avg


def visualize_trust_over_rounds(folder: str) -> None:
    """
    Honest vs. malicious average trust score per round, pooled across every
    FLTrust-enabled config variant in the folder.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = [r for r in load_results(folder) if r["config"]["use_fltrust"]]
    if not results:
        print("No FLTrust-enabled result files found -- skipping trust-over-rounds plot.")
        return

    rounds, honest_avg, malicious_avg = _honest_vs_malicious_by_round(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Honest vs Malicious Avg Trust per Round (pooled across all FLTrust configs)", fontsize=13)
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg trust score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.plot(rounds, honest_avg, color=HONEST_COLOR, linewidth=2, marker="o", markersize=3, label="Honest")
    ax.plot(rounds, malicious_avg, color=MALICIOUS_COLOR, linewidth=2, marker="o", markersize=3, label="Malicious")
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    output_path = os.path.join(folder, "trust_over_rounds.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


def visualize_trust_over_rounds_per_config(folder: str) -> None:
    """
    Honest vs. malicious average trust score per round, one subplot per
    FLTrust-enabled config family (pooling that config's own variants, e.g.
    its different epsilon/topk_ratio values) -- so DP/TopK's effect on trust
    isn't averaged away against plain-FLTrust configs.

    Args:
        folder (str): path to folder containing result JSON files.
    """
    results = [r for r in load_results(folder) if r["config"]["use_fltrust"]]
    if not results:
        print("No FLTrust-enabled result files found -- skipping per-config trust-over-rounds plot.")
        return

    config_groups = defaultdict(list)
    for result in results:
        config_groups[result["config"]["config_id"]].append(result)

    config_ids = sorted(config_groups.keys())
    cols = min(4, len(config_ids))
    rows = (len(config_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4), squeeze=False)
    fig.suptitle("Honest vs Malicious Avg Trust per Round, per Config", fontsize=14)
    axes = axes.flatten()

    for idx, config_id in enumerate(config_ids):
        ax = axes[idx]
        rounds, honest_avg, malicious_avg = _honest_vs_malicious_by_round(config_groups[config_id])

        ax.set_title(f"Config {config_id}", fontsize=10)
        ax.set_xlabel("Round", fontsize=8)
        ax.set_ylabel("Avg trust score", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.plot(rounds, honest_avg, color=HONEST_COLOR, linewidth=1.5, marker="o", markersize=2, label="Honest")
        ax.plot(rounds, malicious_avg, color=MALICIOUS_COLOR, linewidth=1.5, marker="o", markersize=2, label="Malicious")
        ax.legend(fontsize=7, loc="best")

    for idx in range(len(config_ids), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(folder, "trust_over_rounds_per_config.png")
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
                        choices=["all", "lines", "bar", "radar", "radar_per_conf", "lines_per_conf",
                                 "confusion", "f1_chart", "f1_table_per_conf", "label_dist",
                                 "trust_per_client", "trust_rounds", "trust_rounds_per_conf"],
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
    if args.plot in ("all", "confusion"):
        visualize_confusion_matrices(args.folder)
    if args.plot in ("all", "f1_chart"):
        visualize_f1_score(args.folder)
    if args.plot in ("all", "label_dist"):
        visualize_label_distribution(args.folder)
    if args.plot in ("all", "f1_table_per_conf"):
        visualize_f1_tables(args.folder)
    if args.plot in ("all", "trust_per_client"):
        visualize_trust_per_client(args.folder)
    if args.plot in ("all", "trust_rounds"):
        visualize_trust_over_rounds(args.folder)
    if args.plot in ("all", "trust_rounds_per_conf"):
        visualize_trust_over_rounds_per_config(args.folder)