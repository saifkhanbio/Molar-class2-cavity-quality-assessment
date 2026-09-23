"""Extract final O/P U-Net checkpoint metrics and plot their comparison."""

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "latest06apr.ipynb"
CSV_PATH = ROOT / "unet_final_performance.csv"
PNG_PATH = ROOT / "unet_final_performance.png"
SVG_PATH = ROOT / "unet_final_performance.svg"

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_START = re.compile(r"^Epoch (\d+)/(\d+)$")
EARLY_STOP = re.compile(r"^Epoch (\d+): early stopping$")
METRIC = re.compile(
    r"(?<!val_)dice_coef: ([0-9.eE+-]+).*?"
    r"(?<!val_)iou_metric: ([0-9.eE+-]+).*?"
    r"(?<!val_)loss: ([0-9.eE+-]+).*?"
    r"val_dice_coef: ([0-9.eE+-]+).*?"
    r"val_iou_metric: ([0-9.eE+-]+).*?"
    r"val_loss: ([0-9.eE+-]+)"
)


def output_text(cell):
    chunks = []
    for output in cell.get("outputs", []):
        chunks.extend(output.get("text", []))
        chunks.extend(output.get("data", {}).get("text/plain", []))
    return ANSI_ESCAPE.sub("", "".join(chunks))


def latest_training_cell(notebook, prefix):
    signature = f"model = {prefix}_build_refined_unet()"
    checkpoint = f'"{prefix}_unet_refined.h5"'
    candidates = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if signature in source and checkpoint in source and cell.get("outputs"):
            candidates.append((index, cell))
    if not candidates:
        raise RuntimeError(f"No completed training cell found for {prefix}_unet_refined.h5")
    return candidates[-1]


def extract_run(notebook, prefix, cavity_type):
    cell_index, cell = latest_training_cell(notebook, prefix)
    current_epoch = None
    total_epochs = None
    stopped_epoch = None
    records = []

    for raw_line in output_text(cell).splitlines():
        line = raw_line.strip()
        start_match = EPOCH_START.match(line)
        if start_match:
            current_epoch = int(start_match.group(1))
            total_epochs = int(start_match.group(2))
            continue
        stop_match = EARLY_STOP.match(line)
        if stop_match:
            stopped_epoch = int(stop_match.group(1))
            continue
        metric_match = METRIC.search(line)
        if metric_match and current_epoch is not None:
            values = [float(value) for value in metric_match.groups()]
            records.append({
                "checkpoint_epoch": current_epoch,
                "train_dice": values[0],
                "train_iou": values[1],
                "train_loss": values[2],
                "val_dice": values[3],
                "val_iou": values[4],
                "val_loss": values[5],
            })

    if not records:
        raise RuntimeError(f"No epoch metrics found for {prefix}_unet_refined.h5")

    best = max(records, key=lambda row: row["val_iou"])
    return {
        "model": f"{prefix}_unet",
        "cavity_type": cavity_type,
        "model_file": f"{prefix}_unet_refined.h5",
        "performance_scope": "best validation checkpoint",
        "selection_metric": "val_iou",
        "checkpoint_epoch": best["checkpoint_epoch"],
        "training_stopped_epoch": stopped_epoch or records[-1]["checkpoint_epoch"],
        "maximum_epochs": total_epochs,
        "train_loss": best["train_loss"],
        "train_dice": best["train_dice"],
        "train_iou": best["train_iou"],
        "val_loss": best["val_loss"],
        "val_dice": best["val_dice"],
        "val_iou": best["val_iou"],
        "notebook_cell_index": cell_index,
    }


def write_csv(rows):
    fields = list(rows[0])
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.4f}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def add_bar_labels(ax, bars, percentage=False):
    for bar in bars:
        value = bar.get_height()
        label = f"{value * 100:.1f}%" if percentage else f"{value:.3f}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#26323B",
        )


def write_plot(rows):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "axes.titleweight": "bold",
    })
    colors = ["#D95F32", "#178F7A"]
    labels = [f"{row['model']}\n(epoch {row['checkpoint_epoch']})" for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Final U-Net Performance", fontsize=18, fontweight="bold", y=0.97)
    fig.text(0.5, 0.915, "Best saved checkpoint on the validation split",
             ha="center", color="#5B6770", fontsize=10.5)

    width = 0.30
    dice_bars = axes[0].bar(x - width / 2, [row["val_dice"] for row in rows],
                            width, label="Dice", color=colors, edgecolor="white", linewidth=1)
    iou_bars = axes[0].bar(x + width / 2, [row["val_iou"] for row in rows],
                           width, label="IoU", color=colors, alpha=0.58,
                           edgecolor="#26323B", linewidth=0.8, hatch="//")
    add_bar_labels(axes[0], dice_bars, percentage=True)
    add_bar_labels(axes[0], iou_bars, percentage=True)
    axes[0].set_title("Segmentation accuracy", fontsize=12, pad=12)
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    axes[0].legend(frameon=False, loc="upper center", ncol=2)

    loss_bars = axes[1].bar(x, [row["val_loss"] for row in rows], 0.52,
                            color=colors, edgecolor="white", linewidth=1)
    add_bar_labels(axes[1], loss_bars)
    axes[1].set_title("Validation loss", fontsize=12, pad=12)
    axes[1].set_ylabel("BCE + Dice + edge loss")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(row["val_loss"] for row in rows) * 1.35)

    for ax in axes:
        ax.grid(axis="y", color="#D7DDE1", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#A8B0B6")
        ax.spines["bottom"].set_color("#A8B0B6")

    fig.text(0.5, 0.025,
             "Source: latest06apr.ipynb. No separate held-out test evaluation was recorded.",
             ha="center", color="#6B747C", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.17, wspace=0.34)
    fig.savefig(PNG_PATH, dpi=300, facecolor="white")
    fig.savefig(SVG_PATH, facecolor="white")
    plt.close(fig)


def main():
    with NOTEBOOK.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    rows = [
        extract_run(notebook, "O", "Occlusal"),
        extract_run(notebook, "P", "Proximal"),
    ]
    write_csv(rows)
    write_plot(rows)
    for row in rows:
        print(
            f"{row['model']}: epoch {row['checkpoint_epoch']}, "
            f"val_dice={row['val_dice']:.4f}, val_iou={row['val_iou']:.4f}, "
            f"val_loss={row['val_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
