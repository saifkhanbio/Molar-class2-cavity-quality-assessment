#!/usr/bin/env python3
"""Plot recorded checkpoint performance for all three segmentation models."""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/final-model-performance-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
RESULTS = SCRIPTS / "outputs"


def load_metrics(path):
    with path.open(newline="") as stream:
        records = {row["model"]: row for row in csv.DictReader(stream)}
    rows = [records[name] for name in ("O_unet", "P_unet", "CU_unet")]
    for row in rows:
        row["checkpoint_epoch"] = int(row["checkpoint_epoch"])
        for key in ("val_dice", "val_iou", "val_loss"):
            row[key] = float(row[key])
        assert 0 <= row["val_iou"] <= row["val_dice"] <= 1
    return rows


def add_labels(axis, bars, percentage=False):
    for bar in bars:
        value = bar.get_height()
        axis.annotate(f"{value:.1%}" if percentage else f"{value:.3f}",
                      (bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 8), textcoords="offset points", ha="center",
                      va="bottom", fontsize=11, weight="bold", color="#26323B")


def draw(rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "axes.titleweight": "bold", "font.size": 11})
    colors = ["#D95F32", "#178F7A", "#7661B5"]
    labels = [f"{row['model']}\n{row['cavity_type']} · epoch {row['checkpoint_epoch']}"
              for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.suptitle("Final U-Net Performance", fontsize=24, weight="bold", y=.965)
    fig.text(.5, .905, "Best saved checkpoint on the validation split", ha="center",
             fontsize=14, color="#5B6770")
    width = .30
    dice = axes[0].bar(x - width/2, [r["val_dice"] for r in rows], width,
                       color=colors, edgecolor="white", linewidth=1)
    iou = axes[0].bar(x + width/2, [r["val_iou"] for r in rows], width,
                      color=colors, alpha=.58, edgecolor="#26323B", linewidth=.8,
                      hatch="//")
    add_labels(axes[0], dice, percentage=True)
    add_labels(axes[0], iou, percentage=True)
    axes[0].set(title="Segmentation performance", ylabel="Score", ylim=(0, 1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_yticks(np.linspace(0, 1, 6))
    axes[0].legend(handles=[Patch(facecolor="#69757D", edgecolor="white", label="Dice"),
                            Patch(facecolor="#C5CBD0", edgecolor="#69757D", hatch="//", label="IoU")],
                   loc="upper center", ncol=2, frameon=False)
    losses = axes[1].bar(x, [r["val_loss"] for r in rows], .52, color=colors,
                         edgecolor="white", linewidth=1)
    add_labels(axes[1], losses)
    axes[1].set(title="Validation loss", ylabel="BCE + Dice + edge loss", ylim=(0, .50))
    axes[1].set_yticks(np.arange(0, .51, .10))
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=10, pad=7)
        axis.set_title(axis.get_title(), fontsize=16, pad=15)
        axis.grid(axis="y", color="#D7DDE1", linewidth=.8, alpha=.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#A8B0B6")
    fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.14, wspace=.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white",
                metadata={"Title": "Final U-Net Performance",
                          "Description": "Occlusal, proximal and cusp segmentation: validation Dice, IoU and loss at the best saved checkpoints."})
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=SCRIPTS / "model_performance/model_performance.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "Picture1_with_cusp.png")
    args = parser.parse_args()
    rows = load_metrics(args.csv)
    draw(rows, args.output)
    for row in rows:
        print(f"{row['cavity_type']}: epoch {row['checkpoint_epoch']}, "
              f"Dice {row['val_dice']:.2%}, IoU {row['val_iou']:.2%}, loss {row['val_loss']:.4f}")
    print(args.output)


if __name__ == "__main__":
    main()
