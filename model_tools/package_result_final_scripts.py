#!/usr/bin/env python3
"""Collect the generators, checkpoint files and recorded model performance."""
import ast
import copy
import csv
import hashlib
import html
import importlib.metadata
import json
import marshal
import os
from pathlib import Path
import re
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/result-final-package-mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extract_unet_performance as metrics
import h5py

TARGET = ROOT / "Result_final/scripts"
STAGE = ROOT / ".result_final_scripts_staging"
SOURCE_SNAPSHOT = ROOT / "STLFILES/STL_landmarks/work/landmark_sources"
INVENTORY = []


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(source, relative, purpose):
    destination = STAGE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    original_hash = sha256(source)
    assert sha256(destination) == original_hash
    INVENTORY.append({"path": relative, "source": str(source.relative_to(ROOT)), "purpose": purpose,
                      "bytes": source.stat().st_size, "sha256": original_hash, "copy_method": "exact"})


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_records(notebook, prefix):
    index, cell = metrics.latest_training_cell(notebook, prefix)
    records, retained = [], []
    current = None
    checkpoint_lines = []
    for raw in metrics.output_text(cell).splitlines():
        line = raw.strip()
        epoch = metrics.EPOCH_START.match(line)
        if epoch:
            current = int(epoch[1])
            retained.append(line)
            continue
        match = metrics.METRIC.search(line)
        if match and current is not None:
            vals = list(map(float, match.groups()))
            records.append(dict(model_file=prefix+"_unet_refined.h5", epoch=current,
                train_dice=vals[0], train_iou=vals[1], train_loss=vals[2],
                val_dice=vals[3], val_iou=vals[4], val_loss=vals[5]))
            retained.append(line)
        elif re.match(r"Epoch \d+: val_iou_metric improved .*saving model to", line):
            retained.append(line)
            checkpoint_lines.append(line)
        elif metrics.EARLY_STOP.match(line) or line.startswith("Restoring model weights from the end of the best epoch:"):
            retained.append(line)
    return index, records, "\n".join(retained)+"\n", checkpoint_lines


def hdf_description(path):
    with h5py.File(path, "r") as model:
        cfg = json.loads(model.attrs["model_config"])
        training = json.loads(model.attrs["training_config"])
        layers = cfg["config"]["layers"]
        iteration = int(model["optimizer_weights/adam/iteration"][()])
        return dict(keras_file_version=str(model.attrs["keras_version"]), layer_count=len(layers),
                    model_name=cfg["config"]["name"], optimizer_iterations=iteration,
                    training_configuration=training, architecture_configuration=cfg)


def draw_performance(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), layout="constrained")
    fig.suptitle("Segmentation models · selected-checkpoint performance", fontsize=16, weight="bold")
    for shift, key, label, color in [(-.18, "val_dice", "Dice", "#1877a5"), (.18, "val_iou", "IoU", "#29966a")]:
        bars = axes[0].bar(x+shift, [r[key] for r in rows], .34, color=color, label=label)
        axes[0].bar_label(bars, labels=[f"{100*r[key]:.2f}%" for r in rows], padding=4, fontsize=9)
    labels = [r["cavity_type"]+f"\nEpoch {r['checkpoint_epoch']}" for r in rows]
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Validation score")
    axes[0].legend(frameon=False, loc="upper right")
    bars = axes[1].bar(x, [r["val_loss"] for r in rows], .55, color="#db8050")
    axes[1].bar_label(bars, labels=[f"{r['val_loss']:.4f}" for r in rows], padding=4)
    axes[1].set(xticks=x, xticklabels=labels, ylim=(0, .5), ylabel="Validation loss (BCE + Dice + edge)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e9ed")
        ax.set_axisbelow(True)
    fig.supxlabel("Recorded validation-split metrics from latest06apr.ipynb · checkpoint selected by validation IoU", fontsize=9)
    fig.savefig(output / "model_performance.png", dpi=300, facecolor="white")
    fig.savefig(output / "model_performance.svg", facecolor="white")
    plt.close(fig)


def main():
    assert not TARGET.exists() and not STAGE.exists()
    STAGE.mkdir()
    used = {
        "final_avg_efd_score.py": "Occlusal/proximal EFD measurements and overlays",
        "isthmus_disc.py": "Isthmus discovery and saved widths",
        "cusp_geometry.py": "Cusp separation, centers and pairing geometry",
        "cus_ist_ratio.py": "Intercuspal distances, matched isthmus ratios and overlays",
        "landmark_cav_con_comb_smooth.py": "Landmark- and mask-assisted cavity floor selection and measurements",
        "publication_cavity_depth.py": "Current depth, floor regularity and relative-depth PNG/HTML figures",
        "gingival_depth_from_pulpal.py": "Pulpal-reference gingival measurements and historical figures",
        "final_CQS_overall.py": "Current seven-component CQS scores and reports",
        "export_result_final.py": "Filtering, numerical HTML recovery and final presentation rendering",
        "extract_unet_performance.py": "Original notebook performance extractor for occlusal/proximal checkpoints",
    }
    for filename, purpose in used.items():
        copy_file(ROOT / filename, "analysis/"+filename, purpose)
    for name in ["final_cav_con_comb_smooth.py", "cav_con_comb_smooth.py"]:
        copy_file(SOURCE_SNAPSHOT / name, "analysis/"+name, "Underlying STL measurement and detection functions")
    copy_file(ROOT / "__pycache__/refined_cav_con_comb_smooth.cpython-39.pyc",
              "analysis/refined_cav_con_comb_smooth.pyc", "Exact compiled refined detector v1.2; CPython 3.9")
    copy_file(SOURCE_SNAPSHOT / "refined_cav_con_comb_smooth.py",
              "source_snapshots/refined_cav_con_comb_smooth_v1_0.py", "Earlier source snapshot; distinct from the v1.2 runtime")
    for path in sorted((ROOT / "RESULTS-21-09-2026/_archive").rglob("*.py")):
        relative = path.relative_to(ROOT / "RESULTS-21-09-2026/_archive")
        copy_file(path, "archive/"+str(relative), "Historical generator or associated test for archived result versions")
    landmark_dir = ROOT / "STLFILES/STL_landmarks/outputs/image_stl_landmarks/scripts"
    for path in sorted(landmark_dir.iterdir()):
        if path.suffix == ".py" or path.name == "requirements.txt":
            copy_file(path, "landmarks/"+path.name, "Image/STL landmark pipeline")
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        copy_file(path, "tests/"+path.name, "Existing workflow tests")
    copy_file(ROOT / "work/validate_result_final.py", "export_checks/validate_result_final.py", "Original filtered-export checks")
    copy_file(ROOT / "work/check_segmentation_model_provenance.py", "model_tools/check_segmentation_model_provenance.py", "Checkpoint-to-mask comparison tool")
    copy_file(Path(__file__).resolve(), "model_tools/package_result_final_scripts.py", "Package assembly and three-model performance extraction")

    notebook_path = ROOT / "latest06apr.ipynb"
    notebook = json.loads(notebook_path.read_text())
    cleaned = copy.deepcopy(notebook)
    cleaned["metadata"].pop("widgets", None)
    for cell in cleaned["cells"]:
        cell.pop("attachments", None)
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    performance = STAGE / "model_performance"
    performance.mkdir()
    summaries, histories, model_records, agreements = [], [], [], []
    for prefix, kind, modality in [("O", "Occlusal", "occlusal"), ("P", "Proximal", "proximal"), ("CU", "Cusp", "cusp")]:
        summary = metrics.extract_run(notebook, prefix, kind)
        index, history, log, checkpoints = training_records(notebook, prefix)
        cleaned["cells"][index]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": log.splitlines(keepends=True)}]
        (performance / (prefix+"_training_metrics.txt")).write_text(log)
        histories.extend(history)
        source = ROOT / summary["model_file"]
        copy_file(source, "models/"+source.name, kind+" mask prediction checkpoint")
        description = hdf_description(source)
        architecture = "U-Net decoder with ResNet50 encoder" if modality == "occlusal" else "Refined U-Net"
        inference = json.loads(Path("/tmp/result_model_"+modality+"_all.json").read_text())
        steps = description["optimizer_iterations"] / summary["checkpoint_epoch"]
        assert steps == round(steps)
        for record in inference["records"]:
            agreements.append({"model_file": source.name, "cavity_type": kind, **record,
                               "pixel_agreement_pct": 100*(1-record["different_pixels"]/record["pixels"])})
        summary.update(architecture=architecture, selection_metric="val_iou_metric",
                       checkpoint_optimizer_iterations=description["optimizer_iterations"],
                       training_steps_per_epoch=int(steps), sha256=sha256(source),
                       inference_preprocessing="OpenCV BGR then ResNet50 preprocess_input" if modality == "occlusal" else "OpenCV BGR to RGB, then divide by 255",
                       threshold=0.5, input_height=256, input_width=256, input_channels=3)
        summaries.append(summary)
        best_record = next(r for r in history if r["epoch"] == summary["checkpoint_epoch"])
        assert all(best_record[k] == summary[k] for k in ["train_dice","train_iou","train_loss","val_dice","val_iou","val_loss"])
        selected_log = next(line for line in checkpoints if line.startswith(f"Epoch {summary['checkpoint_epoch']}:"))
        model_records.append({**summary, **description, "checkpoint_log": selected_log,
                              "notebook_sha256": sha256(notebook_path), "parameter_count": inference["parameter_count"],
                              "comparison_tensorflow_version": inference["tensorflow_version"],
                              "prediction_folder": {"occlusal":"pred_O_M_masks_folder","proximal":"pred_P_M_masks_folder","cusp":"pred_O_M_cusp_molar_mask"}[modality]})
    output_notebook = STAGE / "notebooks/latest06apr.ipynb"
    output_notebook.parent.mkdir()
    output_notebook.write_text(json.dumps(cleaned, indent=1, ensure_ascii=False)+"\n")
    assert all(a["source"] == b["source"] for a,b in zip(notebook["cells"],cleaned["cells"]))
    INVENTORY.append({"path": "notebooks/latest06apr.ipynb", "source": "latest06apr.ipynb",
                      "purpose": "Original notebook code plus cleaned training-metric streams; other stored outputs removed",
                      "bytes": output_notebook.stat().st_size, "sha256": sha256(output_notebook), "copy_method": "source_preserved_outputs_filtered"})
    write_csv(performance / "model_performance.csv", summaries)
    write_csv(performance / "training_history.csv", histories)
    write_csv(performance / "mask_correspondence.csv", agreements)
    (performance / "model_details.json").write_text(json.dumps(model_records, indent=2)+"\n")
    draw_performance(summaries, performance)
    rows = "".join(f"<tr><td>{r['cavity_type']}</td><td>{r['architecture']}</td><td>{r['checkpoint_epoch']}</td>"
                   f"<td>{100*r['val_dice']:.2f}%</td><td>{100*r['val_iou']:.2f}%</td><td>{r['val_loss']:.4f}</td>"
                   f"<td><a href='../models/{r['model_file']}'>{r['model_file']}</a></td></tr>" for r in summaries)
    (performance / "index.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Model performance</title>
<style>body{{font:16px/1.6 Arial;color:#253343;max-width:1300px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}a{{color:#176ca0}}</style>
<h1>Segmentation model performance</h1><p>Recorded validation-split measurements at the checkpoint selected by validation IoU.</p>
<p><a href="model_performance.csv">Summary CSV</a> · <a href="training_history.csv">Epoch history</a> · <a href="model_details.json">Model configuration and provenance</a> · <a href="mask_correspondence.csv">Mask correspondence</a></p>
<table><tr><th>Task</th><th>Architecture</th><th>Epoch</th><th>Validation Dice</th><th>Validation IoU</th><th>Validation loss</th><th>Checkpoint</th></tr>{rows}</table>
<img src="model_performance.png" alt="Validation metrics for the three segmentation checkpoints">
<p>Training metrics come from the stored notebook epoch logs. Loss combines binary cross-entropy, Dice loss, and a Sobel edge term.
Inference uses 256 × 256 color images and a probability threshold of 0.5. The occlusal checkpoint uses ResNet50 preprocessing; proximal and cusp checkpoints use RGB values divided by 255.</p>
<p>The correspondence table compares new CPU predictions with the 63 retained mask images. Differences range from 0 to 4 pixels per 65,536-pixel mask; the exported masks remain unchanged.</p></html>''')
    environment = {}
    for package in ["numpy","scipy","matplotlib","Pillow","networkx","scikit-image","numpy-stl","plotly","pyvista","beautifulsoup4","h5py"]:
        environment[package] = importlib.metadata.version(package)
    (STAGE / "analysis_environment.json").write_text(json.dumps({"python":sys.version.split()[0],"packages":environment},indent=2)+"\n")
    (STAGE / "requirements-analysis.txt").write_text("\n".join(f"{p}=={v}" for p,v in environment.items())+"\n")
    (STAGE / "requirements-inference.txt").write_text("tensorflow==2.18.0\nkeras==3.10.0\nopencv-python>=4.7\nnumpy>=1.26,<2.1\nh5py\nPillow\n")
    write_csv(STAGE / "script_inventory.csv", INVENTORY)
    (STAGE / "README.md").write_text('''# Result generators and segmentation models

This package accompanies the 21 retained student cases. `script_inventory.csv` records the source path, purpose, size and SHA-256 of every copied source/checkpoint. Python source files and HDF5 checkpoints are exact copies. The notebook keeps every source cell and the three training-metric streams; stored sample figures and other execution outputs were removed.

## Result-to-script map

| Results | Generating code |
| --- | --- |
| Predicted occlusal, proximal and cusp masks | `notebooks/latest06apr.ipynb`; the three files in `models/` |
| EFD scores and contour figures | `analysis/final_avg_efd_score.py` |
| Isthmus widths, cusp pairs and ratios | `analysis/isthmus_disc.py`, `analysis/cusp_geometry.py`, `analysis/cus_ist_ratio.py` |
| Registered cavity floor detection and measurements | `analysis/landmark_cav_con_comb_smooth.py`, `analysis/refined_cav_con_comb_smooth.pyc`, `analysis/final_cav_con_comb_smooth.py` |
| Publication depth and regularity figures | `analysis/publication_cavity_depth.py` |
| Historical pulpal-reference measurements | `analysis/gingival_depth_from_pulpal.py`; versioned generators in `archive/` |
| CQS scores and charts | `analysis/final_CQS_overall.py` |
| Filtered presentation package and regenerated displays | `analysis/export_result_final.py` |
| Image/STL landmarks | `landmarks/derive_landmarks.py`, `landmarks/export_landmarks.py`, `landmarks/reproduce.py` |

## Models and performance

Open `model_performance/index.html`. `model_performance.csv` contains all three selected-checkpoint measurements; `training_history.csv` contains the recorded epoch histories. `model_details.json` links each checkpoint hash to its architecture, optimizer iteration, notebook cell and saved-checkpoint log entry. The three HDF5 files come from the repository root, matching the notebook's model paths.

The occlusal checkpoint is a U-Net-style decoder with a ResNet50 encoder. Its input follows the notebook's OpenCV BGR + ResNet50 preprocessing. Proximal and cusp inference convert BGR to RGB and divide by 255. All outputs use a threshold of 0.5 on 256 × 256 inputs. Performance values are recorded validation-split metrics, selected using validation IoU. The separate mask-correspondence table records the CPU comparison with existing masks.

## Source versions and execution

The exact refined-detector v1.2 source file is no longer present in the repository. Its original CPython 3.9 bytecode is included as `analysis/refined_cav_con_comb_smooth.pyc`, which imports directly in Python 3.9. The earlier v1.0 source is retained separately in `source_snapshots/`. The recovered `final_cav_con_comb_smooth.py` and `cav_con_comb_smooth.py` match the corresponding cached compiled code. `archive/` contains the historical generators and associated tests for the archived result versions.

These are repository-layout source copies. To rerun them, restore their original locations from the inventory or supply their input/output arguments from the original project. The export utility and several notebook cells use repository-relative paths. Select notebook sections individually. Input meshes, training images, reference masks, registered landmarks and measurement caches remain external inputs; this package does not contain those datasets. The filtered tables omit metadata required by some original analyzers, so regeneration starts from the original project inputs.

`requirements-analysis.txt` records the analysis environment's package versions. `requirements-inference.txt` lists the TensorFlow/Keras inference versions used for the checkpoint comparison plus its image-loading dependencies. `tests/` and `export_checks/` contain the existing checks; original result artifacts remain unchanged by this packaging step.
''')
    for path in STAGE.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    model_files = list((STAGE / "models").glob("*.h5"))
    assert len(model_files) == 3 and len(agreements) == 63 and len(histories) == 630
    assert max(row["different_pixels"] for row in agreements) <= 4
    for record in INVENTORY:
        assert sha256(STAGE / record["path"]) == record["sha256"]
        if record["copy_method"] == "exact":
            assert sha256(ROOT / record["source"]) == record["sha256"]
    print(json.dumps({"copied_items":len(INVENTORY),"python_files":len(list(STAGE.rglob('*.py'))),
                      "models":len(model_files),"training_epochs":len(histories),"mask_comparisons":len(agreements),
                      "total_bytes":sum(p.stat().st_size for p in STAGE.rglob('*') if p.is_file())},indent=2))


if __name__ == "__main__":
    main()
