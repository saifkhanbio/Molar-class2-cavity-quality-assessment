#!/usr/bin/env python3
"""Tailor the packaged notebook and synchronize its provenance references."""
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Result_final/scripts"
TARGET = PACKAGE / "notebooks/latest06apr.ipynb"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    original = json.loads((ROOT / "latest06apr.ipynb").read_text())
    packaged = json.loads(TARGET.read_text())
    assert len(packaged["cells"]) == 96, "Expected the initial packaged notebook"
    backup = ROOT / "work/latest06apr_packaged_before_tailoring.ipynb"
    assert not backup.exists()
    backup.write_bytes(TARGET.read_bytes())
    cells, retained, changes = [], [], {}

    def markdown(text, label):
        cells.append({"cell_type": "markdown", "id": label,
                      "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(keepends=True)})

    def code(text, label):
        ast.parse(text)
        cells.append({"cell_type": "code", "id": label, "metadata": {},
                      "source": text.strip().splitlines(keepends=True),
                      "execution_count": None, "outputs": []})

    def keep(index, text=None, reason=None):
        cell = copy.deepcopy(packaged["cells"][index])
        cell["id"] = f"source-cell-{index}"
        cell["metadata"]["original_notebook_cell_index"] = index
        if text is not None:
            cell["source"] = text.strip().splitlines(keepends=True)
            changes[index] = reason or "Section heading updated"
        elif cell["cell_type"] == "markdown":
            source = "".join(cell["source"]).lstrip("# ").strip()
            cell["source"] = ["### " + source + "\n"]
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            cell["execution_count"] = None
            if cell.get("outputs"):
                cell["metadata"]["output_provenance"] = "Recorded original training run; not rerun during tailoring"
        cells.append(cell)
        retained.append(index)

    def source(index):
        return "".join(packaged["cells"][index]["source"])

    markdown("""
    # Final artifact workflow

    This notebook retains data preparation and the three segmentation models used
    with `Result_final`: occlusal ResNet50 U-Net, proximal refined U-Net, and cusp
    refined U-Net. Shape, isthmus, depth, and CQS calculations use the linked Python
    scripts at the end.

    Run **Environment and paths**, then the relevant **Predict retained masks**
    cell to use the supplied checkpoints. Data preparation and training are
    separate workflows; choose those sections when preparing data or retraining.
    Prediction writes to `notebook_predictions/`, and new training checkpoints to
    `notebook_training/`, under the project root.

    Training outputs are the original recorded logs, not results of a new run.
    [Model performance](../model_performance/index.html) and
    [section provenance](notebook_sections.json) accompany this notebook.
    """, "workflow-guide")
    keep(0, "## 1. Environment and paths")
    keep(1, """import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input
""", "Keep imports needed by the retained sections; remove unused Torch, TensorFlow Hub and VGG imports")
    keep(2)
    keep(3)
    code("""# Run from the project root or a directory beneath it.
PROJECT_ROOT = next(
    (p for p in (Path.cwd(), *Path.cwd().parents)
     if (p / 'samples_stu_O').is_dir()
     and (p / 'Result_final/scripts/models').is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Set PROJECT_ROOT to the original dental-analysis project directory.')
SCRIPT_ROOT = PROJECT_ROOT / 'Result_final/scripts'
MODEL_DIR = SCRIPT_ROOT / 'models'
PREDICTION_OUTPUT_ROOT = PROJECT_ROOT / 'notebook_predictions'
TRAINING_OUTPUT_DIR = PROJECT_ROOT / 'notebook_training'
TARGET_SIZE = (input_width, input_height)
os.chdir(PROJECT_ROOT)
""", "project-paths")
    markdown("""
    ## 2. Data preparation, alignment and upscaling

    The original cropping, EDSR upscaling, image/mask alignment and validation-data
    preparation code is retained. Set each section's input/output directories for
    the required dataset before running it. Folder capitalization must match the
    filesystem. EDSR requires `opencv-contrib-python` and the external `EDSR_x4.pb`;
    its path is specified in each upscaling cell. The preparation sections use
    their original output directories. Training and prediction below can use
    already prepared data without rerunning this section.
    """, "preparation-guide")
    for index in range(4, 28):
        if index == 18:
            keep(index, "### Align training images, cavity masks and cusp masks")
        else:
            keep(index)

    markdown("""
    ## 3. Predict retained masks with the final checkpoints

    Run the environment/path cells first, then any prediction cell independently.
    Each cell selects the corresponding filenames already present in `Result_final`,
    loads its own checkpoint, and writes binary masks to `notebook_predictions/`.
    Occlusal input uses OpenCV BGR followed by ResNet50 preprocessing. Proximal and
    cusp input uses RGB divided by 255. All masks use a probability threshold of 0.5
    and a 256 × 256 image size.
    """, "prediction-guide")
    for heading, index, label, model_file, samples, folder in [
        (65, 66, "Occlusal cavity", "O_unet_refined.h5", "samples_stu_O", "pred_O_M_masks_folder"),
        (67, 68, "Proximal cavity", "P_unet_refined.h5", "samples_stu_P", "pred_P_M_masks_folder"),
        (82, 83, "Cusps", "CU_unet_refined.h5", "samples_stu_O", "pred_O_M_cusp_molar_mask"),
    ]:
        keep(heading, "### " + label)
        loop = source(index)[source(index).index("for fname in sorted(os.listdir(samples_folder)):"):]
        loop = loop.replace("('.png','.jpg')", "('.png', '.jpg', '.jpeg')")
        loop = loop.replace("    base    = os.path.splitext(fname)[0]\n",
                            "    base    = os.path.splitext(fname)[0]\n"
                            "    if base + '_mask.png' not in retained_mask_names:\n"
                            "        continue\n")
        setup = f"""model = load_model(str(MODEL_DIR / '{model_file}'), compile=False)
samples_folder = PROJECT_ROOT / '{samples}'
pred_folder = PREDICTION_OUTPUT_ROOT / '{folder}'
pred_folder.mkdir(parents=True, exist_ok=True)
retained_mask_names = {{p.name for p in (SCRIPT_ROOT.parent / '{folder}').glob('*.png')}}
TARGET_SIZE = (256, 256)

"""
        keep(index, setup + loop, "Explicit packaged checkpoint and sample/output paths; select retained cases; original preprocessing and threshold preserved")

    markdown("""
    ## 4. Model training and recorded histories

    These sections retain the architectures, losses, augmentation settings and
    training schedules associated with the final checkpoints. Run the shared
    generators first, then the complete section for the selected model in order.
    Model definitions no longer delete existing checkpoints or run temporary
    debugging predictions. New training saves to `notebook_training/`; section 3
    continues to load the supplied checkpoints in `models/`.

    The unit-scaled and ResNet generators have separate names so model sections
    can be selected independently. The original unit-scaled generator keeps its
    recorded OpenCV BGR training convention; the RGB inference convention above
    is preserved separately. Existing training logs and performance values remain
    historical measurements. A new training run produces its own metrics.
    """, "training-guide")
    markdown("### Shared image/mask resizing", "shared-resize")
    keep(36)
    keep(41, "### Unit-scaled generator for proximal and cusp training")
    keep(42, source(42).replace("class SegmentationDataGenerator(", "class UnitScaleSegmentationDataGenerator("),
         "Name the unit-scaled generator explicitly to prevent cross-section replacement")
    keep(59, "### ResNet50 generator for occlusal training")
    keep(60, source(60).replace("class SegmentationDataGenerator(", "class ResNetSegmentationDataGenerator("),
         "Name the ResNet generator explicitly to prevent cross-section replacement")

    def architecture(index):
        text = source(index)
        text = text[:text.index("    return model") + len("    return model\n")]
        text = text.replace("from skimage import morphology, util\n", "").replace("from scipy import interpolate\n", "")
        keep(index, text, "Keep architecture/loss/metric definitions; remove checkpoint deletion and temporary debugging block")

    def train(index, model_file):
        text = "TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n" + source(index)
        text = text.replace('"' + model_file + '", save_best_only=True',
                            'str(TRAINING_OUTPUT_DIR / "' + model_file + '"), save_best_only=True')
        keep(index, text, "Store new training checkpoints separately; preserve training schedule and recorded output")

    markdown("### Proximal refined U-Net", "proximal-training")
    for index in [28, 29, 32, 33, 37, 38, 43]:
        keep(index)
    keep(44, source(44).replace("SegmentationDataGenerator(", "UnitScaleSegmentationDataGenerator("),
         "Select the explicit unit-scaled generator")
    keep(49)
    architecture(50)
    keep(51)
    train(52, "P_unet_refined.h5")

    markdown("### Occlusal U-Net with ResNet50 encoder", "occlusal-training")
    for index in [30, 31, 34, 35]:
        keep(index)
    keep(57, "### Define the final occlusal architecture")
    keep(58)
    keep(61)
    keep(62, source(62).split("# preview")[0].replace("SegmentationDataGenerator(", "ResNetSegmentationDataGenerator("),
         "Select the ResNet generator and remove development preview loops")
    keep(63, "### Train the occlusal decoder")
    train(64, "O_unet_refined.h5")

    keep(71, "### Cusp refined U-Net")
    keep(72)
    keep(73, source(73).replace("'pred_M_cusp_molar_mask'", "'pred_O_M_cusp_molar_mask'"),
         "Use the final occlusal cusp output folder name")
    keep(74)
    keep(75, source(75).replace("SegmentationDataGenerator(", "UnitScaleSegmentationDataGenerator("),
         "Select unit-scaled cusp inputs explicitly instead of inheriting the ResNet generator")
    keep(78)
    architecture(79)
    keep(80)
    train(81, "CU_unet_refined.h5")

    markdown("""
    ## 5. Generate measurements and result figures

    Continue with the packaged scripts for the final calculations. Their CLI
    options specify the original inputs and output folders.

    | Result | Script |
    | --- | --- |
    | Occlusal/proximal EFD scores and contours | [final_avg_efd_score.py](../analysis/final_avg_efd_score.py) |
    | Isthmus detection and widths | [isthmus_disc.py](../analysis/isthmus_disc.py) |
    | Cusp pairing, distances and isthmus ratios | [cus_ist_ratio.py](../analysis/cus_ist_ratio.py) and [cusp_geometry.py](../analysis/cusp_geometry.py) |
    | Cavity floor detection and measurement | [landmark_cav_con_comb_smooth.py](../analysis/landmark_cav_con_comb_smooth.py) |
    | Depth and regularity PNG/HTML figures | [publication_cavity_depth.py](../analysis/publication_cavity_depth.py) |
    | CQS scores and tables | [final_CQS_overall.py](../analysis/final_CQS_overall.py) |
    | Filtered result presentation | [export_result_final.py](../analysis/export_result_final.py) |

    [Package instructions](../README.md) describe external mesh, landmark and
    reference-mask inputs, plus the Python 3.9 detector dependency.
    """, "downstream-artifacts")

    removed = sorted(set(range(96)) - set(retained))
    assert removed == [39, 40, 45, 46, 47, 48, 53, 54, 55, 56, 69, 70, 76, 77, *range(84, 96)]
    assert len(retained) == len(set(retained))
    packaged["cells"] = cells
    packaged["nbformat_minor"] = max(5, packaged["nbformat_minor"])
    packaged["metadata"]["workflow_scope"] = "Final segmentation artifacts and retained preparation/training workflows"
    packaged["metadata"]["original_notebook_sha256"] = sha256(ROOT / "latest06apr.ipynb")
    TARGET.write_text(json.dumps(packaged, indent=1, ensure_ascii=False) + "\n")
    mapping = {c["metadata"]["original_notebook_cell_index"]: i for i, c in enumerate(cells)
               if "original_notebook_cell_index" in c["metadata"]}
    provenance = {
        "original_notebook": "latest06apr.ipynb (repository root)",
        "original_sha256": sha256(ROOT / "latest06apr.ipynb"),
        "packaged_sha256": sha256(TARGET),
        "original_cells": 96, "retained_original_cells": len(retained), "tailored_cells": len(cells),
        "removed_original_cells": removed,
        "removed_sections": ["Superseded plain occlusal U-Net", "Duplicate occlusal generator setup",
                             "Development-only augmentation previews", "Legacy EFD scoring experiment",
                             "Legacy isthmus implementations and cusp-overlay instructions",
                             "Empty depth/3D headings and outdated scoring outline"],
        "cell_map": [{"original_index": old, "packaged_index": new,
                      "change": changes.get(old, "Code retained; markdown heading level normalized" if original['cells'][old]['cell_type'] == 'markdown' else "Source unchanged")}
                     for old, new in sorted(mapping.items())],
        "historical_training_output_cells": {str(old): mapping[old] for old in [52, 64, 81]},
        "training_or_prediction_executed_during_tailoring": False,
    }
    (PACKAGE / "notebooks/notebook_sections.json").write_text(json.dumps(provenance, indent=2) + "\n")

    inventory_path = PACKAGE / "script_inventory.csv"
    with inventory_path.open() as stream:
        inventory = list(csv.DictReader(stream))
    for row in inventory:
        if row["path"] == "notebooks/latest06apr.ipynb":
            row.update(bytes=TARGET.stat().st_size, sha256=sha256(TARGET),
                       copy_method="tailored_final_workflow", purpose="Final model workflows, retained preparation and original training logs; cell map in notebooks/notebook_sections.json")
    tool = PACKAGE / "model_tools/tailor_result_final_notebook.py"
    shutil.copy2(Path(__file__).resolve(), tool)
    inventory.append(dict(path="model_tools/tailor_result_final_notebook.py",
                          source="work/tailor_result_final_notebook.py",
                          purpose="Notebook section selection and provenance update after package assembly",
                          bytes=tool.stat().st_size, sha256=sha256(tool), copy_method="exact"))
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    for path in [PACKAGE / "model_performance/model_performance.csv", PACKAGE / "model_performance/model_details.json"]:
        if path.suffix == ".csv":
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
        else:
            rows = json.loads(path.read_text())
        for row in rows:
            row["notebook_cell_index_scope"] = "original_repository_notebook"
            row["packaged_notebook_cell_index"] = mapping[int(row["notebook_cell_index"])]
            if path.suffix == ".json":
                row["packaged_notebook_sha256"] = sha256(TARGET)
        if path.suffix == ".csv":
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in ["original_cells", "retained_original_cells", "tailored_cells", "removed_original_cells"]}, indent=2))


if __name__ == "__main__":
    main()
