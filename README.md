# Molar Class II Cavity Quality Assessment

Python workflows for dental image segmentation, cavity shape comparison,
isthmus/cusp measurements, STL cavity depth and floor regularity, and composite
Cavity Quality Scores (CQS). The current sources were collected from
`Result_final/scripts` in the development project.

This repository contains source code only. Images, masks, meshes, landmarks,
measurement caches, trained model weights, recorded performance values, and
generated results are not included. The notebook has no saved cell outputs,
execution counts, attachments, or training histories.

## Code layout

| Workflow | Source |
| --- | --- |
| Preprocessing, alignment, EDSR upscaling, training and prediction | `latest06apr.ipynb` |
| Occlusal and proximal EFD shape scores | `final_avg_efd_score.py` |
| Isthmus discovery and widths | `isthmus_disc.py` |
| Cusp separation, opposite-side pairing and width/distance ratios | `cusp_geometry.py`, `cus_ist_ratio.py` |
| STL floor detection and measurements | `cav_con_comb_smooth.py`, `final_cav_con_comb_smooth.py`, `landmark_cav_con_comb_smooth.py` |
| Depth, floor regularity and common-reference gingival-minus-pulpal figures | `publication_cavity_depth.py` |
| Earlier local pulpal-plane reference workflow | `gingival_depth_from_pulpal.py` |
| Composite quality scoring | `final_CQS_overall.py` |
| Image/STL registration and landmark export | `landmarks/` |
| Performance extraction and plotting | `extract_unet_performance.py`, `model_tools/plot_final_model_performance.py` |
| Presentation export, packaging and checks | `export_result_final.py`, `model_tools/`, `export_checks/` |
| Synthetic regression tests | `tests/` |
| Earlier detector source, retained for reference | `source_snapshots/` |

Analysis modules and the notebook are placed at the repository root so their
existing imports resolve. Supporting tools retain their subdirectories.
Superseded result-specific archives and compiled Python files are omitted.

## Installation

The analysis requirements record the development environment (Python 3.9).
Create a separate environment for TensorFlow or landmark registration because
their NumPy/OpenCV requirements differ.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

For the notebook, install `requirements-notebook.txt` plus Jupyter in its own
environment. `requirements-inference.txt` is the smaller inference dependency
list; use one OpenCV distribution per environment. Install
`landmarks/requirements.txt` separately for landmark fitting. Interactive
plots use Plotly; PNG mesh rendering requires a working headless VTK/OpenGL
environment.

## External inputs and segmentation

Start Jupyter from the checkout and review the notebook's path cell. Set
`DENTAL_PROJECT_ROOT` to an external project directory if inputs live elsewhere;
set `DENTAL_MODEL_DIR` to its checkpoint directory. Prediction requires:

- `samples_stu_O/`: occlusal images, also used for cusp prediction.
- `samples_stu_P/`: proximal images.
- `O_unet_refined.h5`, `P_unet_refined.h5`, `CU_unet_refined.h5`: external checkpoints.

Prediction preserves the input filename stem and adds `_mask.png`. Saved masks
are binary foreground-white/background-black images. Keep these measurement
inputs separate from colored presentation figures. Preparation/training cells
retain their original directory settings; update them before execution. EDSR
upscaling additionally needs an external `EDSR_x4.pb` file.

The occlusal model uses BGR images with ResNet50 preprocessing. Proximal and
cusp inference use RGB divided by 255. The original unit-scaled training
generator uses BGR; this historical distinction is retained and should be
checked when retraining or substituting checkpoints.

## Measurement commands

Run from the repository root. Use `--help` to inspect each script's input and
output options; defaults retain the original development directory names.

```bash
python final_avg_efd_score.py --cavity-type occlusal \
  --pred-folder /path/to/occlusal_masks --ref-folder /path/to/reference_masks \
  --samples-folder /path/to/occlusal_images --output-dir outputs/efd_occlusal

python isthmus_disc.py --mask-dir /path/to/occlusal_masks \
  --cusp-dir /path/to/cusp_masks --image-dir /path/to/occlusal_images \
  --output-dir outputs/isthmus

python cus_ist_ratio.py --isthmus-json outputs/isthmus/isthmus_details.json \
  --mask-dir /path/to/occlusal_masks --cusp-dir /path/to/cusp_masks \
  --image-dir /path/to/occlusal_images --output-dir outputs/ratios

python publication_cavity_depth.py --stl-dir /path/to/meshes \
  --measurements-dir /path/to/unfiltered_measurements --out-dir outputs/depth

python final_CQS_overall.py --occlusal-efd-csv /path/to/occlusal_scores.csv \
  --proximal-efd-csv /path/to/proximal_scores.csv \
  --depth-csv outputs/depth/depth_summary.csv \
  --ratio-csv outputs/ratios/ratios.csv --out-dir outputs/cqs
```

Repeat EFD scoring with `--cavity-type proximal` and the corresponding inputs.
Measurements use original `O_<id>` / `P_<id>` naming, optionally `_st`, and
unfiltered intermediate schemas. Presentation labels such as `MCL2_sample_*`
and filtered tables are final exports, not interchangeable measurement inputs.
STL coordinates are assumed to be millimeters; mask widths require verified
pixel calibration for conversion to millimeters.

CQS uses `average_matching_score_pct` for both EFD components. The implemented
weights are 25% occlusal EFD, 25% proximal EFD, 10% occlusal depth, 10%
pulpal-to-gingival depth, 20% isthmus/intercuspal ratio, and 5% for each floor's
regularity. These are configurable research scoring assumptions, not a
clinically validated grading instrument.

## Source availability and testing

**The exact refined cavity detector v1.2 source is unavailable in the supplied
package.** Its compiled-only copy is excluded from this source repository.
`landmark_cav_con_comb_smooth.py` depends on
`refined_cav_con_comb_smooth.py` and therefore cannot run until that exact module
is supplied externally. `source_snapshots/refined_cav_con_comb_smooth_v1_0.py`
is an older source snapshot, not an equivalent substitute. The standalone
baseline/final analyzers and publication renderer are available as source;
the renderer requires previously generated, unfiltered measurement caches.

The default unittest suite uses synthetic shapes and surfaces. One optional
duplicate-mask regression is skipped when its external masks are absent.
Detector-dependent tests are under `tests/optional_detector/`; run them only
after supplying the missing detector source:

```bash
python -m unittest discover -s tests/optional_detector -p 'test_*.py' -v
```

Historical packaging, relabeling and export tools expect the original project
layout and case-selection rules. They do not download missing inputs. Metric
extraction requires an externally retained notebook with completed training
logs; the cleared notebook here intentionally contains none. Performance
plotting accepts an external metrics CSV through `--csv`.

## Keeping the repository source-only

Do not commit datasets, checkpoints, caches, generated tables/figures, or
executed notebook outputs. `.gitignore` excludes common input/output formats
and directories. Review `git diff --cached` before committing; clear notebook
outputs again after local execution. Dependencies and scripts are retained,
while the original local `Result_final` package is left unchanged.
