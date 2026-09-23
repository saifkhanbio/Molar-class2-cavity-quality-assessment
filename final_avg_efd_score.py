#!/usr/bin/env python3
"""Occlusal and proximal EFD shape matching with explicit best/average scores.

Defaults: occlusal masks/references, order 18. Use --cavity-type proximal for
proximal masks. A recognized --pred-folder also selects matching defaults.
The score is 100 * max(0, 1 - aligned normalized EFD residual). Exact duplicate
contours score 100%; this is a geometric similarity, not clinical accuracy.
Translation, uniform scale, contour start and rotation are ignored; shape and
aspect ratio are retained. Only the dominant exterior boundary is scored.

EFD formulation: Kuhl & Giardina (1982), doi:10.1016/0146-664X(82)90034-X.
This implementation integrates a closed piecewise-linear contour and aligns
all harmonics jointly rather than fixing a potentially unstable first ellipse.
Dependencies: numpy, scipy, scikit-image, Pillow, matplotlib (plots only).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize_scalar
from skimage.measure import find_contours

VERSION = "2"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_paths(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_binary_mask(path):
    """Accept both 0/1 and 0/255 masks without changing their geometry."""
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("L"))
    return pixels > (0.5 if pixels.max(initial=0) <= 1 else 127)


def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask.shape, dtype=np.int64).tobytes() +
                          np.packbits(mask).tobytes()).hexdigest()


def signed_area(contour):
    return 0.5 * float(np.sum(contour[:, 0] * np.roll(contour[:, 1], -1) -
                              contour[:, 1] * np.roll(contour[:, 0], -1)))


def efd_coefficients(contour, order=18):
    """Integrate non-DC harmonics over a closed contour's normalized arc length.

    Return [harmonic, coordinate (x,y), basis (cos,sin)], with unit coefficient
    norm. Dropping the DC term removes translation. Norm scaling is isotropic.
    """
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("EFD order must be a positive integer.")
    points = np.asarray(contour, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Expected at least three finite 2D contour points.")
    # Explicitly close the polygon; discard repeated vertices, including an
    # already repeated closing point. Counterclockwise traversal is canonical.
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]]
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-12:
        points = points[:-1]
    points = points - points.mean(axis=0)
    area = signed_area(points)
    if len(points) < 3 or abs(area) < 1e-12:
        raise ValueError("Degenerate contour has no enclosed area.")
    if area < 0:
        points = points[::-1]
    steps = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(steps, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Degenerate contour segment.")
    perimeter = float(lengths.sum())
    knots = np.r_[0.0, np.cumsum(lengths)] / perimeter
    n = np.arange(1, order + 1, dtype=float)
    phase = 2 * np.pi * n[:, None] * knots[None, :]
    velocity = steps / lengths[:, None]
    factor = perimeter / (2 * np.pi ** 2 * n ** 2)
    cos_coeff = factor[:, None] * (np.diff(np.cos(phase), axis=1) @ velocity)
    sin_coeff = factor[:, None] * (np.diff(np.sin(phase), axis=1) @ velocity)
    coeff = np.stack((cos_coeff, sin_coeff), axis=-1)
    norm = float(np.linalg.norm(coeff))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("EFD coefficients have zero or invalid norm.")
    return coeff / norm


def extract_descriptor(path, order=18):
    mask = read_binary_mask(path)
    labels, count = ndi.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        raise ValueError("Empty foreground mask.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(np.argmax(sizes))
    area = int(component.sum())
    if area < 9:
        raise ValueError("Foreground is too small for a stable contour (<9 pixels).")
    # Padding closes contours even when they touch the image edge.
    contours = find_contours(np.pad(component, 1), 0.5, fully_connected="high")
    contour = max((c[:, ::-1] - 1 for c in contours), key=lambda c: abs(signed_area(c)))
    warnings = []
    discarded = int(mask.sum()) - area
    if discarded:
        warnings.append("secondary_components_not_scored")
    if discarded / int(mask.sum()) >= 0.05:
        warnings.append("substantial_disconnected_foreground_review")
    holes = int((ndi.binary_fill_holes(component) & ~component).sum())
    if holes:
        warnings.append("holes_not_scored_by_exterior_efd")
    if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
        warnings.append("foreground_touches_image_boundary")
    return {"filename": Path(path).name, "path": str(path), "mask": mask,
            "mask_hash": mask_hash(mask), "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "coefficients": efd_coefficients(contour, order), "component_area_px": area,
            "foreground_area_px": int(mask.sum()), "discarded_area_px": discarded,
            "hole_area_px": holes, "component_count": int(count), "warnings": warnings}


def align_descriptors(first, second):
    """Minimize normalized residual over cyclic phase and proper 2D rotation."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (2, 2):
        raise ValueError("Descriptor orders/shapes must match.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Descriptors must be finite.")
    if min(np.linalg.norm(a), np.linalg.norm(b)) <= 1e-12:
        raise ValueError("Descriptor cannot have zero norm.")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    if np.allclose(a, b, rtol=0, atol=1e-13):
        return {"distance": 0.0, "similarity_pct": 100.0, "phase_radians": 0.0,
                "rotation_radians": 0.0, "aligned_coefficients": b.copy()}
    n = np.arange(1, len(a) + 1)
    quarter = b @ np.array([[0., -1.], [1., 0.]])
    dot0 = np.sum(a * b, axis=(1, 2))
    dot1 = np.sum(a * quarter, axis=(1, 2))
    cross0 = np.sum(a[:, 1] * b[:, 0] - a[:, 0] * b[:, 1], axis=1)
    cross1 = np.sum(a[:, 1] * quarter[:, 0] - a[:, 0] * quarter[:, 1], axis=1)

    def correlation(phase):
        cos, sin = np.cos(np.asarray(phase)[..., None] * n), np.sin(np.asarray(phase)[..., None] * n)
        dot = cos @ dot0 + sin @ dot1
        cross = cos @ cross0 + sin @ cross1
        return np.hypot(dot, cross), dot, cross

    size = max(256, 16 * len(a))
    grid = np.arange(size) * (2 * np.pi / size)
    values = correlation(grid)[0]
    peaks = np.flatnonzero((values >= np.roll(values, 1)) & (values > np.roll(values, -1)))
    if not len(peaks):
        peaks = np.array([int(np.argmax(values))])
    step = 2 * np.pi / size
    best_phase = float(grid[np.argmax(values)])
    best_value = float(values.max())
    for index in peaks:
        fit = minimize_scalar(lambda p: -float(correlation(p)[0]),
                              bounds=(grid[index] - step, grid[index] + step), method="bounded",
                              options={"xatol": 1e-12})
        if -fit.fun > best_value:
            best_value, best_phase = float(-fit.fun), float(fit.x)
    _, dot, cross = correlation(best_phase)
    angle = float(np.arctan2(cross, dot))
    cos, sin = np.cos(n * best_phase), np.sin(n * best_phase)
    phase_matrices = np.stack((np.stack((cos, -sin), axis=1),
                               np.stack((sin, cos), axis=1)), axis=1)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    aligned = rotation @ (b @ phase_matrices)
    distance = float(np.linalg.norm(a - aligned))
    if distance < 1e-8:
        distance = 0.0
    return {"distance": distance, "similarity_pct": 100.0 * max(0.0, 1.0 - distance),
            "phase_radians": best_phase % (2 * np.pi), "rotation_radians": angle,
            "aligned_coefficients": aligned}


def reconstruct(coefficients, count=512):
    t = np.linspace(0, 2 * np.pi, count)
    phase = np.arange(1, len(coefficients) + 1)[:, None] * t
    return np.einsum("ncb,nbt->tc", coefficients, np.stack((np.cos(phase), np.sin(phase)), axis=1))


def load_folder(folder, order):
    valid, invalid = [], []
    for path in image_paths(folder):
        try:
            valid.append(extract_descriptor(path, order))
        except (ValueError, OSError) as error:
            invalid.append({"filename": path.name, "error": str(error)})
    return valid, invalid


def unique_references(references):
    """Deduplicate exact masks so repeated files do not reweight the average."""
    groups = {}
    for reference in references:
        groups.setdefault(reference["mask_hash"], []).append(reference)
    return list(groups.values())


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_path_for_prediction(prediction, samples_folder):
    stem = Path(prediction).stem
    if stem.endswith("_mask"):
        stem = stem[:-5]
    matches = [path for path in image_paths(samples_folder) if path.stem == stem] if samples_folder.is_dir() else []
    return matches[0] if matches else None


def save_plot(prediction, sample_path, reference, match, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), layout="constrained")
    mask = prediction["mask"]
    rgb = np.ones((*mask.shape, 3)); rgb[mask] = np.array([46, 175, 80]) / 255
    axes[1].imshow(rgb, interpolation="nearest")
    cavity_type = summary.get("cavity_type", "")
    axes[1].set_title(f"Predicted {cavity_type} cavity".replace("  ", " "))
    if sample_path:
        with Image.open(sample_path) as source:
            tooth = np.asarray(source.convert("RGB")) / 255
        axes[0].imshow(tooth)
        axes[0].set_title("Sample tooth")
        if tooth.shape[:2] == mask.shape:
            overlay = tooth.copy(); overlay[mask] = 0.45 * tooth[mask] + 0.55 * np.array([0, 1, 0])
            axes[2].imshow(overlay)
            axes[2].set_title("Tooth + predicted cavity")
        else:
            axes[2].text(.5, .5, "Grid mismatch\nOverlay withheld", ha="center", va="center")
    else:
        axes[0].text(.5, .5, "Sample unavailable", ha="center", va="center")
    a = reconstruct(prediction["coefficients"])
    b = reconstruct(match["aligned_coefficients"])
    axes[3].plot(a[:, 0], a[:, 1], color="#2eaf50", label="Prediction", lw=2)
    axes[3].plot(b[:, 0], b[:, 1], color="#0057d9", label="Best reference", lw=1.5, ls="--")
    axes[3].set_aspect("equal"); axes[3].invert_yaxis()
    axes[3].set_title(f"Aligned EFD contours\n{reference['filename']}")
    axes[3].legend(loc="lower left", fontsize=8)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(prediction["filename"])
    caption = (f"Best reference: {summary['similarity_pct']:.2f}%   |   "
               f"Mean of {summary['unique_reference_count']} unique references: {summary['average_matching_score_pct']:.2f}%   |   "
               f"EFD order: {summary['efd_order']}")
    if prediction["warnings"]:
        caption += "\n" + "; ".join(prediction["warnings"])
    fig.supxlabel(caption, fontsize=9)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def public_descriptor(item):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in item.items() if key != "mask"}


def score_masks(pred_folder, ref_folder, samples_folder, order, output_dir, plots=True, cavity_type=""):
    predictions, invalid_predictions = load_folder(pred_folder, order)
    references, invalid_references = load_folder(ref_folder, order)
    if not references:
        raise ValueError("No valid reference contours.")
    if not predictions and not invalid_predictions:
        raise ValueError("No prediction image files.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if plots:
        (output_dir / "plots").mkdir(exist_ok=True)
    groups = unique_references(references)
    rows, comparisons, duplicate_checks = [], [], []
    fields = ["sample", "status", "cavity_type", "image_mapping", "efd_order", "best_reference", "best_reference_aliases",
              "efd_distance", "similarity_pct", "average_matching_score_pct", "average_all_files_pct",
              "corresponding_reference", "corresponding_reference_score_pct", "unique_reference_count",
              "reference_file_count", "foreground_area_px", "discarded_area_px", "scored_component_fraction", "warnings", "error"]
    for prediction in predictions:
        matches = [align_descriptors(prediction["coefficients"], group[0]["coefficients"]) for group in groups]
        best_index = min(range(len(groups)), key=lambda i: (matches[i]["distance"], groups[i][0]["filename"]))
        best, reference = matches[best_index], groups[best_index][0]
        scores = [m["similarity_pct"] for m in matches]
        corresponding_stem = Path(prediction["filename"]).stem.removesuffix("_mask")
        corresponding = [(ref["filename"], match["similarity_pct"]) for group, match in zip(groups, matches)
                         for ref in group if Path(ref["filename"]).stem == corresponding_stem]
        sample_path = sample_path_for_prediction(prediction["filename"], samples_folder)
        image_mapping = "missing_sample"
        if sample_path:
            with Image.open(sample_path) as source:
                image_mapping = "same_grid" if source.size == prediction["mask"].shape[::-1] else "grid_mismatch"
        warnings = list(prediction["warnings"])
        if image_mapping != "same_grid":
            warnings.append(image_mapping + "_overlay_unavailable")
        row = {"sample": prediction["filename"], "status": "REVIEW" if warnings else "OK",
               "cavity_type": cavity_type, "image_mapping": image_mapping,
               "efd_order": order, "best_reference": reference["filename"],
               "best_reference_aliases": ";".join(r["filename"] for r in groups[best_index]),
               "efd_distance": best["distance"], "similarity_pct": best["similarity_pct"],
               "average_matching_score_pct": float(np.mean(scores)),
               "average_all_files_pct": float(np.average(scores, weights=[len(g) for g in groups])),
               "corresponding_reference": corresponding[0][0] if corresponding else "",
               "corresponding_reference_score_pct": corresponding[0][1] if corresponding else "",
               "unique_reference_count": len(groups), "reference_file_count": len(references),
               "foreground_area_px": prediction["foreground_area_px"],
               "discarded_area_px": prediction["discarded_area_px"],
               "scored_component_fraction": prediction["component_area_px"] / prediction["foreground_area_px"],
               "warnings": ";".join(warnings), "error": ""}
        rows.append(row)
        for group, match in zip(groups, matches):
            comparisons.append({"sample": prediction["filename"], "reference": group[0]["filename"],
                                "reference_aliases": ";".join(r["filename"] for r in group),
                                "efd_distance": match["distance"], "similarity_pct": match["similarity_pct"],
                                "phase_radians": match["phase_radians"], "rotation_radians": match["rotation_radians"],
                                "reference_warnings": ";".join(group[0]["warnings"])})
        if plots:
            save_plot(prediction, sample_path, reference,
                      best, row, output_dir / "plots" / f"{Path(prediction['filename']).stem}_efd_match.png")
        print(f"{prediction['filename']}: best={best['similarity_pct']:.2f}% mean={row['average_matching_score_pct']:.2f}%")
    for item in invalid_predictions:
        rows.append({"sample": item["filename"], "status": "INVALID_MASK", "cavity_type": cavity_type,
                     "efd_order": order, "error": item["error"]})
    # Direct prediction-to-prediction comparison makes duplicate guarantees
    # testable without confusing them with an average against unrelated refs.
    pairwise = []
    for first, second in combinations(predictions, 2):
        match = align_descriptors(first["coefficients"], second["coefficients"])
        exact = first["mask_hash"] == second["mask_hash"]
        pairwise.append({"first": first["filename"], "second": second["filename"],
                         "exact_mask_duplicate": exact, "efd_distance": match["distance"],
                         "similarity_pct": match["similarity_pct"]})
        if exact:
            duplicate_checks.append(pairwise[-1])
    write_csv(output_dir / "efd_similarity_scores.csv", rows, fields)
    write_csv(output_dir / "reference_comparisons.csv", comparisons,
              ["sample", "reference", "reference_aliases", "efd_distance", "similarity_pct",
               "phase_radians", "rotation_radians", "reference_warnings"])
    write_csv(output_dir / "prediction_pairwise_scores.csv", pairwise,
              ["first", "second", "exact_mask_duplicate", "efd_distance", "similarity_pct"])
    details = {"version": VERSION, "cavity_type": cavity_type, "efd_order": order,
               "pred_folder": str(pred_folder), "ref_folder": str(ref_folder), "samples_folder": str(samples_folder),
               "score_definition": "100 * max(0, 1 - normalized all-harmonic residual after phase and proper rotation alignment)",
               "average_definition": "mean across unique exact binary reference masks; all-files mean exported separately",
               "invariances": ["translation", "uniform scale", "rotation", "contour start", "traversal direction"],
               "limitations": "Dominant exterior only; ignores absolute size/location and holes. Similarity is not calibrated clinical quality. No reflection matching.",
               "references": [public_descriptor(r) for r in references],
               "predictions": [public_descriptor(r) for r in predictions],
               "invalid_predictions": invalid_predictions, "invalid_references": invalid_references,
               "exact_duplicate_checks": duplicate_checks, "results": rows}
    (output_dir / "details.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} samples; {len(groups)} unique references; {len(duplicate_checks)} duplicate pairs to {output_dir}")
    return details


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cavity-type", choices=("occlusal", "proximal"), default=None,
                        help="Default: infer from --pred-folder, otherwise occlusal.")
    parser.add_argument("--pred-folder", type=Path)
    parser.add_argument("--ref-folder", type=Path)
    parser.add_argument("--samples-folder", type=Path)
    parser.add_argument("--efd-order", type=int, default=18)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    inferred = None
    if args.pred_folder:
        if args.pred_folder.name.startswith("pred_O_"):
            inferred = "occlusal"
        elif args.pred_folder.name.startswith("pred_P_"):
            inferred = "proximal"
    if args.cavity_type and inferred and args.cavity_type != inferred:
        parser.error("--cavity-type conflicts with the recognized prediction folder.")
    args.cavity_type = args.cavity_type or inferred or "occlusal"
    prefix = "O" if args.cavity_type == "occlusal" else "P"
    args.pred_folder = args.pred_folder or Path(f"pred_{prefix}_M_masks_folder")
    args.ref_folder = args.ref_folder or Path(f"Scoring_ref_{prefix}_M_mask")
    args.samples_folder = args.samples_folder or Path(f"samples_stu_{prefix}")
    args.output_dir = args.output_dir or Path("final_avg_efd_results") / args.cavity_type
    opposite = "P" if prefix == "O" else "O"
    if (args.ref_folder.name == f"Scoring_ref_{opposite}_M_mask" or
            args.samples_folder.name == f"samples_stu_{opposite}"):
        parser.error("Reference/sample folders must match the selected cavity type.")
    if args.efd_order < 1:
        parser.error("--efd-order must be positive.")
    for path in (args.pred_folder, args.ref_folder):
        if not path.is_dir():
            parser.error(f"Missing directory: {path}")
    if any(args.output_dir.resolve() == path.resolve() for path in (args.pred_folder, args.ref_folder, args.samples_folder)):
        parser.error("Use a separate output directory, not an input folder.")
    return args


def main():
    args = parse_args()
    try:
        details = score_masks(args.pred_folder, args.ref_folder, args.samples_folder, args.efd_order,
                              args.output_dir, plots=not args.no_plots, cavity_type=args.cavity_type)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return int(bool(details["invalid_predictions"]))


if __name__ == "__main__":
    raise SystemExit(main())
