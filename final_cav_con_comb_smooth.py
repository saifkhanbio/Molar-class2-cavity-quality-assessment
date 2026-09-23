#!/usr/bin/env python3
"""Measure occlusal/proximal floor depth and detrended floor unevenness.

Standalone refinement of cav_con_comb_smooth.py. Detection is inherited; final
measurements use strict floor triangles, physical surface-area weighting, a
common fitted occlusal reference, the original 0.75 mm occlusal subtraction,
and explicit missing data.
Run: python3 final_cav_con_comb_smooth.py --stl_dir STLFILES
See the generated measurement_definitions.md and run_manifest.json.
"""

import argparse
import csv
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import Delaunay, cKDTree
from stl import mesh as stlmesh

# ---------------- CONFIG ----------------
TOP_Z_FRAC = 0.50
MIN_POINTS = 50
PROXIMAL_MIN_POINTS = 5
PROXIMAL_SEED_MIN_POINTS = 5
PROXIMAL_BOTTOM_CROP_FRAC = 0.10
PROXIMAL_RASTER_RES_MM = 0.05
PROXIMAL_RASTER_DILATE_CELLS = 7
PROXIMAL_OCCLUSAL_OVERLAP_FRAC = 0.07
PROXIMAL_FLOOR_MIN_BELOW_OCCLUSAL_FLOOR_MM = 0.10
PROXIMAL_FLOOR_MAX_BELOW_OCCLUSAL_FLOOR_MM = 5.00
STL_SUBDIVIDE_MAX_EDGE_MM = 0.6
STL_SUBDIVIDE_MAX_ITERS = 6
PROXIMAL_FINAL_CONNECT_DIST_MM = 0.40
PROXIMAL_FLOOR_MIN_BOUNDARY_DIST_MM = 0.0
PROXIMAL_FLOOR_MAX_DEPTH_SPAN_MM = 2.00
PROXIMAL_GINGIVAL_FLOOR_PERCENTILE = 95.0
PROXIMAL_FLOOR_MIN_NZ = 0.80
PROXIMAL_FLOOR_TARGET_POINTS = 1000
PROXIMAL_FLOOR_FINAL_MIN_POINTS = 200
PROXIMAL_FLOOR_FALLBACK_MIN_NZ = 0.55
PROXIMAL_FALLBACK_EXPAND_FRACS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
PROXIMAL_FLOOR_FALLBACK_MAX_DIST_MM = 3.0
PROXIMAL_FLOOR_FALLBACK_CONE_COS = 0.15
PROXIMAL_FLOOR_FALLBACK_DEPTH_MARGIN_MM = 0.40
PROXIMAL_FLOOR_FALLBACK_SAMPLE_DIVS = 5
OCCLUSAL_FLOOR_INSET_FRAC = 0.05
OCCLUSAL_FLOOR_MIN_NZ = 0.75
OCCLUSAL_FLOOR_MIN_POINTS = 50
OCCLUSAL_FLOOR_DEPTH_MARGIN_MM = 0.25
DEPTH_HTML_MIN_MM = 0.0
DEPTH_HTML_MAX_MM = 4.0
FOOTPRINT_RASTER_RES_MM = 0.05
FOOTPRINT_RASTER_MARGIN_MM = 2.0
FOOTPRINT_PAD_FRAC = 0.03
COMPONENT_JOIN_GRID_CELLS = 4
REFIT_DEPTH_MAD = 1.5
CAVITY_DEPTH_MAD = 2.0
CAVITY_DEPTH_FALLBACK_MADS = tuple(round(2.1 + 0.05 * i, 2) for i in range(29))
CANDIDATE_GRID_N = 120
BOUNDARY_ELONGATION_RATIO = 3.0
BOUNDARY_AXIS_ALIGNMENT = 0.55
MAX_COMPONENT_Z_SPAN_FRAC = 0.45
MIN_COMPONENT_EDGE_DIST_FRAC = 0.10
BBOX_FACE_INWARD_STEP_MM = 0.25
BBOX_MAX_FACE_MOVES = 400
BBOX_BOTTOM_BOUNDARY_MARGIN_MM = 1.0
MIN_CAVITY_BOX_X_SPAN_MM = 1.5
MIN_CAVITY_BOX_Y_SPAN_MM = 1.5
BBOX_COLLAPSE_AREA_RATIO = 0.35
VERSION = "1.2"
DEPTH_OUTPUT_OFFSET_MM = 0.75
MIN_MEASUREMENT_TRIANGLES = 12
MIN_MEASUREMENT_AREA_MM2 = 0.05


def weighted_quantile(values, weights, q):
    """Interpolated quantile of a finite, positive-weight empirical surface."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    order = np.argsort(values[valid], kind="stable")
    v, w = values[valid][order], weights[valid][order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cdf, v))


def fit_floor_plane(points, area_weights):
    """Huber IRLS orthogonal plane; retain ALL residuals in reported metrics.

    Robust weights stabilize plane orientation only. They never replace the
    physical area weights when computing roughness, so defects are not erased.
    """
    pts, weights = np.asarray(points, float), np.asarray(area_weights, float)
    if len(pts) < 3 or not np.isfinite(pts).all() or not np.isfinite(weights).all():
        raise ValueError("A floor plane needs finite points and area weights")
    if np.any(weights <= 0):
        raise ValueError("Floor weights must be positive")
    robust = np.ones(len(pts))
    for _ in range(30):
        w = weights * robust
        center = np.average(pts, axis=0, weights=w)
        centered = pts - center
        eigenvalues, eigenvectors = np.linalg.eigh((centered * w[:, None]).T @ centered / w.sum())
        if eigenvalues[1] <= max(eigenvalues[2] * 1e-10, 1e-14):
            raise ValueError("Floor points do not span a two-dimensional surface")
        normal = eigenvectors[:, 0]
        if normal[2] < 0:
            normal = -normal
        residuals = centered @ normal
        median = weighted_quantile(residuals, weights, 0.5)
        scale = 1.4826 * weighted_quantile(np.abs(residuals - median), weights, 0.5)
        # A floor that is exactly planar apart from a defect still needs robust
        # fitting; a small numerical floor avoids a zero-MAD division.
        threshold = 1.345 * max(scale, 1e-8)
        new_robust = np.minimum(1.0, threshold / np.maximum(np.abs(residuals - median), 1e-15))
        if np.max(np.abs(new_robust - robust)) < 1e-6:
            break
        robust = new_robust
    return center, normal, residuals


def select_measurement_triangles(vectors, detected_points, min_nz):
    """Keep complete, nondegenerate floor faces inside the detected support.

    Membership is tested geometrically to handle repeated STL vertices. Steep
    faces are excluded even if the detector used a relaxed floor-normal cutoff.
    No new surface is interpolated across gaps in the selected floor.
    """
    vectors = np.asarray(vectors, float)
    if detected_points is None or len(detected_points) < 3:
        return np.empty((0, 3, 3))
    tree = cKDTree(np.unique(np.asarray(detected_points, float), axis=0))
    distance, _ = tree.query(vectors.reshape(-1, 3), distance_upper_bound=1e-5)
    supported = (distance.reshape(-1, 3) <= 1e-5).all(axis=1)
    cross = np.cross(vectors[:, 1] - vectors[:, 0], vectors[:, 2] - vectors[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    nz = np.abs(cross[:, 2]) / np.maximum(double_area, 1e-20)
    triangles = vectors[supported & (nz >= min_nz) & (double_area > 1e-12)]
    # Exact duplicate facets (including reversed winding) must not add area.
    if len(triangles):
        canonical = np.sort(np.ascontiguousarray(triangles).view(
            np.dtype((np.void, 3 * triangles.dtype.itemsize))).reshape(-1, 3), axis=1)
        _, first = np.unique(canonical, axis=0, return_index=True)
        triangles = triangles[np.sort(first)]
    return triangles


def measure_floor(triangles, reference_coeffs, depth_offset_mm=0.0):
    """Area-weighted floor metrics in mm, using a common Z-up reference.

    Raw depth is z_reference(x,y) - z_floor. With a positive reporting offset,
    subtract it and clamp to zero, as in the original occlusal pipeline. A
    separate field preserves the tangent-normal approximation with the same
    reporting correction. Geometry and plane residuals never use the offset.
    Three-point triangle quadrature integrates squared plane residuals exactly
    for each planar STL face. Quantiles are quadrature approximations.
    """
    if not np.isfinite(depth_offset_mm) or depth_offset_mm < 0:
        raise ValueError("Depth offset must be finite and nonnegative")
    triangles = np.asarray(triangles, float)
    area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = np.isfinite(triangles).all(axis=(1, 2)) & (area > 1e-12)
    triangles, area = triangles[valid], area[valid]
    if len(triangles) < MIN_MEASUREMENT_TRIANGLES or area.sum() < MIN_MEASUREMENT_AREA_MM2:
        return None
    bary = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6],
                     [1 / 6, 1 / 6, 2 / 3]])
    points = np.einsum("qv,tvc->tqc", bary, triangles).reshape(-1, 3)
    weights = np.repeat(area / 3, 3)
    depths = quadric_z(points[:, 0], points[:, 1], reference_coeffs) - points[:, 2]
    vertices = triangles.reshape(-1, 3)
    vertex_depths = quadric_z(vertices[:, 0], vertices[:, 1], reference_coeffs) - vertices[:, 2]
    raw_depths = depths.copy()
    normal_depths = compute_depth(points, reference_coeffs)
    if depth_offset_mm > 0:
        depths = np.maximum(depths - depth_offset_mm, 0.0)
        vertex_depths = np.maximum(vertex_depths - depth_offset_mm, 0.0)
        normal_depths = np.maximum(normal_depths - depth_offset_mm, 0.0)
    center, normal, residuals = fit_floor_plane(points, weights)
    absolute = np.abs(residuals)
    axis = principal_axis(points[:, :2])
    labels = region_labels(points, axis)
    regions = []
    for i in range(4):
        mask = labels == i
        regional = {"Region": i + 1, "Area_mm2": float(weights[mask].sum()),
                    "SampleCount": int(mask.sum())}
        # Empty regions have missing measurements, never a fabricated zero.
        regional["DepthMedian_mm"] = weighted_quantile(depths[mask], weights[mask], 0.5)
        regional["DepthP95_mm"] = weighted_quantile(depths[mask], weights[mask], 0.95)
        regions.append(regional)
    medians = [r["DepthMedian_mm"] for r in regions]
    regional_sd = float(np.std(medians)) if np.isfinite(medians).all() else float("nan")
    metrics = {
        "FloorArea_mm2": float(area.sum()), "FloorTriangleCount": int(len(triangles)),
        "DepthOffset_mm": float(depth_offset_mm),
        "DepthMean_mm": float(np.average(depths, weights=weights)),
        "DepthMedian_mm": weighted_quantile(depths, weights, 0.5),
        "DepthP95_mm": weighted_quantile(depths, weights, 0.95),
        "DepthMax_mm": float(vertex_depths.max()),
        "DepthMin_mm": float(vertex_depths.min()),
        "ApproxNormalDepthMedian_mm": weighted_quantile(normal_depths, weights, 0.5),
        "FloorResidualRMS_mm": float(np.sqrt(np.average(residuals ** 2, weights=weights))),
        "FloorResidualMeanAbs_mm": float(np.average(absolute, weights=weights)),
        "FloorResidualAbsP95_mm": weighted_quantile(absolute, weights, 0.95),
        "FloorTilt_deg": float(np.degrees(np.arccos(np.clip(normal[2], -1, 1)))),
        "RegionalDepthMedianSD_mm": regional_sd,
        "NegativeDepthAreaFraction": float(weights[raw_depths < 0].sum() / weights.sum()),
        "DepthClippedAreaFraction": (float(weights[raw_depths < depth_offset_mm].sum() / weights.sum())
                                     if depth_offset_mm > 0 else 0.0),
    }
    return {"metrics": metrics, "points": points, "weights": weights,
            "depths": depths, "raw_depths": raw_depths, "depth_offset_mm": float(depth_offset_mm),
            "residuals": residuals, "plane_center": center,
            "plane_normal": normal, "labels": labels, "regions": regions,
            "triangles": triangles}


MEASUREMENT_DEFINITIONS = """# Final cavity depth and floor measurements

All STL coordinates are assumed to be millimeters and the tooth long axis is
assumed to be approximately Z. Use --scale-to-mm for an explicit uniform unit
conversion. Occlusal depths retain the original reporting correction:
reported_depth = max(raw_depth - 0.75 mm, 0). Proximal depths have no offset.
The original source script and STL inputs are not modified.

## Depth

Occlusal and proximal depths are vertical (Z-axis) separations from the SAME
robust quadratic occlusal reference surface: z_reference(x,y) - z_floor.
This estimated surface is not a manually identified central fossa or an
unprepared scan; the values require anatomical validation before clinical
thresholds are applied. Proximal depth is gingival-floor depth from that surface,
not axial-wall thickness. The previous normal-distance approximation is
exported separately as ApproxNormalDepthMedian_mm with the same occlusal-only
offset and zero clamp. The reporting correction is applied to individual depths
before computing every depth statistic, regional statistic and plotted value.
DepthOffset_mm records the subtraction for each floor. Raw depths remain in the
NPZ files. NegativeDepthAreaFraction diagnoses raw reference depths;
DepthClippedAreaFraction reports the surface fraction clamped by the correction.
Neither floor selection nor geometric smoothness uses the reporting offset.

DepthMedian_mm is the primary typical-floor measurement. DepthP95_mm is a
separate deep-floor statistic. Both use physical triangle-area weighting and
three-point quadrature. DepthMax_mm and DepthMin_mm are the actual extrema at
selected floor mesh vertices, not averages of regional extrema. P95 describes
the whole selected floor, not a mean of four regional P95 values. Inspect the
maximum and the depth map as well as the median for localized over-preparation.

## Smoothness and tilt

FloorResidualRMS_mm is the surface-area-weighted RMS orthogonal distance to a
robustly fitted floor plane; lower values indicate a more even floor. All
selected residuals contribute, including defects. Huber weights affect plane
fitting only. A perfectly flat tilted floor has zero residual roughness and a
nonzero FloorTilt_deg. Mean absolute and P95 absolute residuals are also given.
This measures floor unevenness at the STL resolution, including waviness; it
is not calibrated microscopic surface roughness and contains scanner noise.
RegionalDepthMedianSD_mm remains a separate depth-uniformity diagnostic and
must not be interpreted as surface smoothness. No arbitrary 0-100 smoothness
score or clinical pass/fail threshold is assigned.

## Surface selection and weighting

The inherited cavity detector proposes the floor regions. Measurements require
complete, nondegenerate triangles within those regions and strict face-normal
cutoffs (|nz| >= 0.75 occlusal, >= 0.80 proximal). Walls and partially supported
boundary triangles are excluded. Exact duplicate facets are removed. Each
triangle contributes in proportion to physical area, not its number of STL
vertices. Subdivision cannot add surface detail missing from the scan.
No interpolation bridges floor gaps. At least 12 triangles and 0.05 mm^2 are
required; these are computational support thresholds, not clinical criteria.
Undetected or insufficient floors remain missing. Missing regional statistics
are never filled with zero. Detector fallback/box warnings are retained.

## Files and downstream scoring

summary.csv has one row for EVERY input, including missing regions and errors.
regions.csv reports four regions per measurable floor; details.json records
the fitted reference and floor planes. Each *_measurements.npz stores selected
floor triangles, quadrature points, area weights, depths and plane residuals.
Depth HTML, residual HTML, PNG review panels and index.html allow inspection.
run_manifest.json records parameters, software versions and input SHA-256s.

Use the new explicit depth and FloorResidualRMS_mm columns when revising CQS.
Old tolerances for SD of regional maxima do not transfer to plane residual RMS.
The old CQS CSV names are intentionally not aliased to these different metrics.
Run geometric tests with:
python3 -m unittest discover -s tests -p 'test_final_cav_con_comb_smooth.py' -v
"""


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, data):
    path.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n")


def write_table(path, rows, first_columns=()):
    fields = list(first_columns)
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_measurement_views(vectors, results, out_dir, stem):
    """Offline HTML and static panels show the exact surfaces being measured."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    go, pio = _try_plotly()
    fig = go.Figure()
    _add_mesh_background(fig, go, vectors)
    limit = max((float(np.max(np.abs(r["residuals"]))) for r in results.values()
                 if r is not None), default=0.01)
    limit = max(limit, 0.001)
    png, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    background = vectors.reshape(-1, 3)[::max(1, vectors.size // 90000)]
    depth_max = max((float(r["depths"].max()) for r in results.values() if r is not None), default=1)
    for i, (name, result) in enumerate(results.items()):
        for ax in axes[i]:
            ax.scatter(background[:, 0], background[:, 1], c="0.85", s=0.5)
            ax.set_aspect("equal")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
        if result is None:
            for ax in axes[i]:
                ax.set_title(name + ": floor unavailable")
            continue
        points = result["points"]
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            name=name + " floor residuals", marker=dict(size=3, color=result["residuals"],
            colorscale="RdBu", cmin=-limit, cmax=limit, showscale=i == 0,
            colorbar=dict(title="Plane residual (mm)", x=0.80)),
            customdata=np.column_stack([result["depths"], result["residuals"], result["weights"]]),
            hovertemplate="Depth=%{customdata[0]:.4f} mm<br>Residual=%{customdata[1]:.4f} mm"
                          "<br>Area weight=%{customdata[2]:.6f} mm²<extra></extra>"))
        for j, (values, cmap, vmin, vmax) in enumerate([
            (result["depths"], "viridis", 0, depth_max),
            (result["residuals"], "coolwarm", -limit, limit)]):
            mappable = axes[i, j].scatter(points[:, 0], points[:, 1], c=values,
                                         s=3, cmap=cmap, vmin=vmin, vmax=vmax)
            png.colorbar(mappable, ax=axes[i, j], label="mm", shrink=0.8)
        metrics = result["metrics"]
        axes[i, 0].set_title(f"{name} depth: median {metrics['DepthMedian_mm']:.3f} mm\n"
                             f"P95 {metrics['DepthP95_mm']:.3f}; max {metrics['DepthMax_mm']:.3f}")
        axes[i, 1].set_title(f"{name} floor residual RMS {metrics['FloorResidualRMS_mm']:.4f} mm\n"
                             f"Tilt {metrics['FloorTilt_deg']:.2f}°")
    png.suptitle(stem + " — occlusal depth −0.75 mm (clamped at 0); proximal unchanged")
    png.savefig(out_dir / f"{stem}_review.png", dpi=130)
    plt.close(png)
    _apply_plotly_sidebar_layout(fig, title=stem + " floor residuals (mm)", reserve_colorbar=True)
    pio.write_html(fig, file=str(out_dir / f"{stem}_smoothness.html"),
                   include_plotlyjs="directory", config=_plotly_html_config(), auto_open=False)


def process_final(stl_path, out_dir, scale_to_mm=1.0, plots=True):
    """Keep the original detection workflow; measure selected floor faces anew."""
    obj = stlmesh.Mesh.from_file(str(stl_path))
    original_vectors = np.asarray(obj.vectors, dtype=float) * scale_to_mm
    if not len(original_vectors) or not np.isfinite(original_vectors).all():
        raise ValueError("STL is empty or contains non-finite coordinates")
    vectors = subdivide(original_vectors)
    verts = vectors.reshape(-1, 3)
    nz_abs = face_nz_per_vertex(vectors)
    z = verts[:, 2]
    occl = verts[z >= z.max() - TOP_Z_FRAC * np.ptp(z)]
    row = {"Tooth": stl_path.stem, "Status": "unavailable", "OcclusalStatus": "not_detected",
           "ProximalStatus": "not_detected", "Warnings": "", "Error": ""}
    detail = {"input": str(stl_path.resolve()), "original_triangles": len(original_vectors),
              "subdivided_triangles": len(vectors)}
    if len(occl) < MIN_POINTS:
        row["Warnings"] = "too_few_occlusal_points"
        return row, [], detail
    depth = compute_depth(occl, fit_robust_quadric(occl))
    candidates = []
    for k in _detection_mads():
        candidate = build_occlusal_candidate(verts, occl, vectors, depth, k)
        if candidate is not None:
            candidates.append(candidate)
            if candidate["usable"]:
                break
    if not candidates:
        row["Warnings"] = "occlusal_region_not_detected"
        return row, [], detail
    selected = best_candidate(candidates)
    if not selected["inside"].any():
        row["Warnings"] = "empty_occlusal_box"
        return row, [], detail
    aligned = selected["aligned"]
    common_vectors = aligned["stl_vectors"]
    # One reference is used for BOTH floors in this common coordinate frame.
    reference = fit_robust_quadric(aligned["occl"])
    occlusal_points, _, _ = recover_occlusal_floor(selected, nz_abs)
    warnings = []
    if selected["warning"]:
        warnings.append("occlusal_detection_box_requires_review")
    if occlusal_points is None:
        occlusal_points = aligned["cavity"][selected["inside"]]
        warnings.append("occlusal_floor_uses_detected_component_fallback")
    proximal = build_proximal(selected, nz_abs)
    if proximal is not None and proximal["expand_frac"] > 0:
        warnings.append("proximal_detection_uses_expanded_search")
    inputs = [("Occlusal", occlusal_points, OCCLUSAL_FLOOR_MIN_NZ),
              ("Proximal", None if proximal is None else proximal["cavity_in_occl_frame"],
               PROXIMAL_FLOOR_MIN_NZ)]
    results, regions, arrays = {}, [], {}
    detail.update(reference_coefficients=reference, reference_frame="occlusal_aligned_xy_original_z",
                  alignment_angle_rad=aligned["theta"], alignment_origin_xy=aligned["origin"],
                  detection_depth_mad=selected["depth_mad"],
                  proximal_expansion_fraction=None if proximal is None else proximal["expand_frac"])
    for name, points, min_nz in inputs:
        triangles = select_measurement_triangles(common_vectors, points, min_nz)
        result = measure_floor(triangles, reference,
                               depth_offset_mm=DEPTH_OUTPUT_OFFSET_MM if name == "Occlusal" else 0.0)
        results[name] = result
        detail[name] = {"detected_vertex_occurrences": 0 if points is None else len(points),
                        "strict_floor_triangles": len(triangles), "minimum_abs_normal_z": min_nz}
        if result is None:
            row[name + "Status"] = "not_detected" if points is None else "insufficient_floor_support"
            warnings.append(name.lower() + "_floor_unavailable")
            continue
        row[name + "Status"] = "measured"
        row.update({name + key: value for key, value in result["metrics"].items()})
        detail[name].update(plane_center=result["plane_center"], plane_normal=result["plane_normal"],
                            metrics=result["metrics"])
        if result["metrics"]["NegativeDepthAreaFraction"] > 0:
            warnings.append(name.lower() + "_negative_depth_reference_requires_review")
        if not np.isfinite(result["metrics"]["RegionalDepthMedianSD_mm"]):
            warnings.append(name.lower() + "_empty_longitudinal_region")
        for regional in result["regions"]:
            regions.append({"Tooth": stl_path.stem, "Cavity": name, **regional})
        for key in ["points", "weights", "depths", "raw_depths", "depth_offset_mm",
                    "residuals", "plane_center", "plane_normal", "triangles"]:
            arrays[name.lower() + "_" + key] = result[key]
    measured = sum(r is not None for r in results.values())
    row["Status"] = "complete" if measured == 2 else "partial" if measured else "unavailable"
    row["Warnings"] = ";".join(warnings)
    detail["warnings"] = warnings
    arrays["reference_coefficients"] = reference
    np.savez_compressed(out_dir / f"{stl_path.stem}_measurements.npz", **arrays)
    save_bbox_csv(selected["bbox"], out_dir / f"{stl_path.stem}_bbox.csv")
    save_bbox_csv(selected["initial_bbox"], out_dir / f"{stl_path.stem}_initial_bbox.csv")
    if plots:
        o, p = results["Occlusal"], results["Proximal"]
        op, od = (None, None) if o is None else (o["points"], o["depths"])
        pp, pd = (None, None) if p is None else (p["points"], p["depths"])
        export_depth_html(common_vectors, op, od, pp, pd, selected["bbox"],
                          out_dir / f"{stl_path.stem}_depth.html",
                          stl_path.stem + " depth (mm; occlusal −0.75, clamped at 0; proximal unchanged)",
                          occl_metrics=None if o is None else o["metrics"],
                          prox_metrics=None if p is None else p["metrics"])
        export_combined_html(common_vectors, op, pp, out_dir / f"{stl_path.stem}_combined.html",
                             occl_depth=od, prox_depth=pd,
                             occl_metrics=None if o is None else o["metrics"],
                             prox_metrics=None if p is None else p["metrics"])
        export_measurement_views(common_vectors, results, out_dir, stl_path.stem)
    return row, regions, detail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stl_dir", "--stl-dir", type=Path, default=Path("STLFILES"))
    ap.add_argument("--out_dir", "--out-dir", type=Path, default=Path("final_cav_con_comb_smooth_results"))
    ap.add_argument("--scale-to-mm", type=float, default=1.0,
                    help="Multiply STL coordinates by this factor before processing")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    if not np.isfinite(args.scale_to_mm) or args.scale_to_mm <= 0:
        ap.error("--scale-to-mm must be finite and positive")
    if not args.stl_dir.is_dir():
        ap.error("Input STL directory does not exist")
    files = sorted((f for f in args.stl_dir.iterdir() if f.is_file() and f.suffix.lower() == ".stl"),
                   key=lambda p: p.name.lower())
    if not files:
        ap.error("No input STL files found")
    if len({f.stem.lower() for f in files}) != len(files):
        ap.error("Input STL stems must be unique to prevent output collisions")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        ap.error("Output directory must be new or empty; choose --out_dir to preserve previous results")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "measurement_definitions.md").write_text(MEASUREMENT_DEFINITIONS)
    import scipy
    manifest = {"script_version": VERSION, "script_sha256": sha256(Path(__file__)),
                "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                "scale_to_mm": args.scale_to_mm, "occlusal_depth_offset_mm": DEPTH_OUTPUT_OFFSET_MM,
                "proximal_depth_offset_mm": 0.0, "occlusal_depth_clamp_min_mm": 0.0,
                "depth_direction": "vertical_z", "reference": "shared_robust_occlusal_quadric",
                "parameters": {k: v for k, v in globals().items() if k.isupper()
                               and isinstance(v, (int, float, tuple))},
                "inputs": [{"path": str(f.resolve()), "sha256": sha256(f)} for f in files]}
    write_json(args.out_dir / "run_manifest.json", manifest)
    rows, regions, details = [], [], {}
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
        try:
            row, regional, detail = process_final(path, args.out_dir, args.scale_to_mm, not args.no_plots)
            regions.extend(regional)
            details[path.stem] = detail
        except Exception as exc:
            traceback.print_exc()
            row = {"Tooth": path.stem, "Status": "error", "OcclusalStatus": "error",
                   "ProximalStatus": "error", "Warnings": "", "Error": str(exc)}
            details[path.stem] = {"error": str(exc), "traceback": traceback.format_exc()}
        rows.append(row)
        print(f"  {row['Status']}: O={row['OcclusalStatus']}, P={row['ProximalStatus']}", flush=True)
        write_table(args.out_dir / "summary.csv", rows,
                    ["Tooth", "Status", "OcclusalStatus", "ProximalStatus", "Warnings", "Error"])
        write_table(args.out_dir / "regions.csv", regions, ["Tooth", "Cavity", "Region"])
        write_json(args.out_dir / "details.json", details)
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                    counts={status: sum(r["Status"] == status for r in rows)
                            for status in ["complete", "partial", "unavailable", "error"]})
    write_json(args.out_dir / "run_manifest.json", manifest)
    from html import escape
    links = []
    for row in rows:
        stem = row["Tooth"]
        artifacts = [f'<a href="{escape(stem + suffix, quote=True)}">{label}</a>'
                     for suffix, label in [("_review.png", "PNG"), ("_depth.html", "Depth"),
                                           ("_smoothness.html", "Floor residuals"),
                                           ("_combined.html", "Combined")]
                     if (args.out_dir / (stem + suffix)).exists()]
        links.append(f"<tr><td>{escape(stem)}</td><td>{escape(row['Status'])}</td>"
                     f"<td>{' | '.join(artifacts)}</td><td>{escape(row['Warnings'])}</td></tr>")
    (args.out_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Final cavity measurements</title>'
        '<h1>Final cavity measurements</h1><p>Occlusal depth: max(raw depth − 0.75 mm, 0). '
        'Proximal depth: unchanged. '
        'Floor residual RMS: mm, lower is smoother. Review flagged detection regions.</p>'
        '<p><a href="summary.csv">Summary CSV</a> | <a href="regions.csv">Regions CSV</a> | '
        '<a href="measurement_definitions.md">Definitions</a></p>'
        '<table border="1" cellpadding="6"><tr><th>Tooth</th><th>Status</th><th>Views</th>'
        '<th>Warnings</th></tr>' + ''.join(links) + '</table>')
    print(f"Saved {len(rows)} input records to {args.out_dir}; {manifest['counts']}")
    return 1 if manifest["counts"]["error"] else 0


# ---------------- STL DENSIFICATION ----------------
def subdivide(vectors, max_edge=STL_SUBDIVIDE_MAX_EDGE_MM,
              max_iters=STL_SUBDIVIDE_MAX_ITERS):
    """1->4 midpoint subdivision until every edge <= max_edge.

    Sub-triangles are coplanar with their parent so |nz| is preserved.
    """
    for _ in range(max_iters):
        v0, v1, v2 = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        max_e = np.maximum.reduce([
            np.linalg.norm(v1 - v0, axis=1),
            np.linalg.norm(v2 - v1, axis=1),
            np.linalg.norm(v0 - v2, axis=1),
        ])
        m = max_e > max_edge
        if not m.any():
            return vectors
        keep = vectors[~m]
        s0, s1, s2 = vectors[m, 0], vectors[m, 1], vectors[m, 2]
        m01, m12, m20 = 0.5 * (s0 + s1), 0.5 * (s1 + s2), 0.5 * (s2 + s0)
        vectors = np.concatenate([
            keep,
            np.stack([s0, m01, m20], axis=1),
            np.stack([m01, s1, m12], axis=1),
            np.stack([m20, m12, s2], axis=1),
            np.stack([m01, m12, m20], axis=1),
        ], axis=0)
    return vectors


def face_nz_per_vertex(stl_vectors):
    """|nz| of unit face normal, replicated across each triangle's 3 vertices."""
    v0, v1, v2 = stl_vectors[:, 0], stl_vectors[:, 1], stl_vectors[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    nz = np.abs(n[:, 2]) / (np.linalg.norm(n, axis=1) + 1e-12)
    return np.repeat(nz, 3)


# ---------------- QUADRIC SURFACE ----------------
def fit_quadric(x, y, z):
    A = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    return np.linalg.lstsq(A, z, rcond=None)[0]


def quadric_z(x, y, c):
    return c[0] * x * x + c[1] * y * y + c[2] * x * y + c[3] * x + c[4] * y + c[5]


def compute_depth(p, c):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    z_ref = quadric_z(x, y, c)
    dzdx = 2 * c[0] * x + c[2] * y + c[3]
    dzdy = 2 * c[1] * y + c[2] * x + c[4]
    return (z_ref - z) / np.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)


def fit_robust_quadric(pts):
    c = fit_quadric(pts[:, 0], pts[:, 1], pts[:, 2])
    for _ in range(2):
        d = compute_depth(pts, c)
        med = np.median(d)
        mad = np.median(np.abs(d - med)) + 1e-9
        m = d <= med + REFIT_DEPTH_MAD * mad
        if m.sum() < MIN_POINTS:
            break
        c = fit_quadric(pts[m, 0], pts[m, 1], pts[m, 2])
    return c


def robust_threshold(d, k=CAVITY_DEPTH_MAD):
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-9
    return med + k * mad


# ---------------- ROTATION HELPERS ----------------
def rotate_xy(points, theta, origin):
    """Rotate the XY components of `points` (shape (...,3)) around `origin`."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    out = points.copy()
    out[..., :2] = (out[..., :2] - origin) @ R.T + origin
    return out


def principal_axis(xy):
    pts = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    v = vt[0]
    return -v if v[1] < 0 else v


def angle_to_y(axis):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    return np.pi / 2.0 - np.arctan2(a[1], a[0])


def region_labels(points, axis):
    proj = points[:, :2] @ axis
    t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    return np.clip((t * 4).astype(int), 0, 3)


# ---------------- OCCLUSAL CAVITY COMPONENT ----------------
def select_cavity_component(occl, depth, mask):
    pts = occl[mask]
    d = depth[mask]
    if len(pts) < MIN_POINTS:
        return None, None

    xy = pts[:, :2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    n_grid = CANDIDATE_GRID_N
    dx = xmax - xmin + 1e-9
    dy = ymax - ymin + 1e-9
    gx = np.clip(((xy[:, 0] - xmin) / dx * n_grid).astype(int), 0, n_grid - 1)
    gy = np.clip(((xy[:, 1] - ymin) / dy * n_grid).astype(int), 0, n_grid - 1)

    grid = np.zeros((n_grid, n_grid), dtype=bool)
    grid[gx, gy] = True
    labeled, num = label(grid)
    if num == 0:
        return None, None
    labels = labeled[gx, gy]

    all_xy = occl[:, :2]
    tooth_min, tooth_max = all_xy.min(axis=0), all_xy.max(axis=0)
    tooth_span = tooth_max - tooth_min + 1e-9
    slab_z_span = occl[:, 2].max() - occl[:, 2].min() + 1e-9

    best_score, best_label = -np.inf, None
    for lab in range(1, num + 1):
        keep = labels == lab
        n = int(keep.sum())
        if n < MIN_POINTS:
            continue
        comp_xy = xy[keep]
        comp_pts = pts[keep]
        comp_d = d[keep]
        if (comp_pts[:, 2].max() - comp_pts[:, 2].min()) / slab_z_span > MAX_COMPONENT_Z_SPAN_FRAC:
            continue
        center = comp_xy.mean(axis=0)
        edge_dist = np.minimum(center - tooth_min, tooth_max - center)
        min_edge_frac = float((edge_dist / tooth_span).min())
        if min_edge_frac < MIN_COMPONENT_EDGE_DIST_FRAC:
            continue
        centered = comp_xy - center
        _, svals, vt = np.linalg.svd(centered, full_matrices=False)
        compactness = (
            min(1.0, (svals[1] + 1e-9) / (svals[0] + 1e-9) * 3.0)
            if len(svals) > 1 else 0.5
        )
        if len(svals) > 1:
            elong = (svals[0] + 1e-9) / (svals[1] + 1e-9)
            bnd = np.array([
                [tooth_min[0] - center[0], 0.0],
                [tooth_max[0] - center[0], 0.0],
                [0.0, tooth_min[1] - center[1]],
                [0.0, tooth_max[1] - center[1]],
            ])
            nb = bnd[np.argmin(np.linalg.norm(bnd, axis=1))]
            nb /= np.linalg.norm(nb) + 1e-9
            la = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
            if elong >= BOUNDARY_ELONGATION_RATIO and abs(float(la @ nb)) >= BOUNDARY_AXIS_ALIGNMENT:
                continue
        edge_score = float(np.clip(min_edge_frac, 0.05, 1.0))
        mean_d = float(comp_d.mean())
        max_d = float(comp_d.max())
        score = (n ** 0.7) * mean_d * (0.5 * mean_d + max_d) * compactness * edge_score
        if score > best_score:
            best_score, best_label = score, lab

    if best_label is None:
        return None, None
    near = distance_transform_edt(labeled != best_label) <= COMPONENT_JOIN_GRID_CELLS
    nearby = np.unique(labeled[near])
    keep = np.isin(labels, nearby) & (labels != 0)
    return pts[keep], d[keep]


# ---------------- CAVITY BBOX ----------------
def cavity_bbox(cavity, all_verts, coeffs):
    xy_cav = cavity[:, :2]
    sx = all_verts[:, 0].max() - all_verts[:, 0].min()
    sy = all_verts[:, 1].max() - all_verts[:, 1].min()
    pad = np.array([sx, sy]) * FOOTPRINT_PAD_FRAC
    cmin, cmax = xy_cav.min(axis=0), xy_cav.max(axis=0)
    aabb = (
        (all_verts[:, 0] >= cmin[0] - pad[0]) & (all_verts[:, 0] <= cmax[0] + pad[0]) &
        (all_verts[:, 1] >= cmin[1] - pad[1]) & (all_verts[:, 1] <= cmax[1] + pad[1])
    )
    try:
        hull = Delaunay(xy_cav)
        inside = (hull.find_simplex(all_verts[:, :2]) >= 0) | aabb
        hull_xy = xy_cav[hull.convex_hull.flatten()]
    except Exception:
        inside = aabb
        hull_xy = xy_cav
    verts_in = all_verts[inside]
    xy_all = np.vstack([xy_cav, verts_in[:, :2]]) if len(verts_in) else xy_cav
    xmin, ymin = xy_all.min(axis=0)
    xmax, ymax = xy_all.max(axis=0)
    zmax = float(max(
        quadric_z(xy_cav[:, 0], xy_cav[:, 1], coeffs).max(),
        quadric_z(hull_xy[:, 0], hull_xy[:, 1], coeffs).max(),
    ))
    z_min_candidates = [cavity[:, 2].min()]
    if len(verts_in):
        z_min_candidates.append(verts_in[:, 2].min())
    zmin = float(min(z_min_candidates))
    if zmax <= zmin:
        zmax = float(max(cavity[:, 2].max(), zmin + 1e-6))
    return np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])


BBOX_FACES = [
    ("XMIN_YZ", [0, 3, 4, 7]), ("XMAX_YZ", [1, 2, 5, 6]),
    ("YMIN_XZ", [0, 1, 4, 5]), ("YMAX_XZ", [2, 3, 6, 7]),
    ("ZMIN_XY", [0, 1, 2, 3]), ("ZMAX_XY", [4, 5, 6, 7]),
]


def save_bbox_csv(bbox, path):
    np.savetxt(path, bbox, delimiter=",", header="X,Y,Z", comments="")


def save_bbox_faces_csv(bbox, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Face", "CenterX", "CenterY", "CenterZ"])
        for name, idx in BBOX_FACES:
            cx, cy, cz = bbox[idx].mean(axis=0)
            w.writerow([name, f"{cx:.6f}", f"{cy:.6f}", f"{cz:.6f}"])


def bbox_span(bbox):
    return bbox.max(axis=0) - bbox.min(axis=0)


def points_in_aabb(p, b, tol=1e-9):
    return np.all((p >= b.min(axis=0) - tol) & (p <= b.max(axis=0) + tol), axis=1)


def inset_bbox_xy(bbox, frac):
    out = bbox.copy()
    bmin = bbox.min(axis=0)
    bmax = bbox.max(axis=0)
    inset = (bmax[:2] - bmin[:2]) * frac
    center = 0.5 * (bmin[:2] + bmax[:2])
    new_min = np.minimum(bmin[:2] + inset, center)
    new_max = np.maximum(bmax[:2] - inset, center)
    for axis in (0, 1):
        lo = bbox[:, axis] <= center[axis]
        out[lo, axis] = new_min[axis]
        out[~lo, axis] = new_max[axis]
    return out


def expand_bbox_xy(bbox, frac):
    out = bbox.copy()
    bmin = bbox.min(axis=0)
    bmax = bbox.max(axis=0)
    pad = (bmax - bmin) * frac
    center = 0.5 * (bmin + bmax)
    new_min = bmin - pad
    new_max = bmax + pad
    for axis in (0, 1, 2):
        lo = bbox[:, axis] <= center[axis]
        out[lo, axis] = new_min[axis]
        out[~lo, axis] = new_max[axis]
    return out


# ---------------- STL FOOTPRINT (XY) ----------------
def projected_footprint(stl_vectors, with_distance=False):
    tri_xy = stl_vectors[:, :, :2]
    fp = {
        "tri_xy": tri_xy,
        "tri_min": tri_xy.min(axis=1),
        "tri_max": tri_xy.max(axis=1),
        "zmin": float(stl_vectors[:, :, 2].min()),
        "zmax": float(stl_vectors[:, :, 2].max()),
    }
    if with_distance:
        fp["distance_field"] = _footprint_distance_field(tri_xy)
    return fp


def _footprint_distance_field(tri_xy):
    xy = tri_xy.reshape(-1, 2)
    origin = xy.min(axis=0) - FOOTPRINT_RASTER_MARGIN_MM
    xy_max = xy.max(axis=0) + FOOTPRINT_RASTER_MARGIN_MM
    res = FOOTPRINT_RASTER_RES_MM
    shape = np.maximum(np.ceil((xy_max - origin) / res).astype(int) + 1, 2)
    mask = np.zeros(tuple(shape), dtype=bool)

    for tri in tri_xy:
        lo = np.clip(np.floor((tri.min(axis=0) - origin) / res).astype(int) - 1,
                     0, shape - 1)
        hi = np.clip(np.ceil((tri.max(axis=0) - origin) / res).astype(int) + 1,
                     0, shape - 1)
        if np.any(hi < lo):
            continue
        ix = np.arange(lo[0], hi[0] + 1)
        iy = np.arange(lo[1], hi[1] + 1)
        X, Y = np.meshgrid(origin[0] + ix * res, origin[1] + iy * res, indexing="ij")
        a, b, c = tri
        v0 = c - a
        v1 = b - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            ti = np.clip(np.round((tri - origin) / res).astype(int), 0, shape - 1)
            mask[ti[:, 0], ti[:, 1]] = True
            continue
        v2x, v2y = X - a[0], Y - a[1]
        u = (v2x * v1[1] - v1[0] * v2y) / den
        v = (v0[0] * v2y - v2x * v0[1]) / den
        inside = (u >= -1e-9) & (v >= -1e-9) & ((u + v) <= 1.0 + 1e-9)
        mask[np.ix_(ix, iy)] |= inside

    return {
        "origin": origin, "res": res, "shape": shape, "mask": mask,
        "inside_dist": distance_transform_edt(mask) * res,
        "outside_dist": distance_transform_edt(~mask) * res,
    }


def footprint_distance(xy, fp):
    df = fp["distance_field"]
    xy = np.asarray(xy, float)
    flat = xy.reshape(-1, 2)
    idx = np.round((flat - df["origin"]) / df["res"]).astype(int)
    in_grid = (
        (idx[:, 0] >= 0) & (idx[:, 0] < df["shape"][0]) &
        (idx[:, 1] >= 0) & (idx[:, 1] < df["shape"][1])
    )
    inside = np.zeros(len(flat), dtype=bool)
    dist = np.full(len(flat), np.inf, dtype=float)
    vi = idx[in_grid]
    vmask = df["mask"][vi[:, 0], vi[:, 1]]
    dval = np.where(
        vmask,
        df["inside_dist"][vi[:, 0], vi[:, 1]],
        df["outside_dist"][vi[:, 0], vi[:, 1]],
    )
    inside[in_grid] = vmask
    dist[in_grid] = dval
    return inside.reshape(xy.shape[:-1]), dist.reshape(xy.shape[:-1])


# ---------------- BBOX RESIZE TO BACK BOTTOM ONTO STL ----------------
def _point_in_tri(p, tri, eps=1e-9):
    a, b, c = tri
    v0, v1, v2 = c - a, b - a, p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < eps:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return u >= -eps and v >= -eps and u + v <= 1.0 + eps


def _xy_in_footprint(p, fp):
    cands = np.where(
        (p[0] >= fp["tri_min"][:, 0] - 1e-9) & (p[0] <= fp["tri_max"][:, 0] + 1e-9) &
        (p[1] >= fp["tri_min"][:, 1] - 1e-9) & (p[1] <= fp["tri_max"][:, 1] + 1e-9)
    )[0]
    return any(_point_in_tri(p, fp["tri_xy"][i]) for i in cands)


def _margin_offsets(margin):
    if margin <= 0:
        return np.array([[0.0, 0.0]])
    d = margin / np.sqrt(2.0)
    return np.array([
        [0, 0], [margin, 0], [-margin, 0], [0, margin], [0, -margin],
        [d, d], [d, -d], [-d, d], [-d, -d],
    ])


def _bottom_validity(bbox, fp):
    bottom = bbox[[0, 1, 2, 3]]
    if bottom[:, 2].min() < fp["zmin"] or bottom[:, 2].max() > fp["zmax"]:
        return [False, False, False, False]
    offsets = _margin_offsets(BBOX_BOTTOM_BOUNDARY_MARGIN_MM)
    out = []
    for v in bottom:
        ok = True
        for o in offsets:
            if not _xy_in_footprint(v[:2] + o, fp):
                ok = False
                break
        out.append(ok)
    return out


def _bottom_score(bbox, fp):
    bottom = bbox[[0, 1, 2, 3]]
    if bottom[:, 2].min() < fp["zmin"] or bottom[:, 2].max() > fp["zmax"]:
        return 0
    offsets = _margin_offsets(BBOX_BOTTOM_BOUNDARY_MARGIN_MM)
    s = 0
    for v in bottom:
        for o in offsets:
            if _xy_in_footprint(v[:2] + o, fp):
                s += 1
    return s


def _adjacent_faces_for_invalid(validity):
    adj = {0: (0, 2), 1: (1, 2), 2: (1, 3), 3: (0, 3)}
    faces = set()
    for i, ok in enumerate(validity):
        if not ok:
            faces.update(adj[i])
    return sorted(faces)


def _apply_face_moves(bbox, moves):
    out = bbox.copy()
    s = BBOX_FACE_INWARD_STEP_MM
    if moves[0]:
        out[[0, 3, 4, 7], 0] += moves[0] * s
    if moves[1]:
        out[[1, 2, 5, 6], 0] -= moves[1] * s
    if moves[2]:
        out[[0, 1, 4, 5], 1] += moves[2] * s
    if moves[3]:
        out[[2, 3, 6, 7], 1] -= moves[3] * s
    return out


def _moves_within_limits(bbox, moves):
    if any(m > BBOX_MAX_FACE_MOVES for m in moves):
        return False
    s = BBOX_FACE_INWARD_STEP_MM
    sx = bbox[:, 0].max() - bbox[:, 0].min()
    sy = bbox[:, 1].max() - bbox[:, 1].min()
    return (
        sx - (moves[0] + moves[1]) * s >= MIN_CAVITY_BOX_X_SPAN_MM and
        sy - (moves[2] + moves[3]) * s >= MIN_CAVITY_BOX_Y_SPAN_MM
    )


def resize_bbox(bbox, fp):
    """Greedy inward face-move search until the four bottom corners sit on the
    STL footprint with the configured margin. Falls back to the best score."""
    if all(_bottom_validity(bbox, fp)):
        return bbox, (0, 0, 0, 0), False
    moves = (0, 0, 0, 0)
    best_bbox, best_score = bbox, _bottom_score(bbox, fp)
    cur = bbox
    for _ in range(BBOX_MAX_FACE_MOVES * 4):
        cands = _adjacent_faces_for_invalid(_bottom_validity(cur, fp))
        if not cands:
            break
        opts = []
        for i in cands:
            cm = list(moves); cm[i] += 1
            cm = tuple(cm)
            if not _moves_within_limits(bbox, cm):
                continue
            cb = _apply_face_moves(bbox, cm)
            opts.append((_bottom_score(cb, fp), -sum(cm), i, cm, cb))
        if not opts:
            break
        s, _, _, moves, cur = max(opts)
        if s >= best_score:
            best_score, best_bbox = s, cur
        if all(_bottom_validity(cur, fp)):
            return cur, moves, False
    return best_bbox, moves, True


def _bbox_collapsed(initial, resized):
    initial_area = float(max(bbox_span(initial)[0], 0.0) * max(bbox_span(initial)[1], 0.0))
    resized_area = float(max(bbox_span(resized)[0], 0.0) * max(bbox_span(resized)[1], 0.0))
    if initial_area <= 1e-9:
        return False
    span = bbox_span(resized)
    near_min = (
        span[0] <= MIN_CAVITY_BOX_X_SPAN_MM + BBOX_FACE_INWARD_STEP_MM or
        span[1] <= MIN_CAVITY_BOX_Y_SPAN_MM + BBOX_FACE_INWARD_STEP_MM
    )
    severe = (resized_area / initial_area) < BBOX_COLLAPSE_AREA_RATIO
    return near_min and severe


# ---------------- OCCLUSAL CANDIDATE BUILD ----------------
def _detection_mads():
    seen = []
    for v in [CAVITY_DEPTH_MAD, *CAVITY_DEPTH_FALLBACK_MADS]:
        if not any(abs(v - s) < 1e-9 for s in seen):
            seen.append(v)
    return seen


def _align_to_y(cavity, verts, occl, stl_vectors):
    axis = principal_axis(cavity[:, :2])
    origin = cavity[:, :2].mean(axis=0)
    theta = angle_to_y(axis)
    return {
        "theta": theta,
        "origin": origin,
        "rotation_deg": float(np.degrees(theta)),
        "cavity": rotate_xy(cavity, theta, origin),
        "verts": rotate_xy(verts, theta, origin),
        "occl": rotate_xy(occl, theta, origin),
        "stl_vectors": rotate_xy(stl_vectors, theta, origin),
    }


def build_occlusal_candidate(verts, occl, stl_vectors, depth, depth_mad):
    t = robust_threshold(depth, depth_mad)
    cavity, depth_cav = select_cavity_component(occl, depth, depth >= t)
    if cavity is None:
        return None
    aligned = _align_to_y(cavity, verts, occl, stl_vectors)
    coeffs_aligned = fit_robust_quadric(aligned["occl"])
    initial_bbox = cavity_bbox(aligned["cavity"], aligned["verts"], coeffs_aligned)
    fp = projected_footprint(aligned["stl_vectors"])
    bbox, moves, warning = resize_bbox(initial_bbox, fp)
    inside = points_in_aabb(aligned["cavity"], bbox)
    no_inside = not inside.any()
    collapsed = warning and _bbox_collapsed(initial_bbox, bbox)
    if warning and not bbox_meets_min_xy_span(bbox):
        # fallback to initial bbox (mirrors original "using_initial_bbox_after_failed_resize")
        bbox = initial_bbox
        moves = (0, 0, 0, 0)
        inside = points_in_aabb(aligned["cavity"], bbox)
    span = bbox_span(bbox)
    bottom_valid = sum(_bottom_validity(bbox, fp))
    has_warning = warning or no_inside or collapsed
    usable = (
        not has_warning and inside.any() and bottom_valid == 4 and
        span[0] >= MIN_CAVITY_BOX_X_SPAN_MM and span[1] >= MIN_CAVITY_BOX_Y_SPAN_MM
    )
    return {
        "depth_mad": depth_mad,
        "depth_cav": depth_cav,
        "inside": inside,
        "aligned": aligned,
        "initial_bbox": initial_bbox,
        "bbox": bbox,
        "moves": moves,
        "warning": has_warning,
        "bottom_valid": bottom_valid,
        "n_inside": int(inside.sum()),
        "usable": usable,
    }


def bbox_meets_min_xy_span(bbox):
    sp = bbox_span(bbox)
    return sp[0] >= MIN_CAVITY_BOX_X_SPAN_MM and sp[1] >= MIN_CAVITY_BOX_Y_SPAN_MM


def best_candidate(cands):
    usable = [c for c in cands if c["usable"]]
    if usable:
        return usable[0]
    return max(cands, key=lambda c: (
        not c["warning"], c["bottom_valid"], c["n_inside"],
        -abs(c["depth_mad"] - CAVITY_DEPTH_MAD),
    ))


# ---------------- PROXIMAL FLOOR DETECTION ----------------
def largest_connected_component_mask(points, max_link_mm=PROXIMAL_FINAL_CONNECT_DIST_MM):
    """Return a mask for the largest spatially connected point component."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.ones(1, dtype=bool)

    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=np.uint8)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    tree = cKDTree(points[:, :3])
    for a, b in tree.query_pairs(r=max_link_mm):
        union(a, b)

    roots = np.array([find(i) for i in range(n)], dtype=int)
    unique, counts = np.unique(roots, return_counts=True)
    best_root = unique[np.argmax(counts)]
    return roots == best_root


def select_proximal_component(verts, depth, mask):
    src_idx = np.flatnonzero(mask)
    pts = verts[mask]
    d = depth[mask]
    if len(pts) < PROXIMAL_MIN_POINTS:
        return None, None, None

    keep = largest_connected_component_mask(pts)
    if int(keep.sum()) < PROXIMAL_MIN_POINTS:
        return None, None, None
    return pts[keep], d[keep], src_idx[keep]


def _augment_small_proximal_floor(verts, depth, pre_nz_mask, nz_abs,
                                  cavity_idx, min_nz):
    """Add nearby floor candidates away from the low-nz wall side."""
    if cavity_idx is None or len(cavity_idx) >= PROXIMAL_FLOOR_TARGET_POINTS:
        return cavity_idx, 0
    if len(cavity_idx) < PROXIMAL_MIN_POINTS:
        return cavity_idx, 0

    selected = np.zeros(len(verts), dtype=bool)
    selected[cavity_idx] = True
    center = verts[cavity_idx, :2].mean(axis=0)

    low_nz = pre_nz_mask & (nz_abs < min_nz)
    low_idx = np.flatnonzero(low_nz)
    if len(low_idx) < PROXIMAL_MIN_POINTS:
        return cavity_idx, 0

    def away_from_wall(idx):
        rel = verts[idx, :2] - center
        dist = np.linalg.norm(rel, axis=1)
        weights = np.maximum(min_nz - nz_abs[idx], 0.0) / np.maximum(dist, 0.1)
        wall_vec = (rel * weights[:, None]).sum(axis=0)
        wall_norm = np.linalg.norm(wall_vec)
        if wall_norm <= 1e-9:
            return None
        return -wall_vec / wall_norm

    low_rel = verts[low_idx, :2] - center
    low_dist = np.linalg.norm(low_rel, axis=1)
    near_low = low_dist <= PROXIMAL_FLOOR_FALLBACK_MAX_DIST_MM
    away_dirs = []
    if near_low.sum() >= PROXIMAL_MIN_POINTS:
        away_dirs.append(away_from_wall(low_idx[near_low]))
    away_dirs.append(away_from_wall(low_idx))
    away_dirs = [a for a in away_dirs if a is not None]
    if not away_dirs:
        return cavity_idx, 0

    sel_d = depth[cavity_idx]
    sel_min = float(sel_d.min())
    sel_max = float(sel_d.max())
    sel_med = float(np.median(sel_d))
    depth_ok = (
        (depth >= sel_min - PROXIMAL_FLOOR_FALLBACK_DEPTH_MARGIN_MM) &
        (depth <= sel_max + PROXIMAL_FLOOR_FALLBACK_DEPTH_MARGIN_MM)
    )

    fallback = (
        pre_nz_mask & ~selected &
        (nz_abs >= min_nz) &
        depth_ok
    )
    cand_idx = np.flatnonzero(fallback)
    if len(cand_idx) == 0:
        return cavity_idx, 0

    rel = verts[cand_idx, :2] - center
    dist = np.linalg.norm(rel, axis=1)
    in_radius = (dist > 1e-9) & (dist <= PROXIMAL_FLOOR_FALLBACK_MAX_DIST_MM)
    cone_steps = [PROXIMAL_FLOOR_FALLBACK_CONE_COS, 0.0, -0.25, -0.50]
    need = PROXIMAL_FLOOR_TARGET_POINTS - len(cavity_idx)
    chosen = None
    best = None

    for away in away_dirs:
        proj = rel @ away
        cos_to_away = proj / np.maximum(dist, 1e-9)
        for cone in cone_steps:
            stage = in_radius & (cos_to_away >= cone)
            n_stage = int(stage.sum())
            if n_stage == 0:
                continue
            if best is None or n_stage > int(best.sum()):
                best = stage
            if n_stage >= need:
                chosen = stage
                break
            if chosen is not None:
                break
        if chosen is not None:
            break

    if chosen is None:
        chosen = best
    if chosen is None:
        return cavity_idx, 0

    cand_idx = cand_idx[chosen]
    rel = verts[cand_idx, :2] - center
    dist = np.linalg.norm(rel, axis=1)
    depth_delta = np.abs(depth[cand_idx] - sel_med)
    score = dist + 0.5 * depth_delta - 0.5 * nz_abs[cand_idx]
    order = np.argsort(score)
    add_idx = cand_idx[order[:need]]
    if len(add_idx) == 0:
        return cavity_idx, 0
    return np.concatenate([cavity_idx, add_idx]), int(len(add_idx))


def _barycentric_samples(divs=PROXIMAL_FLOOR_FALLBACK_SAMPLE_DIVS):
    bary = []
    for i in range(divs + 1):
        for j in range(divs + 1 - i):
            k = divs - i - j
            if max(i, j, k) == divs:
                continue
            bary.append((i / divs, j / divs, k / divs))
    return np.asarray(bary, dtype=float)


def _sample_small_proximal_floor(stl_vectors, coeffs, fp, initial_bbox,
                                 resized_bbox, crop_z, occl_floor, cavity_pts,
                                 cavity_d, pre_nz_mask, nz_abs, min_nz):
    """Resample strict flat triangles to stabilize small gingival floors."""
    if len(cavity_pts) >= PROXIMAL_FLOOR_TARGET_POINTS:
        return cavity_pts, cavity_d, 0
    if len(cavity_pts) < PROXIMAL_MIN_POINTS:
        return cavity_pts, cavity_d, 0

    verts = stl_vectors.reshape(-1, 3)
    center = cavity_pts[:, :2].mean(axis=0)
    low_nz = pre_nz_mask & (nz_abs < min_nz)
    low_idx = np.flatnonzero(low_nz)
    if len(low_idx) < PROXIMAL_MIN_POINTS:
        return cavity_pts, cavity_d, 0

    def away_from_wall(idx):
        rel = verts[idx, :2] - center
        dist = np.linalg.norm(rel, axis=1)
        weights = np.maximum(min_nz - nz_abs[idx], 0.0) / np.maximum(dist, 0.1)
        wall_vec = (rel * weights[:, None]).sum(axis=0)
        wall_norm = np.linalg.norm(wall_vec)
        if wall_norm <= 1e-9:
            return None
        return -wall_vec / wall_norm

    low_rel = verts[low_idx, :2] - center
    low_dist = np.linalg.norm(low_rel, axis=1)
    near_low = low_dist <= PROXIMAL_FLOOR_FALLBACK_MAX_DIST_MM
    away_dirs = []
    if near_low.sum() >= PROXIMAL_MIN_POINTS:
        away_dirs.append(away_from_wall(low_idx[near_low]))
    away_dirs.append(away_from_wall(low_idx))
    away_dirs = [a for a in away_dirs if a is not None]
    if not away_dirs:
        return cavity_pts, cavity_d, 0

    tri_pre = pre_nz_mask.reshape(-1, 3).any(axis=1)
    face_nz = nz_abs.reshape(-1, 3).min(axis=1)
    tri_mask = tri_pre & (face_nz >= min_nz)
    tris = stl_vectors[tri_mask]
    tri_nz = face_nz[tri_mask]
    if len(tris) == 0:
        return cavity_pts, cavity_d, 0

    bary = _barycentric_samples()
    if len(bary) == 0:
        return cavity_pts, cavity_d, 0
    samples = np.einsum("bj,tjc->tbc", bary, tris).reshape(-1, 3)
    sample_nz = np.repeat(tri_nz, len(bary))
    _, unique_idx = np.unique(np.round(samples, 6), axis=0, return_index=True)
    samples = samples[unique_idx]
    sample_nz = sample_nz[unique_idx]

    sample_depth = compute_depth(samples, coeffs)
    inside_fp, bd = footprint_distance(samples[:, :2], fp)
    zone = points_in_aabb(samples, initial_bbox) & ~points_in_aabb(samples, resized_bbox)
    sample_mask = (
        zone &
        (samples[:, 2] >= crop_z) &
        inside_fp &
        (bd >= PROXIMAL_FLOOR_MIN_BOUNDARY_DIST_MM) &
        (sample_depth >= occl_floor + PROXIMAL_FLOOR_MIN_BELOW_OCCLUSAL_FLOOR_MM) &
        (sample_depth <= occl_floor + PROXIMAL_FLOOR_MAX_BELOW_OCCLUSAL_FLOOR_MM) &
        (sample_nz >= min_nz) &
        (sample_depth >= float(cavity_d.min()) - PROXIMAL_FLOOR_FALLBACK_DEPTH_MARGIN_MM) &
        (sample_depth <= float(cavity_d.max()) + PROXIMAL_FLOOR_FALLBACK_DEPTH_MARGIN_MM)
    )
    cand_pts = samples[sample_mask]
    cand_d = sample_depth[sample_mask]
    cand_nz = sample_nz[sample_mask]
    if len(cand_pts) == 0:
        return cavity_pts, cavity_d, 0

    dist_to_seed = np.linalg.norm(
        cand_pts[:, None, :2] - cavity_pts[None, :, :2], axis=2
    ).min(axis=1)
    new_enough = dist_to_seed > (PROXIMAL_RASTER_RES_MM * 0.25)
    cand_pts = cand_pts[new_enough]
    cand_d = cand_d[new_enough]
    cand_nz = cand_nz[new_enough]
    if len(cand_pts) == 0:
        return cavity_pts, cavity_d, 0

    rel = cand_pts[:, :2] - center
    dist = np.linalg.norm(rel, axis=1)
    in_radius = (dist > 1e-9) & (dist <= PROXIMAL_FLOOR_FALLBACK_MAX_DIST_MM)
    cone_steps = [PROXIMAL_FLOOR_FALLBACK_CONE_COS, 0.0, -0.25, -0.50, -1.0]
    need = PROXIMAL_FLOOR_TARGET_POINTS - len(cavity_pts)
    chosen = None
    best = None

    for away in away_dirs:
        cos_to_away = (rel @ away) / np.maximum(dist, 1e-9)
        for cone in cone_steps:
            stage = in_radius & (cos_to_away >= cone)
            n_stage = int(stage.sum())
            if n_stage == 0:
                continue
            if best is None or n_stage > int(best.sum()):
                best = stage
            if n_stage >= need:
                chosen = stage
                break
        if chosen is not None:
            break

    if chosen is None:
        chosen = best
    if chosen is None:
        return cavity_pts, cavity_d, 0

    cand_pts = cand_pts[chosen]
    cand_d = cand_d[chosen]
    cand_nz = cand_nz[chosen]
    dist = dist[chosen]
    depth_delta = np.abs(cand_d - float(np.median(cavity_d)))
    score = dist + 0.5 * depth_delta - 0.5 * cand_nz
    order = np.argsort(score)
    add_n = min(need, len(order))
    add_pts = cand_pts[order[:add_n]]
    add_d = cand_d[order[:add_n]]
    if len(add_pts) == 0:
        return cavity_pts, cavity_d, 0
    return np.vstack([cavity_pts, add_pts]), np.concatenate([cavity_d, add_d]), int(len(add_pts))


def _build_proximal_once(occl_sel, nz_abs, expand_frac=0.0,
                         min_nz=PROXIMAL_FLOOR_MIN_NZ,
                         max_below_occlusal_mm=PROXIMAL_FLOOR_MAX_BELOW_OCCLUSAL_FLOOR_MM):
    aligned = occl_sel["aligned"]
    verts_f = aligned["verts"]
    occl_f = aligned["occl"]
    stl_f = aligned["stl_vectors"]
    initial_bbox = occl_sel["initial_bbox"]
    resized_bbox = occl_sel["bbox"]

    coeffs = fit_robust_quadric(occl_f)
    all_depth = compute_depth(verts_f, coeffs)
    expand_frac = min(expand_frac, PROXIMAL_OCCLUSAL_OVERLAP_FRAC)
    search_outer_bbox = expand_bbox_xy(initial_bbox, expand_frac)
    search_excluded_bbox = inset_bbox_xy(resized_bbox, expand_frac)
    zone = (
        points_in_aabb(verts_f, search_outer_bbox) &
        ~points_in_aabb(verts_f, search_excluded_bbox)
    )
    if zone.sum() < PROXIMAL_MIN_POINTS:
        return None

    cropped = zone

    fp = projected_footprint(stl_f, with_distance=True)
    inside_fp, bd = footprint_distance(verts_f[:, :2], fp)

    occl_floor = float(occl_sel["depth_cav"][occl_sel["inside"]].max())
    pre_nz_band = (
        cropped & inside_fp &
        (bd >= PROXIMAL_FLOOR_MIN_BOUNDARY_DIST_MM) &
        (all_depth >= occl_floor + PROXIMAL_FLOOR_MIN_BELOW_OCCLUSAL_FLOOR_MM) &
        (all_depth <= occl_floor + max_below_occlusal_mm)
    )
    band = (
        pre_nz_band &
        (nz_abs >= min_nz)
    )
    if band.sum() < PROXIMAL_MIN_POINTS:
        return None

    cavity_pts, cavity_d, cavity_idx = select_proximal_component(
        verts_f, all_depth, band
    )
    if cavity_pts is None:
        return None
    if len(cavity_pts) < PROXIMAL_MIN_POINTS:
        return None
    fallback_added = 0
    cavity_pts = verts_f[cavity_idx]
    cavity_d = all_depth[cavity_idx]

    pax = principal_axis(cavity_pts[:, :2])
    porigin = cavity_pts[:, :2].mean(axis=0)
    ptheta = angle_to_y(pax)
    return {
        "verts_rot": rotate_xy(verts_f, ptheta, porigin),
        "stl_vectors_rot": rotate_xy(stl_f, ptheta, porigin),
        "cavity_rot": rotate_xy(cavity_pts, ptheta, porigin),
        "cavity_in_occl_frame": cavity_pts,
        "depth": cavity_d,
        "outer_bbox": rotate_xy(search_outer_bbox, ptheta, porigin),
        "excluded_bbox": rotate_xy(resized_bbox, ptheta, porigin),
        "canonical_outer_bbox": search_outer_bbox,
        "canonical_excluded_bbox": resized_bbox,
        "proximal_box_region": rotate_xy(verts_f[cropped], ptheta, porigin),
        "floor_candidate_region": rotate_xy(verts_f[band], ptheta, porigin),
        "rotation_deg": float(np.degrees(ptheta)),
        "fallback_added": fallback_added,
        "expand_frac": expand_frac,
    }


def build_proximal(occl_sel, nz_abs, min_nz=PROXIMAL_FLOOR_MIN_NZ):
    best = _build_proximal_once(occl_sel, nz_abs, expand_frac=0.0, min_nz=min_nz)
    if best is not None and len(best["cavity_rot"]) >= PROXIMAL_FLOOR_TARGET_POINTS:
        return best

    for frac in PROXIMAL_FALLBACK_EXPAND_FRACS:
        proximal = _build_proximal_once(
            occl_sel, nz_abs,
            expand_frac=frac,
            min_nz=PROXIMAL_FLOOR_FALLBACK_MIN_NZ,
            max_below_occlusal_mm=np.inf,
        )
        if proximal is None:
            continue
        if best is None or len(proximal["cavity_rot"]) > len(best["cavity_rot"]):
            best = proximal
        if len(proximal["cavity_rot"]) >= PROXIMAL_FLOOR_TARGET_POINTS:
            return proximal

    if best is None or len(best["cavity_rot"]) < PROXIMAL_FLOOR_FINAL_MIN_POINTS:
        return None
    return best


def recover_occlusal_floor(occl_sel, nz_abs):
    """Recover the occlusal floor from the full mesh inside the occlusal box.

    The initial occlusal component is intentionally detected from the upper
    tooth slab, which can capture walls but miss a deep floor.  This pass keeps
    the proximal separation by staying inside a slightly inset occlusal bbox.
    """
    aligned = occl_sel["aligned"]
    verts_f = aligned["verts"]
    occl_f = aligned["occl"]
    bbox = occl_sel["bbox"]
    selected_depth = occl_sel["depth_cav"][occl_sel["inside"]]

    coeffs = fit_robust_quadric(occl_f)
    all_depth = compute_depth(verts_f, coeffs)
    floor_box = inset_bbox_xy(bbox, OCCLUSAL_FLOOR_INSET_FRAC)
    base = (
        points_in_aabb(verts_f, floor_box) &
        (nz_abs >= OCCLUSAL_FLOOR_MIN_NZ)
    )
    if base.sum() < OCCLUSAL_FLOOR_MIN_POINTS:
        return None, None, 0

    depth_cut = max(
        float(np.median(selected_depth) - OCCLUSAL_FLOOR_DEPTH_MARGIN_MM),
        float(selected_depth.min() - OCCLUSAL_FLOOR_DEPTH_MARGIN_MM),
    )
    floor_mask = base & (all_depth >= depth_cut)
    if floor_mask.sum() < OCCLUSAL_FLOOR_MIN_POINTS:
        floor_mask = base & (all_depth >= float(selected_depth.min() - OCCLUSAL_FLOOR_DEPTH_MARGIN_MM))
    if floor_mask.sum() < OCCLUSAL_FLOOR_MIN_POINTS:
        return None, None, 0

    floor_pts = verts_f[floor_mask]
    floor_depth = all_depth[floor_mask]
    keep = largest_connected_component_mask(floor_pts)
    if int(keep.sum()) < OCCLUSAL_FLOOR_MIN_POINTS:
        return None, None, 0
    return floor_pts[keep], floor_depth[keep], int(keep.sum())


# ---------------- HTML EXPORT ----------------
REGION_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _try_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        return go, pio
    except ImportError:
        print("[WARN] plotly not installed")
        return None, None


def _plotly_html_config():
    return {
        "responsive": True,
        "editable": True,
        "edits": {
            "legendPosition": True,
            "legendText": False,
            "colorbarPosition": True,
            "colorbarTitleText": False,
            "titleText": False,
            "axisTitleText": False,
        },
    }


def _add_bbox(fig, go, bbox, color, name, width=2):
    for k, (i, j) in enumerate(BBOX_EDGES):
        fig.add_trace(go.Scatter3d(
            x=[bbox[i, 0], bbox[j, 0]],
            y=[bbox[i, 1], bbox[j, 1]],
            z=[bbox[i, 2], bbox[j, 2]],
            mode="lines", name=name,
            line=dict(color=color, width=width),
            showlegend=k == 0, hoverinfo="skip",
        ))


def _add_mesh_background(fig, go, stl_vectors):
    """Plotly Mesh3d of the STL surface; renders far faster than dense markers."""
    if stl_vectors is None or not len(stl_vectors):
        return
    flat = stl_vectors.reshape(-1, 3)
    n = len(stl_vectors)
    faces = np.arange(3 * n, dtype=int).reshape(n, 3)
    fig.add_trace(go.Mesh3d(
        x=flat[:, 0], y=flat[:, 1], z=flat[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="lightgray", opacity=0.25, flatshading=True,
        name="mesh", showlegend=False, hoverinfo="skip",
    ))


def _add_cavity_text_label(fig, go, points, text):
    if points is None or not len(points):
        return
    center = points.mean(axis=0)
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode="text", text=[text], textposition="middle center",
        textfont=dict(color="black", size=16),
        name=text, showlegend=False, hoverinfo="skip",
    ))


def _apply_plotly_sidebar_layout(fig, title=None, reserve_colorbar=False,
                                 legend_position="right"):
    scene_xmax = 0.78 if reserve_colorbar else 0.84
    if legend_position == "top":
        legend = dict(
            x=0.50, y=0.995, xanchor="center", yanchor="bottom",
            orientation="h", bgcolor="rgba(255,255,255,0.86)",
            bordercolor="rgba(0,0,0,0.18)", borderwidth=1,
            itemsizing="constant",
        )
        margin = dict(l=0, r=80 if reserve_colorbar else 10, t=50 if title else 24, b=0)
    else:
        legend_x = 0.86 if reserve_colorbar else 0.82
        legend = dict(
            x=legend_x, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.86)", bordercolor="rgba(0,0,0,0.18)",
            borderwidth=1, itemsizing="constant",
        )
        margin = dict(l=0, r=260 if reserve_colorbar else 220, t=45 if title else 10, b=0)
    layout = dict(
        scene=dict(domain=dict(x=[0.0, scene_xmax], y=[0.0, 1.0]), aspectmode="data"),
        legend=legend,
        margin=margin,
    )
    if title:
        layout["title"] = dict(text=title)
    fig.update_layout(**layout)


def export_region_html(stl_vectors, cavity, depth, labels, bbox, out_path,
                       excluded_bbox=None, label_prefix="Region"):
    go, pio = _try_plotly()
    if go is None:
        return
    fig = go.Figure()
    _add_mesh_background(fig, go, stl_vectors)
    for i in range(4):
        m = labels == i
        if not m.any():
            continue
        rmax = float(depth[m].max())
        fig.add_trace(go.Scatter3d(
            x=cavity[m, 0], y=cavity[m, 1], z=cavity[m, 2],
            mode="markers",
            name=f"{label_prefix} {i + 1}: max {rmax:.2f} mm",
            marker=dict(size=3, color=REGION_COLORS[i]),
            customdata=depth[m],
            hovertemplate=f"{label_prefix} {i + 1}<br>depth=%{{customdata:.2f}} mm<extra></extra>",
        ))
    _add_bbox(fig, go, bbox, "red", "Cavity box")
    if excluded_bbox is not None:
        _add_bbox(fig, go, excluded_bbox, "orange", "Excluded occlusal box")
    _apply_plotly_sidebar_layout(fig, legend_position="top")
    pio.write_html(fig, file=str(out_path), auto_open=False,
                   config=_plotly_html_config(), include_plotlyjs="directory")


def export_proximal_box_region_html(region_pts, out_path,
                                    floor_pts=None, floor_depth=None,
                                    floor_labels=None, floor_candidate_pts=None):
    go, pio = _try_plotly()
    if go is None:
        return
    fig = go.Figure()
    if len(region_pts):
        fig.add_trace(go.Scatter3d(
            x=region_pts[:, 0], y=region_pts[:, 1], z=region_pts[:, 2],
            mode="markers", name="proximal box region",
            marker=dict(size=2, opacity=0.35, color="slategray"),
        ))
    if floor_candidate_pts is not None and len(floor_candidate_pts):
        fig.add_trace(go.Scatter3d(
            x=floor_candidate_pts[:, 0], y=floor_candidate_pts[:, 1],
            z=floor_candidate_pts[:, 2],
            mode="markers", name="floor candidate window",
            marker=dict(size=2, opacity=0.25, color="black"),
            hoverinfo="skip",
        ))
    if floor_pts is not None and floor_depth is not None and floor_labels is not None:
        for i in range(4):
            m = floor_labels == i
            if not m.any():
                continue
            p = float(np.percentile(floor_depth[m], PROXIMAL_GINGIVAL_FLOOR_PERCENTILE))
            fig.add_trace(go.Scatter3d(
                x=floor_pts[m, 0], y=floor_pts[m, 1], z=floor_pts[m, 2],
                mode="markers",
                name=f"D{i + 1} floor P95 {p:.2f} mm",
                marker=dict(size=5, color=REGION_COLORS[i]),
                customdata=floor_depth[m],
                hovertemplate=f"D{i + 1}<br>depth=%{{customdata:.2f}} mm<extra></extra>",
            ))
    _apply_plotly_sidebar_layout(fig, legend_position="top")
    pio.write_html(fig, file=str(out_path), auto_open=False,
                   config=_plotly_html_config(), include_plotlyjs="directory")


def depth_legend(name, depths, metrics=None):
    """Use the CSV statistics in views, distinguishing raw sample summaries."""
    if metrics is not None:
        return (f"{name} median {metrics['DepthMedian_mm']:.2f} | "
                f"P95 {metrics['DepthP95_mm']:.2f} | max {metrics['DepthMax_mm']:.2f} mm")
    return f"{name} sample mean {np.mean(depths):.2f} | sample max {np.max(depths):.2f} mm"


def export_combined_html(stl_vectors, occl_pts, prox_pts, out_path,
                         occl_depth=None, prox_depth=None, occl_metrics=None, prox_metrics=None):
    go, pio = _try_plotly()
    if go is None:
        return
    fig = go.Figure()
    _add_mesh_background(fig, go, stl_vectors)
    if occl_pts is not None and len(occl_pts):
        occl_avg = float(np.mean(occl_depth)) if occl_depth is not None and len(occl_depth) else 0.0
        occl_max = float(np.max(occl_depth)) if occl_depth is not None and len(occl_depth) else 0.0
        fig.add_trace(go.Scatter3d(
            x=occl_pts[:, 0], y=occl_pts[:, 1], z=occl_pts[:, 2],
            mode="markers", name=depth_legend("Occlusal", occl_depth, occl_metrics),
            marker=dict(size=3, color="red"),
            customdata=occl_depth,
            hovertemplate="Occlusal<br>depth=%{customdata:.2f} mm<extra></extra>",
        ))
    if prox_pts is not None and len(prox_pts):
        prox_avg = float(np.mean(prox_depth)) if prox_depth is not None and len(prox_depth) else 0.0
        prox_max = float(np.max(prox_depth)) if prox_depth is not None and len(prox_depth) else 0.0
        fig.add_trace(go.Scatter3d(
            x=prox_pts[:, 0], y=prox_pts[:, 1], z=prox_pts[:, 2],
            mode="markers", name=depth_legend("Proximal", prox_depth, prox_metrics),
            marker=dict(size=3, color="green"),
            customdata=prox_depth,
            hovertemplate="Proximal<br>depth=%{customdata:.2f} mm<extra></extra>",
        ))
    _add_cavity_text_label(fig, go, occl_pts, "OCC")
    _add_cavity_text_label(fig, go, prox_pts, "PROX")
    _apply_plotly_sidebar_layout(fig, legend_position="top")
    pio.write_html(fig, file=str(out_path), auto_open=False,
                   config=_plotly_html_config(), include_plotlyjs="directory")


def export_depth_html(stl_vectors, occl_pts, occl_depth, prox_pts, prox_depth,
                      bbox, out_path, title, occl_metrics=None, prox_metrics=None):
    go, pio = _try_plotly()
    if go is None:
        return
    fig = go.Figure()
    _add_mesh_background(fig, go, stl_vectors)

    color_min = DEPTH_HTML_MIN_MM
    color_max = max(1.0, max((float(np.max(d)) for d in (occl_depth, prox_depth)
                              if d is not None and len(d)), default=1.0))
    if occl_pts is not None and len(occl_pts):
        occl_avg = float(np.mean(occl_depth)) if occl_depth is not None and len(occl_depth) else 0.0
        occl_max = float(np.max(occl_depth)) if occl_depth is not None and len(occl_depth) else 0.0
        fig.add_trace(go.Scatter3d(
            x=occl_pts[:, 0], y=occl_pts[:, 1], z=occl_pts[:, 2],
            mode="markers",
            name=depth_legend("Occlusal", occl_depth, occl_metrics),
            marker=dict(
                size=3, color=occl_depth, colorscale="Turbo",
                cmin=color_min, cmax=color_max,
                colorbar=dict(title="Depth (mm)", x=0.795, len=0.76),
            ),
            customdata=occl_depth,
            hovertemplate="Occlusal<br>depth=%{customdata:.2f} mm<extra></extra>",
        ))
    if prox_pts is not None and len(prox_pts):
        prox_avg = float(np.mean(prox_depth)) if prox_depth is not None and len(prox_depth) else 0.0
        prox_max = float(np.max(prox_depth)) if prox_depth is not None and len(prox_depth) else 0.0
        fig.add_trace(go.Scatter3d(
            x=prox_pts[:, 0], y=prox_pts[:, 1], z=prox_pts[:, 2],
            mode="markers",
            name=depth_legend("Proximal", prox_depth, prox_metrics),
            marker=dict(
                size=3, symbol="diamond", color=prox_depth, colorscale="Turbo",
                cmin=color_min, cmax=color_max, showscale=False,
            ),
            customdata=prox_depth,
            hovertemplate="Proximal<br>depth=%{customdata:.2f} mm<extra></extra>",
        ))
    _add_cavity_text_label(fig, go, occl_pts, "OCC")
    _add_cavity_text_label(fig, go, prox_pts, "PROX")
    _apply_plotly_sidebar_layout(
        fig, title=title, reserve_colorbar=True, legend_position="top"
    )
    pio.write_html(fig, file=str(out_path), auto_open=False,
                   config=_plotly_html_config(), include_plotlyjs="directory")


if __name__ == "__main__":
    raise SystemExit(main())
