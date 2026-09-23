#!/usr/bin/env python3
"""Cavity analysis guided by supplied image/STL correspondences and predictions.

The supplied landmarks are model-derived, not independent anatomical ground
truth. Keep the original and refined analyzers alongside this script.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import refined_cav_con_comb_smooth as core

measurement = core.measurement
VERSION = "1.2"
DEFAULT_LANDMARKS = Path("STLFILES/STL_landmarks/outputs/image_stl_landmarks")


def largest_mask_component(mask):
    """Retain the main preparation and report disconnected prediction islands."""
    labels, count = ndi.label(mask, structure=np.ones((3, 3)))
    if not count:
        return mask, 0, 0.
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    selected = labels == areas.argmax()
    return selected, count, float(1 - selected.sum() / mask.sum())


def resize_matrix(source_size, destination_size):
    """Pixel-center affine map matching image resizing, in x/y order."""
    scale = np.asarray(destination_size, float) / source_size
    return np.array([[scale[0], 0, (scale[0] - 1) / 2],
                     [0, scale[1], (scale[1] - 1) / 2], [0, 0, 1]])


def warp_image(image, matrix, shape, order=1, background=255):
    """Resample a source image with a forward x/y homogeneous affine map."""
    inverse = np.linalg.inv(matrix)
    return ndi.affine_transform(np.asarray(image, float), inverse[:2, :2][::-1, ::-1],
                                offset=inverse[:2, 2][::-1], output_shape=shape,
                                order=order, mode="constant", cval=background)


def fit_image_affine(raw_image, sample_image):
    """Register source rendering to its processed prediction-image grid.

    Fits only the 2D image transform; it does not refit the supplied 3D camera.
    Black augmentation padding is excluded from the foreground comparison.
    """
    size = 128
    raw = np.asarray(raw_image.convert("L").resize((size, size)), float)
    target = np.asarray(sample_image.convert("L").resize((size, size)), float)
    source_mask = core.image_tooth_mask(raw_image.resize((size, size)))
    target_mask = core.image_tooth_mask(sample_image.resize((size, size)))
    if source_mask is None or target_mask is None:
        raise ValueError("Unable to identify source/processed tooth silhouette")
    source_center = np.argwhere(source_mask)[:, ::-1].mean(axis=0)
    target_center = np.argwhere(target_mask)[:, ::-1].mean(axis=0)
    target_points = np.argwhere(target_mask)[:, ::-1]
    source_points = np.argwhere(source_mask)[:, ::-1] - source_center

    def transform(parameters):
        angle, sx, sy, shear, dx, dy = parameters
        c, s = np.cos(angle), np.sin(angle)
        linear = np.array([[c, -s], [s, c]]) @ np.array([[np.exp(sx), shear], [0, np.exp(sy)]])
        result = np.eye(3)
        result[:2, :2] = linear
        result[:2, 2] = target_center + [dx, dy] - linear @ source_center
        return result

    def evaluate(parameters, details=False):
        matrix = transform(parameters)
        moved = warp_image(raw, matrix, target.shape)
        moved_mask = warp_image(source_mask, matrix, target.shape, order=0, background=0) > .5
        intersection = moved_mask & target_mask
        iou = intersection.sum() / max(1, (moved_mask | target_mask).sum())
        use = ndi.binary_erosion(intersection, iterations=1)
        correlation = np.corrcoef(moved[use], target[use])[0, 1] if use.sum() > 100 else 0
        if not np.isfinite(correlation):
            correlation = 0
        loss = 1 - iou + 2 * (1 - correlation)
        return (float(iou), float(correlation), matrix) if details else loss

    proposals = []
    for degrees in range(-180, 180, 15):
        angle = np.radians(degrees)
        c, s = np.cos(angle), np.sin(angle)
        rotated = source_points @ np.array([[c, s], [-s, c]])
        scale = np.sqrt(np.var(target_points, axis=0) / np.maximum(np.var(rotated, axis=0), 1))
        parameters = np.r_[angle, np.log(scale), 0., 0., 0.]
        proposals.append((evaluate(parameters), parameters))
    solutions = []
    for _, parameters in sorted(proposals, key=lambda value: value[0])[:3]:
        bounds = [(parameters[0] - .35, parameters[0] + .35), (-.5, .5), (-.5, .5),
                  (-.25, .25), (-8, 8), (-8, 8)]
        result = minimize(evaluate, parameters, method="Powell", bounds=bounds,
                          options={"maxiter": 12, "xtol": .0005, "ftol": 1e-5})
        solutions.append((result.fun, result.x))
    _, parameters = min(solutions, key=lambda value: value[0])
    iou, correlation, normalized = evaluate(parameters, details=True)
    matrix = (np.linalg.inv(resize_matrix(sample_image.size, (size, size))) @ normalized @
              resize_matrix(raw_image.size, (size, size)))
    return matrix, {"image_affine_iou": iou, "image_affine_correlation": correlation,
                    "image_affine_parameters": parameters.tolist()}


class LandmarkBundle:
    """Read source provenance and validate, without trusting circular fit errors."""
    def __init__(self, folder, root):
        self.folder, self.root = Path(folder), Path(root)
        self.cameras = json.loads((self.folder / "camera_params.json").read_text())
        records = json.loads((self.folder / "provenance.json").read_text())["records"]
        self.records = {(int(r["case_id"]), r["view"]): r for r in records}
        self.landmarks = {}
        for row in csv.DictReader((self.folder / "landmarks.csv").open()):
            self.landmarks.setdefault((int(row["case_id"]), row["view"]), []).append(row)
        self.cache = {}

    def source(self, relative):
        path = (self.root / relative.replace("\\", "/")).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("Landmark source path is outside the project")
        return path

    def guidance(self, geometry, vectors, path, number, name, sample_folder, mask_folder, scale):
        letter = name[0]
        record = self.records.get((number, letter))
        info = {"status": "missing_landmark_record", "use_for_selection": False}
        if record is None:
            return info, None, None, None
        camera = self.cameras[str(number)][name.lower()]
        raw_path = self.source(camera["source_image"])
        if measurement.sha256(path) != record["mesh_sha256"]:
            raise ValueError(f"Landmark mesh hash mismatch: {path.name}")
        if measurement.sha256(raw_path) != record["image_sha256"]:
            raise ValueError(f"Landmark source image hash mismatch: {raw_path.name}")
        sample_path = core.matching_file(sample_folder, number)
        mask_path = core.matching_file(mask_folder, number)
        if sample_path is None or mask_path is None:
            info["status"] = "missing_prediction_image_or_mask"
            return info, None, None, None
        raw = Image.open(raw_path).convert("RGB")
        sample = Image.open(sample_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        if list(raw.size) != camera["image_size"]:
            raise ValueError("Landmark camera image dimensions differ from source")
        info.update(source_image=str(raw_path), source_sha256=record["image_sha256"],
                    image_path=str(sample_path.resolve()), mask_path=str(mask_path.resolve()),
                    image_sha256=measurement.sha256(sample_path), mask_sha256=measurement.sha256(mask_path),
                    human_verified=False, landmark_source="model_derived_correspondences",
                    silhouette_iou=camera["silhouette_iou"], shading_correlation=camera["shading_correlation"])
        # C corners from reference masks must never substitute for predictions.
        # They audit the supplied camera/mesh coordinates only.
        rows = self.landmarks.get((number, name.lower()), [])
        usable = [r for r in rows if r["stl_x"] != ""]
        reconstruction_errors = []
        for row in usable:
            triangle = vectors[int(row["face_index_zero_based"])] / scale
            bary = np.array([float(row[f"barycentric_{i}"]) for i in range(3)])
            point = np.array([float(row["stl_" + axis]) for axis in "xyz"])
            reconstruction_errors.append(float(np.linalg.norm(bary @ triangle - point)))
        max_error = max(reconstruction_errors, default=0.)
        if max_error > 1e-5:
            raise ValueError("Landmark XYZ/triangle reconstruction mismatch")
        info.update(landmark_count=len(rows), mapped_landmark_count=len(usable),
                    landmark_reconstruction_max_native_units=max_error,
                    landmarks_snapped_over_2px=sum(float(r["boundary_snap_px"]) > 2 for r in rows))
        key = (record["image_sha256"], info["image_sha256"])
        if key not in self.cache:
            self.cache[key] = fit_image_affine(raw, sample)
        raw_to_sample, fit = self.cache[key]
        info.update(fit)
        # Account explicitly for the prediction-grid resize (including O_45).
        raw_to_mask = resize_matrix(sample.size, mask_image.size) @ raw_to_sample
        matrix = raw_to_mask @ np.asarray(camera["matrix"], float)
        matrix[:, :3] /= scale
        matrix /= np.linalg.norm(matrix[2, :3])
        info.update(matrix=matrix, toward_camera=camera["toward_camera"],
                    image_size=list(mask_image.size), raw_to_prediction_matrix=raw_to_mask,
                    status="landmark_camera_prediction_registered_requires_review",
                    use_for_selection=True)
        if fit["image_affine_correlation"] < .97 or fit["image_affine_iou"] < .94:
            info.update(status="prediction_image_registration_rejected", use_for_selection=False)
        if camera["silhouette_iou"] < .97 or camera["shading_correlation"] < .94:
            info.update(status="landmark_camera_fit_rejected", use_for_selection=False)
        display = sample.resize(mask_image.size)
        mask, component_count, discarded = largest_mask_component(np.asarray(mask_image) > 127)
        info.update(predicted_mask_component_count=component_count,
                    discarded_mask_island_fraction=discarded)
        tooth = core.image_tooth_mask(display)
        if not mask.any() or tooth is None:
            info.update(status="empty_mask_or_tooth", use_for_selection=False)
            return info, None, None, None
        inside = float((mask & ndi.binary_dilation(tooth, iterations=2)).sum() / mask.sum())
        info["mask_inside_tooth_fraction"] = inside
        if inside < .90:
            info.update(status="prediction_mask_outside_tooth", use_for_selection=False)
        support, visible = core.mask_face_support(geometry, info, mask)
        return info, support, visible, {"image": display, "mask": mask, "tooth": tooth}


def masked_crown_reference(geometry, grid, guidance, baseline):
    """Exclude registered preparation surfaces from the depth-reference fit.

    Outline/cavity landmarks are not treated as healthy cusp heights. The result
    remains an estimated unprepared crown surface, not a measured original tooth.
    """
    excluded = np.zeros(grid["height"].shape, bool)
    for info, support, _, _ in guidance.values():
        if not info.get("use_for_selection") or support is None:
            continue
        xy = np.floor((geometry["centers"][support, :2] - grid["origin"]) / grid["spacing"]).astype(int)
        excluded[xy[:, 0], xy[:, 1]] = True
    excluded = ndi.binary_dilation(excluded, iterations=3)
    eligible = grid["footprint"] & (grid["edge_distance"] >= .65)
    zmin, zmax = geometry["vertices"][:, 2].min(), geometry["vertices"][:, 2].max()
    eligible &= grid["height"] >= zmin + .40 * (zmax - zmin)
    keep = eligible & ~excluded
    diagnostics = {"method": "prediction_excluded_robust_crown_quadric",
                   "excluded_reference_fraction": float((eligible & excluded).sum() / max(1, eligible.sum())),
                   "reference_grid_samples": int(keep.sum())}
    if keep.sum() < 100 or keep.sum() < .35 * eligible.sum():
        diagnostics["method"] = "original_reference_insufficient_unmasked_crown"
        return baseline, diagnostics
    ix = np.argwhere(keep)
    points = np.column_stack([grid["origin"] + (ix + .5) * grid["spacing"],
                              grid["height"][ix[:, 0], ix[:, 1]]])
    origin = points[:, :2].mean(axis=0)
    centered = points.copy()
    centered[:, :2] -= origin
    if np.linalg.matrix_rank(np.column_stack([centered[:, 0] ** 2, centered[:, 1] ** 2,
                                              centered[:, 0] * centered[:, 1], centered[:, :2],
                                              np.ones(len(centered))])) < 6:
        diagnostics["method"] = "original_reference_degenerate_unmasked_support"
        return baseline, diagnostics
    coefficients = measurement.fit_robust_quadric(centered)
    for _ in range(3):
        residual = measurement.quadric_z(*centered[:, :2].T, coefficients) - centered[:, 2]
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        selected = residual < med + max(1.5 * mad, .20)
        if selected.sum() < 100:
            break
        coefficients = measurement.fit_quadric(*centered[selected].T)
    a, b, c, d, e, f = coefficients
    ox, oy = origin
    result = np.array([a, b, c, d - 2 * a * ox - c * oy, e - 2 * b * oy - c * ox,
                       f + a * ox ** 2 + b * oy ** 2 + c * ox * oy - d * ox - e * oy])
    reference_change = (measurement.quadric_z(*points[:, :2].T, result) -
                        measurement.quadric_z(*points[:, :2].T, baseline))
    diagnostics["reference_change_median_mm"] = float(np.median(reference_change))
    diagnostics["reference_change_abs_p95_mm"] = float(np.quantile(abs(reference_change), .95))
    return result, diagnostics


def select_landmark_floors(candidates, geometry, fields, guidance):
    """Keep a separate proximal box; partition only an actually shared floor."""
    occlusal, proximal = core.choose_candidates(candidates, geometry, fields)
    notes = []
    if occlusal is None:
        return None, None, {"notes": ["no_supported_floor_candidates"]}
    ranked = {}
    for name in ["Occlusal", "Proximal"]:
        info, support, _, _ = guidance[name]
        ranked[name] = []
        if not info.get("use_for_selection") or support is None:
            continue
        for candidate in candidates:
            ids = candidate["ids"]
            area = geometry["area"][ids]
            covered = float(area[support[ids]].sum())
            fraction = covered / area.sum()
            candidate[name.lower() + "_mask_coverage"] = fraction
            flat_covered = float(area[support[ids] & (fields["nz"][ids] >= .60)].sum())
            if fraction >= (.15 if name == "Occlusal" else .05) and flat_covered >= .20:
                score = fraction * np.sqrt(area.sum())
                if name == "Occlusal":
                    score *= candidate["edge_distance_mm"]
                ranked[name].append((score, candidate))
        ranked[name].sort(key=lambda item: item[0], reverse=True)
    if ranked["Occlusal"]:
        occlusal = ranked["Occlusal"][0][1]
        notes.append("occlusal_selected_using_registered_prediction")
    if ranked["Proximal"]:
        # An occlusal floor can be seen through the proximal opening. Prefer a
        # distinct supported box; do not discard it by splitting that overlap.
        distinct = [item for item in ranked["Proximal"] if item[1] is not occlusal]
        proximal = (distinct or ranked["Proximal"])[0][1]
        notes.append("proximal_selected_using_registered_prediction")
    elif ranked["Occlusal"]:
        _, proximal = core.choose_candidates(candidates, geometry, fields, occlusal=occlusal)
    pinfo, support, _, _ = guidance["Proximal"]
    if (proximal is None or proximal is occlusal) and pinfo.get("use_for_selection"):
        ids = occlusal["ids"]
        # A side view can also see the far occlusal wall through the opening.
        # Only peripheral support can define the gingival/terminal partition.
        part = support[ids] & (fields["edge"][ids] <= 2.4)
        toward = np.asarray(pinfo.get("toward_camera", [0., 0., 0.]), float)[:2]
        if np.linalg.norm(toward) > .5:
            toward /= np.linalg.norm(toward)
            position = geometry["centers"][ids, :2] @ toward
            part &= position >= position.max() - 2.4
            if geometry["area"][ids[part]].sum() >= .20:
                # The side camera sees separated strips around occluding walls.
                # Those strips locate the inward boundary of one terminal box;
                # do not turn unseen intervening floor into occlusal islands.
                cut = measurement.weighted_quantile(position[part], geometry["area"][ids[part]], .05)
                part = position >= cut
        if min(geometry["area"][ids[part]].sum(), geometry["area"][ids[~part]].sum()) >= .20:
            proximal = core.summarize_candidate(ids[part], geometry, fields, "registered_prediction_partition")
            occlusal = core.summarize_candidate(ids[~part], geometry, fields, "registered_prediction_partition")
            notes.append("continuous_floor_partitioned_using_proximal_prediction_requires_review")
        elif proximal is occlusal:
            proximal = None
    if proximal is None:
        split = core.partition_opening(occlusal, geometry, fields)
        if split:
            occlusal, proximal, _ = split
            notes.append("geometry_opening_partition_requires_review")
        else:
            notes.append("proximal_floor_unresolved")
    # Retain mostly upward faces for floor measurement; walls remain available
    # in the full cavity export. A fallback is reported, never silently applied.
    selections = []
    for name, candidate in [("Occlusal", occlusal), ("Proximal", proximal)]:
        if candidate is None:
            selections.append(None)
            continue
        ids = candidate["ids"]
        info, support, _, _ = guidance[name]
        if name == "Occlusal" and info.get("use_for_selection") and support is not None:
            seeds = ids[support[ids]]
            if geometry["area"][seeds].sum() >= .20:
                allowed = np.zeros(len(geometry["area"]), bool)
                allowed[ids] = True
                distance = dijkstra(core.graph_for_faces(geometry, allowed), directed=False,
                                    indices=seeds, min_only=True, limit=.30)
                clipped = ids[np.isfinite(distance[ids])]
                if geometry["area"][clipped].sum() >= .20:
                    ids = clipped
                    notes.append(name.lower() + "_floor_limited_to_mask_with_0.30mm_surface_margin")
        flatter = ids[fields["nz"][ids] >= .60]
        if geometry["area"][flatter].sum() >= .20:
            candidate = core.summarize_candidate(flatter, geometry, fields,
                                                 candidate.get("source", "prediction_supported_component"))
        else:
            notes.append(name.lower() + "_steep_floor_fallback_requires_review")
        selections.append(candidate)
    return *selections, {"notes": notes}


def analyze_one(path, out, args, bundle):
    vectors = np.asarray(measurement.stlmesh.Mesh.from_file(str(path)).vectors, float) * args.scale_to_mm
    geometry = core.mesh_geometry(vectors)
    number = core.case_number(path)
    guidance = {}
    for name in ["Occlusal", "Proximal"]:
        letter = name[0].lower()
        guidance[name] = bundle.guidance(geometry, vectors, path, number, name,
                                         getattr(args, letter + "_images"),
                                         getattr(args, letter + "_masks"), args.scale_to_mm)
    candidates, baseline, fields = core.discover_candidates(geometry, guidance)
    occlusal, proximal, selection = select_landmark_floors(candidates, geometry, fields, guidance)
    reference, reference_info = masked_crown_reference(geometry, fields["grid"], guidance, baseline)
    regions = core.grow_cavity_regions(geometry, fields, occlusal, proximal)
    notes = list(selection["notes"])
    row = {"Tooth": path.stem, "Status": "unavailable", "OcclusalStatus": "not_detected",
           "ProximalStatus": "not_detected", "ReferenceMethod": reference_info["method"],
           "ReferenceExcludedFraction": reference_info["excluded_reference_fraction"]}
    arrays = {"reference_coefficients": reference, "baseline_reference_coefficients": baseline}
    results, regional = {}, []
    for i, (name, candidate) in enumerate([("Occlusal", occlusal), ("Proximal", proximal)]):
        info, support, _, _ = guidance[name]
        row[name + "MaskStatus"] = info["status"]
        row[name + "MaskUsed"] = bool(info.get("use_for_selection"))
        row[name + "ImageCorrelation"] = info.get("image_affine_correlation", "")
        row[name + "CavityRegionArea_mm2"] = float(geometry["area"][regions[i]].sum())
        if not info.get("use_for_selection"):
            notes.append(name.lower() + "_" + info["status"])
        result = None if candidate is None else measurement.measure_floor(
            geometry["vectors"][candidate["ids"]], reference,
            depth_offset_mm=measurement.DEPTH_OUTPUT_OFFSET_MM if name == "Occlusal" else 0.)
        results[name] = result
        if result is None:
            continue
        row[name + "Status"] = "measured_requires_review"
        row[name + "SelectionMethod"] = candidate.get("source", "geometry_component")
        row.update({name + key: value for key, value in result["metrics"].items()})
        prefix = name.lower() + "_"
        for key in ["points", "weights", "depths", "raw_depths", "depth_offset_mm", "residuals",
                    "plane_center", "plane_normal", "triangles"]:
            arrays[prefix + key] = result[key]
        arrays[prefix + "floor_face_indices"] = candidate["ids"]
        arrays[prefix + "cavity_face_indices"] = regions[i]
        base_depth = measurement.quadric_z(*result["points"][:, :2].T, baseline) - result["points"][:, 2]
        arrays[prefix + "reference_only_depth_change"] = result["raw_depths"] - base_depth
        row[name + "ReferenceOnlyMedianDepthChange_mm"] = measurement.weighted_quantile(
            result["raw_depths"] - base_depth, result["weights"], .5)
        if support is not None:
            area = geometry["area"][candidate["ids"]]
            row[name + "FloorMaskSupportedFraction"] = float(np.average(support[candidate["ids"]], weights=area))
        regional.extend({"Tooth": path.stem, "Cavity": name, **r} for r in result["regions"])
        if result["metrics"]["FloorTilt_deg"] > 45:
            notes.append(name.lower() + "_high_floor_tilt_review")
        if result["metrics"]["NegativeDepthAreaFraction"] > .01:
            notes.append(name.lower() + "_negative_reference_depth_review")
        # Audit the actual exported measurements, not just successful execution.
        expected = np.maximum(result["raw_depths"] - .75, 0) if name == "Occlusal" else result["raw_depths"]
        np.testing.assert_allclose(result["depths"], expected, atol=1e-10)
        residual = (result["points"] - result["plane_center"]) @ result["plane_normal"]
        np.testing.assert_allclose(result["residuals"], residual, atol=1e-10)
    if occlusal is not None and proximal is not None:
        assert not np.intersect1d(occlusal["ids"], proximal["ids"]).size
        assert not np.intersect1d(*regions).size
    count = sum(result is not None for result in results.values())
    row["Status"] = "complete_requires_review" if count == 2 else "partial_requires_review" if count else "unavailable"
    row["Warnings"] = ";".join(notes)
    row["Error"] = ""
    np.savez_compressed(out / (path.stem + "_measurements.npz"), **arrays)
    if not args.no_plots:
        core.export_detection(geometry, candidates, [occlusal, proximal], regions, out / (path.stem + "_detection.html"))
        core.export_registration(geometry, guidance, results, out / (path.stem + "_registration.png"))
        measurement.export_measurement_views(geometry["vectors"], results, out, path.stem)
        o, p = results["Occlusal"], results["Proximal"]
        measurement.export_combined_html(geometry["vectors"], None if o is None else o["points"],
            None if p is None else p["points"], out / (path.stem + "_combined.html"),
            occl_depth=None if o is None else o["depths"], prox_depth=None if p is None else p["depths"],
            occl_metrics=None if o is None else o["metrics"], prox_metrics=None if p is None else p["metrics"])
    detail = {"selection": selection, "reference": reference_info,
              "reference_coefficients": reference, "baseline_reference_coefficients": baseline,
              "candidates": [{k: v for k, v in c.items() if k != "ids"} for c in candidates]}
    return row, regional, detail, {name: value[0] for name, value in guidance.items()}


METHODS = """# Landmark-assisted cavity analysis

This run uses the supplied model-derived camera matrices and 2D/3D correspondences.
Input mesh and raw-image SHA-256 hashes and landmark triangle/barycentric coordinates
are checked before use. These checks establish file/coordinate consistency, not
independent anatomical accuracy. All supplied landmarks have human_verified=False.
T points are outline extrema, not cusp tips. C points are mask corners, not known
healthy crown heights. They are not used as independent depth-reference anchors.

## Predicted masks and image registration

Each raw rendering is registered to its processed prediction image with a 2D affine
transform. Black augmentation padding is excluded. Foreground IoU must be >=0.94
and shading correlation >=0.97. Supplied camera fit must have silhouette IoU >=0.97
and shading correlation >=0.94. Pixel-center resize transforms account for the mask
grid explicitly, including the different source-image size for O_45. Prediction masks
are read from pred_O_M_masks_folder and pred_P_M_masks_folder. Reference masks used
to create some supplied C landmarks are never substituted for current predictions.
The largest connected mask component defines the main preparation; disconnected
prediction islands are reported and excluded. This assumes one preparation per view.

Composed perspective matrices map original STL coordinates (after scale-to-mm)
directly to prediction pixels. An approximate depth buffer excludes hidden surfaces;
it is not exact triangle ray tracing. All registrations and boundaries require review.
Good fitted image agreement does not establish unique pose or clinical accuracy.

## Cavity and floor discovery

Connected crown depressions provide candidate floors. Registered masks select their
corresponding components, with an interior-location preference for the occlusal floor
to prevent swapping it with a strongly visible proximal box. A distinct supported
proximal box is retained even when the
proximal view also sees part of the occlusal floor. A shared floor can be partitioned
using proximal-mask support within 2.4 mm of the crown outline and the opening facing
the proximal camera; distant visible floor fragments are excluded. The inward extent
of these supported faces defines a contiguous terminal band, filling side-view
occlusion gaps. A separate supported proximal component is retained in full because
the camera may not see its whole floor. Occlusal selection stays within 0.30 mm along
the mesh from mask-supported faces. Mostly upward-facing facets
(|nz| >=0.60) reduce wall inclusion; fallback use and high fitted tilt are flagged.
Geometry provides fallback candidates when a mapping is rejected. The script assumes
the occlusal direction is approximately +Z and assigns one proximal region per tooth.
The exported cavity surface includes grown walls; measurements use only selected floors.

## Depth and smoothness

The reference quadric is fitted to spatially balanced crown samples, excluding the
registered preparation footprint plus a 0.30 mm margin, with iterative depression
rejection. Insufficient or degenerate unmasked support retains the original reference.
This is still an estimate of the original crown. No intact pre-preparation scan or
independent true-depth measurements are provided. Better registration/floor selection
does not by itself prove more accurate absolute depths.

Occlusal depth remains max(reference_Z - floor_Z - 0.75 mm, 0). Proximal depth is
reference_Z - floor_Z without an offset. The inherited area-weighted plane-residual
RMS smoothness definition is unchanged. The CSV separately reports the reference-only
depth change on the same selected faces; other before/after differences can also arise
from selecting a different floor. Coordinates and STL inputs are not deformed.

## Reproduce and inspect

Run python3 landmark_cav_con_comb_smooth.py --out-dir NEW_EMPTY_DIRECTORY.
Use --cases 35 36 for a subset, --no-plots for numeric results, and --help for paths.
summary.csv contains measurements and warnings; registrations.json contains composed
camera matrices and image-fit diagnostics. NPZ files retain face indices, raw/corrected
depths, plane fits and both reference quadrics. Open index.html for the visual review.
The original final_cav_con_comb_smooth.py and refined_cav_con_comb_smooth.py are imported
for existing helpers and remain unchanged. STL units are assumed to be millimeters.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl-dir", type=Path, default=Path("STLFILES"))
    parser.add_argument("--out-dir", type=Path, default=Path("landmark_cav_con_comb_smooth_results"))
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--o-images", type=Path, default=Path("samples_stu_O"))
    parser.add_argument("--p-images", type=Path, default=Path("samples_stu_P"))
    parser.add_argument("--o-masks", type=Path, default=Path("pred_O_M_masks_folder"))
    parser.add_argument("--p-masks", type=Path, default=Path("pred_P_M_masks_folder"))
    parser.add_argument("--scale-to-mm", type=float, default=1.)
    parser.add_argument("--cases", type=int, nargs="+")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--image-transform-cache", type=Path,
                        help="Reuse this script's image affine fits, keyed by source/image hashes")
    args = parser.parse_args(argv)
    if not np.isfinite(args.scale_to_mm) or args.scale_to_mm <= 0:
        parser.error("Scale must be finite and positive")
    paths = sorted(p for p in args.stl_dir.iterdir() if p.suffix.lower() == ".stl")
    if args.cases:
        paths = [p for p in paths if core.case_number(p) in args.cases]
    if not paths:
        parser.error("No matching STL inputs")
    if len({p.stem.lower() for p in paths}) != len(paths):
        parser.error("Duplicate STL names would overwrite outputs")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be new or empty")
    bundle = LandmarkBundle(args.landmarks, args.source_root)
    if args.image_transform_cache:
        for key, value in json.loads(args.image_transform_cache.read_text()).items():
            bundle.cache[tuple(key.split(":"))] = (np.asarray(value[0]), value[1])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "methods.md").write_text(METHODS)
    protected = [Path(core.__file__), Path(measurement.__file__), args.landmarks / "camera_params.json",
                 args.landmarks / "landmarks.csv", args.landmarks / "provenance.json", *paths]
    manifest = {"version": VERSION, "script_sha256": measurement.sha256(Path(__file__)),
                "command": sys.argv, "started_utc": datetime.now(timezone.utc).isoformat(),
                "occlusal_offset_mm": .75, "scale_to_mm": args.scale_to_mm,
                "protected_inputs": {str(p.resolve()): measurement.sha256(p) for p in protected}}
    measurement.write_json(args.out_dir / "run_manifest.json", manifest)
    rows, regions, details, registrations = [], [], {}, {}
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.stem}", flush=True)
        try:
            row, regional, detail, registration = analyze_one(path, args.out_dir, args, bundle)
            regions.extend(regional)
            details[path.stem], registrations[path.stem] = detail, registration
        except Exception as error:
            import traceback
            traceback.print_exc()
            row = {"Tooth": path.stem, "Status": "error", "Error": str(error)}
        rows.append(row)
        print(row["Status"], row.get("Warnings", ""), flush=True)
        measurement.write_table(args.out_dir / "summary.csv", rows, ["Tooth", "Status"])
        measurement.write_table(args.out_dir / "regions.csv", regions, ["Tooth", "Cavity", "Region"])
        measurement.write_json(args.out_dir / "details.json", details)
        measurement.write_json(args.out_dir / "registrations.json", registrations)
        measurement.write_json(args.out_dir / "image_transforms.json",
                               {":".join(key): value for key, value in bundle.cache.items()})
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["counts"] = {s: sum(r["Status"] == s for r in rows) for s in sorted({r["Status"] for r in rows})}
    manifest["protected_inputs_unchanged"] = all(measurement.sha256(Path(p)) == digest
                                                 for p, digest in manifest["protected_inputs"].items())
    measurement.write_json(args.out_dir / "run_manifest.json", manifest)
    from html import escape
    html = ['<!doctype html><meta charset="utf-8"><title>Landmark-assisted cavity review</title>',
            '<h1>Landmark-assisted cavity review</h1><p>Estimated anatomy and reference surfaces: review before scoring.</p>',
            '<p><a href="summary.csv">Measurements CSV</a> | <a href="methods.md">Methods</a> | '
            '<a href="registrations.json">Registration diagnostics</a></p><table border="1" cellpadding="6">']
    for row in rows:
        stem = row["Tooth"]
        links = []
        for suffix, title in [("_registration.png", "Mask alignment"), ("_detection.html", "Cavity surfaces"),
                              ("_review.png", "Depth/smoothness"), ("_combined.html", "3D depth"),
                              ("_smoothness.html", "3D smoothness")]:
            if (args.out_dir / (stem + suffix)).exists():
                links.append(f'<a href="{escape(stem + suffix)}">{title}</a>')
        html.append(f'<tr><td>{escape(stem)}</td><td>{escape(row["Status"])}</td><td>{" | ".join(links)}</td>'
                    f'<td>{escape(row.get("Warnings", ""))}</td></tr>')
    (args.out_dir / "index.html").write_text("\n".join(html) + "</table>")
    print(manifest["counts"], flush=True)
    return int(any(r["Status"] == "error" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
