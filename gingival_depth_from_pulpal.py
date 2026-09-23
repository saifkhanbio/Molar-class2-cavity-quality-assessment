#!/usr/bin/env python3
"""Display pulpal depth and signed gingival depth for student-preparation STLs.

Reads saved landmark/mask-assisted floor geometry without rerunning detection.
No ideal-preparation targets, clipping or offset are applied to relative gingival
depth. Displayed pulpal depth retains the existing occlusal metric and its offset.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


VERSION = "1.1.1"
PREFIX = "GingivalFromPulpal"
BLUE = "#3978a8"
GREEN = "#14866f"
INK = "#253343"


def weighted_quantile(values, weights, q):
    """Area-weighted midpoint-CDF quantile, matching the existing exporter."""
    order = np.argsort(values, kind="stable")
    x, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(np.interp(q, (np.cumsum(w) - .5*w)/w.sum(), x))


def axis_frame(axis):
    """An orthonormal frame whose third row points toward the occlusal surface."""
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError("Measurement axis must be a finite nonzero 3-vector")
    u = axis / np.linalg.norm(axis)
    seed = np.eye(3)[np.argmin(abs(u))]
    x = np.cross(seed, u)
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(u, x), u])


def quadrature(triangles):
    """Three degree-two quadrature points per original, unmodified triangle."""
    tri = np.asarray(triangles, float)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or not len(tri) or not np.isfinite(tri).all():
        raise ValueError("A finite nonempty triangle surface is required")
    area = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    if area.sum() <= 0 or np.any(area <= 0):
        raise ValueError("Floor contains zero-area triangles")
    bary = np.full((3, 3), 1/6)
    np.fill_diagonal(bary, 2/3)
    return np.einsum("ij,tjk->tik", bary, tri).reshape(-1, 3), np.repeat(area/3, 3), area


def fit_plane(points, weights, robust=True):
    """Area-weighted orthogonal plane; optional Huber IRLS of plane residuals.

    Robust weighting applies only to the reference fit. Every gingival point
    and its original area weight participates in reported depth statistics.
    """
    points, weights = np.asarray(points, float), np.asarray(weights, float)
    if len(points) < 3 or not np.isfinite(points).all() or not np.isfinite(weights).all():
        raise ValueError("Insufficient or non-finite reference samples")
    if np.any(weights <= 0):
        raise ValueError("Reference area weights must be positive")
    effective = weights.copy()
    last_normal = None
    for _ in range(40 if robust else 1):
        center = np.average(points, axis=0, weights=effective)
        delta = points-center
        covariance = (delta*effective[:, None]).T @ delta / effective.sum()
        values, vectors = np.linalg.eigh(covariance)
        if values[1] < 1e-12:
            raise ValueError("Reference samples are collinear; a plane is undefined")
        normal = vectors[:, 0]
        residual = delta @ normal
        scale = max(1e-8, 1.4826*weighted_quantile(abs(residual), weights, .5))
        factors = np.minimum(1., 1.345*scale/np.maximum(abs(residual), 1e-15))
        if last_normal is not None and 1-abs(normal @ last_normal) < 1e-12:
            break
        last_normal = normal
        effective = weights*factors
    return {"center": center, "normal": normal,
            "rms_mm": float(np.sqrt(np.average(residual**2, weights=weights))),
            "effective_area_fraction": float(effective.sum()/weights.sum()),
            "minor_spread_mm": float(np.sqrt(max(values[1], 0)))}


def signed_depth(points, plane, axis):
    """Distance along +axis from each floor sample to a reference-plane hit."""
    denominator = float(plane["normal"] @ axis)
    if abs(denominator) < .05:
        raise ValueError("Reference plane nearly parallel to measurement axis (|n.u| < 0.05)")
    return (plane["center"]-np.asarray(points)) @ plane["normal"] / denominator


def distribution(depths, weights):
    return {"Mean_mm": float(np.average(depths, weights=weights)),
            "Median_mm": weighted_quantile(depths, weights, .5),
            "P05_mm": weighted_quantile(depths, weights, .05),
            "P95_mm": weighted_quantile(depths, weights, .95),
            "SD_mm": float(np.sqrt(np.average((depths-np.average(depths, weights=weights))**2, weights=weights))),
            "NegativeAreaFraction": float(weights[depths < 0].sum()/weights.sum())}


def measure(data, axis=(0., 0., 1.), local_fraction=.55, local_radius_mm=1.5):
    """Build a spatially local pulpal reference without an ideal depth prior."""
    frame = axis_frame(axis)
    axis = frame[2]
    if not 0 < local_fraction <= 1 or local_radius_mm <= 0:
        raise ValueError("Local fraction must be in (0, 1]; radius must be positive")
    otri, gtri = data["occlusal_triangles"], data["proximal_triangles"]
    op, ow, oa = quadrature(otri)
    gp, gw, ga = quadrature(gtri)
    # Audit sample geometry against the source exporter when these arrays exist.
    for name, points, weights in [("occlusal", op, ow), ("proximal", gp, gw)]:
        if name+"_points" in data:
            np.testing.assert_allclose(points, data[name+"_points"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(weights, data[name+"_weights"], atol=1e-9, rtol=0)
    gxy = gtri.reshape(-1, 3) @ frame[:2].T
    distances = cKDTree(gxy).query(otri.mean(axis=1) @ frame[:2].T)[0]
    # Whole faces, closest in the plane perpendicular to the measurement axis.
    # Selection never depends on measured depth, residual sign or an ideal range.
    target_area = min(oa.sum(), max(local_fraction*oa.sum(), 1.0))
    order = np.argsort(distances, kind="stable")
    limit_index = min(np.searchsorted(np.cumsum(oa[order]), target_area), len(oa)-1)
    radius = max(local_radius_mm, float(distances[order[limit_index]]))
    selected = distances <= radius + 1e-12
    local_mask = np.repeat(selected, 3)
    flags = []
    method = "local_area_weighted_huber_plane"
    try:
        local = fit_plane(op[local_mask], ow[local_mask])
        xy = op[local_mask] @ frame[:2].T
        covariance = np.cov(xy.T, aweights=ow[local_mask], bias=True)
        if np.sqrt(np.linalg.eigvalsh(covariance)[0]) < .10:
            raise ValueError("Local footprint too narrow to extrapolate a stable plane")
        depths = signed_depth(gp, local, axis)
    except ValueError as exc:
        flags.append("local_reference_unstable: " + str(exc))
        selected[:] = True
        local_mask[:] = True
        local = fit_plane(op, ow)
        depths = signed_depth(gp, local, axis)
        method = "whole_floor_robust_plane_fallback"
        radius = float(distances.max())
    global_plane = fit_plane(op, ow)
    try:
        global_depths = signed_depth(gp, global_plane, axis)
        global_median = weighted_quantile(global_depths, gw, .5)
    except ValueError:
        global_depths = np.full(len(gp), np.nan)
        global_median = float("nan")
        flags.append("whole_floor_comparison_plane_nearly_parallel_to_axis")
    ordinary = fit_plane(op[local_mask], ow[local_mask], robust=False)
    try:
        ordinary_depths = signed_depth(gp, ordinary, axis)
        robust_shift = abs(weighted_quantile(depths, gw, .5)-weighted_quantile(ordinary_depths, gw, .5))
    except ValueError:
        ordinary_depths = np.full(len(gp), np.nan)
        robust_shift = float("nan")
        flags.append("ordinary_plane_nearly_parallel_to_axis")
    stats = distribution(depths, gw)
    sensitivity = abs(stats["Median_mm"]-global_median)
    axis_cos = abs(float(local["normal"] @ axis))
    extrapolation = cKDTree(op[local_mask] @ frame[:2].T).query(gp @ frame[:2].T)[0]
    # Thresholds mark method sensitivity/support, never student quality.
    if sensitivity > .25:
        flags.append("local_vs_whole_floor_median_differs_over_0.25mm")
    if np.isfinite(robust_shift) and robust_shift > .25:
        flags.append("robust_vs_ordinary_median_differs_over_0.25mm")
    if axis_cos < .5:
        flags.append("axis_projection_amplification_over_2")
    if local["minor_spread_mm"] < .15:
        flags.append("narrow_reference_support")
    p95_distance = weighted_quantile(extrapolation, gw, .95)
    if p95_distance > 2.0:
        flags.append("reference_extrapolation_distance_p95_over_2mm")
    observations = []
    if stats["NegativeAreaFraction"] > 0:
        observations.append("signed_negative_depths_retained; inspect floor relation and labels")
    if local["rms_mm"] > .25:
        observations.append("pulpal_surface_departs_from_plane; may reflect student preparation")
    for name in ("occlusal", "proximal"):
        plane = fit_plane(*quadrature(data[name+"_triangles"])[:2], robust=False)
        tilt = np.degrees(np.arccos(np.clip(abs(plane["normal"] @ axis), 0, 1)))
        if tilt > 45:
            observations.append(name+" surface is steep; retain geometry and verify anatomical label")
    vertices = signed_depth(gtri.reshape(-1, 3), local, axis)
    metrics = {PREFIX+k: v for k, v in stats.items()}
    metrics.update({PREFIX+"Status": "measured_geometry_review" if flags else "measured_requires_anatomical_review",
                    PREFIX+"ReferenceMethod": method,
                    PREFIX+"MinVertex_mm": float(vertices.min()), PREFIX+"MaxVertex_mm": float(vertices.max()),
                    PREFIX+"WholeFloorMedian_mm": global_median,
                    PREFIX+"LocalGlobalSensitivity_mm": sensitivity,
                    PREFIX+"RobustOrdinarySensitivity_mm": robust_shift,
                    PREFIX+"ReferenceArea_mm2": float(ow[local_mask].sum()),
                    PREFIX+"ReferenceAreaFraction": float(ow[local_mask].sum()/ow.sum()),
                    PREFIX+"ReferenceRadius_mm": radius,
                    PREFIX+"ReferenceResidualRMS_mm": local["rms_mm"],
                    PREFIX+"ReferenceEffectiveAreaFraction": local["effective_area_fraction"],
                    PREFIX+"ReferenceTilt_deg": float(np.degrees(np.arccos(np.clip(axis_cos, 0, 1)))),
                    PREFIX+"AxisProjectionAmplification": 1/axis_cos,
                    PREFIX+"ExtrapolationDistanceP95_mm": p95_distance,
                    PREFIX+"FloorArea_mm2": float(ga.sum()),
                    PREFIX+"Offset_mm": 0., PREFIX+"Axis": ",".join(f"{v:.8g}" for v in axis),
                    PREFIX+"GeometryFlags": "; ".join(flags),
                    PREFIX+"Observations": "; ".join(observations),
                    PREFIX+"AnatomicallyValidated": False, PREFIX+"IdealRangeApplied": False})
    # A real quadrature point nearest the median; its arrow has its own value.
    i = int(np.argmin(abs(depths-stats["Median_mm"])))
    foot = gp[i]
    hit = foot + depths[i]*axis
    np.testing.assert_allclose((hit-local["center"]) @ local["normal"], 0., atol=1e-8)
    result = {"metrics": metrics, "points": gp, "weights": gw, "depths": depths,
              "global_depths": global_depths, "ordinary_depths": ordinary_depths,
              "plane": local, "global_plane": global_plane, "frame": frame,
              "local_triangles": otri[selected], "reference_points": op[local_mask],
              "reference_weights": ow[local_mask], "selected_face_mask": selected,
              "arrow_point": foot, "arrow_hit": hit, "arrow_depth_mm": float(depths[i]),
              "flags": flags, "observations": observations,
              "floor_geometry_sha256": hashlib.sha256(otri.tobytes()+gtri.tobytes()).hexdigest()}
    return result


def attach_pulpal_measurements(result, data, source_row):
    """Expose the existing occlusal/pulpal metric without subtracting offset twice.

    The unoffset value is exported separately and is not silently substituted
    for the established occlusal depth. Both use the same saved floor samples.
    """
    points = data["occlusal_points"]
    weights = data["occlusal_weights"]
    depths = data["occlusal_depths"]
    raw = data["occlusal_raw_depths"]
    offset = float(data["occlusal_depth_offset_mm"])
    np.testing.assert_allclose(depths, np.maximum(raw-offset, 0.) if offset > 0 else raw,
                               rtol=0, atol=1e-9)
    median = weighted_quantile(depths, weights, .5)
    np.testing.assert_allclose(median, float(source_row["OcclusalDepthMedian_mm"]), atol=1e-9, rtol=0)
    metrics = {"PulpalDepth"+key: value for key, value in distribution(depths, weights).items()}
    metrics.update({"PulpalDepthUnoffset"+key: value for key, value in distribution(raw, weights).items()})
    metrics.update(PulpalDepthOffset_mm=offset,
                   PulpalDepthStatus=source_row["OcclusalStatus"],
                   PulpalDepthReferenceMethod=source_row.get("ReferenceMethod", ""),
                   PulpalDepthDefinition="existing_occlusal_depth_with_saved_offset",
                   PulpalDepthAxis="0,0,1")
    result["metrics"].update(metrics)
    i = int(np.argmin(abs(depths-median)))
    result.update(pulpal_points=points, pulpal_weights=weights, pulpal_depths=depths,
                  pulpal_unoffset_depths=raw, pulpal_arrow_point=points[i].copy(),
                  pulpal_arrow_hit=points[i]+[0., 0., depths[i]],
                  pulpal_arrow_depth_mm=float(depths[i]), pulpal_offset_mm=offset)


def save_measurement(path, result):
    fields = {k: result[k] for k in ["points", "weights", "depths", "global_depths", "ordinary_depths",
              "frame", "local_triangles", "reference_points", "reference_weights",
              "selected_face_mask", "arrow_point", "arrow_hit", "arrow_depth_mm"]}
    for key in ["pulpal_points", "pulpal_weights", "pulpal_depths", "pulpal_unoffset_depths",
                "pulpal_arrow_point", "pulpal_arrow_hit", "pulpal_arrow_depth_mm", "pulpal_offset_mm"]:
        if key in result:
            fields[key] = result[key]
    for prefix, plane in [("reference", result["plane"]), ("whole_floor_reference", result["global_plane"])]:
        fields[prefix+"_center"] = plane["center"]
        fields[prefix+"_normal"] = plane["normal"]
    np.savez_compressed(path, **fields)


def prepare_surfaces(case, result, pub):
    for region in pub.REGIONS:
        surface = case["floors_world"][region]
        values = signed_depth(surface.points, result["plane"], result["frame"][2])
        case["floors"][region]["Relative depth (mm)"] = values
    patch = pub.polydata(result["local_triangles"])
    patch.points = (patch.points-case["center"]) @ case["basis"].T
    return patch


def marker_camera_scale(focus, scale, eye, markers, aspect):
    """Expand the fixed camera frame only when a depth marker would be clipped."""
    eye = np.asarray(eye, float)/np.linalg.norm(eye)
    up = np.array([0., 0., 1.])
    up -= (up @ eye)*eye
    up /= np.linalg.norm(up)
    right = np.cross(up, eye)
    delta = np.asarray(markers)-focus
    needed = max(float(np.max(abs(delta @ up))), float(np.max(abs(delta @ right)))/aspect)
    return max(scale, needed*1.05) if needed > scale else scale


def render_mesh(case, result, pub, limits, isolated=False):
    plotter = pub.pv.Plotter(off_screen=True, window_size=(1400, 1150))
    plotter.set_background("white")
    if not isolated:
        plotter.add_mesh(case["context"], color="#dddeda", opacity=.25,
                         smooth_shading=True, ambient=.5)
    plotter.add_mesh(case["floors"]["occlusal"], color=BLUE, lighting=False)
    plotter.add_mesh(case["local_patch"], color=GREEN, lighting=False)
    plotter.add_mesh(case["floors"]["proximal"], scalars="Relative depth (mm)",
                     cmap="viridis", clim=limits, lighting=False, show_scalar_bar=False)
    arrow = (np.array([result["arrow_point"], result["arrow_hit"]])-case["center"]) @ case["basis"].T
    plotter.add_lines(arrow, color=INK, width=4)
    plotter.add_points(arrow, color=INK, point_size=10, render_points_as_spheres=True)
    pulpal_arrow = (np.array([result["pulpal_arrow_point"], result["pulpal_arrow_hit"]])-case["center"]) @ case["basis"].T
    plotter.add_lines(pulpal_arrow, color=BLUE, width=4)
    plotter.add_points(pulpal_arrow, color=BLUE, point_size=10, render_points_as_spheres=True)
    points = np.concatenate([case["floors"][r].points for r in pub.REGIONS]) if isolated else case["full"].points
    focus = (points.min(axis=0)+points.max(axis=0))/2
    eye = np.array([.8, -1.6, 1.65])
    plotter.camera_position = [focus+eye*15, focus, [0, 0, 1]]
    plotter.enable_parallel_projection()
    scale = max(np.ptp(points, axis=0))*.63
    plotter.camera.parallel_scale = marker_camera_scale(focus, scale, eye, np.concatenate([arrow, pulpal_arrow]), 1400/1150)
    plotter.enable_anti_aliasing("ssaa")
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def cut_segments(surface, origin, direction, axis):
    cut = surface.slice(normal=np.cross(direction, axis), origin=origin)
    segments, i = [], 0
    while i < len(cut.lines):
        count = cut.lines[i]
        points = cut.points[cut.lines[i+1:i+count+1]] - origin
        xy = np.column_stack([points @ direction, points @ axis])
        segments.extend(np.stack([xy[:-1], xy[1:]], axis=1))
        i += count+1
    return np.asarray(segments)


def draw_pulpal_section(ax, case, result, pub):
    """An actual +Z STL section through the displayed pulpal sample point."""
    from matplotlib.collections import LineCollection
    origin = result["pulpal_arrow_point"]
    direction = result["arrow_point"]-origin
    direction[2] = 0.
    direction = direction/np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-8 else case["basis"][0]
    for surface, color, width in [(case["full_world"], "#b1b6bc", .65),
                                  (case["floors_world"]["occlusal"], BLUE, 2)]:
        segments = cut_segments(surface, origin, direction, np.array([0., 0., 1.]))
        if segments.size:
            ax.add_collection(LineCollection(segments, colors=color, linewidths=width))
    floor = case["floors_world"]["occlusal"].points
    x = (floor-origin) @ direction
    h = np.linspace(x.min()-.4, x.max()+.4, 150)
    reference = pub.quadric_z(origin+h[:, None]*direction, case["data"]["reference_coefficients"])-origin[2]
    ax.plot(h, reference, color="#748698", lw=1., ls="--", label="Estimated crown reference")
    ax.plot(h, reference-result["pulpal_offset_mm"], color=BLUE, lw=1.2, ls=":", label="Reference − saved offset")
    d = result["pulpal_arrow_depth_mm"]
    ax.annotate("", (0, d), (0, 0), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3))
    ax.scatter([0, 0], [0, d], color=BLUE, s=12, zorder=5)
    ax.annotate(f"Pulpal point\n{d:.2f} mm", (0, d/2), xytext=(9, 0), textcoords="offset points",
                fontsize=8, va="center", bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    ax.set(xlim=(h.min(), h.max()), ylim=(min((floor-origin)[:, 2].min(), 0)-.35, max(reference.max(), d)+.3),
           xlabel="Distance along section (mm)", ylabel="Height relative to pulpal point (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("C  Pulpal depth · actual STL section", loc="left", fontsize=9, weight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.95,
              fontsize=6.5, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.spines[["top", "right"]].set_visible(False)


def write_png(case, result, output, pub, limits, dpi):
    plt = pub.plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 10.3))
    fig.subplots_adjust(left=.085, right=.96, bottom=.16, top=.89, hspace=.52, wspace=.28)
    m = result["metrics"]
    fig.suptitle(f"{case['stem']} | Pulpal and gingival cavity depths", x=.07, ha="left", fontsize=16, weight="bold", y=.978)
    fig.text(.07, .94, f"Pulpal median  {m['PulpalDepthMedian_mm']:.2f} mm   |   "
             f"Gingival-from-pulpal median  {m[PREFIX+'Median_mm']:.2f} mm", fontsize=11)
    fig.text(.07, .918, f"Pulpal unoffset median: {m['PulpalDepthUnoffsetMedian_mm']:.2f} mm. "
             f"Displayed pulpal depth retains the {m['PulpalDepthOffset_mm']:.2f} mm offset.", fontsize=8)
    for ax, isolated, title in [(axes[0, 0], False, "A  Original tooth and measured floors"),
                               (axes[0, 1], True, "B  Floors and local pulpal reference patch")]:
        ax.imshow(render_mesh(case, result, pub, limits, isolated))
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=9, weight="bold", pad=6)
    draw_pulpal_section(axes[1, 0], case, result, pub)
    ax = axes[1, 1]
    ax.hist(result["pulpal_depths"], bins=35, weights=100*result["pulpal_weights"]/result["pulpal_weights"].sum(),
            color=BLUE, edgecolor="white", linewidth=.3)
    ax.axvline(m["PulpalDepthMedian_mm"], color=INK, lw=1.3, label="Area-weighted median")
    ax.set(xlabel="Pulpal depth with saved offset (mm)", ylabel="Pulpal floor area (%)")
    ax.set_title("D  Pulpal depth distribution", loc="left", fontsize=9, weight="bold")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.spines[["top", "right"]].set_visible(False)
    origin, axis = result["arrow_point"], result["frame"][2]
    direction = result["plane"]["center"]-origin
    direction -= (direction @ axis)*axis
    direction = direction/np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-8 else result["frame"][0]
    ax = axes[2, 0]
    cuts = []
    for surface, color, width in [(case["full_world"], "#b1b6bc", .65),
                                  (case["floors_world"]["occlusal"], BLUE, 2),
                                  (case["floors_world"]["proximal"], "#9c6b18", 2)]:
        segments = cut_segments(surface, origin, direction, axis)
        if segments.size:
            ax.add_collection(LineCollection(segments, colors=color, linewidths=width))
            cuts.append(segments)
    if not cuts:
        raise ValueError("No STL cross-section through the measured point")
    points = np.concatenate([case["floors_world"][r].points for r in pub.REGIONS])
    x = (points-origin) @ direction
    h = np.linspace(x.min()-.6, x.max()+.6, 150)
    z = signed_depth(origin+h[:, None]*direction, result["plane"], axis)
    ax.plot(h, z, ls="--", lw=1.2, color=GREEN)
    d = result["arrow_depth_mm"]
    ax.annotate("", (0, d), (0, 0), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.scatter([0, 0], [0, d], color=INK, s=12, zorder=5)
    ax.annotate(f"Point depth\n{d:.2f} mm", (0, d/2), xytext=(10, 0), textcoords="offset points",
                fontsize=8, va="center", bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    heights = (points-origin) @ axis
    ax.set(xlim=(h.min(), h.max()), ylim=(min(heights.min(), z.min(), 0)-.4, max(heights.max(), z.max(), d)+.4),
           xlabel="Distance along section (mm)", ylabel="Height relative to marked floor point (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("E  Gingival depth · actual STL section", loc="left", fontsize=9, weight="bold")
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2, 1]
    ax.hist(result["depths"], bins=35, weights=100*result["weights"]/result["weights"].sum(),
            color="#688aa3", edgecolor="white", linewidth=.3)
    ax.axvline(m[PREFIX+"Median_mm"], color=INK, lw=1.3, label="Local reference median")
    if np.isfinite(m[PREFIX+"WholeFloorMedian_mm"]):
        ax.axvline(m[PREFIX+"WholeFloorMedian_mm"], color=GREEN, lw=1.2, ls="--", label="Whole-floor reference median")
    ax.set(xlabel="Signed gingival depth from pulpal reference (mm)", ylabel="Gingival floor area (%)")
    ax.set_title("F  Gingival depth distribution", loc="left", fontsize=9, weight="bold")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.spines[["top", "right"]].set_visible(False)
    cax = fig.add_axes([.62, .098, .28, .011])
    cb = fig.colorbar(pub.matplotlib.cm.ScalarMappable(norm=Normalize(*limits), cmap="viridis"), cax=cax, orientation="horizontal")
    cb.set_label("Signed gingival depth (mm); common batch scale", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=4, label="Pulpal floor"),
                        Line2D([0], [0], color=GREEN, lw=4, label="Reference patch / plane"),
                        Line2D([0], [0], color="#9c6b18", lw=3, label="Gingival floor in section")],
               loc="lower left", bbox_to_anchor=(.075, .069), frameon=False, fontsize=7.5)
    fig.text(.07, .045, "Pulpal: existing occlusal metric with saved offset. Gingival: signed floor-to-plane distance, no offset.", fontsize=7.5)
    fig.text(.07, .029, "Arrows mark actual samples near the medians. Student measurements and anatomical review flags are preserved.", fontsize=7.5)
    if result["flags"]:
        fig.text(.07, .013, "Reference sensitivity/support flagged — see HTML and CSV; this is not a preparation-quality grade.", fontsize=7.5, color="#8d591b")
    path = output / f"{case['stem']}_gingival_depth.png"
    fig.savefig(path, dpi=dpi, facecolor="white", metadata={"Title": f"{case['stem']} pulpal and gingival depths"})
    plt.close(fig)
    with pub.Image.open(path) as picture:
        picture.thumbnail((1380, 1545))
        picture.convert("RGB").save(output / f"{case['stem']}_preview.jpg", quality=92)
    return path


def write_html(case, result, output, pub, limits):
    import base64
    go = pub.go
    fig = go.Figure()
    fig.add_trace(pub.mesh_trace(case["context"], "Tooth (click legend to hide)", limits[1]))
    fig.data[-1].opacity = .22
    for surface, name, color in [(case["floors"]["occlusal"], "Pulpal floor", BLUE),
                                  (case["local_patch"], "Local reference patch", GREEN)]:
        trace = pub.mesh_trace(surface, name, limits[1])
        trace.color = color
        fig.add_trace(trace)
    g = case["floors"]["proximal"].copy()
    g["Depth (mm)"] = g["Relative depth (mm)"]
    fig.add_trace(pub.mesh_trace(g, "Gingival floor: signed depth", limits[1], colored=True))
    arrow = (np.array([result["arrow_point"], result["arrow_hit"]])-case["center"]) @ case["basis"].T
    fig.add_trace(go.Scatter3d(x=arrow[:, 0], y=arrow[:, 1], z=arrow[:, 2], mode="lines+markers+text",
                             text=["", f"Gingival {result['arrow_depth_mm']:.2f} mm"], textposition="top center",
                             line=dict(color=INK, width=6), marker=dict(size=3, color=INK), name="Gingival point depth"))
    arrow = (np.array([result["pulpal_arrow_point"], result["pulpal_arrow_hit"]])-case["center"]) @ case["basis"].T
    fig.add_trace(go.Scatter3d(x=arrow[:, 0], y=arrow[:, 1], z=arrow[:, 2], mode="lines+markers+text",
                             text=["", f"Pulpal {result['pulpal_arrow_depth_mm']:.2f} mm"], textposition="top center",
                             line=dict(color=BLUE, width=6), marker=dict(size=3, color=BLUE), name="Pulpal point depth (saved offset)"))
    # Plane patch above the combined floor footprint, hidden initially for clarity.
    frame, origin = result["frame"], result["plane"]["center"]
    cloud = np.concatenate([case["data"][r+"triangles"].reshape(-1, 3) for r in ["occlusal_", "proximal_"]])
    xy = (cloud-origin) @ frame[:2].T
    xx, yy = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), 12),
                         np.linspace(xy[:, 1].min(), xy[:, 1].max(), 12))
    plane_points = origin+xx.ravel()[:, None]*frame[0]+yy.ravel()[:, None]*frame[1]
    plane_points += signed_depth(plane_points, result["plane"], frame[2])[:, None]*frame[2]
    shown = (plane_points-case["center"]) @ case["basis"].T
    fig.add_trace(go.Surface(x=shown[:, 0].reshape(xx.shape), y=shown[:, 1].reshape(xx.shape),
                             z=shown[:, 2].reshape(xx.shape), showscale=False, opacity=.25,
                             colorscale=[[0, GREEN], [1, GREEN]], name="Extended pulpal reference plane",
                             showlegend=True, visible="legendonly"))
    pulpal_cloud = result["pulpal_points"]
    cx, cy = np.meshgrid(np.linspace(pulpal_cloud[:, 0].min(), pulpal_cloud[:, 0].max(), 12),
                         np.linspace(pulpal_cloud[:, 1].min(), pulpal_cloud[:, 1].max(), 12))
    crown = np.column_stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)])
    crown[:, 2] = pub.quadric_z(crown, case["data"]["reference_coefficients"])-result["pulpal_offset_mm"]
    shown_crown = (crown-case["center"]) @ case["basis"].T
    fig.add_trace(go.Surface(x=shown_crown[:, 0].reshape(cx.shape), y=shown_crown[:, 1].reshape(cx.shape),
                             z=shown_crown[:, 2].reshape(cx.shape), showscale=False, opacity=.25,
                             colorscale=[[0, BLUE], [1, BLUE]], name="Pulpal crown reference minus saved offset",
                             showlegend=True, visible="legendonly"))
    hidden = dict(visible=False, showbackground=False)
    fig.update_layout(template="plotly_white", height=700, margin=dict(l=0, r=20, t=25, b=0),
                      scene=dict(aspectmode="data", xaxis=hidden, yaxis=hidden, zaxis=hidden,
                                 camera=dict(eye=dict(x=1., y=-1.8, z=1.7), projection=dict(type="orthographic"))),
                      coloraxis=dict(colorscale="viridis", cmin=limits[0], cmax=limits[1],
                                     colorbar=dict(title="Signed depth<br>(mm)", len=.65)),
                      legend=dict(orientation="h", y=-.03))
    plot = fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})
    preview = base64.b64encode((output/f"{case['stem']}_preview.jpg").read_bytes()).decode()
    m = result["metrics"]
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in result["flags"]+result["observations"])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{case['stem']} | Pulpal and gingival depths</title><style>
body{{font:16px/1.55 Arial;color:{INK};background:#f4f6f8;margin:auto;max-width:1150px;padding:25px}}
section{{background:white;border-radius:10px;padding:20px;margin:18px 0}}img{{width:100%}}a{{color:#28688e}}h1{{line-height:1.2}}
.value{{font-size:30px;font-weight:bold}}small{{font-size:15px}}details{{margin:20px 0}}
</style><a href="index.html">All specimens</a> · <a href="{case['stem']}_gingival_depth.png">Publication PNG</a>
<h1>{case['stem']} · Pulpal and gingival cavity depths</h1>
<section><h2>Pulpal depth of the occlusal preparation</h2>
<span class="value">{m['PulpalDepthMedian_mm']:.3f} <small>mm area-weighted median</small></span>
<p>5th–95th percentile: {m['PulpalDepthP05_mm']:.3f} to {m['PulpalDepthP95_mm']:.3f} mm.</p>
<p>This is the existing occlusal depth with its saved {m['PulpalDepthOffset_mm']:.2f} mm offset already applied.
Unoffset crown-to-pulpal median: {m['PulpalDepthUnoffsetMedian_mm']:.3f} mm. These are two definitions of depth to the same floor.</p></section>
<section><h2>Gingival depth from the pulpal reference plane</h2>
<span class="value">{m[PREFIX+'Median_mm']:.3f} <small>mm area-weighted median</small></span>
<p>5th–95th percentile: {m[PREFIX+'P05_mm']:.3f} to {m[PREFIX+'P95_mm']:.3f} mm · Whole-floor reference median: {m[PREFIX+'WholeFloorMedian_mm']:.3f} mm</p>
<p>Student preparation — no ideal range applied. Signed measurements, irregularity and inclined floors are retained.</p></section>
<section><img alt="Pulpal and gingival depths with measured floors, actual STL sections and both depth distributions" src="data:image/jpeg;base64,{preview}"></section>
<section>{plot}<p>Rotate and zoom. Click legend entries to hide the tooth or show the extended reference plane. Geometry is unchanged.</p></section>
<section><h2>Measurement interpretation</h2>
<p>Pulpal depth = max(estimated crown-reference Z − pulpal-floor Z − saved offset, 0), along original +Z.
Its saved offset is applied once. The separately listed unoffset depth uses the original signed crown-to-floor samples.</p>
<p>Gingival depth is the signed distance along axis {html.escape(m[PREFIX+'Axis'])},
from each gingival-floor sample to the extrapolated local pulpal plane. Positive means the gingival point lies below that reference.
Negative values may reflect floor relationships, plane extrapolation or incorrect labels; they are not clipped or automatically rejected.</p>
<p>The green patch comprises the spatially closest pulpal faces. A robust area-weighted plane reduces the influence of isolated reference irregularities;
all gingival samples retain their original area weights. Reference-patch area: {m[PREFIX+'ReferenceArea_mm2']:.2f} mm².
Reference residual RMS: {m[PREFIX+'ReferenceResidualRMS_mm']:.3f} mm. Local versus whole-floor median difference:
{m[PREFIX+'LocalGlobalSensitivity_mm']:.3f} mm. This is method sensitivity, not a confidence interval.</p>
<p>No estimated crown surface or 0.75 mm correction enters the relative gingival measurement. Existing crown-referenced depths remain in the combined CSV.
The anatomical axis and model-derived floor labels are not independently validated. No CQS quality grade is assigned here.</p>
<h2>Geometry review and observations</h2><ul>{notes or '<li>No additional geometric sensitivity flag; anatomical review is still required.</li>'}</ul>
<p>Flags concern measurement support or sensitivity. Surface tilt, shallow preparation, deep preparation and roughness can be real student features.
Nothing is moved or reselected to achieve an expected preparation depth.</p></section></html>'''
    (output/f"{case['stem']}_gingival_depth.html").write_text(page)


METHODS = """# Pulpal depth and gingival depth from the pulpal floor

This analysis measures student preparations, without ideal-depth thresholds,
clipping, absolute-value conversion, offsets or score-driven floor changes.
It reuses both saved landmark/prediction-guided floor labels on original STL
triangles. It does not rerun segmentation or claim that these labels are verified.

The default measurement direction is original STL +Z, toward the occlusal side.
This is an assumed orientation, not a landmark-validated anatomical tooth axis.
`--axis X Y Z` accepts another independently established direction.

The pulpal-to-gingival measurement definition is also described in Figure 1C of
Azhari et al. (2024), https://doi.org/10.1016/j.sdentj.2024.07.005 . That study
does not validate this automated floor detector or the local-plane algorithm.

## Reference and measurement

Choose pulpal faces closest to the gingival floor in projection perpendicular to
the axis: include faces within at least 1.5 mm and at least 55% of pulpal area
(or 1 mm², capped at available area). Fit an area-weighted orthogonal plane with
Huber IRLS. Expand to the whole pulpal floor only if local support is degenerate
or its transverse spread is below 0.10 mm. These are numerical support rules,
not preparation-quality targets. Original floor geometry is never modified.

For each gingival quadrature point g, compute
`h = dot(n, c - g) / dot(n, u)` using pulpal plane center c, normal n and unit
axis u. Positive h means below the reference. Retain every gingival sample and
its area weight, including negative and extreme values. The new offset is zero.
The existing occlusal 0.75 mm correction is retained for the displayed pulpal
depth. It is not applied to relative gingival depth.

## Displayed pulpal depth

`PulpalDepthMedian_mm` is an explicit alias of the existing
`OcclusalDepthMedian_mm`, reproduced from the saved area-weighted floor samples.
Pulpal depth = max(estimated crown Z - pulpal floor Z - saved offset, 0).
The saved offset is 0.75 mm in this dataset and is applied once. Source STL,
floor geometry and original depth columns are unchanged. A separate
`PulpalDepthUnoffsetMedian_mm` is computed from the original raw sample depths,
not by adding an offset to a possibly clipped summary statistic.

Every PNG/HTML now displays both pulpal and relative gingival medians, separate
real STL sections and distributions, and both actual-point depth markers.
Unoffset pulpal depth is also listed explicitly for reference. The two pulpal
definitions measure depth to the same floor; they are not independent anatomy.

Report area-weighted mean, median, 5th/95th percentiles, SD, signed negative-area
fraction and exact vertex extrema. Repeat using a whole-floor robust plane and
an ordinary local plane to disclose reference sensitivity. Differences are not
confidence intervals. All measurements require independent anatomical review.

## Review without forcing ideal preparations

No depth range triggers rejection or automatic floor relabeling. Tilt and plane
residuals can represent real student preparation features. Signed negative values
are observations, not proof of an algorithm error. Review geometry when reference
choice shifts the median by >0.25 mm, axis amplification exceeds two, the reference
is narrow, or 95th-percentile distance to reference support exceeds 2 mm. These
heuristics are unvalidated support flags, not clinical pass/fail thresholds.
Planes nearly parallel to the axis (|n.u| < 0.05) are numerically unresolved;
failures are explicit rather than assigned a fabricated depth.

The plane is extended across the proximal box; the upper arrow endpoint is on
this mathematical reference, not necessarily on a physically present surface.
Use the actual STL section, isolated floors and reference-sensitivity columns to
review it. Both local and global planes are saved in the NPZ for reproduction.

## Outputs and CQS

`gingival_depth_summary.csv`: new measurements and review fields.
`depth_summary_with_gingival.csv`: original columns plus new measurements.
`*_gingival_measurements.npz`: samples, weights, both planes, selected faces and arrow.
`*_gingival_depth.png/html`: publication figure and offline interactive 3D view.
`validation.json` and `run_manifest.json`: checks, failures and input hashes.

Use `GingivalFromPulpalMedian_mm` for the proposed proximal-depth component of CQS
(10% weight), after reference/floor validation and calibration of a matching
educational rubric. No grading thresholds or final CQS are imposed here. The
previous crown-referenced 2.5–3.5 mm band does not apply automatically.

Run from the repository root:
`python3 gingival_depth_from_pulpal.py --out-dir NEW_EMPTY_DIRECTORY`
For a subset add `--cases O_1 O_35 O_41`; `--no-plots` exports numeric results only.
Tests: `python3 -m unittest discover -s tests -p test_gingival_depth_from_pulpal.py`.
Dependencies: numpy, scipy and the publication_cavity_depth.py exporter dependencies.
"""


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-dir", type=Path, default=Path("landmark_cav_con_comb_smooth_results"))
    parser.add_argument("--stl-dir", type=Path, default=Path("STLFILES"))
    parser.add_argument("--out-dir", type=Path, default=Path("RESULTS-21-09-2026/gingival_depth_from_pulpal_results"))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--axis", nargs=3, type=float, default=[0., 0., 1.])
    parser.add_argument("--local-fraction", type=float, default=.55)
    parser.add_argument("--local-radius-mm", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    axis_frame(args.axis)
    if args.dpi < 150 or not 0 < args.local_fraction <= 1 or args.local_radius_mm <= 0:
        parser.error("Require dpi >= 150, 0 < local-fraction <= 1, local-radius-mm > 0")
    if args.out_dir.resolve() in [args.stl_dir.resolve(), args.measurements_dir.resolve()]:
        parser.error("Output must be separate from inputs")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Use a new empty output directory to preserve existing results")
    # Reuse only provenance/render helpers, never the unavailable old detector imports.
    import publication_cavity_depth as pub
    rows = {r["Tooth"]: r for r in csv.DictReader((args.measurements_dir/"summary.csv").open())}
    names = sorted(args.cases or rows, key=pub.natural_key)
    if len(names) != len(set(names)) or set(names)-set(rows):
        parser.error("Unknown or duplicate case identifiers")
    source_manifest = json.loads((args.measurements_dir/"run_manifest.json").read_text())
    registrations = json.loads((args.measurements_dir/"registrations.json").read_text())
    protected_paths = [args.measurements_dir/k for k in ["summary.csv", "registrations.json", "run_manifest.json"]]
    protected_paths += [args.measurements_dir/f"{n}_measurements.npz" for n in names]
    protected_paths += [args.stl_dir/f"{n}.stl" for n in names]
    protected = {str(p.resolve()): pub.sha256(p) for p in protected_paths}
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": VERSION, "command": sys.argv, "started_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": pub.sha256(__file__), "exporter_sha256": pub.sha256(pub.__file__),
                "protected_inputs": protected, "axis": args.axis, "anatomical_axis_validated": False,
                "local_fraction": args.local_fraction, "local_radius_mm": args.local_radius_mm,
                "ideal_range_applied": False, "relative_offset_mm": 0., "dpi": args.dpi, "cases": []}
    results, records, failures = {}, [], []
    for name in names:
        print(f"Measure {name}", flush=True)
        try:
            with np.load(args.measurements_dir/f"{name}_measurements.npz", allow_pickle=False) as saved:
                data = {k: saved[k] for k in saved.files}
            result = measure(data, args.axis, args.local_fraction, args.local_radius_mm)
            # Validate against source STL and all existing depth statistics now.
            case = pub.load_case(args.stl_dir/f"{name}.stl", args.measurements_dir, rows[name], registrations[name], source_manifest)
            attach_pulpal_measurements(result, data, rows[name])
            result["provenance"] = {"source_statistic_max_error": case["max_statistic_error"],
                                    "stl_sha256": case["stl_sha256"], "measurement_sha256": case["npz_sha256"]}
            results[name] = result
            records.append({"Tooth": name, **result["metrics"], PREFIX+"Error": ""})
            save_measurement(output/f"{name}_gingival_measurements.npz", result)
        except (ValueError, KeyError, AssertionError) as exc:
            failures.append({"Tooth": name, "error": str(exc)})
            records.append({"Tooth": name, PREFIX+"Status": "unresolved", PREFIX+"Error": str(exc)})
            print(f"  UNRESOLVED: {exc}", flush=True)
    write_csv(output/"gingival_depth_summary.csv", records)
    write_csv(output/"depth_summary_with_gingival.csv", [{**rows[r["Tooth"]], **r} for r in records])
    # A common batch scale includes all signed vertex depths, without truncation.
    low = min((r["metrics"][PREFIX+"MinVertex_mm"] for r in results.values()), default=0.)
    high = max((r["metrics"][PREFIX+"MaxVertex_mm"] for r in results.values()), default=1.)
    limits = (math.floor(low*2)/2, math.ceil(high*2)/2)
    if limits[0] == limits[1]:
        limits = (limits[0]-.5, limits[1]+.5)
    manifest["color_limits_mm"] = limits
    for index, (name, result) in enumerate(results.items(), 1):
        print(f"[{index}/{len(results)}] Export {name}", flush=True)
        if not args.no_plots:
            case = pub.load_case(args.stl_dir/f"{name}.stl", args.measurements_dir, rows[name], registrations[name], source_manifest)
            case["local_patch"] = prepare_surfaces(case, result, pub)
            png = write_png(case, result, output, pub, limits, args.dpi)
            write_html(case, result, output, pub, limits)
            with pub.Image.open(png) as picture:
                expected = (round(9.2*args.dpi), round(10.3*args.dpi))
                if any(abs(a-b) > 1 for a, b in zip(picture.size, expected)):
                    raise ValueError("Unexpected PNG dimensions")
        manifest["cases"].append({"Tooth": name, **result["provenance"],
                                  "arrow_depth_mm": result["arrow_depth_mm"],
                                  "pulpal_arrow_depth_mm": result["pulpal_arrow_depth_mm"],
                                  "pulpal_offset_mm": result["pulpal_offset_mm"],
                                  "flags": result["flags"], "observations": result["observations"]})
    unchanged = all(pub.sha256(p) == digest for p, digest in protected.items())
    if not unchanged:
        raise ValueError("Protected inputs changed during processing")
    # Re-read generated artifacts to validate the actual files, not just memory.
    recompute_error = 0.
    duplicates = {}
    for name, result in results.items():
        with np.load(output/f"{name}_gingival_measurements.npz", allow_pickle=False) as saved:
            plane = {"center": saved["reference_center"], "normal": saved["reference_normal"]}
            calc = signed_depth(saved["points"], plane, saved["frame"][2])
            np.testing.assert_allclose(calc, saved["depths"], atol=1e-9, rtol=0)
            recompute_error = max(recompute_error, abs(weighted_quantile(calc, saved["weights"], .5)-result["metrics"][PREFIX+"Median_mm"]))
            pulpal_calc = np.maximum(saved["pulpal_unoffset_depths"]-saved["pulpal_offset_mm"], 0.)
            np.testing.assert_allclose(pulpal_calc, saved["pulpal_depths"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(weighted_quantile(pulpal_calc, saved["pulpal_weights"], .5),
                                       float(rows[name]["OcclusalDepthMedian_mm"]), atol=1e-9, rtol=0)
        duplicates.setdefault(result["floor_geometry_sha256"], []).append(name)
    duplicate_checks = []
    for group in duplicates.values():
        if len(group) > 1:
            medians = [results[n]["metrics"][PREFIX+"Median_mm"] for n in group]
            error = float(np.ptp(medians))
            assert error < 1e-9
            duplicate_checks.append({"cases": group, "median_difference_mm": error})
    combined = list(csv.DictReader((output/"depth_summary_with_gingival.csv").open()))
    assert all(all(row[k] == v for k, v in rows[row["Tooth"]].items()) for row in combined)
    checks = {"requested": len(names), "measured": len(results), "unresolved": failures,
              "input_files_unchanged": unchanged, "original_csv_columns_unchanged": True,
              "saved_plane_depths_recomputed": True, "median_recompute_max_error_mm": recompute_error,
              "source_stl_floor_membership_and_disjointness_checked": True,
              "source_quadrature_reproduced": True, "ideal_depth_constraints_applied": False,
              "pulpal_medians_match_existing_occlusal_medians": True,
              "pulpal_saved_offset_applied_once": True,
              "anatomically_validated": False, "clinical_accuracy_tested": False,
              "identical_floor_geometry_checks": duplicate_checks,
              "png_count": len(list(output.glob("*_gingival_depth.png"))),
              "html_count": len(list(output.glob("*_gingival_depth.html")))}
    (output/"validation.json").write_text(json.dumps(checks, indent=2))
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), protected_inputs_unchanged=unchanged, failures=failures)
    (output/"run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output/"README.md").write_text(METHODS)
    cards = []
    for row in records:
        name = row["Tooth"]
        if name in results:
            value = row[PREFIX+"Median_mm"]
            link = f"{name}_gingival_depth.html" if not args.no_plots else "gingival_depth_summary.csv"
            picture = f'<img src="{name}_preview.jpg" alt="{name} depth figure">' if not args.no_plots else ""
            flag = " · Geometry review" if results[name]["flags"] else ""
            cards.append(f'<a href="{link}">{picture}<b>{name}</b>Pulpal: {row["PulpalDepthMedian_mm"]:.3f} mm<br>'
                         f'Gingival from pulpal: {value:.3f} mm{flag}</a>')
        else:
            cards.append(f'<div>{name}: unresolved — {html.escape(row[PREFIX+"Error"])}</div>')
    (output/"index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student preparations | Pulpal and gingival depths</title><style>
body{font:16px/1.5 Arial;color:#253343;margin:30px;background:#f4f6f8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:20px}
.grid a,.grid div{background:white;padding:15px;border-radius:10px;color:#253343;text-decoration:none}img{width:100%}b{display:block}</style>
<h1>Pulpal and gingival cavity depths</h1><p>Pulpal depth retains the saved 0.75 mm offset. Relative gingival depth has no offset. Student measurements and review flags are preserved.</p>
<p><a href="gingival_depth_summary.csv">Relative-depth CSV</a> · <a href="depth_summary_with_gingival.csv">Combined depth CSV</a> · <a href="README.md">Methods</a></p>
<div class="grid">'''+"".join(cards)+"</div></html>")
    print(f"Finished {len(results)}/{len(names)} cases: {output/'index.html'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
