#!/usr/bin/env python3
"""Export measured cavity depths as print figures and standalone interactive HTML.

Run from the repository root: python3 publication_cavity_depth.py
Requires numpy, numpy-stl, pyvista/VTK, matplotlib, Pillow and plotly. Reads saved
landmark-assisted measurements; does not import or rerun the detection scripts.
"""

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/publication-cavity-mpl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/publication-cavity-mesa")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pyvista as pv
from stl import mesh as stl_mesh


VERSION = "2.1"
PUBLICATION_DEPTH_REDUCTION_MM = 0.75
REGIONS = ("occlusal", "proximal")
INK = "#253343"
MUTED = "#586779"
REFERENCE = "#536c85"
CMAP = "viridis"
SMOOTH_CMAP = "magma"
PULPAL_COLOR = "#1976a3"
GINGIVAL_COLOR = "#bd5725"
FLOOR_LABELS = {"occlusal": "Occlusal pulpal floor", "proximal": "Proximal gingival floor"}
OVERVIEW_EYE = np.array([0.65, -1.25, 1.7])
DETAIL_EYES = {"occlusal": np.array([0., 0., 1.]),
               "proximal": np.array([0., -.65, 1.])}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def natural_key(value):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", value)]


def quadric_z(points, coefficients):
    x, y = np.asarray(points)[..., 0], np.asarray(points)[..., 1]
    a, b, c, d, e, f = coefficients
    return a*x*x + b*y*y + c*x*y + d*x + e*y + f


def reported_depth(points, coefficients, offset):
    raw = quadric_z(points, coefficients) - np.asarray(points)[..., 2]
    return np.maximum(raw - offset, 0.) if offset > 0 else raw


def publication_reference_z(points, coefficients):
    """Reference used for both displayed floors, including the agreed reduction."""
    return quadric_z(points, coefficients) - PUBLICATION_DEPTH_REDUCTION_MM


def publication_depth(points, coefficients):
    """Reduce both raw depths equally; retain signed values without clipping."""
    return publication_reference_z(points, coefficients) - np.asarray(points)[..., 2]


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - .5*weights) / weights.sum()
    return float(np.interp(quantile, cdf, values))


def common_reference_metrics(data):
    """Apply the same crown-reference convention to both floors, without clipping.

    The relative depth is the difference of regional area-weighted medians,
    not the median distance from gingival points to a fitted pulpal plane.
    """
    coefficients = data["reference_coefficients"]
    metrics, samples = {}, {}
    for region in REGIONS:
        name = region.title()
        points, weights = data[region+"_points"], data[region+"_weights"]
        if not len(points) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(f"{region}: valid positive area weights are required")
        raw_depths = reported_depth(points, coefficients, 0.)
        np.testing.assert_allclose(raw_depths, data[region+"_raw_depths"], rtol=0, atol=1e-8)
        depths = publication_depth(points, coefficients)
        if not np.all(np.isfinite(depths)):
            raise ValueError(f"{region}: non-finite reference depth")
        vertices = data[region+"_triangles"].reshape(-1, 3)
        vertex_depths = publication_depth(vertices, coefficients)
        residuals = (points-data[region+"_plane_center"]) @ data[region+"_plane_normal"]
        np.testing.assert_allclose(residuals, data[region+"_residuals"], rtol=0, atol=1e-8)
        values = {"DepthMean_mm": np.average(depths, weights=weights),
                  "DepthMedian_mm": weighted_quantile(depths, weights, .5),
                  "DepthP95_mm": weighted_quantile(depths, weights, .95),
                  "DepthMin_mm": vertex_depths.min(), "DepthMax_mm": vertex_depths.max(),
                  "NegativeDepthAreaFraction": np.average(depths < 0, weights=weights),
                  "DepthClippedAreaFraction": 0.,
                  "FloorResidualRMS_mm": np.sqrt(np.average(residuals**2, weights=weights)),
                  "FloorResidualMeanAbs_mm": np.average(abs(residuals), weights=weights),
                  "FloorResidualAbsP95_mm": weighted_quantile(abs(residuals), weights, .95)}
        metrics.update({name+key: float(value) for key, value in values.items()})
        samples[region] = {"depths": depths, "residuals": residuals}
    metrics["GingivalMinusPulpalDepth_mm"] = metrics["ProximalDepthMedian_mm"]-metrics["OcclusalDepthMedian_mm"]
    metrics["GingivalMinusPulpalMethod"] = "difference_of_area_weighted_medians_same_crown_reference"
    metrics["DepthReference"] = "shared_crown_reference_original_STL_Z"
    return metrics, samples


def prepare_publication_case(case):
    """Update the export view while preserving the source loader's API and data."""
    source = case["row"]
    metrics, samples = common_reference_metrics(case["data"])
    row = {key: source[key] for key in ("Tooth", "Status", "ReferenceMethod", "Warnings", "Error")}
    for region in REGIONS:
        name = region.title()
        for suffix in ("Status", "FloorArea_mm2", "FloorTriangleCount", "FloorTilt_deg", "SelectionMethod"):
            row[name+suffix] = source[name+suffix]
        np.testing.assert_allclose(metrics[name+"FloorResidualRMS_mm"],
                                   float(source[name+"FloorResidualRMS_mm"]), rtol=0, atol=1e-8)
        points = case["data"][region+"_points"]
        index = int(np.argmin(abs(samples[region]["depths"]-metrics[name+"DepthMedian_mm"])))
        case["representative"][region] = {"point": points[index].copy(),
                "depth_mm": float(samples[region]["depths"][index])}
        world = case["floors_world"][region]
        world["Depth (mm)"] = publication_depth(world.points, case["data"]["reference_coefficients"])
        world["Plane deviation (mm)"] = abs((world.points-case["data"][region+"_plane_center"]) @
                                             case["data"][region+"_plane_normal"])
        for key in ("Depth (mm)", "Plane deviation (mm)"):
            case["floors"][region][key] = world[key].copy()
    row.update(metrics)
    case.update(row=row, samples=samples)
    if metrics["GingivalMinusPulpalDepth_mm"] < 0:
        case["notes"].append("Signed gingival-minus-pulpal depth is negative; inspect the floor labels and reference geometry.")
    return case


def polydata(triangles):
    points, indices = np.unique(triangles.reshape(-1, 3), axis=0,
                                return_inverse=True)
    faces = np.column_stack([np.full(len(triangles), 3), indices.reshape(-1, 3)])
    return pv.PolyData(points, faces.ravel())


def review_notes(row):
    notes = []
    for region in REGIONS:
        name = region.title()
        tilt = float(row[name + "FloorTilt_deg"])
        area = float(row[name + "FloorArea_mm2"])
        if tilt > 45 or area < 1:
            notes.append(f"{name} floor selection requires review "
                         f"(fitted plane tilt {tilt:.1f}°, area {area:.2f} mm²).")
    return notes


def load_case(stl_path, measurement_dir, row, registration, source_manifest):
    """Check provenance, geometry and numeric summaries before drawing anything."""
    expected_hash = source_manifest.get("protected_inputs", {}).get(str(stl_path.resolve()))
    actual_hash = sha256(stl_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(f"STL changed since measurement: {stl_path}")
    npz_path = measurement_dir / f"{stl_path.stem}_measurements.npz"
    with np.load(npz_path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    scale = float(source_manifest.get("scale_to_mm", 1.))
    triangles = stl_mesh.Mesh.from_file(str(stl_path)).vectors.astype(float)*scale
    full = polydata(triangles)
    # Match by coordinates, not face indices: STL deduplication may reorder faces.
    vertex_lookup = {tuple(p): i for i, p in enumerate(full.points)}
    source_faces = full.faces.reshape(-1, 4)[:, 1:]
    source_keys = {tuple(sorted(f)) for f in source_faces}
    measured_keys = set()
    errors = {}
    representative = {}
    floors_world = {}
    for region in REGIONS:
        name = region.title()
        tri = data[region + "_triangles"]
        keys = {tuple(sorted(vertex_lookup[tuple(p)] for p in t)) for t in tri}
        if not keys.issubset(source_keys):
            raise ValueError(f"{stl_path.stem}: {region} floor is not on source mesh")
        if measured_keys.intersection(keys):
            raise ValueError(f"{stl_path.stem}: occlusal/proximal floors overlap")
        measured_keys.update(keys)
        floor = polydata(tri)
        offset = float(data[region + "_depth_offset_mm"])
        floor["Depth (mm)"] = reported_depth(floor.points, data["reference_coefficients"], offset)
        floors_world[region] = floor
        depths, weights = data[region + "_depths"], data[region + "_weights"]
        calculated = reported_depth(data[region + "_points"], data["reference_coefficients"], offset)
        np.testing.assert_allclose(calculated, depths, rtol=0, atol=1e-8)
        area = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1).sum()
        metrics = {"DepthMean_mm": np.average(depths, weights=weights),
                   "DepthMedian_mm": weighted_quantile(depths, weights, .5),
                   "DepthP95_mm": weighted_quantile(depths, weights, .95),
                   "DepthMax_mm": floor["Depth (mm)"].max(),
                   "DepthMin_mm": floor["Depth (mm)"].min(),
                   "FloorArea_mm2": area, "DepthOffset_mm": offset}
        for metric, value in metrics.items():
            errors[name + metric] = abs(float(row[name + metric]) - value)
            if errors[name + metric] > 1e-7:
                raise ValueError(f"{stl_path.stem}: saved CSV/geometry mismatch: {name}{metric}")
        # Pick an actual measured point nearest the area-weighted median; never
        # label a fabricated point/arrow with the summary median.
        index = int(np.argmin(abs(depths - float(row[name + "DepthMedian_mm"]))))
        representative[region] = {"point": data[region + "_points"][index].copy(),
                                  "depth_mm": float(depths[index]), "offset_mm": offset}
    keep = np.array([tuple(sorted(f)) not in measured_keys for f in source_faces])
    context = pv.PolyData(full.points.copy(),
                         np.column_stack([np.full(keep.sum(), 3), source_faces[keep]]).ravel())
    # Rigid display transform: +Z is preserved; proximal opening faces -Y.
    toward = np.asarray(registration["Proximal"]["toward_camera"], float)
    toward[2] = 0.
    toward /= np.linalg.norm(toward)
    basis = np.array([[-toward[1], toward[0], 0.], -toward, [0., 0., 1.]])
    center = (full.points.min(axis=0) + full.points.max(axis=0))/2

    def display(surface):
        result = surface.copy()
        result.points = (surface.points - center) @ basis.T
        return result

    return {"stem": stl_path.stem, "row": row, "data": data, "full_world": full,
            "full": display(full), "context": display(context),
            "floors_world": floors_world,
            "floors": {r: display(f) for r, f in floors_world.items()},
            "representative": representative, "center": center, "basis": basis,
            "notes": review_notes(row), "max_statistic_error": max(errors.values()),
            "stl_sha256": actual_hash, "npz_sha256": sha256(npz_path)}


def render_surface(case, region=None, pixels=1400, vmax=6.5, vmin=0., smoothness=False):
    """Orthographic, unmodified meshes; colored floors are not light-shaded."""
    plotter = pv.Plotter(off_screen=True, window_size=(pixels, pixels))
    plotter.set_background("white")
    if region is None:
        plotter.add_mesh(case["context"], color="#e4e4dd", smooth_shading=True,
                         ambient=.4, diffuse=.6, specular=.12)
        surfaces, eye = case["floors"], OVERVIEW_EYE
        bounds_points = case["full"].points
    else:
        surfaces, eye = {region: case["floors"][region]}, DETAIL_EYES[region]
        bounds_points = surfaces[region].points
    for surface in surfaces.values():
        scalar = "Plane deviation (mm)" if smoothness else "Depth (mm)"
        plotter.add_mesh(surface, scalars=scalar, cmap=SMOOTH_CMAP if smoothness else CMAP, clim=(vmin, vmax),
                         lighting=False, show_scalar_bar=False, interpolate_before_map=True)
        edge = surface.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                             manifold_edges=False, non_manifold_edges=False)
        if edge.n_points:
            plotter.add_mesh(edge, color=INK, line_width=1.3, lighting=False)
    eye = eye / np.linalg.norm(eye)
    focal = (bounds_points.min(axis=0) + bounds_points.max(axis=0))/2
    up_hint = np.array([0., 1., 0.]) if region == "occlusal" else np.array([0., 0., 1.])
    right = np.cross(up_hint, eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    projected = np.column_stack([(bounds_points-focal) @ right, (bounds_points-focal) @ up])
    parallel_scale = float(np.max(np.ptp(projected, axis=0))*.65)
    plotter.camera_position = [focal + eye*30, focal, up]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = parallel_scale
    plotter.enable_anti_aliasing("ssaa")
    picture = plotter.screenshot(return_img=True)
    plotter.close()
    return picture, parallel_scale


def section_geometry(case):
    """A real vertical STL section through the two arrow sample locations."""
    p = case["representative"]["occlusal"]["point"]
    q = case["representative"]["proximal"]["point"]
    direction = q-p
    direction[2] = 0.
    if np.linalg.norm(direction) < 1e-6:
        direction = -case["basis"][1].copy()
    direction /= np.linalg.norm(direction)
    normal = np.cross(direction, [0., 0., 1.])
    cut = case["full_world"].slice(normal=normal, origin=p)
    segments = []
    i = 0
    while i < len(cut.lines):
        count = cut.lines[i]
        points = cut.points[cut.lines[i+1:i+count+1]]
        coords = np.column_stack([(points-p) @ direction, points[:, 2]])
        segments.extend(np.stack([coords[:-1], coords[1:]], axis=1))
        i += count+1
    if not segments:
        raise ValueError(f"{case['stem']}: no mesh cross-section")
    return p, direction, np.asarray(segments)


def draw_section(ax, case, vmax):
    p, direction, segments = section_geometry(case)
    z_origin = float(case["full_world"].points[:, 2].min())
    segments[:, :, 1] -= z_origin
    xmin, xmax = segments[:, :, 0].min(), segments[:, :, 0].max()
    h = np.linspace(xmin, xmax, 350)
    path = p + h[:, None]*direction
    reference = publication_reference_z(path, case["data"]["reference_coefficients"]) - z_origin
    ax.add_collection(LineCollection(segments, colors="#7b848d", linewidths=.65, zorder=1))
    ax.plot(h, reference, color=REFERENCE, lw=1.2, ls=(0, (4, 2)), zorder=2)
    for region in REGIONS:
        record = case["representative"][region]
        point, depth = record["point"], record["depth_mm"]
        x = float((point-p) @ direction)
        z = point[2]-z_origin
        top = z + depth
        color = PULPAL_COLOR if region == "occlusal" else GINGIVAL_COLOR
        ax.scatter([x], [z], s=23, c=[color], edgecolors=INK, linewidths=.5, zorder=5)
        ax.annotate("", (x, top), (x, z), arrowprops=dict(arrowstyle="<->", lw=1., color=INK), zorder=4)
        ax.plot([x-.13, x+.13], [top, top], color=INK, lw=.8)
        shift, align = (-.28, "right") if region == "occlusal" else (.28, "left")
        label = "Pulpal" if region == "occlusal" else "Gingival"
        ax.text(x+shift, (z+top)/2, f"{label}\n{depth:.2f} mm", ha=align,
                va="center", fontsize=7, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.93, pad=1.4), zorder=6)
    ax.set_xlim(xmin-.35, xmax+.35)
    ax.set_ylim(0, max(reference.max(), segments[:, :, 1].max())+.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance along section (mm)", fontsize=7, labelpad=2)
    ax.set_ylabel("Height (mm)", fontsize=7, labelpad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#c7ced5")
    ax.tick_params(labelsize=6, length=2.5, color="#bcc4cd")
    ax.grid(axis="y", color="#edf0f3", linewidth=.5)


def draw_depth_difference(ax, case):
    """A reference-normalized schematic, explicitly distinct from an STL section."""
    row = case["row"]
    pulpal, gingival = [float(row[r.title()+"DepthMedian_mm"]) for r in REGIONS]
    delta = float(row["GingivalMinusPulpalDepth_mm"])
    ax.axhline(0, color=REFERENCE, ls="--", lw=1.)
    for x, value, label, color in ((.2, pulpal, "Pulpal", PULPAL_COLOR),
                                   (.58, gingival, "Gingival", GINGIVAL_COLOR)):
        ax.plot([x-.10, x+.10], [value, value], color=color, lw=2.5)
        ax.annotate("", (x, value), (x, 0), arrowprops=dict(arrowstyle="<->", color=color, lw=1.1))
        ax.text(x, value+.13, f"{label}\n{value:.2f} mm", ha="center", va="top", color=color, fontsize=8)
    ax.plot([.69, .88], [pulpal, pulpal], color=MUTED, lw=.7, ls=":")
    ax.plot([.69, .88], [gingival, gingival], color=MUTED, lw=.7, ls=":")
    if abs(delta) < .35:
        # Tiny gaps cannot accommodate two arrowheads; use an exact-height bracket.
        ax.plot([.84, .84], [pulpal, gingival], color=INK, lw=1.2)
        for value in (pulpal, gingival):
            ax.plot([.825, .855], [value, value], color=INK, lw=1.2)
    else:
        ax.annotate("", (.84, gingival), (.84, pulpal), arrowprops=dict(arrowstyle="<->", lw=1.3, color=INK))
    ax.text(.87, (pulpal+gingival)/2, f"Δ\n{delta:+.2f}", fontsize=8, va="center", ha="left")
    ax.set(xlim=(0, 1.08), ylim=(max(pulpal, gingival, 0)+1.1, min(pulpal, gingival, 0)-.35))
    ax.set_ylabel("Depth below reference (mm)", fontsize=8)
    ax.set_xticks([])
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#c7ced5")


def write_png(case, output, dpi, vmax, vmin=0., smooth_max=2.):
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                                "axes.labelcolor": MUTED, "xtick.color": MUTED,
                                "ytick.color": MUTED, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(10.5, 9.8), facecolor="white")
    row = case["row"]
    fig.text(.045, .966, f"{case['stem']}  |  Cavity depth and floor regularity", fontsize=17, weight="bold")
    fig.text(.045, .941, "Shared crown reference · landmark- and mask-guided floor selection", fontsize=10, color=MUTED)
    panels = [(None, "A", "Tooth overview", .04, .695),
              ("occlusal", "B", FLOOR_LABELS["occlusal"], .365, .695),
              ("proximal", "C", FLOOR_LABELS["proximal"], .69, .695),
              ("occlusal", "F", "Pulpal floor smoothness", .04, .115),
              ("proximal", "G", "Gingival floor smoothness", .365, .115)]
    for region, letter, title, x, y in panels:
        smooth = letter in ("F", "G")
        fig.text(x, y+.212, f"{letter}  {title}", fontsize=10, weight="bold")
        ax = fig.add_axes([x, y, .285, .207])
        picture, scale = render_surface(case, region, pixels=max(800, int(dpi*3)),
                                         vmax=smooth_max if smooth else vmax,
                                         vmin=0. if smooth else vmin, smoothness=smooth)
        ax.imshow(picture)
        ax.set_axis_off()
        bar_mm = 2. if region is None else 1.
        bar_fraction = bar_mm/(2*scale)
        ax.plot([.06, .06+bar_fraction], [.035, .035], transform=ax.transAxes, color=INK, lw=1.5)
        ax.text(.06+bar_fraction/2, .055, f"{bar_mm:g} mm", transform=ax.transAxes, ha="center", fontsize=7)
        if smooth:
            fig.text(x, y-.010, f"RMS  {row[region.title()+'FloorResidualRMS_mm']:.3f} mm", fontsize=11, weight="bold")
            fig.text(x, y-.028, "Absolute distance from fitted floor plane", fontsize=8, color=MUTED)
        elif region:
            name = region.title()
            fig.text(x, y-.012, f"Median  {row[name+'DepthMedian_mm']:.2f} mm", fontsize=11, weight="bold")
            fig.text(x, y-.030, f"95th percentile  {row[name+'DepthP95_mm']:.2f} mm", fontsize=8, color=MUTED)
        else:
            fig.text(x, y-.012, "Color = depth from shared crown reference", fontsize=8)
            fig.text(x, y-.030, "Gray = remaining tooth surface", fontsize=8, color=MUTED)
    cax = fig.add_axes([.36, .631, .60, .012])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.text(.04, .631, "Depth (mm) · same scale across cases", fontsize=9)
    fig.text(.04, .578, "D  Vertical STL section", fontsize=10, weight="bold")
    section = fig.add_axes([.09, .398, .52, .155])
    draw_section(section, case, vmax)
    fig.legend(handles=[Line2D([0], [0], color="#7b848d", lw=.8, label="STL section"),
                            Line2D([0], [0], color=REFERENCE, ls="--", label="Shared crown reference")],
                   loc="upper left", bbox_to_anchor=(.18, .575), ncol=2, fontsize=7, frameon=False)
    fig.text(.04, .348, "Arrows show actual sample depths near regional medians; height zero is the lowest STL vertex.", fontsize=7.5, color=MUTED)
    fig.text(.69, .578, "E  Gingival − pulpal depth", fontsize=10, weight="bold")
    comparison = fig.add_axes([.744, .388, .222, .175])
    draw_depth_difference(comparison, case)
    fig.text(.69, .366, f"Δ = {row['ProximalDepthMedian_mm']:.3f} − {row['OcclusalDepthMedian_mm']:.3f} = {row['GingivalMinusPulpalDepth_mm']:+.3f} mm", fontsize=9, weight="bold")
    fig.text(.69, .348, "Difference of medians · schematic", fontsize=7.5, color=MUTED)
    fig.text(.69, .327, "H  Floor residual distributions", fontsize=10, weight="bold")
    hist = fig.add_axes([.748, .145, .215, .16])
    bins = np.linspace(0, smooth_max, 36)
    for region, color, label in (("occlusal", PULPAL_COLOR, "Pulpal"), ("proximal", GINGIVAL_COLOR, "Gingival")):
        weights = case["data"][region+"_weights"]
        hist.hist(abs(case["samples"][region]["residuals"]), weights=100*weights/weights.sum(),
                  bins=bins, histtype="step", color=color, lw=1.3, label=label)
    hist.set_xlabel("Absolute plane residual (mm)", fontsize=8)
    hist.set_ylabel("Floor area (%) per bin", fontsize=8)
    hist.tick_params(labelsize=7)
    hist.spines[["top", "right"]].set_visible(False)
    hist.legend(fontsize=8, frameon=False)
    cax = fig.add_axes([.26, .062, .39, .010])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, smooth_max), cmap=SMOOTH_CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.text(.04, .061, "Plane deviation (mm)", fontsize=9)
    fig.text(.69, .061, "Smoothness here measures floor planarity;\nit does not measure microscopic roughness.", fontsize=8, color=MUTED)
    footer = "Area-weighted statistics. Both floor depths use the same crown reference and original STL +Z axis."
    fig.text(.04, .022, footer, fontsize=8, color=MUTED)
    if case["notes"]:
        fig.text(.04, .007, "Review flag: inspect anatomical floor selection and reference. See HTML for details.", fontsize=8, color="#9a5c13")
    png = output/f"{case['stem']}_depth.png"
    fig.savefig(png, dpi=dpi, metadata={"Title": f"{case['stem']} cavity depth and floor regularity",
                    "Description": "Shared crown reference. Gingival-minus-pulpal median depth and floor-plane residuals."})
    plt.close(fig)
    with Image.open(png) as picture:
        picture.thumbnail((2100, 1960), Image.Resampling.LANCZOS)
        picture.convert("RGB").save(output/f"{case['stem']}_preview.jpg", quality=95)
    complete = output/f"{case['stem']}_complete.png"
    if case["stem"] == "O_37" or complete.exists():
        shutil.copyfile(png, complete)
    return png


def mesh_trace(surface, name, vmax, colored=False, smoothness=False):
    vertices = surface.points
    faces = surface.faces.reshape(-1, 4)[:, 1:]
    options = dict(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                   i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name,
                   flatshading=False, showscale=False)
    if colored:
        scalar = "Plane deviation (mm)" if smoothness else "Depth (mm)"
        options.update(intensity=surface[scalar], intensitymode="vertex",
                       coloraxis="coloraxis", lighting=dict(ambient=1., diffuse=0., specular=0.),
                       hovertemplate=name+"<br>"+scalar+": %{intensity:.3f}<extra></extra>")
    else:
        options.update(color="#deded6", hoverinfo="skip",
                       lighting=dict(ambient=.55, diffuse=.65, specular=.15, roughness=.75))
    return go.Mesh3d(**options)


def preview_camera(view="overview"):
    """Use the PNG camera orientation in both interactive viewers and controls."""
    region = {"overview": None, "top": "occlusal", "proximal": "proximal"}[view]
    eye = np.asarray(OVERVIEW_EYE if region is None else DETAIL_EYES[region], dtype=float)
    direction = eye/np.linalg.norm(eye)
    up_hint = np.array([0., 1., 0.]) if region == "occlusal" else np.array([0., 0., 1.])
    right = np.cross(up_hint, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    # Keep a comfortable framing distance; orthographic orientation matches PNG.
    return {"eye": dict(zip("xyz", (direction*2.55).tolist())),
            "up": dict(zip("xyz", up.tolist())), "center": dict(x=0., y=0., z=0.),
            "projection": {"type": "orthographic"}}


def case_camera(case, view="overview", initial=False):
    """Preserve existing views; preview alignment was requested only for O_37."""
    if case["stem"] == "O_37":
        return preview_camera(view)
    eyes = {"overview": dict(x=1.05, y=-1.65, z=1.65), "top": dict(x=0., y=0., z=2.5),
            "proximal": dict(x=0., y=-2., z=1.6)}
    camera = {"eye": eyes[view], "projection": {"type": "orthographic"}}
    if not initial:
        camera.update(center=dict(x=0., y=0., z=0.),
                      up=dict(x=0., y=1., z=0.) if view == "top" else dict(x=0., y=0., z=1.))
    return camera


def interactive_figure(case, vmax, vmin=0., smoothness=False):
    fig = go.Figure()
    fig.add_trace(mesh_trace(case["context"], "Tooth surface", vmax))
    for region in REGIONS:
        fig.add_trace(mesh_trace(case["floors"][region], FLOOR_LABELS[region], vmax, True, smoothness))
    if not smoothness:
        for region in REGIONS:
            record = case["representative"][region]
            a = record["point"].copy()
            b = a.copy()
            b[2] += record["depth_mm"]
            points = (np.array([a, b])-case["center"]) @ case["basis"].T
            color = PULPAL_COLOR if region == "occlusal" else GINGIVAL_COLOR
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="lines+markers+text", line=dict(color=color, width=5), marker=dict(size=3, color=color),
                text=["", f"{region.title()} {record['depth_mm']:.2f} mm"], textposition="top center",
                name=FLOOR_LABELS[region]+" depth marker", visible="legendonly"))
        cloud = np.concatenate([case["floors"][r].points for r in REGIONS])
        xx, yy = np.meshgrid(np.linspace(cloud[:, 0].min()-.3, cloud[:, 0].max()+.3, 24),
                            np.linspace(cloud[:, 1].min()-.3, cloud[:, 1].max()+.3, 24))
        world = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)]) @ case["basis"]+case["center"]
        zz = publication_reference_z(world, case["data"]["reference_coefficients"]).reshape(xx.shape)-case["center"][2]
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=.22, showscale=False,
                     colorscale=[[0, REFERENCE], [1, REFERENCE]], name="Shared crown reference",
                     hoverinfo="skip", visible="legendonly", showlegend=True))
    axis = dict(visible=False, showbackground=False)
    fig.update_layout(template="plotly_white", margin=dict(l=0, r=50, b=0, t=20), height=650,
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data", bgcolor="white",
                   camera=case_camera(case, initial=True)),
        coloraxis=dict(colorscale=SMOOTH_CMAP if smoothness else CMAP, cmin=vmin, cmax=vmax,
                       colorbar=dict(title="Plane deviation (mm)" if smoothness else "Depth (mm)", thickness=16, len=.6)),
        legend=dict(orientation="h", y=-.02), paper_bgcolor="white", font=dict(family="Arial", color=INK))
    return fig


def write_html(case, output, vmax, vmin=0., smooth_max=2.):
    plots = []
    for smooth in (False, True):
        fig = interactive_figure(case, smooth_max if smooth else vmax, 0. if smooth else vmin, smooth)
        plots.append(fig.to_html(full_html=False, include_plotlyjs=not smooth,
                     div_id="smoothness-viewer" if smooth else "depth-viewer",
                     config={"displaylogo": False, "responsive": True,
                             "toImageButtonOptions": {"format": "png", "filename": case["stem"]+("_smoothness" if smooth else "_depth"),
                                                      "width": 1800, "height": 1500, "scale": 2}}))
    row = case["row"]
    preview = base64.b64encode((output/f"{case['stem']}_preview.jpg").read_bytes()).decode()
    cards = []
    for region in REGIONS:
        name = region.title()
        cards.append(f'<div class="card"><h2>{FLOOR_LABELS[region]}</h2><strong>{row[name+"DepthMedian_mm"]:.3f} mm</strong>'
                     f'<p>Area-weighted median depth<br>95th percentile {row[name+"DepthP95_mm"]:.3f} mm<br>'
                     f'Floor residual RMS {row[name+"FloorResidualRMS_mm"]:.3f} mm</p></div>')
    cards.append(f'<div class="card"><h2>Gingival − pulpal depth</h2><strong>{row["GingivalMinusPulpalDepth_mm"]:+.3f} mm</strong>'
                 f'<p>{row["ProximalDepthMedian_mm"]:.3f} − {row["OcclusalDepthMedian_mm"]:.3f} mm<br>Difference of regional medians</p></div>')
    warnings = "".join(f'<p class="warning">{html.escape(n)}</p>' for n in case["notes"])
    if row.get("Warnings"):
        warnings += '<details><summary>Source detection notes</summary><p>'+html.escape(row["Warnings"])+"</p></details>"
    cameras = json.dumps({name: case_camera(case, name) for name in ("overview", "top", "proximal")})
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{case['stem']} · Cavity depth and smoothness</title>
<style>body{{margin:0;background:#f4f6f8;color:{INK};font:15px/1.6 Arial,sans-serif}}
main{{max-width:1250px;margin:auto;padding:32px 24px}}h1{{font-size:30px;margin:0}}h2{{font-size:19px}}
.muted,small{{color:{MUTED}}}a{{color:#315e86}}.cards{{display:flex;gap:16px;margin:24px 0}}
.card{{flex:1;background:white;border:1px solid #dfe5eb;border-radius:10px;padding:20px}}strong{{font-size:28px}}
.figure,.viewer{{background:white;border-radius:10px;padding:16px;margin-top:24px}}.figure img{{width:100%;height:auto}}
.warning{{background:#fff2db;padding:12px 18px}}footer{{font-size:13px;color:{MUTED};margin-top:25px}}
button{{padding:8px 12px;background:white;border:1px solid #cbd4dd;border-radius:5px;cursor:pointer}}details{{margin:20px 0}}
@media(max-width:680px){{.cards{{display:block}}.card{{margin-bottom:12px}}main{{padding:16px 10px}}}}
@media print{{body{{background:white}}.viewer,nav{{display:none}}}}</style></head><body><main>
<nav><a href="index.html">All specimens</a> · <a href="{case['stem']}_depth.png" download>Publication PNG</a> · <a href="depth_summary.csv">Measurement CSV</a></nav>
<h1>{case['stem']} · Cavity depth and floor regularity</h1><p class="muted">Shared crown reference · original STL +Z axis</p>
<div class="cards">{''.join(cards)}</div>{warnings}
<section class="figure"><img alt="Eight panels: tooth, pulpal and gingival depth maps, STL section, gingival-minus-pulpal depth, two smoothness maps and residual distributions" src="data:image/jpeg;base64,{preview}"></section>
<section class="viewer"><h2>Interactive cavity depths</h2>
<button onclick="view('depth-viewer','overview')">Overview</button> <button onclick="view('depth-viewer','top')">Top</button>
<button onclick="view('depth-viewer','proximal')">Proximal</button>{plots[0]}</section>
<section class="viewer"><h2>Interactive floor smoothness</h2>
<button onclick="view('smoothness-viewer','overview')">Overview</button> <button onclick="view('smoothness-viewer','top')">Top</button>
<button onclick="view('smoothness-viewer','proximal')">Proximal</button>{plots[1]}</section>
<p class="muted">Drag to rotate; scroll to zoom. Click legend entries to hide the tooth or show reference markers. Color scales are shared across all specimens.</p>
<details open><summary><b>Measurement interpretation</b></summary>
<p>Both floor depths use the same crown reference and the original STL +Z direction. Statistics are computed from the floor samples with area weights. Signed depth values are retained.</p>
<p><b>Gingival − pulpal depth = proximal gingival median depth − occlusal pulpal median depth.</b> Panel E aligns the reference to zero to show this arithmetic. It is a schematic, not an anatomical section. The fitted crown reference can vary with position, so this difference of regional medians is not a point-to-point floor distance or the earlier local-pulpal-plane measurement.</p>
<p>Panel D is a real STL section through actual sampled points near each regional median. Its arrows show the measured sample depths; the headline values are regional medians. Original measured floor triangles are retained.</p>
<p>Smoothness maps show absolute perpendicular distance from each floor's own fitted plane. RMS uses area-weighted sample residuals; lower RMS means greater planarity. This measures floor regularity, not microscopic roughness. Surface colors interpolate vertex values; the distributions use area-weighted samples.</p>
<p>Student preparation geometry is retained, including signed negative depth differences. Review notes identify uncertain floor labels or reference support; they do not automatically establish a preparation defect.</p></details>
<footer>Exporter v{VERSION} · Source STL SHA-256: {case['stl_sha256']}</footer></main>
<script>const previewCameras = {cameras};
function view(id,name){{Plotly.relayout(id,{{'scene.camera':previewCameras[name]}});}}</script></body></html>'''
    (output/f"{case['stem']}_depth.html").write_text(page)


def write_index(output, records, vmax, vmin=0.):
    cards = "".join(f'<a href="{r["tooth"]}_depth.html"><img src="{r["tooth"]}_preview.jpg" '
                    f'alt="{r["tooth"]} cavity depth"><b>{r["tooth"]}</b>'
                    f'<span>{" · Review flag" if r["review_notes"] else ""}</span></a>' for r in records)
    (output / "index.html").write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cavity depth atlas</title>
<style>body{{font:16px/1.6 Arial;color:#253343;background:#f4f6f8;max-width:1400px;margin:40px auto;padding:0 24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:22px}}
a{{display:block;background:white;border:1px solid #dbe2e9;padding:16px;border-radius:10px;text-decoration:none;color:#253343}}
img{{width:100%;height:auto}}span{{color:#9a5c13}}</style></head><body>
<h1>Cavity depth and floor regularity atlas</h1><p>{len(records)} specimens · Shared {vmin:g}–{vmax:g} mm depth scale</p>
<p>Both floors use the same crown reference. Each figure includes smoothness and gingival-minus-pulpal median depth.</p>
<p><a href="depth_summary.csv">Download updated depth and smoothness measurements</a></p>
<div class="grid">{cards}</div><p>Estimated reference depths; anatomical floor selections require validation.</p></body></html>''')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stl-dir", type=Path, default=Path("STLFILES"))
    parser.add_argument("--measurements-dir", type=Path, default=Path("landmark_cav_con_comb_smooth_results"))
    parser.add_argument("--out-dir", type=Path, default=Path("publication_cavity_depth_results"))
    parser.add_argument("--cases", nargs="+", help="Optional specimen stems, e.g. O_1 O_35")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--color-max", type=float, help="Global depth limit; must include all measured vertices")
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error("Use at least 150 dpi (600 recommended for publication).")
    if args.out_dir.resolve() in [args.measurements_dir.resolve(), args.stl_dir.resolve()]:
        parser.error("Write to a separate output directory.")
    rows = {r["Tooth"]: r for r in csv.DictReader((args.measurements_dir/"summary.csv").open())}
    registrations = json.loads((args.measurements_dir/"registrations.json").read_text())
    source_manifest = json.loads((args.measurements_dir/"run_manifest.json").read_text())
    paths = sorted([p for p in args.stl_dir.iterdir() if p.suffix.lower() == ".stl"], key=lambda p: natural_key(p.stem))
    if args.cases:
        unknown = set(args.cases) - {p.stem for p in paths}
        if unknown:
            parser.error(f"Unknown STLs: {sorted(unknown)}")
        paths = [p for p in paths if p.stem in args.cases]
    if not paths:
        parser.error("No STL inputs found.")
    # Choose scales from displayed vertex depths for ALL cases, even for a subset.
    minimum, maximum, residual_maximum = 0., 0., 0.
    for stem in rows:
        with np.load(args.measurements_dir/f"{stem}_measurements.npz", allow_pickle=False) as saved:
            for region in REGIONS:
                vertices = saved[region+"_triangles"].reshape(-1, 3)
                depths = publication_depth(vertices, saved["reference_coefficients"])
                residual = abs((vertices-saved[region+"_plane_center"]) @ saved[region+"_plane_normal"])
                minimum = min(minimum, float(depths.min()))
                maximum = max(maximum, float(depths.max()))
                residual_maximum = max(residual_maximum, float(residual.max()))
    vmin = math.floor(minimum*2)/2
    vmax = args.color_max if args.color_max is not None else math.ceil(maximum*2)/2
    smooth_max = max(.1, math.ceil(residual_maximum*10)/10)
    if not np.isfinite(vmax) or vmax < maximum or vmax <= vmin:
        parser.error(f"Color scale must include measured maximum {maximum:.4f} mm.")
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    records, publication_rows = [], []
    protected = {str(p): sha256(p) for p in [args.measurements_dir/"summary.csv",
                 args.measurements_dir/"registrations.json", args.measurements_dir/"run_manifest.json"]}
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.stem}: validate, render PNG, export HTML", flush=True)
        case = load_case(path, args.measurements_dir, rows[path.stem], registrations[path.stem], source_manifest)
        prepare_publication_case(case)
        publication_rows.append(case["row"])
        png = write_png(case, output, args.dpi, vmax, vmin, smooth_max)
        write_html(case, output, vmax, vmin, smooth_max)
        with Image.open(png) as picture:
            size = list(picture.size)
        records.append({"tooth": path.stem, "png": png.name, "html": f"{path.stem}_depth.html",
                        "png_pixels": size, "dpi": args.dpi, "stl_sha256": case["stl_sha256"],
                        "measurement_sha256": case["npz_sha256"], "review_notes": case["notes"],
                        "max_statistic_error": case["max_statistic_error"],
                        "gingival_minus_pulpal_depth_mm": case["row"]["GingivalMinusPulpalDepth_mm"],
                        "display_rotation": case["basis"].tolist(), "display_center": case["center"].tolist(),
                        "arrow_depth_mm": {r: case["representative"][r]["depth_mm"] for r in REGIONS}})
        if sha256(path) != case["stl_sha256"] or sha256(args.measurements_dir/f"{path.stem}_measurements.npz") != case["npz_sha256"]:
            raise RuntimeError("Source inputs changed during export")
        del case
    for path, digest in protected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Measurement input changed: {path}")
    write_index(output, records, vmax, vmin)
    manifest = {"exporter_version": VERSION, "command": sys.argv,
                "script_sha256": sha256(__file__), "source_measurements": str(args.measurements_dir.resolve()),
                "color_scale_mm": [vmin, vmax], "colormap": CMAP, "input_files_unchanged": True,
                "smoothness_color_scale_mm": [0., smooth_max], "smoothness_colormap": SMOOTH_CMAP,
                "reference": "shared crown reference for both floors",
                "relative_depth_formula": "ProximalDepthMedian_mm - OcclusalDepthMedian_mm",
                "checks": ["STL provenance hashes", "floor triangles on original STL", "disjoint floor regions",
                           "source depth formula reproduces saved samples", "source CSV statistics reproduced from saved geometry",
                           "displayed depths use the shared reference", "smoothness residuals and RMS unchanged",
                           "relative depth equals difference of regional medians",
                           "no color clipping", "PNG dimensions", "unchanged measurement inputs"],
                "cases": records}
    (output/"render_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # This CSV describes the NEW figures; original source values stay in the input folder.
    with (output/"depth_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(publication_rows[0]))
        writer.writeheader()
        writer.writerows(publication_rows)
    (output/"README.md").write_text(
        "# Publication cavity-depth figures\n\n"
        f"{len(records)} specimens. Open `index.html` to browse; every `*_depth.html` includes "
        "an embedded figure preview and standalone, offline interactive 3D viewer. "
        f"Full-resolution PNGs are {args.dpi} dpi, 10.5 × 9.8 inches.\n\n"
        "A: original tooth with measured floor surfaces. B/C: isolated floor maps; "
        "different magnifications, with orthographic millimeter scale bars. "
        "D: real vertical STL section through two actual sampled points near the "
        "regional area-weighted medians; arrows show those point depths. "
        "E: gingival-minus-pulpal median depth, drawn as a reference-normalized schematic. "
        "F/G: absolute fitted-plane residual maps for each floor. H: area-weighted residual distributions.\n\n"
        "Both floor depths use the same crown reference and the original STL +Z direction. "
        "GingivalMinusPulpalDepth_mm = ProximalDepthMedian_mm − OcclusalDepthMedian_mm. "
        "This is a difference of regional area-weighted medians, not a local pulpal-plane "
        "distance or a point-to-point floor separation. Negative differences are retained.\n\n"
        "Smoothness means area-weighted perpendicular residual RMS from each floor's "
        "own fitted plane (planarity, not microscopic roughness); these measurements are unchanged. "
        f"Shared depth color scale: {vmin:g}–{vmax:g} mm; absolute plane residual: 0–{smooth_max:g} mm. "
        "Surface colors interpolate vertex values; statistics use area-weighted samples. Surfaces are not decimated.\n\n"
        f"Source: `{args.measurements_dir}`. Estimates and anatomical "
        "labels require independent validation. High-tilt (>45°) or small (<1 mm²) "
        "floor selections are visibly flagged; see each HTML and the manifest.\n\n"
        "`depth_summary.csv` contains the depth and floor regularity measurements used in the figures. "
        "Original inputs and existing CQS results are preserved. "
        "`render_manifest.json` "
        "records input hashes, numeric checks, display rotations, and arrow values. "
        "Regenerate with `python3 publication_cavity_depth.py` from the repository root.\n")
    print(f"Finished {len(records)} specimens: {output/'index.html'}", flush=True)


if __name__ == "__main__":
    main()
