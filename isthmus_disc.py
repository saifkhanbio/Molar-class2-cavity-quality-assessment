#!/usr/bin/env python3
"""Locate isthmuses and labeled fallback estimates in predicted cavity masks.

Run from this repository: python3 isthmus_disc.py
Outputs: one CSV row per mask, detailed JSON, and diagnostic PNG figures.
Widths and coordinates use the mask pixel grid. Millimeters are optional:
  python3 isthmus_disc.py --pixel-size-mm VERIFIED_MASK_PIXEL_SIZE

The detector uses connected centerlines, transverse boundary intersections,
width valleys at several smoothing scales, contour concavity, and cut support.
If no standard neck is found, it tries relaxed valleys, a valid cusp crossing,
then a balanced interior section to report one low-confidence fallback estimate.
Use --no-fallback for conservative detection only. Unmeasurable masks stay empty.
Scores are heuristic evidence scores, not probabilities or clinical validation.
The middle panel shows a green cavity on white with red isthmus lines only.
Optional cusp guidance is internal; eligible intercuspal lines appear only on
the tooth panel. Fallback counts and methods are exported separately from standard
geometric detections, even though both are included in the isthmuses list.

Image mapping defaults to equal-sized, already registered images only. Use
--image-mapping resize ONLY if full-frame resizing is the known mask transform.
Mapping does not undo prior geometric distortion; measurement is in mask space.
An isotropic mm/pixel value is unsuitable for anisotropically distorted masks.

Dependencies: numpy, scipy, scikit-image, networkx, Pillow, matplotlib.
Method references (this script is a proposed combination, not a reproduction):
  https://www.solism.ca/bib/solisetal2012skeleton.pdf
  https://arxiv.org/abs/1409.2104
  https://avrithis.net/data/pub/pdf/journ/J27.cviu18.cuts.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/isthmus-disc-matplotlib")

import networkx as nx
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.measure import find_contours
from skimage.morphology import h_maxima, medial_axis
from skimage.segmentation import watershed


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ALGORITHM_VERSION = "5"


@dataclass(frozen=True)
class Config:
    min_component_area: int = 12
    min_component_fraction: float = 0.02
    min_width_px: float = 3.0
    min_prominence_px: float = 0.8
    min_relative_prominence: float = 0.10
    min_cut_fraction: float = 0.06
    min_separation_fraction: float = 0.12
    max_paths: int = 6
    pixel_size_mm: float | None = None
    cusp_near_min_px: float = 2.0
    cusp_near_width_fraction: float = 0.25
    enable_fallback: bool = True


def read_mask(path: Path, threshold: float = 0.5) -> np.ndarray:
    """Read a binary/probability raster, including PNG values encoded as 0/1."""
    with Image.open(path) as image:
        if image.mode in ("1", "L", "I", "F", "I;16"):
            values = np.asarray(image, dtype=float)
        else:
            values = np.asarray(image.convert("L"), dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Mask must be a finite two-dimensional raster.")
    maximum = float(values.max(initial=0))
    scale = 1.0 if maximum <= 1 else (255.0 if maximum <= 255 else 65535.0)
    return values > threshold * scale


def build_graph(skeleton: np.ndarray, distance: np.ndarray) -> nx.Graph:
    """Build a rotation-independent graph without redundant diagonal triangles."""
    graph = nx.Graph()
    height, width = skeleton.shape
    for row, col in np.argwhere(skeleton):
        node = int(row * width + col)
        graph.add_node(node, rc=(int(row), int(col)), radius=float(distance[row, col]))
    for node, attributes in list(graph.nodes(data=True)):
        row, col = attributes["rc"]
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            rr, cc = row + dr, col + dc
            if not (0 <= rr < height and 0 <= cc < width and skeleton[rr, cc]):
                continue
            if dr and dc and (skeleton[row, cc] or skeleton[rr, col]):
                continue
            graph.add_edge(node, int(rr * width + cc), weight=math.hypot(dr, dc))
    return graph


def prune_spurs(graph: nx.Graph) -> nx.Graph:
    """Remove only short terminal chains, retaining junction connectivity."""
    graph = graph.copy()
    while True:
        remove = set()
        for end in sorted(n for n, degree in graph.degree() if degree == 1):
            chain, previous, current, length = [end], None, end, 0.0
            while True:
                neighbors = [n for n in graph[current] if n != previous]
                if not neighbors:
                    break
                following = neighbors[0]
                length += graph[current][following]["weight"]
                previous, current = current, following
                if graph.degree(current) != 2:
                    break
                chain.append(current)
            if graph.degree(current) >= 3:
                limit = max(2.5, 1.25 * graph.nodes[current]["radius"])
                if length < limit:
                    remove.update(chain)
        if not remove:
            return graph
        graph.remove_nodes_from(remove)


def centerline_paths(graph: nx.Graph, max_paths: int) -> list[list[int]]:
    """Retain paths through substantial branches; there is no vertical-axis rule."""
    options = []
    for nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(nodes)
        ends = sorted(n for n, degree in subgraph.degree() if degree == 1)
        if len(ends) < 2:
            # Closed loops do not have a unique end-to-end passage.
            continue
        if len(ends) > 24:
            # Bound work on severely irregular masks, retaining long branches.
            center = max(subgraph, key=lambda n: subgraph.nodes[n]["radius"])
            lengths = nx.single_source_dijkstra_path_length(subgraph, center, weight="weight")
            ends = sorted(ends, key=lambda n: (-lengths[n], n))[:24]
        for i, start in enumerate(ends):
            _, paths = nx.single_source_dijkstra(subgraph, start, weight="weight")
            for end in ends[i + 1:]:
                path = paths[end]
                score = sum(
                    graph[a][b]["weight"] *
                    ((graph.nodes[a]["radius"] + graph.nodes[b]["radius"]) / 2) ** 0.7
                    for a, b in zip(path, path[1:])
                )
                options.append((score, path))
    options.sort(key=lambda item: (-item[0], item[1][0], item[1][-1]))
    selected, covered = [], set()
    for _, path in options:
        edges = {tuple(sorted(pair)) for pair in zip(path, path[1:])}
        if selected and len(edges - covered) < max(5, 0.15 * len(edges)):
            continue
        selected.append(path)
        covered.update(edges)
        if len(selected) == max_paths:
            break
    return selected


def resample_curve(coords: np.ndarray, step: float = 1.0) -> np.ndarray:
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-6]
    coords, lengths = coords[keep], lengths[keep]
    if len(coords) < 2:
        return coords
    positions = np.linspace(0, lengths[-1], max(2, int(math.ceil(lengths[-1] / step)) + 1))
    return np.column_stack([np.interp(positions, lengths, coords[:, axis]) for axis in (0, 1)])


def cross_sections(coords: np.ndarray, signed_distance: np.ndarray, sigma: float):
    """Intersect both normal rays with the first boundary, at subpixel resolution."""
    smooth = ndi.gaussian_filter1d(coords, sigma=sigma, axis=0, mode="nearest")
    coords = resample_curve(smooth)
    tangent = np.gradient(ndi.gaussian_filter1d(coords, sigma=sigma, axis=0), axis=0)
    magnitude = np.linalg.norm(tangent, axis=1)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / np.maximum(magnitude[:, None], 1e-9)
    # Padding guarantees a background boundary even for masks touching an edge.
    reach = math.hypot(*signed_distance.shape)
    steps = np.arange(0.0, reach + 0.5, 0.5)
    endpoints = []
    for sign in (-1, 1):
        rays = coords[:, None, :] + sign * normal[:, None, :] * steps[None, :, None]
        values = ndi.map_coordinates(signed_distance, rays.reshape(-1, 2).T,
                                     order=1, mode="constant", cval=-1).reshape(len(coords), -1)
        outside = values <= 0
        first = outside.argmax(axis=1)
        valid = (values[:, 0] > 0) & outside.any(axis=1) & (first > 0) & (magnitude > 1e-6)
        preceding = np.maximum(first - 1, 0)
        rows = np.arange(len(coords))
        v0, v1 = values[rows, preceding], values[rows, first]
        alpha = np.divide(v0, v0 - v1, out=np.zeros_like(v0), where=np.abs(v0 - v1) > 1e-9)
        lengths = steps[preceding] + 0.5 * alpha
        end = coords + sign * normal * lengths[:, None]
        end[~valid] = np.nan
        endpoints.append(end)
    widths = np.linalg.norm(endpoints[1] - endpoints[0], axis=1)
    return coords, widths, endpoints


def contour_model(mask: np.ndarray):
    """Smooth a contour for a boundary-concavity check, not for width measurement."""
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    curve = resample_curve(max(contours, key=len))
    curve = ndi.gaussian_filter1d(curve[:-1], sigma=1.2, axis=0, mode="wrap")
    return curve, cKDTree(curve)


def boundary_concavity(model, endpoints: np.ndarray) -> list[float]:
    if model is None:
        return [0.0, 0.0]
    curve, tree = model
    width = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    window = max(3, int(round(width * 0.4)))
    strengths = []
    for i, endpoint in enumerate(endpoints):
        _, index = tree.query(endpoint)
        outward = (endpoint - endpoints[1 - i]) / max(width, 1e-9)
        chord_middle = (curve[(index - window) % len(curve)] +
                        curve[(index + window) % len(curve)]) / 2
        strengths.append(float(np.dot(chord_middle - curve[index], outward) / window))
    return strengths


def cut_support(mask: np.ndarray, endpoints: np.ndarray) -> float:
    """Fraction of the cavity on the smaller side of a complete transverse cut."""
    direction = endpoints[1] - endpoints[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    a = np.rint(endpoints[0] - 2 * direction).astype(int)
    b = np.rint(endpoints[1] + 2 * direction).astype(int)
    rr, cc = line(*a, *b)
    valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    cut = np.zeros_like(mask)
    cut[rr[valid], cc[valid]] = True
    cut = ndi.binary_dilation(cut, structure=np.ones((3, 3)))
    labels, count = ndi.label(mask & ~cut, structure=np.ones((3, 3)))
    if count < 2:
        return 0.0
    areas = np.sort(np.bincount(labels.ravel())[1:])[::-1]
    if areas[:2].sum() < 0.9 * areas.sum():
        return 0.0
    return float(areas[1] / mask.sum())


def analyze_cusps(cusp_mask: np.ndarray, cavity_mask: np.ndarray) -> dict:
    """Recover distinct cusp cores without forcing four regions or two pairs.

    Distance peaks seed watershed only when separated by a substantial saddle.
    Core centers estimate positions in a 2D mask; they are not 3D cusp tips.
    Pairing follows the whole cusp arrangement, with an elongated cavity as
    fallback for a single pair or an ambiguous cusp axis. Each core belongs to
    at most one pair, and two pairs must preserve longitudinal ordering.
    """
    anatomy = {"regions": [], "pairs": [], "warnings": [], "contours_rc": []}
    if cusp_mask.shape != cavity_mask.shape:
        anatomy["warnings"].append("cusp_mask_size_mismatch_ignored")
        return anatomy
    cusp_mask = np.asarray(cusp_mask, dtype=bool)
    components, count = ndi.label(cusp_mask, structure=np.ones((3, 3)))
    anatomy["input_components"] = int(count)
    if not count:
        anatomy["warnings"].append("empty_cusp_mask")
        return anatomy
    sizes = np.bincount(components.ravel())[1:]
    keep = np.flatnonzero(sizes >= max(20, 0.025 * sizes.max())) + 1
    cleaned = np.isin(components, keep)
    if not cleaned.any():
        anatomy["warnings"].append("no_substantial_cusp_regions")
        return anatomy
    distances = ndi.distance_transform_edt(np.pad(cleaned, 1))[1:-1, 1:-1]
    smooth = ndi.gaussian_filter(distances, sigma=0.8)
    height = max(1.0, 0.10 * float(smooth.max()))
    maxima = h_maxima(smooth, height) & cleaned
    markers, marker_count = ndi.label(maxima, structure=np.ones((3, 3)))
    regions = watershed(-smooth, markers, mask=cleaned)
    region_sizes = np.bincount(regions.ravel(), minlength=marker_count + 1)
    minimum_area = max(25, 0.03 * int(cleaned.sum()))
    for region_id in range(1, marker_count + 1):
        if region_sizes[region_id] < minimum_area:
            continue
        core = np.argwhere(markers == region_id)
        center = core[np.argmin(np.linalg.norm(core - core.mean(axis=0), axis=1))]
        if cavity_mask[tuple(center)]:
            anatomy["warnings"].append("cusp_core_overlaps_cavity_ignored")
            continue
        anatomy["regions"].append({"region_id": region_id, "center_rc": center.astype(float).tolist(),
                                   "radius_px": float(distances[tuple(center)]),
                                   "area_px": int(region_sizes[region_id])})
        anatomy["contours_rc"].extend(contour[::2].tolist() for contour in
                                      find_contours((regions == region_id).astype(float), 0.5))
    anatomy["peak_suppression_px"] = height
    anatomy["split_connected_regions"] = max(0, len(anatomy["regions"]) - len(keep))
    if len(anatomy["regions"]) < 2:
        anatomy["warnings"].append("insufficient_distinct_cusp_cores")
        return anatomy
    if len(anatomy["regions"]) > 4:
        anatomy["warnings"].append("more_than_four_cusp_cores_review_pairing")
    points = np.argwhere(cleaned).astype(float)
    origin = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points - origin, full_matrices=False)
    ambiguous_axis = singular[0] < 1.2 * singular[1]
    if (len(anatomy["regions"]) == 2 or ambiguous_axis) and cavity_mask.any():
        # Two opposing cores alone determine a transverse line, not the tooth's
        # longitudinal axis. An elongated cavity can disambiguate a single pair
        # or an almost square arrangement without relying on image orientation.
        cavity_points = np.argwhere(cavity_mask).astype(float)
        _, cavity_singular, cavity_vectors = np.linalg.svd(cavity_points - cavity_points.mean(axis=0),
                                                         full_matrices=False)
        if len(cavity_singular) == 2 and cavity_singular[0] > 1.8 * cavity_singular[1]:
            singular, vectors = cavity_singular, cavity_vectors
            anatomy["warnings"].append("cusp_pair_axis_estimated_from_cavity")
    if singular[0] < 1.2 * singular[1]:
        anatomy["warnings"].append("ambiguous_cusp_arrangement_axis")
        return anatomy
    axis = vectors[0]
    if axis[0] < 0:
        axis = -axis
    transverse = np.array([-axis[1], axis[0]])
    anatomy["axis_rc"] = axis.tolist()
    anatomy["origin_rc"] = origin.tolist()
    options = []
    for first, second in combinations(anatomy["regions"], 2):
        a, b = np.array(first["center_rc"]), np.array(second["center_rc"])
        lateral_a, lateral_b = np.dot(a - origin, transverse), np.dot(b - origin, transverse)
        if lateral_a * lateral_b >= 0:
            continue
        if min(abs(lateral_a), abs(lateral_b)) < 0.2 * min(first["radius_px"], second["radius_px"]):
            continue
        direction = b - a
        separation = float(np.linalg.norm(direction))
        cosine = abs(float(np.dot(direction, transverse))) / max(separation, 1e-9)
        if cosine < math.cos(math.radians(40)):
            continue
        if lateral_a > lateral_b:
            first, second, a, b = second, first, b, a
        quality = 0.65 * cosine + 0.35 * min(1.0, min(first["radius_px"], second["radius_px"]) / 5.0)
        options.append({"region_ids": [first["region_id"], second["region_id"]],
                        "centers_rc": [a.tolist(), b.tolist()], "midpoint_rc": ((a + b) / 2).tolist(),
                        "intercusp_distance_px": separation, "pairing_score": quality,
                        "axis_positions": [float(np.dot(a, axis)), float(np.dot(b, axis))]})
    choices = [[pair] for pair in options]
    for first, second in combinations(options, 2):
        if set(first["region_ids"]) & set(second["region_ids"]):
            continue
        left_order = first["axis_positions"][0] - second["axis_positions"][0]
        right_order = first["axis_positions"][1] - second["axis_positions"][1]
        if left_order * right_order <= 0:
            continue
        choices.append([first, second])
    if not choices:
        anatomy["warnings"].append("no_reliable_opposing_cusp_pair")
        return anatomy
    chosen = max(choices, key=lambda choice: sum(pair["pairing_score"] for pair in choice))
    chosen.sort(key=lambda pair: float(np.dot(pair["midpoint_rc"], axis)))
    for index, pair in enumerate(chosen, 1):
        pair["pair_id"] = index
    anatomy["pairs"] = chosen
    return anatomy


def segment_distance(first, second):
    """Shortest Euclidean distance between two finite 2D segments, in pixels."""
    a, b = np.asarray(first, dtype=float)
    c, d = np.asarray(second, dtype=float)
    u, v = b - a, d - c

    def cross(x, y):
        return float(x[0] * y[1] - x[1] * y[0])

    denominator = cross(u, v)
    if abs(denominator) > 1e-12:
        t, s = cross(c - a, v) / denominator, cross(c - a, u) / denominator
        if 0 <= t <= 1 and 0 <= s <= 1:
            return 0.0

    def point_distance(point, start, end):
        direction = end - start
        squared = float(np.dot(direction, direction))
        t = np.clip(np.dot(point - start, direction) / squared, 0, 1) if squared else 0.0
        return float(np.linalg.norm(point - (start + t * direction)))

    return min(point_distance(a, c, d), point_distance(b, c, d),
               point_distance(c, a, b), point_distance(d, a, b))


def cusp_near_threshold(width_px, config):
    return max(config.cusp_near_min_px, config.cusp_near_width_fraction * width_px)


def cusp_support_at(endpoints, pairs, component_id, config):
    """Support a neck only near a valid cavity crossing in the same component."""
    best_score, pair_id = 0.0, None
    width = float(np.linalg.norm(np.asarray(endpoints)[1] - endpoints[0]))
    threshold = cusp_near_threshold(width, config)
    for pair in pairs:
        if pair["cavity_component_id"] != component_id:
            continue
        offset = segment_distance(endpoints, pair["cavity_endpoints_rc"])
        if offset <= threshold:
            score = pair["pairing_score"] * (1.0 - 0.5 * offset / max(threshold, 1e-9))
            if score > best_score:
                best_score, pair_id = score, pair["pair_id"]
    return float(best_score), pair_id


def cusp_guided_sites(mask: np.ndarray, pairs: list[dict], config: Config) -> list[dict]:
    """Measure where each estimated intercusp line crosses the cavity.

    These are anatomical estimates, not evidence of a local width minimum.
    A missing, ambiguous, terminal, or sub-resolution crossing is not reported
    as a usable site. No connection is invented across disconnected mask pieces.
    """
    signed = ndi.distance_transform_edt(np.pad(mask, 1)) - ndi.distance_transform_edt(np.pad(~mask, 1, constant_values=True))
    labels, _ = ndi.label(mask, structure=np.ones((3, 3)))
    sites = []
    for pair in pairs:
        site = {"pair_id": pair["pair_id"], "kind": "cusp_guided_estimate",
                "measurement_direction": "estimated_intercusp_line", "status": "no_cavity_between_cusps",
                "width_px": None, "width_mm": None, "associated_isthmus_id": None}
        a, b = np.asarray(pair["centers_rc"])
        length = float(np.linalg.norm(b - a))
        t = np.linspace(0.0, 1.0, max(2, int(math.ceil(length / 0.25)) + 1))
        coords = a + t[:, None] * (b - a)
        values = ndi.map_coordinates(signed, (coords + 1).T, order=1, mode="constant", cval=-1)
        intervals, count = ndi.label(values > 0)
        substantial = [np.flatnonzero(intervals == i) for i in range(1, count + 1)
                       if np.count_nonzero(intervals == i) * length / (len(t) - 1) >= 1.0]
        if len(substantial) != 1:
            if len(substantial) > 1:
                site["status"] = "ambiguous_multiple_cavity_crossings"
            sites.append(site)
            continue
        first, last = int(substantial[0][0]), int(substantial[0][-1])
        if first == 0 or last == len(t) - 1:
            site["status"] = "cusp_core_overlaps_cavity"
            sites.append(site)
            continue
        positions = []
        for low, high in ((first - 1, first), (last, last + 1)):
            fraction = -values[low] / (values[high] - values[low])
            positions.append(coords[low] + fraction * (coords[high] - coords[low]))
        endpoints = np.asarray(positions)
        center = endpoints.mean(axis=0)
        width = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        component_id = int(labels[tuple(np.rint(center).astype(int))])
        fraction = cut_support(labels == component_id, endpoints) if component_id else 0.0
        if width < config.min_width_px:
            site["status"] = "below_minimum_width"
        elif fraction < config.min_cut_fraction:
            site["status"] = "insufficient_cavity_on_both_sides"
        else:
            site.update(status="available", center_rc=center.tolist(), endpoints_rc=endpoints.tolist(),
                        width_px=width, width_mm=width * config.pixel_size_mm if config.pixel_size_mm else None,
                        component_id=component_id, cut_fraction=fraction, pairing_score=pair["pairing_score"])
        sites.append(site)
    return sites


def filter_intercuspal_sites(sites, isthmuses, config):
    """Keep near-neck crossings, or possible isthmuses only if none were found.

    Rejected measurements are retained under rejected_crossing in the JSON for
    auditing, but are neither drawn nor exported as usable measurement widths.
    """
    for site in sites:
        site["draw_intercuspal_line"] = False
        site["interpretation"] = "unavailable"
        if site["status"] != "available":
            continue
        if not isthmuses:
            site.update(draw_intercuspal_line=True, interpretation="possible_isthmus")
            continue
        matches = []
        for candidate in isthmuses:
            if candidate["component_id"] != site["component_id"]:
                continue
            distance = segment_distance(candidate["endpoints_rc"], site["endpoints_rc"])
            threshold = cusp_near_threshold(candidate["width_px"], config)
            matches.append((distance, threshold, candidate["isthmus_id"]))
        eligible = [match for match in matches if match[0] <= match[1]]
        if matches:
            distance, threshold, nearest_id = min(eligible or matches)
            site.update(nearest_isthmus_distance_px=distance, near_threshold_px=threshold,
                        nearest_isthmus_id=nearest_id)
        if eligible:
            matched = next(candidate for candidate in isthmuses if candidate["isthmus_id"] == nearest_id)
            interpretation = "near_fallback_isthmus" if matched.get("is_fallback") else "near_detected_isthmus"
            site.update(draw_intercuspal_line=True, interpretation=interpretation,
                        associated_isthmus_id=nearest_id)
        else:
            site["status"] = "dropped_not_near_detected_isthmus"
            site["rejected_crossing"] = {key: site.pop(key) for key in
                ("center_rc", "endpoints_rc", "width_px", "width_mm")}
            site.update(width_px=None, width_mm=None, associated_isthmus_id=None)
    return sites


def profile_candidates(profile, mask, contour, config, component_id, cusp_pairs, relaxed=False):
    """Find width valleys; the explicit fallback pass permits weaker evidence."""
    candidates = []
    profile["rejections"] = []

    def reject(reason, index=None):
        profile["rejections"].append({"reason": reason, "path_index": index})

    valid_labels, count = ndi.label(np.isfinite(profile["widths"]))
    for run in range(1, count + 1):
        indices = np.flatnonzero(valid_labels == run)
        if len(indices) < 12:
            reject("insufficient_centerline_length")
            continue
        widths = profile["widths"][indices]
        base_sigma = max(1.0, 0.08 * float(np.median(widths)), 0.01 * len(widths))
        curves = [ndi.gaussian_filter1d(widths, base_sigma * scale, mode="nearest")
                  for scale in (1.0, 2.0, 3.5)]
        peak_sets = [find_peaks(-curve, prominence=config.min_prominence_px,
                               width=(None, None), rel_height=0.5) for curve in curves]
        if not len(peak_sets[0][0]):
            reject("no_width_valley")
        for peak_number, peak in enumerate(peak_sets[0][0]):
            properties = peak_sets[0][1]
            # Prominence uses both shoulders of this valley. A fixed spatial
            # window can sit entirely inside a broad neck and report no valley.
            prominence = float(properties["prominences"][peak_number])
            width = float(curves[0][peak])
            original_index = int(indices[peak])
            if width < config.min_width_px:
                reject("below_minimum_width", original_index)
                continue
            margin = max(2.0 if relaxed else 3.0, 0.08 * len(widths), (0.35 if relaxed else 0.6) * width)
            if peak < margin or peak > len(widths) - 1 - margin:
                reject("terminal_taper_region", original_index)
                continue
            tolerance = max(3.0, 0.30 * width)
            low = properties["left_ips"][peak_number]
            high = properties["right_ips"][peak_number]
            persistence = 1
            for peaks, other_properties in peak_sets[1:]:
                # A broad valley's lowest sample can move under smoothing even
                # when the same valley is preserved. Match overlapping basins
                # as well as close minima; avoid merging two neighboring necks.
                overlap = np.maximum(0, np.minimum(high, other_properties["right_ips"]) -
                                     np.maximum(low, other_properties["left_ips"]))
                shorter_span = np.minimum(high - low, other_properties["widths"])
                same_basin = (overlap >= 0.5 * shorter_span) & (overlap > 0)
                nearby = np.abs(peaks - peak) <= tolerance
                persistence += int(np.any(same_basin | nearby))
            if persistence < (1 if relaxed else 2):
                reject("unstable_across_scales", original_index)
                continue
            relative = prominence / max(width, 1e-9)
            if prominence < config.min_prominence_px or relative < config.min_relative_prominence:
                reject("insufficient_relative_narrowing", original_index)
                continue
            # For a flat valley, report its representative middle and preserve
            # the near-minimum interval, rather than an arbitrary plateau edge.
            near_minimum = width + min(0.5, 0.10 * prominence)
            first = last = int(peak)
            while first > 0 and curves[0][first - 1] <= near_minimum:
                first -= 1
            while last + 1 < len(widths) and curves[0][last + 1] <= near_minimum:
                last += 1
            peak = (first + last) // 2
            index = int(indices[peak])
            endpoints = np.asarray([profile["endpoints"][0][index], profile["endpoints"][1][index]])
            support = cut_support(mask, endpoints)
            if support < config.min_cut_fraction:
                reject("insufficient_cavity_on_both_sides", index)
                continue
            concavities = boundary_concavity(contour, endpoints)
            # A straight-sided passage can acquire a false width valley from
            # oblique skeleton branches at its corners. Even the relaxed pass
            # needs a small amount of boundary narrowing before calling it a valley.
            if relaxed and max(concavities) < 0.003:
                reject("no_boundary_narrowing_for_relaxed_valley", index)
                continue
            # A gently varying asymmetric neck can have only one concave wall.
            if not relaxed and max(concavities) < 0.015 and relative < 0.20:
                reject("weak_boundary_and_width_evidence", index)
                continue
            center = endpoints.mean(axis=0)
            cusp_strength, cusp_pair_id = cusp_support_at(endpoints, cusp_pairs, component_id, config)
            cusp_support = cusp_pair_id is not None
            score = (0.35 * min(relative / 0.35, 1.0) + 0.25 * persistence / 3.0 +
                     0.20 * min(max(0, max(concavities)) / 0.15, 1.0) +
                     0.20 * min(support / 0.25, 1.0))
            score = min(1.0, score + 0.10 * cusp_strength)
            if score < (0.25 if relaxed else 0.45):
                reject("low_combined_evidence", index)
                continue
            candidates.append({
                "component_id": component_id, "path_id": profile["path_id"],
                "path_index": index, "path_length_px": float(len(profile["widths"]) - 1),
                "center_rc": center.tolist(), "endpoints_rc": endpoints.tolist(),
                "width_px": float(np.linalg.norm(endpoints[1] - endpoints[0])),
                "width_mm": None, "evidence_score": float(score),
                "prominence_px": prominence, "relative_prominence": relative,
                "scale_support": persistence, "cut_fraction": support,
                "boundary_concavity": concavities, "cusp_support": cusp_support,
                "cusp_support_strength": cusp_strength, "cusp_pair_id": cusp_pair_id,
                "near_minimum_interval_rc": profile["coords_rc"][[indices[first], indices[last]]].tolist(),
                "valley_span_px": float(high - low),
            })
    return candidates


def select_candidates(candidates, graphs, config):
    """Select the best compatible pair jointly, or the strongest single neck."""
    candidates = sorted(candidates, key=lambda c: (-c["evidence_score"], *c["center_rc"]))
    unique = []
    for candidate in candidates:
        if any(candidate["component_id"] == other["component_id"] and
               np.linalg.norm(np.array(candidate["center_rc"]) - other["center_rc"]) <
               max(3.0, 0.6 * min(candidate["width_px"], other["width_px"])) for other in unique):
            continue
        unique.append(candidate)
    # A bounded set is sufficient for selecting at most two supported locations.
    unique = unique[:40]
    for candidate in unique:
        graph = graphs[candidate["component_id"]]
        candidate["graph_node"] = min(graph, key=lambda n: np.linalg.norm(
            np.array(graph.nodes[n]["rc"]) - candidate["center_rc"]))
    best_pair, best_score = None, -1.0
    for i, first in enumerate(unique):
        for second in unique[i + 1:]:
            if first["component_id"] != second["component_id"]:
                compatible = True
            else:
                graph = graphs[first["component_id"]]
                try:
                    path = nx.shortest_path(graph, first["graph_node"], second["graph_node"], weight="weight")
                except nx.NetworkXNoPath:
                    continue
                distance = sum(graph[a][b]["weight"] for a, b in zip(path, path[1:]))
                separation = max(4.0, 0.8 * max(first["width_px"], second["width_px"]),
                                 config.min_separation_fraction * min(first["path_length_px"], second["path_length_px"]))
                radii = [graph.nodes[n]["radius"] for n in path]
                # Two points in one flat trough are one isthmus, not a pair.
                recovery = max(radii, default=0) - max(radii[0], radii[-1])
                compatible = distance >= separation and recovery >= max(0.5, 0.05 * max(radii))
            score = first["evidence_score"] + second["evidence_score"]
            if compatible and score > best_score:
                best_pair, best_score = [first, second], score
    selected = best_pair if best_pair else unique[:1]
    # Deterministic spatial labels, not mesial/distal anatomical assignments.
    selected.sort(key=lambda c: tuple(c["center_rc"]))
    for i, candidate in enumerate(selected, 1):
        candidate["isthmus_id"] = i
        if config.pixel_size_mm is not None:
            candidate["width_mm"] = candidate["width_px"] * config.pixel_size_mm
    return selected, unique


def fallback_isthmus(profiles, labels, kept, sites, config):
    """Select one auditable estimate after the standard detector finds nothing.

    Try relaxed valleys, valid cusp crossings, then a nonterminal interior cut.
    Empty/noise masks and shapes without usable cross-sections remain unresolved.
    """
    audit = []
    if not kept:
        return None, [{"stage": "fallback", "outcome": "no_substantial_cavity"}]
    relaxed_config = replace(config, min_width_px=min(config.min_width_px, 2.0),
                             min_prominence_px=min(config.min_prominence_px, 0.3),
                             min_relative_prominence=min(config.min_relative_prominence, 0.04),
                             min_cut_fraction=min(config.min_cut_fraction, 0.04))
    largest = max(kept, key=lambda component_id: np.count_nonzero(labels == component_id))
    component = labels == largest
    contour = contour_model(component)
    eligible_profiles = [profile for profile in profiles if profile["component_id"] == largest]
    valleys, rejections = [], Counter()
    for profile in eligible_profiles:
        work = dict(profile)
        valleys.extend(profile_candidates(work, component, contour, relaxed_config, largest, [], relaxed=True))
        rejections.update(rejection["reason"] for rejection in work["rejections"])
    audit.append({"stage": "relaxed_valley", "candidate_count": len(valleys),
                  "thresholds": {"min_width_px": relaxed_config.min_width_px,
                                 "min_prominence_px": relaxed_config.min_prominence_px,
                                 "min_relative_prominence": relaxed_config.min_relative_prominence,
                                 "min_cut_fraction": relaxed_config.min_cut_fraction,
                                 "minimum_scale_support": 1, "minimum_score": 0.25,
                                 "minimum_boundary_concavity": 0.003}, "rejections": dict(rejections)})
    if valleys:
        candidate = max(valleys, key=lambda item: item["evidence_score"])
        candidate["detection_method"] = "fallback_relaxed_valley"
    else:
        crossings = [site for site in sites if site["status"] == "available"
                     and site["component_id"] == largest]
        audit.append({"stage": "cusp_crossing", "candidate_count": len(crossings),
                      "thresholds": {"min_width_px": config.min_width_px,
                                     "min_cut_fraction": config.min_cut_fraction}})
        if crossings:
            site = max(crossings, key=lambda item: (item["pairing_score"], item["cut_fraction"]))
            candidate = {key: site[key] for key in ("center_rc", "endpoints_rc", "width_px",
                         "width_mm", "component_id", "cut_fraction")}
            candidate.update(detection_method="fallback_cusp_crossing", cusp_pair_id=site["pair_id"],
                             cusp_support=True, cusp_support_strength=site["pairing_score"],
                             evidence_score=None, path_id=None, path_index=None)
        else:
            cuts = []
            for profile in eligible_profiles:
                widths = np.asarray(profile["widths"])
                count = len(widths)
                if count < 12:
                    continue
                low, high = int(math.ceil(0.20 * (count - 1))), int(math.floor(0.80 * (count - 1)))
                indices = [index for index in range(low, high + 1)
                           if np.isfinite(widths[index]) and widths[index] >= relaxed_config.min_width_px]
                reference_width = max((widths[index] for index in indices), default=1.0)
                for index in indices:
                    endpoints = np.asarray([profile["endpoints"][0][index], profile["endpoints"][1][index]])
                    fraction = cut_support(component, endpoints)
                    if fraction < 0.10:
                        continue
                    centrality = 1.0 - 2.0 * abs(index / (count - 1) - 0.5)
                    narrowness = max(0.0, 1.0 - widths[index] / reference_width)
                    rank = 0.45 * narrowness + 0.35 * min(1.0, fraction / 0.5) + 0.20 * centrality
                    cuts.append({"center_rc": endpoints.mean(axis=0).tolist(),
                                 "endpoints_rc": endpoints.tolist(), "width_px": float(widths[index]),
                                 "component_id": largest, "cut_fraction": fraction,
                                 "path_id": profile["path_id"], "path_index": index,
                                 "fallback_rank": float(rank), "evidence_score": None,
                                 "detection_method": "fallback_interior_section", "cusp_support": False,
                                 "cusp_support_strength": 0.0, "cusp_pair_id": None})
            audit.append({"stage": "interior_section", "candidate_count": len(cuts),
                          "thresholds": {"min_width_px": relaxed_config.min_width_px,
                                         "path_fraction_range": [0.20, 0.80], "min_cut_fraction": 0.10},
                          "ranking": "0.45*narrowness + 0.35*area_balance + 0.20*centrality"})
            if not cuts:
                audit.append({"stage": "fallback", "outcome": "no_valid_interior_cross_section"})
                return None, audit
            candidate = max(cuts, key=lambda item: item["fallback_rank"])
    candidate.update(isthmus_id=1, is_fallback=True, confidence_label="low",
                     width_mm=candidate["width_px"] * config.pixel_size_mm if config.pixel_size_mm else None)
    audit.append({"stage": "selection", "outcome": candidate["detection_method"],
                  "center_rc": candidate["center_rc"], "width_px": candidate["width_px"]})
    return candidate, audit


def detect_isthmuses(mask: np.ndarray, config: Config | None = None,
                     cusp_mask: np.ndarray | None = None) -> dict:
    """Pure analysis API; does not write files or require a matching tooth image."""
    config = config or Config()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) == 0:
        raise ValueError("Expected a nonempty 2D mask array.")
    result = {"status": "no_supported_isthmus", "warnings": [], "isthmuses": [],
              "candidates": [], "profiles": [], "cusp_midpoints_rc": [],
              "rejection_summary": {}, "review_reasons": [], "cusp_anatomy": {},
              "cusp_guided_sites": [], "fallback_used": False, "fallback_audit": []}
    if not mask.any():
        result.update(status="empty_mask", clean_mask=mask.copy())
        result["review_reasons"] = ["empty_mask"]
        return result
    labels, _ = ndi.label(mask, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]
    cutoff = max(config.min_component_area, config.min_component_fraction * int(areas.max()))
    kept = [i + 1 for i, area in enumerate(areas) if area >= cutoff]
    clean = np.isin(labels, kept)
    result["clean_mask"] = clean
    result["components_kept"] = len(kept)
    result["discarded_foreground_pixels"] = int(mask.sum() - clean.sum())
    if len(kept) > 1:
        result["warnings"].append("multiple_cavity_components")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        result["warnings"].append("cavity_touches_image_border")
    anatomy = analyze_cusps(cusp_mask, clean) if cusp_mask is not None else {}
    result["cusp_anatomy"] = anatomy
    result["warnings"].extend(anatomy.get("warnings", []))
    pairs = anatomy.get("pairs", [])
    result["cusp_midpoints_rc"] = [pair["midpoint_rc"] for pair in pairs]
    sites = cusp_guided_sites(clean, pairs, config) if pairs else []
    valid_sites = {}
    for site in sites:
        if site["status"] == "available":
            # Preserve original component IDs, including gaps left by removed noise.
            site["component_id"] = int(labels[tuple(np.rint(site["center_rc"]).astype(int))])
            valid_sites[site["pair_id"]] = site
    padded_pairs = [{**pair,
                     "cavity_endpoints_rc": (np.asarray(valid_sites[pair["pair_id"]]["endpoints_rc"]) + 3).tolist(),
                     "cavity_component_id": valid_sites[pair["pair_id"]]["component_id"]}
                    for pair in pairs if pair["pair_id"] in valid_sites]
    graphs, candidates = {}, []
    for component_id in kept:
        original_component = labels == component_id
        # Padding is internal only; coordinates are translated back before return.
        component = np.pad(original_component, 3)
        distance = ndi.distance_transform_edt(component)
        signed = distance - ndi.distance_transform_edt(~component)
        smooth = ndi.gaussian_filter(component.astype(float), sigma=0.65) >= 0.5
        if ndi.label(smooth, structure=np.ones((3, 3)))[1] != 1:
            smooth = component
        skeleton = medial_axis(smooth, rng=0)
        graph = prune_spurs(build_graph(skeleton, distance))
        graphs[component_id] = graph
        paths = centerline_paths(graph, config.max_paths)
        if not paths:
            result["warnings"].append(f"component_{component_id}_no_open_centerline")
            continue
        contour = contour_model(component)
        for path in paths:
            coords = np.array([graph.nodes[n]["rc"] for n in path], dtype=float)
            if len(coords) < 12:
                continue
            coords = resample_curve(coords)
            sigma = max(1.0, 0.18 * float(np.median([graph.nodes[n]["radius"] for n in path])))
            coords, widths, endpoints = cross_sections(coords, signed, sigma)
            profile = {"path_id": len(result["profiles"]), "component_id": component_id,
                       "coords_rc": coords, "widths": widths, "endpoints": endpoints}
            found = profile_candidates(profile, component, contour, config, component_id,
                                       padded_pairs)
            for candidate in found:
                candidate["center_rc"] = (np.array(candidate["center_rc"]) - 3).tolist()
                candidate["endpoints_rc"] = (np.array(candidate["endpoints_rc"]) - 3).tolist()
                candidate["near_minimum_interval_rc"] = (np.array(candidate["near_minimum_interval_rc"]) - 3).tolist()
            candidates.extend(found)
            profile["coords_rc"] -= 3
            profile["endpoints"] = [end - 3 for end in endpoints]
            result["profiles"].append(profile)
        for node in graph:
            graph.nodes[node]["rc"] = tuple(np.array(graph.nodes[node]["rc"]) - 3)
    selected, candidates = select_candidates(candidates, graphs, config)
    for candidate in candidates:
        candidate.update(detection_method="geometric_neck", is_fallback=False, confidence_label="standard")
    result["rejection_summary"] = dict(Counter(
        rejection["reason"] for profile in result["profiles"] for rejection in profile["rejections"]))
    if not selected:
        result["review_reasons"] = sorted(result["rejection_summary"]) or ["no_usable_centerline"]
        if config.enable_fallback:
            fallback, audit = fallback_isthmus(result["profiles"], labels, kept, sites, config)
            result["fallback_audit"] = audit
            if fallback is not None:
                selected = [fallback]
                result["fallback_used"] = True
                result["review_reasons"].append("low_confidence_fallback_requires_review")
    result.update(isthmuses=selected, candidates=candidates)
    result["cusp_guided_sites"] = filter_intercuspal_sites(sites, selected, config)
    if result["fallback_used"]:
        result["status"] = "one_isthmus_fallback"
    elif selected:
        result["status"] = "two_isthmuses" if len(selected) == 2 else "one_isthmus"
    return result


def matching_image(mask_path: Path, image_dir: Path) -> Path | None:
    stem = mask_path.stem[:-5] if mask_path.stem.endswith("_mask") else mask_path.stem
    matches = sorted(p for p in image_dir.glob(stem + ".*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return matches[0] if matches else None


def map_to_image(result, mask_shape, image_size, mapping):
    """Attach source coordinates only when the selected mapping is applicable."""
    if image_size is None:
        result["image_mapping"] = "missing_image"
        result["warnings"].append("missing_matching_tooth_image")
        return
    width, height = image_size
    same_size = (height, width) == tuple(mask_shape)
    if not same_size and mapping != "resize":
        result["image_mapping"] = "unverified_size_mismatch"
        result["warnings"].append("image_mask_transform_required")
        return
    result["image_mapping"] = "identity_assumed_registered" if same_size else "full_frame_resize_user_selected"
    scales = np.array([height / mask_shape[0], width / mask_shape[1]])
    result["image_scale_rc"] = scales.tolist()
    if not np.isclose(scales[0], scales[1]):
        result["warnings"].append("anisotropic_resize_mask_widths_are_not_source_widths")
    measurements = result["isthmuses"] + [site for site in result.get("cusp_guided_sites", [])
                                         if site["status"] == "available"]
    for candidate in measurements:
        endpoints = (np.array(candidate["endpoints_rc"]) + 0.5) * scales - 0.5
        center = (np.array(candidate["center_rc"]) + 0.5) * scales - 0.5
        candidate["image_center_rc"] = center.tolist()
        candidate["image_endpoints_rc"] = endpoints.tolist()
        candidate["width_image_px"] = float(np.linalg.norm(endpoints[1] - endpoints[0]))


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items() if key != "clean_mask"}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_figure(result, image_path, output_path):
    """Scientific diagnostic: original image, detected cuts, and width profiles."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/isthmus-disc-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    image = None
    if image_path is not None:
        with Image.open(image_path) as source:
            image = np.array(source.convert("RGB"))
        axes[0].imshow(image)
    else:
        axes[0].text(0.5, 0.5, "Matching tooth image unavailable", ha="center")
    axes[0].set_title("Tooth + mapped locations" if result.get("image_mapping", "").startswith(
        ("identity", "full_frame")) else "Tooth (mapping unverified)")
    mask_rgb = np.ones((*result["clean_mask"].shape, 3), dtype=float)
    mask_rgb[result["clean_mask"]] = np.array([46, 175, 80]) / 255.0
    axes[1].imshow(mask_rgb, interpolation="nearest")
    displayed_pairs = {site["pair_id"] for site in result.get("cusp_guided_sites", [])
                       if site.get("draw_intercuspal_line")}
    for pair in result.get("cusp_anatomy", {}).get("pairs", []):
        if pair["pair_id"] not in displayed_pairs:
            continue
        points = np.asarray(pair["centers_rc"])
        if "image_scale_rc" in result:
            mapped = (points + 0.5) * result["image_scale_rc"] - 0.5
            axes[0].plot(mapped[:, 1], mapped[:, 0], color="#4899dc", lw=0.8, ls=":", marker="+", ms=6)
    for profile in result["profiles"]:
        axes[2].plot(profile["widths"], lw=1, alpha=0.55, label=f"Path {profile['path_id'] + 1}")
    for i, candidate in enumerate(result["isthmuses"]):
        color = "#e00000"
        prefix = "F" if candidate.get("is_fallback") else "I"
        label = f"{prefix}{i + 1}: {candidate['width_px']:.2f} px"
        for axis, key in ((axes[1], "endpoints_rc"), (axes[0], "image_endpoints_rc")):
            if key not in candidate:
                continue
            ends = np.asarray(candidate[key])
            center = ends.mean(axis=0)
            axis.plot(ends[:, 1], ends[:, 0], color=color, lw=2, marker="o", ms=3,
                      ls="--" if candidate.get("is_fallback") else "-",
                      label=label if axis is axes[1] else None)
            axis.annotate(f"{prefix}{i + 1}", (center[1], center[0]), xytext=(5, 5),
                          textcoords="offset points", color=color, weight="bold")
        if candidate.get("path_index") is not None:
            axes[2].scatter(candidate["path_index"], candidate["width_px"], color=color,
                            s=45, zorder=5, label=label)
    for site in result.get("cusp_guided_sites", []):
        if not site.get("draw_intercuspal_line") or site.get("associated_isthmus_id") is not None:
            continue
        prefix = "P" if site["interpretation"] == "possible_isthmus" else "S"
        for axis, key in ((axes[0], "image_endpoints_rc"),):
            if key not in site:
                continue
            ends = np.asarray(site[key])
            center = ends.mean(axis=0)
            axis.plot(ends[:, 1], ends[:, 0], color="#e00000", lw=1.8, ls="--", marker="x", ms=4)
            axis.annotate(f"{prefix}{site['pair_id']}", (center[1], center[0]), xytext=(5, -10),
                          textcoords="offset points", color="#e00000", fontsize=8)
    if result["profiles"]:
        axes[2].legend(fontsize=7)
    if result["isthmuses"]:
        axes[1].legend(loc="lower left", fontsize=7)
    mask_label = "Fallback isthmus (low confidence)" if result.get("fallback_used") else result["status"].replace("_", " ")
    axes[1].set_title(f"Predicted cavity\n{mask_label}")
    axes[2].set(xlabel="Distance along each centerline (approximately px)", ylabel="Cross-sectional width (mask px)")
    axes[2].grid(alpha=0.2)
    for axis in axes[:2]:
        axis.axis("off")
    fig.suptitle(result["filename"], fontsize=11)
    if result["warnings"]:
        fig.supxlabel("; ".join(result["warnings"]), fontsize=7)
    elif result.get("review_reasons"):
        fig.supxlabel("Review: " + "; ".join(result["review_reasons"]), fontsize=7)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def csv_row(result):
    row = {"filename": result["filename"], "sample_image": result.get("sample_image", ""),
           "status": result["status"], "n_isthmuses": len(result["isthmuses"]),
           "n_geometric_isthmuses": sum(not item.get("is_fallback", False) for item in result["isthmuses"]),
           "n_fallback_isthmuses": sum(item.get("is_fallback", False) for item in result["isthmuses"]),
           "fallback_used": result.get("fallback_used", False),
           "n_cusp_pairs": len(result.get("cusp_anatomy", {}).get("pairs", [])),
           "n_cusp_guided_sites": sum(site["status"] == "available" for site in result.get("cusp_guided_sites", [])),
           "n_possible_isthmuses": sum(item.get("is_fallback", False) for item in result["isthmuses"]) +
                                   sum(site.get("interpretation") == "possible_isthmus"
                                       for site in result.get("cusp_guided_sites", [])),
           "image_mapping": result.get("image_mapping", ""),
           "review_reasons": ";".join(result.get("review_reasons", [])),
           "warnings": ";".join(result["warnings"]), "error": result.get("error", "")}
    for index in (1, 2):
        candidate = next((c for c in result["isthmuses"] if c["isthmus_id"] == index), {})
        prefix = f"isthmus{index}_"
        for key in ("width_px", "width_mm", "width_image_px", "evidence_score", "prominence_px",
                    "relative_prominence", "scale_support", "cut_fraction", "component_id", "cusp_support",
                    "cusp_pair_id", "cusp_support_strength", "detection_method", "is_fallback", "confidence_label"):
            row[prefix + key] = candidate.get(key)
        for key, label in (("center_rc", "center"), ("image_center_rc", "image_center")):
            for axis, value in zip(("row", "col"), candidate.get(key, [None, None])):
                row[prefix + label + "_" + axis] = value
        for key, label in (("endpoints_rc", "boundary"), ("image_endpoints_rc", "image_boundary")):
            for endpoint, coords in enumerate(candidate.get(key, [[None, None], [None, None]]), 1):
                for axis, value in zip(("row", "col"), coords):
                    row[f"{prefix}{label}{endpoint}_{axis}"] = value
    return row


def cusp_site_rows(result):
    """A separate table prevents counting an anatomical estimate as a neck."""
    rows = []
    pairs = {pair["pair_id"]: pair for pair in result.get("cusp_anatomy", {}).get("pairs", [])}
    sites = result.get("cusp_guided_sites", []) or [{"status": "no_usable_cusp_pair"}]
    for site in sites:
        pair = pairs.get(site.get("pair_id"), {})
        row = {"filename": result["filename"], "pair_id": site.get("pair_id"), "status": site["status"],
               "kind": "cusp_guided_estimate", "interpretation": site.get("interpretation", "unavailable"),
               "draw_intercuspal_line": site.get("draw_intercuspal_line", False),
               "associated_isthmus_id": site.get("associated_isthmus_id"),
               "nearest_isthmus_id": site.get("nearest_isthmus_id"),
               "nearest_isthmus_distance_px": site.get("nearest_isthmus_distance_px"),
               "near_threshold_px": site.get("near_threshold_px"),
               "width_px": site.get("width_px"), "width_mm": site.get("width_mm"),
               "width_image_px": site.get("width_image_px"), "pairing_score": pair.get("pairing_score"),
               "intercusp_distance_px": pair.get("intercusp_distance_px"),
               "warnings": ";".join(result.get("warnings", []))}
        for key, label in (("center_rc", "center"), ("image_center_rc", "image_center")):
            for axis, value in zip(("row", "col"), site.get(key, [None, None])):
                row[label + "_" + axis] = value
        for key, label in (("endpoints_rc", "boundary"), ("image_endpoints_rc", "image_boundary")):
            for index, coords in enumerate(site.get(key, [[None, None], [None, None]]), 1):
                for axis, value in zip(("row", "col"), coords):
                    row[f"{label}{index}_{axis}"] = value
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask-dir", type=Path, default=Path("pred_O_M_masks_folder"))
    parser.add_argument("--image-dir", type=Path, default=Path("samples_stu_O"))
    parser.add_argument("--output-dir", type=Path, default=Path("isthmus_disc_results"))
    parser.add_argument("--cusp-dir", type=Path,
                        help="Aligned cusp masks for spatial support and separately labeled measurement estimates.")
    parser.add_argument("--glob", default="*_mask.png", help="Input pattern, e.g. O_42_st_mask.png or *.png.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pixel-size-mm", type=float, help="Verified isotropic mm per MASK pixel; omitted by default.")
    parser.add_argument("--image-mapping", choices=("same-size", "resize"), default="same-size",
                        help="Select resize only for a known full-frame image-to-mask resize.")
    parser.add_argument("--min-width-px", type=float, default=3.0)
    parser.add_argument("--min-prominence-px", type=float, default=0.8)
    parser.add_argument("--min-relative-prominence", type=float, default=0.10)
    parser.add_argument("--min-cut-fraction", type=float, default=0.06)
    parser.add_argument("--min-separation-fraction", type=float, default=0.12)
    parser.add_argument("--cusp-near-min-px", type=float, default=2.0,
                        help="Minimum near-neck tolerance in mask pixels (default: 2).")
    parser.add_argument("--cusp-near-width-fraction", type=float, default=0.25,
                        help="Near-neck tolerance as a width fraction, floored by --cusp-near-min-px.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-fallback", dest="enable_fallback", action="store_false",
                        help="Disable lower-confidence estimates when no standard neck is detected.")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between zero and one.")
    for name in ("pixel_size_mm", "min_width_px", "min_prominence_px"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    for name in ("min_relative_prominence", "min_cut_fraction", "min_separation_fraction"):
        if not 0 < getattr(args, name) < 0.5:
            parser.error(f"--{name.replace('_', '-')} must be between zero and 0.5.")
    for name in ("cusp_near_min_px", "cusp_near_width_fraction"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    if not args.mask_dir.is_dir():
        parser.error(f"Mask directory does not exist: {args.mask_dir}")
    if args.cusp_dir is not None and not args.cusp_dir.is_dir():
        parser.error(f"Cusp directory does not exist: {args.cusp_dir}")
    if args.output_dir.resolve() in (args.mask_dir.resolve(), args.image_dir.resolve()):
        parser.error("Use a separate output directory.")
    return args


def main():
    args = parse_args()
    paths = sorted(p for p in args.mask_dir.glob(args.glob) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No mask images matched {args.mask_dir / args.glob}")
    config = Config(**{name: getattr(args, name) for name in (
        "pixel_size_mm", "min_width_px", "min_prominence_px", "min_relative_prominence",
        "min_cut_fraction", "min_separation_fraction", "cusp_near_min_px", "cusp_near_width_fraction",
        "enable_fallback")})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, details, site_rows, errors = [], [], [], 0
    for path in paths:
        try:
            mask = read_mask(path, args.threshold)
            cusp = None
            warnings = []
            if args.cusp_dir is not None:
                cusp_path = args.cusp_dir / path.name
                if cusp_path.exists():
                    cusp = read_mask(cusp_path, args.threshold)
                    if cusp.shape != mask.shape:
                        warnings.append("cusp_mask_size_mismatch_ignored")
                        cusp = None
                else:
                    warnings.append("missing_cusp_mask")
            result = detect_isthmuses(mask, config, cusp)
            result.update(filename=path.name, mask_shape=list(mask.shape))
            result["warnings"].extend(warnings)
            image_path = matching_image(path, args.image_dir)
            image_size = None
            if image_path is not None:
                with Image.open(image_path) as source:
                    image_size = source.size
                result["sample_image"] = str(image_path)
            map_to_image(result, mask.shape, image_size, args.image_mapping)
            if not args.no_plots:
                save_figure(result, image_path, args.output_dir / f"{path.stem}_isthmus.png")
        except Exception as error:
            errors += 1
            result = {"filename": path.name, "status": "error", "isthmuses": [],
                      "warnings": [], "error": f"{type(error).__name__}: {error}"}
        rows.append(csv_row(result))
        if args.cusp_dir is not None:
            site_rows.extend(cusp_site_rows(result))
        details.append(serializable(result))
        widths = ", ".join(f"{c['width_px']:.2f} px" for c in result["isthmuses"])
        print(f"{path.name}: {result['status']}" + (f" ({widths})" if widths else "") +
              (f" [{result['error']}]" if result.get("error") else ""))
    with (args.output_dir / "isthmus_locations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if site_rows:
        with (args.output_dir / "cusp_guided_sites.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(site_rows[0]))
            writer.writeheader()
            writer.writerows(site_rows)
    payload = {"algorithm_version": ALGORITHM_VERSION,
               "config": asdict(config), "arguments": {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items()}, "coordinate_convention": "zero-based row, column; pixel centers",
               "width_units": "width_px: mask pixels; width_image_px: mapped source pixels; width_mm: explicit mask calibration",
               "score_note": "Heuristic evidence score, not a probability. Expert localization validation required.",
               "fallback_note": "When standard detection returns zero, one low-confidence estimate may be selected from relaxed valleys, a cusp crossing, or a balanced interior section. n_isthmuses includes these estimates; n_geometric_isthmuses excludes them. See each result's fallback_audit for thresholds, attempted stages, and selected method.",
               "cusp_site_note": "Only valid cavity crossings near detected isthmuses are displayed. If no isthmus is detected in the mask, valid crossings are labeled possible_isthmus. Proximity uses finite cavity/neck segment distance within max(cusp_near_min_px, cusp_near_width_fraction * neck width). Estimates are not counted as geometric isthmuses.",
               "results": details}
    (args.output_dir / "isthmus_details.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} results to {args.output_dir} ({errors} processing errors).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
