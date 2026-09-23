"""Separate cusp lobes and build disjoint, noncrossing pairs with side checks.

Concavity cuts are used only for analysis; original prediction pixels are never
edited. Centers estimate mask cores, not verified anatomical cusp tips.
"""

from itertools import combinations
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage.draw import line
from skimage.measure import find_contours
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def concavity_defects(mask):
    """Find substantial inward deviations from consecutive convex-hull edges."""
    contours = find_contours(np.pad(mask, 1), 0.5)
    if not contours:
        return []
    contour = max(contours, key=len) - 1
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    if len(contour) < 4:
        return []
    hull = sorted(ConvexHull(contour).vertices)
    minimum = max(2.5, 0.08 * float(ndi.distance_transform_edt(np.pad(mask, 1)).max()))
    defects = []
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        ids = np.arange(start, end if end > start else end + len(contour)) % len(contour)
        a, b = contour[start], contour[end]
        direction = b - a
        points = contour[ids]
        t = np.clip((points - a) @ direction / (direction @ direction), 0, 1)
        inward = points - a - t[:, None] * direction
        distances = np.linalg.norm(inward, axis=1)
        deepest = int(np.argmax(distances))
        depth = float(distances[deepest])
        if depth >= minimum:
            defects.append((points[deepest], depth, inward[deepest] / depth))
    return defects


def split_touching_lobes(mask, depth=0):
    """Cut opposing notches only when both remaining lobes are substantial."""
    if depth >= 3 or mask.sum() < 100:
        return mask.copy(), []
    options = []
    for (a, da, va), (b, db, vb) in combinations(concavity_defects(mask), 2):
        if va @ vb > -0.25:
            continue
        cut = np.zeros_like(mask)
        rr, cc = line(*np.rint(a).astype(int), *np.rint(b).astype(int))
        # Contours can lie half a pixel outside the image border.
        rr, cc = np.clip(rr, 0, mask.shape[0] - 1), np.clip(cc, 0, mask.shape[1] - 1)
        cut[rr, cc] = True
        labels, _ = ndi.label(mask & ~ndi.binary_dilation(cut), np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if len(sizes) < 3:
            continue
        ids = np.argsort(sizes)[-2:]
        areas = sizes[ids]
        if areas.min() < max(50, 0.20 * mask.sum()) or areas.sum() < 0.88 * mask.sum():
            continue
        options.append((min(da, db), a, b, labels, ids, areas))
    if not options:
        return mask.copy(), []
    strength, a, b, labels, ids, areas = max(options, key=lambda option: option[0])
    separated = np.zeros_like(mask)
    audit = [{"endpoints_rc": [a.tolist(), b.tolist()], "concavity_depth_px": strength,
              "child_areas_px": areas.tolist(), "recursion_depth": depth}]
    for region_id in ids:
        part, child_audit = split_touching_lobes(labels == region_id, depth + 1)
        separated |= part
        audit.extend(child_audit)
    return separated, audit


ALLOWED_CUSP_PAIRS = ((1, 3), (2, 4))
ALTERNATE_CUSP_PAIRS = ((1, 4), (2, 3))


def segments_intersect(first, second):
    """Test finite segments, including collinear overlap and endpoint contact."""
    a, b = np.asarray(first, dtype=float)
    c, d = np.asarray(second, dtype=float)
    scale = max(1.0, float(np.linalg.norm(b - a)), float(np.linalg.norm(d - c)))
    tolerance = 1e-9 * scale

    def side(p, q, point):
        u, v = q - p, point - p
        cross = float(u[0] * v[1] - u[1] * v[0])
        return 0 if abs(cross) <= tolerance * scale else (1 if cross > 0 else -1)

    def on_segment(p, q, point):
        return bool(np.all(point >= np.minimum(p, q) - tolerance) and
                    np.all(point <= np.maximum(p, q) + tolerance))

    s1, s2, s3, s4 = side(a, b, c), side(a, b, d), side(c, d, a), side(c, d, b)
    return (s1 * s2 < 0 and s3 * s4 < 0 or
            s1 == 0 and on_segment(a, b, c) or s2 == 0 and on_segment(a, b, d) or
            s3 == 0 and on_segment(c, d, a) or s4 == 0 and on_segment(c, d, b))


def validate_pair_set(pairs):
    """Enforce one pair per cusp and reject touching or crossing pair lines."""
    ids = [i for pair in pairs for i in pair["region_ids"]]
    if any(len(pair["region_ids"]) != 2 for pair in pairs) or len(ids) != len(set(ids)):
        raise ValueError("Each cusp may participate in only one pair.")
    for first, second in combinations(pairs, 2):
        if segments_intersect(first["centers_rc"], second["centers_rc"]):
            raise ValueError("Intercuspal pair lines must not intersect.")


def pairing_plan(regions):
    """Switch both pairs only if the default lines intersect; keep core IDs."""
    centers = {region["region_id"]: np.asarray(region["center_rc"], dtype=float)
               for region in regions}
    plan = {"active_cusp_pairs": [list(ids) for ids in ALLOWED_CUSP_PAIRS],
            "pairing_switched_for_crossing": False, "pairing_degenerate": False}
    if not all(i in centers and centers[i].shape == (2,) and np.isfinite(centers[i]).all()
               for i in (1, 2, 3, 4)):
        return plan
    if segments_intersect([centers[1], centers[3]], [centers[2], centers[4]]):
        plan["pairing_switched_for_crossing"] = True
        plan["active_cusp_pairs"] = [list(ids) for ids in ALTERNATE_CUSP_PAIRS]
        if segments_intersect([centers[1], centers[4]], [centers[2], centers[3]]):
            # Degenerate/overlapping geometry cannot provide two safe lines.
            # Retain only the first alternative and flag the omitted connection.
            plan["active_cusp_pairs"] = [list(ALTERNATE_CUSP_PAIRS[0])]
            plan["pairing_degenerate"] = True
    return plan


def requested_pairs(regions):
    """Use default or crossing-corrected pairs without renumbering cores."""
    by_id = {region["region_id"]: region for region in regions}
    plan = pairing_plan(regions)
    pairs, used = [], set()
    for pair_id, ids in enumerate(plan["active_cusp_pairs"], 1):
        if not all(region_id in by_id for region_id in ids):
            continue
        centers = np.asarray([by_id[region_id]["center_rc"] for region_id in ids], dtype=float)
        distance = float(np.linalg.norm(centers[1] - centers[0]))
        if not np.isfinite(centers).all() or not math.isfinite(distance) or distance <= 0:
            continue
        pairs.append({"pair_id": pair_id, "region_ids": list(ids),
                      "pair_label": f"C{ids[0]}-C{ids[1]}", "centers_rc": centers.tolist(),
                      "midpoint_rc": centers.mean(axis=0).tolist(),
                      "intercusp_distance_px": distance, "pairing_score": 1.0,
                      "pairing_method": "crossing_corrected_labels" if plan["pairing_switched_for_crossing"] else "requested_cusp_labels",
                      "pairing_switched_for_crossing": plan["pairing_switched_for_crossing"],
                      "pair_recovered": False})
        used.update(ids)
    return pairs, [region["region_id"] for region in regions if region["region_id"] not in used]


def cavity_side_model(cavity_mask):
    """Estimate a curved midline from transverse slices of the main cavity.

    The principal direction defines longitudinal position, not image left/right.
    Median-filtered boundary midpoints follow a bent cavity. Beyond the observed
    cavity, extend the end position parallel to its principal axis and record
    the extrapolation; do not infer a new cavity region or alter the mask.
    """
    labels, count = ndi.label(cavity_mask, np.ones((3, 3)))
    if not count:
        return None
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    component = int(np.argmax(areas))
    points = np.argwhere(labels == component).astype(float)
    if len(points) < 3:
        return None
    origin = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    longitudinal = (points-origin) @ axis
    lateral = (points-origin) @ normal
    edges = np.linspace(longitudinal.min()-.01, longitudinal.max()+.01,
                        max(5, int(np.ptp(longitudinal)/3)+2))
    position, midpoint, widths = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (longitudinal >= lo) & (longitudinal < hi)
        if selected.sum() < 2:
            continue
        low, high = np.quantile(lateral[selected], [.05, .95])
        position.append(float(np.median(longitudinal[selected])))
        midpoint.append(float((low+high)/2))
        widths.append(float(high-low))
    if len(position) < 2:
        return None
    midpoint = ndi.gaussian_filter1d(ndi.median_filter(midpoint, size=3, mode="nearest"),
                                    1., mode="nearest")
    return {"origin_rc": origin.tolist(), "axis_rc": axis.tolist(),
            "normal_rc": normal.tolist(), "component_id": component,
            "position_px": position, "midpoint_px": midpoint.tolist(),
            "width_px": widths, "principal_axis_ratio": float(values[-1]/max(values[0], 1e-9)),
            "method": "main_cavity_transverse_boundary_midpoints",
            "extrapolation": "constant_lateral_endpoint_parallel_to_principal_axis"}


def largest_four_regions(regions):
    """Keep at most four cusp lobes by segmented pixel area for pairing.

    Retain every detection for display and audit. Equal areas are resolved by
    original detection ID, independently of input-list order. Four or fewer
    cores are never discarded merely because one is small.
    """
    if len(regions) > 4:
        if any(not math.isfinite(r.get("area_px", float("nan"))) or r["area_px"] <= 0
               for r in regions):
            raise ValueError("Positive cusp-region areas are required to select the largest four.")
        ranked = sorted(regions, key=lambda r: (-r["area_px"], r.get("source_region_id", r["region_id"])))
        retained = {id(r) for r in ranked[:4]}
    else:
        retained = {id(r) for r in regions}
    for region in regions:
        eligible = id(region) in retained
        region.update(pairing_eligible=eligible,
                      pairing_exclusion_reason="" if eligible else "outside_four_largest_cusp_areas")
    return ([r for r in regions if r["pairing_eligible"]],
            [r for r in regions if not r["pairing_eligible"]])


def opposite_side_pairs(regions, cavity_mask):
    """Choose up to two disjoint, noncrossing connections across cavity banks.

    No same-bank pair is admitted, even if its segment intersects a curved mask.
    Actual cavity intersections are preferred, then more transverse/shorter
    connections. Only the four largest segmented cores may compete for pairing.
    Labels C1/C2 belong to bank A and C3/C4 to bank B. Original detection IDs are
    retained separately. Bank names are geometric, not buccal/lingual diagnoses.
    """
    all_regions = regions
    regions, excluded = largest_four_regions(all_regions)
    model = cavity_side_model(cavity_mask)
    plan = {"active_cusp_pairs": [], "pairs": [], "unpaired_region_ids": [],
            "pairing_switched_for_crossing": False, "pairing_degenerate": False,
            "opposite_sides_required": True, "cavity_side_model": model,
            "pairing_audit": [], "cusp_labels_reassigned": False,
            "cusp_size_measure": "segmented_lobe_area_px",
            "n_pairing_cusp_centers": len(regions),
            "excluded_source_region_ids": [r["region_id"] for r in excluded],
            "excluded_region_ids": [r["region_id"] for r in excluded]}
    legacy = pairing_plan(regions)
    old_edges = {tuple(sorted(p["region_ids"])) for p in requested_pairs(regions)[0]}
    for region in all_regions:
        region["source_region_id"] = region["region_id"]
    if model is None:
        plan["unpaired_region_ids"] = [r["region_id"] for r in all_regions]
        plan["opposite_side_status"] = "no_usable_cavity_axis"
        return plan
    origin, axis, normal = [np.asarray(model[k]) for k in ("origin_rc", "axis_rc", "normal_rc")]
    position = np.asarray(model["position_px"])
    for region in all_regions:
        point = np.asarray(region["center_rc"])
        t = float((point-origin) @ axis)
        signed = float((point-origin) @ normal - np.interp(t, position, model["midpoint_px"]))
        # Half-pixel ambiguity band prevents an on-axis core becoming a bank.
        side = -1 if signed < -.5 else (1 if signed > .5 else 0)
        region.update(cavity_side=side, cavity_side_name={-1: "A", 1: "B", 0: "ambiguous"}[side],
                      cavity_side_distance_px=signed, cavity_longitudinal_position_px=t,
                      cavity_side_extrapolated=bool(t < position[0] or t > position[-1]))
    options = []
    for first, second in combinations(regions, 2):
        ids = [first["source_region_id"], second["source_region_id"]]
        audit = {"source_region_ids": ids,
                 "sides": [first["cavity_side"], second["cavity_side"]]}
        if first["cavity_side"] * second["cavity_side"] != -1:
            audit["status"] = "rejected_same_or_ambiguous_side"
            plan["pairing_audit"].append(audit)
            continue
        a, b = sorted([first, second], key=lambda r: r["cavity_side"])
        endpoints = np.array([a["center_rc"], b["center_rc"]])
        distance = float(np.linalg.norm(endpoints[1]-endpoints[0]))
        if not np.isfinite(endpoints).all() or distance <= 0:
            continue
        samples = np.linspace(endpoints[0], endpoints[1], max(3, int(np.ceil(distance*4))+1))
        inside = ndi.map_coordinates(np.asarray(cavity_mask, float), samples.T,
                                     order=0, mode="constant", cval=0) > .5
        overlap = float(inside.sum()*distance/(len(samples)-1))
        axial_gap = abs(a["cavity_longitudinal_position_px"]-b["cavity_longitudinal_position_px"])
        candidate = {"regions": (a, b), "centers_rc": endpoints.tolist(),
                     "distance": distance, "axial_fraction": axial_gap/distance,
                     "cavity_intersection_length_px": overlap,
                     "cavity_intersection_observed": overlap >= 1.0,
                     "source_region_ids": [a["source_region_id"], b["source_region_id"]]}
        options.append(candidate)
        audit.update(status="opposite_sides_candidate", cavity_intersection_length_px=overlap)
        plan["pairing_audit"].append(audit)
    combinations_to_rank = [(p,) for p in options]
    for a, b in combinations(options, 2):
        if set(a["source_region_ids"]) & set(b["source_region_ids"]):
            continue
        if not segments_intersect(a["centers_rc"], b["centers_rc"]):
            combinations_to_rank.append((a, b))

    def ranking(pairs):
        return (-len(pairs), -sum(p["cavity_intersection_observed"] for p in pairs),
                sum(p["axial_fraction"] for p in pairs), sum(p["distance"] for p in pairs),
                tuple(sorted(tuple(sorted(p["source_region_ids"])) for p in pairs)))

    chosen = min(combinations_to_rank, key=ranking) if combinations_to_rank else ()
    used = {i for p in chosen for i in p["source_region_ids"]}
    # Assign available canonical slots to selected cores first, then unused
    # same-bank cores. Extra/ambiguous centers stay visible as C5 and above.
    mapping = {}
    for side, start in [(-1, 1), (1, 3)]:
        bank = [r for r in regions if r["cavity_side"] == side]
        key = lambda r: (r["cavity_longitudinal_position_px"], r["source_region_id"])
        selected = sorted([r for r in bank if r["source_region_id"] in used], key=key)
        spare = sorted([r for r in bank if r["source_region_id"] not in used], key=key)
        canonical = sorted((selected+spare)[:2], key=key)
        for offset, region in enumerate(canonical):
            mapping[region["source_region_id"]] = start+offset
    for region in all_regions:
        if region["source_region_id"] not in mapping:
            mapping[region["source_region_id"]] = max([4, *mapping.values()])+1
        region["region_id"] = mapping[region["source_region_id"]]
    plan["cusp_labels_reassigned"] = any(i != j for i, j in mapping.items())
    plan["source_to_display_cusp_ids"] = {str(i): j for i, j in mapping.items()}
    chosen = sorted(chosen, key=lambda p: sum(r["cavity_longitudinal_position_px"] for r in p["regions"]))
    new_edges = {tuple(sorted(p["source_region_ids"])) for p in chosen}
    plan["pairing_changed_from_legacy"] = new_edges != old_edges
    plan["legacy_pairing_switched_for_crossing"] = legacy["pairing_switched_for_crossing"]
    for pair_id, item in enumerate(chosen, 1):
        ids = [r["region_id"] for r in item["regions"]]
        points = np.asarray(item["centers_rc"])
        pair = {"pair_id": pair_id, "region_ids": ids, "source_region_ids": item["source_region_ids"],
                "pair_label": f"C{ids[0]}-C{ids[1]}", "centers_rc": points.tolist(),
                "midpoint_rc": points.mean(axis=0).tolist(), "intercusp_distance_px": item["distance"],
                "pairing_score": 1.-item["axial_fraction"], "pairing_method": "opposite_cavity_banks_noncrossing",
                "opposite_sides_valid": True, "cavity_sides": [-1, 1],
                "cavity_side_distances_px": [r["cavity_side_distance_px"] for r in item["regions"]],
                "cavity_side_extrapolated": any(r["cavity_side_extrapolated"] for r in item["regions"]),
                "cavity_intersection_length_px": item["cavity_intersection_length_px"],
                "cavity_intersection_observed": item["cavity_intersection_observed"],
                "pair_recovered": False, "pairing_switched_for_crossing": False}
        plan["pairs"].append(pair)
        plan["active_cusp_pairs"].append(ids)
    alternate = {tuple(p) for p in plan["active_cusp_pairs"]} == set(ALTERNATE_CUSP_PAIRS)
    plan["pairing_switched_for_crossing"] = alternate
    for p in plan["pairs"]:
        p["pairing_switched_for_crossing"] = alternate
    plan["unpaired_region_ids"] = sorted(r["region_id"] for r in all_regions if r["source_region_id"] not in used)
    plan["excluded_region_ids"] = [r["region_id"] for r in excluded]
    plan["opposite_side_status"] = "available" if chosen else "no_opposite_side_pair"
    return plan


def complete_disjoint_pairs(regions, cavity_mask, previous_plan):
    """Complete a four-core plan using each center exactly once.

    Rank a first pair together with its forced complement, rejecting any
    intersecting combination. Prefer opposite-side pairs crossing the predicted
    cavity and preserve the existing first pair when equally supported.
    A same-side complement is explicitly marked for display only; it cannot
    supply a ratio. Original detection IDs and all centers are preserved.
    """
    regions = [r for r in regions if r.get("pairing_eligible", True)]
    if len(regions) != 4 or len(previous_plan.get("pairs", [])) != 1:
        return None
    model = previous_plan.get("cavity_side_model")
    if model is None:
        return None
    ordered = sorted(regions, key=lambda r: r["source_region_id"])
    origin, axis, normal = [np.asarray(model[k], float) for k in ("origin_rc", "axis_rc", "normal_rc")]
    centers = np.asarray([r["center_rc"] for r in ordered], float)
    if centers.shape != (4, 2) or not np.isfinite(centers).all():
        return None
    signed = (centers-origin) @ normal
    signs = np.where(signed < -.5, -1, np.where(signed > .5, 1, 0))
    # A saved isthmus can belong to a secondary cavity component. Preserve an
    # existing supported pair there; ratio matching checks the actual component.
    cavity = np.asarray(cavity_mask, float)
    candidates = {}
    previous_edges = {tuple(sorted(p["source_region_ids"])) for p in previous_plan["pairs"]}
    for indices in combinations(range(4), 2):
        endpoints = centers[list(indices)]
        length = float(np.linalg.norm(endpoints[1]-endpoints[0]))
        if length <= 0:
            continue
        samples = np.linspace(endpoints[0], endpoints[1], max(3, int(np.ceil(length*4))+1))
        inside = ndi.map_coordinates(cavity, samples.T, order=0, mode="constant", cval=0) > .5
        overlap = float(inside.sum()*length/(len(samples)-1))
        ids = [ordered[i]["source_region_id"] for i in indices]
        candidates[indices] = {
            "indices": indices, "ids": ids, "centers_rc": endpoints.tolist(),
            "length": length, "overlap": overlap, "intersects_cavity": overlap >= 1.,
            "opposite_sides_valid": bool(signs[indices[0]]*signs[indices[1]] == -1),
            "axial_fraction": float(abs(np.dot(endpoints[1]-endpoints[0], axis))/length),
            "previous_pair": tuple(ids) in previous_edges}
    options = []
    for indices, first in candidates.items():
        if not first["opposite_sides_valid"]:
            continue
        remaining = tuple(i for i in range(4) if i not in indices)
        second = candidates.get(remaining)
        if second is None or segments_intersect(first["centers_rc"], second["centers_rc"]):
            continue
        options.append((first, second))
    if not options:
        return None

    def ranking(option):
        first, second = option
        return (-sum(p["opposite_sides_valid"] for p in option),
                -int(first["intersects_cavity"]), -int(first["previous_pair"]),
                -int(second["intersects_cavity"]), first["axial_fraction"],
                sum(p["length"] for p in option), tuple(first["ids"]))

    chosen = min(options, key=ranking)
    position = np.asarray(model["position_px"])
    for i, region in enumerate(ordered):
        t = float((centers[i]-origin) @ axis)
        region.update(region_id=region["source_region_id"], cavity_side=int(signs[i]),
                      cavity_side_name={-1: "A", 1: "B", 0: "ambiguous"}[int(signs[i])],
                      cavity_side_distance_px=float(signed[i]), cavity_longitudinal_position_px=t,
                      cavity_side_extrapolated=bool(t < position[0] or t > position[-1]))
    straight_model = dict(model, midpoint_px=[0.]*len(position),
                          method="longest_principal_component_through_cavity_centroid",
                          extrapolation="unshifted_straight_principal_axis")
    pairs = []
    for pair_id, item in enumerate(chosen, 1):
        ids, indices = item["ids"], list(item["indices"])
        valid = item["opposite_sides_valid"]
        pairs.append({"pair_id": pair_id, "region_ids": ids, "source_region_ids": ids,
                      "pair_label": f"C{ids[0]}-C{ids[1]}", "centers_rc": item["centers_rc"],
                      "midpoint_rc": np.mean(item["centers_rc"], axis=0).tolist(),
                      "intercusp_distance_px": item["length"], "pairing_score": 1.-item["axial_fraction"],
                      "pairing_method": "first_pair_then_remaining_two", "opposite_sides_valid": valid,
                      "cavity_sides": signs[indices].tolist(),
                      "cavity_side_distances_px": signed[indices].tolist(),
                      "cavity_side_extrapolated": any(ordered[i]["cavity_side_extrapolated"] for i in indices),
                      "cavity_intersection_length_px": item["overlap"],
                      "cavity_intersection_observed": item["intersects_cavity"],
                      "pair_recovered": True, "pairing_switched_for_crossing": False,
                      "shared_cusp_id": None, "shared_cusp_pair": False,
                      "pair_role": "primary" if pair_id == 1 else "remaining_two",
                      "pair_completed_by_exclusion": pair_id == 2,
                      "side_rule_conflict": not valid,
                      "side_reference": "cavity_PC1_centroid_axis"})
    validate_pair_set(pairs)
    return {"pairs": pairs, "active_cusp_pairs": [p["region_ids"] for p in pairs],
            "unpaired_region_ids": [], "cavity_side_model": straight_model,
            "cusp_labels_reassigned": False,
            "source_to_display_cusp_ids": {str(r["region_id"]): r["region_id"] for r in ordered},
            "label_convention": "original_detection_ids_restored_for_disjoint_completion",
            "pairing_switched_for_crossing": False, "pairing_degenerate": False,
            "opposite_sides_required": all(p["opposite_sides_valid"] for p in pairs),
            "opposite_side_status": "complement_requires_review" if any(p["side_rule_conflict"] for p in pairs) else "available",
            "disjoint_pair_completion_used": True,
            "disjoint_pair_completion_audit": {
                "previous_pair_source_ids": [p["source_region_ids"] for p in previous_plan["pairs"]],
                "primary_pair_source_ids": pairs[0]["source_region_ids"],
                "remaining_pair_source_ids": pairs[1]["source_region_ids"],
                "signed_PC1_distances_px": signed.tolist(),
                "all_four_centers_retained": True, "one_pair_per_cusp": True,
                "opposite_sides_required_for_ratio": True,
                "noncrossing_definition": "no shared cusp, intersection, endpoint contact or overlap"}}


def analyze_cusp_geometry(cusp_mask, cavity_mask):
    anatomy = {"regions": [], "pairs": [], "warnings": [], "contours_rc": [], "split_audit": []}
    if cusp_mask.shape != cavity_mask.shape:
        raise ValueError("Cusp and cavity grids must match.")
    labels, count = ndi.label(np.asarray(cusp_mask, bool), np.ones((3, 3)))
    anatomy["input_components"] = int(count)
    if not count:
        anatomy["warnings"].append("empty_cusp_mask")
        return anatomy
    sizes = np.bincount(labels.ravel())[1:]
    # Retain small but substantial isolated lobes (e.g. O_39_st). The later
    # area and interior-radius checks reject speckles without a cusp core.
    keep = np.flatnonzero(sizes >= max(20, 0.01 * sizes.max())) + 1
    separated = np.zeros_like(cusp_mask, dtype=bool)
    for component in keep:
        part, audit = split_touching_lobes(labels == component)
        separated |= part
        anatomy["split_audit"].extend(dict(item, source_component=int(component)) for item in audit)
    distances = ndi.distance_transform_edt(np.pad(separated, 1))[1:-1, 1:-1]
    smooth = ndi.gaussian_filter(distances, 0.8)
    height = max(1.0, 0.10 * float(smooth.max()))
    markers, marker_count = ndi.label(h_maxima(smooth, height) & separated, np.ones((3, 3)))
    basins = watershed(-smooth, markers, mask=separated)
    minimum_area = max(25, 0.01 * int(separated.sum()))
    tensor = np.zeros((2, 2))
    for marker in range(1, marker_count + 1):
        points = np.argwhere(basins == marker)
        if len(points) < minimum_area:
            continue
        core = np.argwhere(markers == marker)
        center = core[np.argmin(np.linalg.norm(core - core.mean(axis=0), axis=1))]
        if cavity_mask[tuple(center)]:
            anatomy["warnings"].append("cusp_core_overlaps_cavity_ignored")
            continue
        radius = float(distances[tuple(center)])
        if radius < 2.5:
            continue
        values, vectors = np.linalg.eigh(np.cov(points.T))
        axis = vectors[:, -1]
        elongation = float((values[1] - values[0]) / max(values.sum(), 1e-9))
        tensor += elongation * np.outer(axis, axis)
        anatomy["regions"].append({"region_id": len(anatomy["regions"]) + 1,
                                   "center_rc": center.astype(float).tolist(), "radius_px": radius,
                                   "area_px": len(points), "major_axis_rc": axis.tolist(),
                                   "elongation": elongation})
        anatomy["contours_rc"].extend(contour[::2].tolist() for contour in
                                     find_contours((basins == marker).astype(float), 0.5))
    anatomy["peak_suppression_px"] = height
    anatomy["split_connected_regions"] = max(0, len(anatomy["regions"]) - len(keep))
    # Keep every detected center unchanged; labels now describe cavity banks.
    values, vectors = np.linalg.eigh(tensor)
    if values[-1] >= 1e-6 and values[-1] >= 1.2 * values[0]:
        axis = vectors[:, -1]
        if axis[0] < 0:
            axis = -axis
        anatomy["axis_rc"] = axis.tolist()
        anatomy["axis_method"] = "equal_lobe_elongation_weighted_orientation"
    if anatomy["regions"]:
        anatomy["origin_rc"] = np.mean([region["center_rc"] for region in anatomy["regions"]], axis=0).tolist()
    anatomy.update(opposite_side_pairs(anatomy["regions"], cavity_mask))
    anatomy["label_convention"] = "C1_C2_bank_A_C3_C4_bank_B_ordered_longitudinally_original_ids_retained"
    recovery = complete_disjoint_pairs(anatomy["regions"], cavity_mask, anatomy)
    if recovery is not None:
        anatomy.update(recovery)
        # Completion restores original IDs on the four retained cores. Give
        # excluded detections fresh display IDs if those would now collide.
        used_ids = {r["region_id"] for r in anatomy["regions"] if r["pairing_eligible"]}
        for region in anatomy["regions"]:
            if not region["pairing_eligible"]:
                region["region_id"] = max([4, *used_ids]) + 1
                used_ids.add(region["region_id"])
                anatomy["unpaired_region_ids"].append(region["region_id"])
                anatomy["source_to_display_cusp_ids"][str(region["source_region_id"])] = region["region_id"]
        anatomy["excluded_region_ids"] = [r["region_id"] for r in anatomy["regions"] if not r["pairing_eligible"]]
        if any(p["side_rule_conflict"] for p in anatomy["pairs"]):
            anatomy["warnings"].append("remaining_two_pair_fails_cavity_side_rule_display_only")
    validate_pair_set(anatomy["pairs"])
    anatomy["pairing_rule"] = "one_pair_per_cusp_remaining_two_form_second_pair"
    if anatomy["n_pairing_cusp_centers"] == 4 and len(anatomy["pairs"]) == 1:
        anatomy["warnings"].append("four_cusp_completion_unavailable_without_crossing")
    if not anatomy["pairs"]:
        anatomy["warnings"].append("no_valid_opposite_side_cusp_pair")
    elif len(anatomy["pairs"]) == 1 and len(anatomy["regions"]) >= 4:
        anatomy["warnings"].append("only_one_opposite_side_pair_supported_by_cavity_model")
    if set(anatomy["unpaired_region_ids"]) - set(anatomy["excluded_region_ids"]):
        anatomy["warnings"].append("unpaired_cusp_cores_after_opposite_side_matching")
    return anatomy
