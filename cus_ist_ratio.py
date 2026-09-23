#!/usr/bin/env python3
"""Match saved isthmus widths to opposing cusp distances in the same mask grid.

Primary ratio: isthmus width / intercuspal distance. The reciprocal is also
exported. Select the closest valid cavity-crossing/isthmus combination, never
the pair with the smallest ratio. Saved widths are not recomputed or replaced.

Run: python3 cus_ist_ratio.py
Default input: isthmus_disc_results_fallback/isthmus_details.json
If absent, reuse saved widths in cus_ist_ratio_results_largest_four/details.json.
Default output: cus_ist_ratio_results_cusp_rescue/
Use --pairing-policy near-only --no-ratio-fallback to withhold distant ratios.

Cusps in each ratio-eligible pair must lie on opposite sides of the cavity midline.
C1/C2 label one cavity bank, C3/C4 the other; original detection IDs are audited.
Keep up to two disjoint noncrossing pairs, including C1-C4/C2-C3 when needed.
Merged-cusp separation and center positions are preserved. Blank ratios use
explicitly flagged relaxed matching, which NEVER relaxes opposite-side validity.
Each cusp may participate in only one pair. For incomplete four-core cases,
a first pair fixes the second pair as the two remaining cusps. Reject crossing
combinations. If the complement fails the cavity-side rule, show and export it
as display-only; ratio matching still requires opposite-side validity.
If more than four cusps are detected, start with the four largest regions by
pixel area. If neither pair meets proximity, try replacing one paired cusp with
an excluded detection. Require opposite sides, a valid cavity crossing, no cusp
reuse and no crossing pair lines. Prefer strict near matches; a closer distant
replacement is permitted only with ratio fallback enabled and remains flagged.
Recovered four-cusp displays use outward labels with leader lines anchored to
their own centers; labels never determine pair membership or measured distances.
Use --no-ratio-fallback to retain the original crossing requirements.
The middle panel contains only green cavity pixels, white background and red
isthmus lines. The third panel shows green cusp regions, blue circles/centers
and blue distance lines on white. The tooth panel shows pair lines only when
nearby; the standalone cusp panel also shows geometric pairs without a valid
cavity crossing. A selected ratio fallback is dotted and labeled.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cus-ist-ratio-matplotlib")
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from cusp_geometry import (ALLOWED_CUSP_PAIRS, ALTERNATE_CUSP_PAIRS,
                           analyze_cusp_geometry, validate_pair_set)
from isthmus_disc import (Config, cusp_guided_sites, cusp_near_threshold,
                          map_to_image, matching_image, read_mask, segment_distance,
                          serializable)


VERSION = "12"
MEASUREMENT_FIELDS = [
    "isthmus_id", "cusp_pair_id", "cusp_pair_label", "cusp_region_ids", "detection_method", "is_fallback", "confidence_label",
    "isthmus_width_px", "intercuspal_distance_px", "ratio_isthmus_to_intercuspal",
    "ratio_intercuspal_to_isthmus", "isthmus_width_mm", "intercuspal_distance_mm",
    "match_distance_px", "center_distance_px", "near_threshold_px", "within_near_tolerance",
    "pairing_method", "pair_recovered", "local_axis_angle_deg",
    "ratio_is_fallback", "ratio_fallback_stage", "match_distance_basis",
    "original_crossing_status", "crossing_checks_relaxed",
    "relaxed_near_threshold_px", "proximity_relaxation_factor", "pairing_switched_for_crossing",
    "opposite_sides_valid", "cusp_source_region_ids", "cavity_side_distances_px",
    "cavity_side_extrapolated", "cavity_intersection_observed",
    "shared_cusp_pair", "shared_cusp_id", "side_reference",
    "pair_role", "pair_completed_by_exclusion", "side_rule_conflict",
    "uses_reinstated_cusp", "reinstated_cusp_ids",
]


def point_segment_distance(point, endpoints):
    a, b = np.asarray(endpoints, dtype=float)
    direction = b - a
    squared = float(np.dot(direction, direction))
    t = np.clip(np.dot(np.asarray(point) - a, direction) / squared, 0, 1) if squared else 0.0
    return float(np.linalg.norm(np.asarray(point) - a - t * direction))


def validate_isthmus(item, mask_shape):
    """Reject stale/inconsistent measurement units rather than silently fixing widths."""
    width = float(item["width_px"])
    endpoints = np.asarray(item["endpoints_rc"], dtype=float)
    center = np.asarray(item["center_rc"], dtype=float)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Isthmus width must be finite and positive.")
    if endpoints.shape != (2, 2) or center.shape != (2,) or not np.isfinite(endpoints).all() or not np.isfinite(center).all():
        raise ValueError("Invalid row/column measurement coordinates.")
    if not np.isclose(np.linalg.norm(endpoints[1] - endpoints[0]), width, rtol=1e-5, atol=1e-5):
        raise ValueError("Saved mask width does not match its boundary endpoints.")
    if not np.allclose(endpoints.mean(axis=0), center, rtol=0, atol=1e-4):
        raise ValueError("Saved isthmus center does not match its endpoints.")
    if (center < 0).any() or (center > np.asarray(mask_shape) - 1).any():
        raise ValueError("Isthmus center is outside the mask grid.")
    if (endpoints < -1).any() or (endpoints > np.asarray(mask_shape)).any():
        raise ValueError("Isthmus endpoints are outside the mask grid.")


def make_measurement(neck, pair, comparison_endpoints, config):
    """Measure saved width / actual center distance; never alter either length."""
    if not pair.get("opposite_sides_valid", False):
        return None
    centers = np.asarray(pair["centers_rc"], dtype=float)
    if centers.shape != (2, 2) or not np.isfinite(centers).all():
        return None
    distance = float(np.linalg.norm(centers[1] - centers[0]))
    width = float(neck["width_px"])
    if not math.isfinite(distance) or distance <= 0 or not math.isfinite(width) or width <= 0:
        return None
    gap = segment_distance(neck["endpoints_rc"], comparison_endpoints)
    threshold = cusp_near_threshold(width, config)
    return {"isthmus_id": neck["isthmus_id"], "cusp_pair_id": pair["pair_id"],
            "cusp_pair_label": "-".join(f"C{i}" for i in sorted(pair["region_ids"])),
            "cusp_region_ids": json.dumps(sorted(pair["region_ids"])),
            "cusp_source_region_ids": json.dumps(pair.get("source_region_ids", [])),
            "opposite_sides_valid": True,
            "cavity_side_distances_px": json.dumps(pair.get("cavity_side_distances_px", [])),
            "cavity_side_extrapolated": pair.get("cavity_side_extrapolated", False),
            "cavity_intersection_observed": pair.get("cavity_intersection_observed", False),
            "shared_cusp_pair": pair.get("shared_cusp_pair", False),
            "shared_cusp_id": pair.get("shared_cusp_id"),
            "side_reference": pair.get("side_reference", "cavity_midline"),
            "pair_role": pair.get("pair_role", "opposite_side_pair"),
            "pair_completed_by_exclusion": pair.get("pair_completed_by_exclusion", False),
            "side_rule_conflict": pair.get("side_rule_conflict", False),
            "uses_reinstated_cusp": pair.get("uses_reinstated_cusp", False),
            "reinstated_cusp_ids": json.dumps(pair.get("reinstated_cusp_ids", [])),
            "detection_method": neck.get("detection_method", "unspecified"),
            "is_fallback": neck.get("is_fallback", False),
            "confidence_label": neck.get("confidence_label", "unspecified"),
            "isthmus_width_px": width, "intercuspal_distance_px": distance,
            "ratio_isthmus_to_intercuspal": width / distance,
            "ratio_intercuspal_to_isthmus": distance / width,
            "isthmus_width_mm": width * config.pixel_size_mm if config.pixel_size_mm else None,
            "intercuspal_distance_mm": distance * config.pixel_size_mm if config.pixel_size_mm else None,
            "match_distance_px": gap,
            "center_distance_px": point_segment_distance(neck["center_rc"], comparison_endpoints),
            "near_threshold_px": threshold, "within_near_tolerance": gap <= threshold,
            "pairing_method": pair.get("pairing_method", "requested_cusp_labels"),
            "pair_recovered": pair.get("pair_recovered", False), "local_axis_angle_deg": None,
            "pairing_switched_for_crossing": pair.get("pairing_switched_for_crossing", False),
            "ratio_is_fallback": False, "ratio_fallback_stage": "none",
            "relaxed_near_threshold_px": threshold, "proximity_relaxation_factor": 1,
            "match_distance_basis": "cavity_crossing_segment",
            "original_crossing_status": "available", "crossing_checks_relaxed": False,
            "selected": False, "isthmus_center_rc": neck["center_rc"],
            "isthmus_endpoints_rc": neck["endpoints_rc"], "cusp_centers_rc": pair["centers_rc"],
            "cavity_crossing_endpoints_rc": comparison_endpoints}


def measurement_order(row):
    return (row["match_distance_px"], row["center_distance_px"],
            row["isthmus_id"], row["cusp_pair_id"])


def closest_ratio(isthmuses, pairs, crossings, config, policy="nearest", active_pairs=ALLOWED_CUSP_PAIRS):
    """Select the closest allowed combination passing the original checks."""
    if policy not in ("nearest", "near-only"):
        raise ValueError("Unknown pairing policy.")
    allowed = {tuple(ids) for ids in active_pairs}
    usable = {site["pair_id"]: site for site in crossings if site["status"] == "available"}
    candidates = []
    for neck in isthmuses:
        for pair in pairs:
            if tuple(sorted(pair.get("region_ids", []))) not in allowed:
                continue
            if "compatible_isthmus_ids" in pair and neck["isthmus_id"] not in pair["compatible_isthmus_ids"]:
                continue
            site = usable.get(pair["pair_id"])
            if site is None or site["component_id"] != neck["component_id"]:
                continue
            row = make_measurement(neck, pair, site["endpoints_rc"], config)
            if row is None:
                continue
            row["eligible"] = row["within_near_tolerance"] or policy == "nearest"
            candidates.append(row)
    candidates.sort(key=measurement_order)
    selected = next((row for row in candidates if row["eligible"]), None)
    if selected is not None:
        selected["selected"] = True
    return selected, candidates


def fallback_ratio(isthmuses, pairs, crossings, config, strict_candidates, active_pairs=ALLOWED_CUSP_PAIRS):
    """Relax proximity, then crossing checks, only when the strict ratio is blank.

    The final stage ranks the real finite cusp-center segments against saved
    width segments. It does not pretend these segments cross the cavity.
    Missing centers, invalid lengths and missing widths remain unmeasurable.
    """
    allowed = {tuple(ids) for ids in active_pairs}
    if strict_candidates:
        candidates = [dict(row) for row in strict_candidates
                      if tuple(json.loads(row["cusp_region_ids"])) in allowed and row.get("opposite_sides_valid", False)]
        stage = "proximity_relaxed"
    else:
        candidates = []
        stage = "cusp_segment_fallback"
        sites = {site["pair_id"]: site for site in crossings}
        for neck in isthmuses:
            for pair in pairs:
                if tuple(sorted(pair.get("region_ids", []))) not in allowed:
                    continue
                row = make_measurement(neck, pair, pair["centers_rc"], config)
                if row is None:
                    continue
                site = sites.get(pair["pair_id"], {})
                reason = site.get("status", "missing_cavity_crossing")
                if reason == "available" and site.get("component_id") != neck["component_id"]:
                    reason = "crossing_in_different_component"
                row.update(match_distance_basis="finite_cusp_center_segment",
                           original_crossing_status=reason, crossing_checks_relaxed=True,
                           cavity_crossing_endpoints_rc=None)
                candidates.append(row)
    candidates.sort(key=measurement_order)
    for row in candidates:
        # Doubling the search tolerance eventually admits every finite measured
        # gap. Retain the original near flag so distant matches remain visible.
        factor = 1
        while row["near_threshold_px"] * factor < row["match_distance_px"]:
            factor *= 2
        row.update(ratio_is_fallback=True, ratio_fallback_stage=stage,
                   relaxed_near_threshold_px=row["near_threshold_px"] * factor,
                   proximity_relaxation_factor=factor, eligible=True, selected=False)
    selected = candidates[0] if candidates else None
    if selected is not None:
        selected["selected"] = True
    return selected, candidates


def rescue_excluded_cusp(anatomy, isthmuses, mask, component_labels, sites,
                         strict_candidates, config, allow_relaxed=True):
    """Try an excluded cusp only after both original pairs miss proximity.

    Replace one endpoint of one pair, preserving the other complete pair and
    all display IDs. Accept only a valid crossing in the saved width component.
    Strict near matches take priority; distant matches must improve the measured
    gap and are separately marked as relaxed-proximity fallback.
    """
    excluded = [r for r in anatomy["regions"] if not r.get("pairing_eligible", True)]
    if (not excluded or not isthmuses or len(anatomy["pairs"]) != 2 or
            any(c["within_near_tolerance"] for c in strict_candidates)):
        return None, None
    model = anatomy.get("cavity_side_model")
    if model is None:
        return None, None
    baseline, _ = fallback_ratio(isthmuses, anatomy["pairs"], sites, config,
                                 strict_candidates, anatomy["active_cusp_pairs"])
    baseline_gap = baseline["match_distance_px"] if baseline else float("inf")
    audit = {"trigger": "no_original_pair_meets_proximity", "accepted": False,
             "baseline_gap_px": baseline_gap if math.isfinite(baseline_gap) else None,
             "trials": []}
    by_id = {r["region_id"]: r for r in anatomy["regions"]}
    origin, axis, normal = [np.asarray(model[k], float) for k in ("origin_rc", "axis_rc", "normal_rc")]
    options = []
    for spare in excluded:
        for old_pair in anatomy["pairs"]:
            for dropped in old_pair["region_ids"]:
                partner = next(i for i in old_pair["region_ids"] if i != dropped)
                ids = sorted([partner, spare["region_id"]])
                centers = np.asarray([by_id[i]["center_rc"] for i in ids], float)
                length = float(np.linalg.norm(centers[1]-centers[0]))
                t = (centers-origin) @ axis
                signed = (centers-origin) @ normal - np.interp(t, model["position_px"], model["midpoint_px"])
                sides = np.where(signed < -.5, -1, np.where(signed > .5, 1, 0))
                trial = {"region_ids": ids, "reinstated_cusp_id": spare["region_id"],
                         "displaced_cusp_id": dropped}
                audit["trials"].append(trial)
                if length <= 0 or not np.isfinite(length) or sides[0]*sides[1] != -1:
                    trial["status"] = "rejected_side_or_length"
                    continue
                replacement = {**deepcopy(old_pair), "region_ids": ids,
                    "source_region_ids": [by_id[i]["source_region_id"] for i in ids],
                    "pair_label": "-".join(f"C{i}" for i in ids), "centers_rc": centers.tolist(),
                    "midpoint_rc": centers.mean(axis=0).tolist(), "intercusp_distance_px": length,
                    "pairing_score": 1.-float(abs(t[1]-t[0])/length),
                    "pairing_method": "excluded_cusp_proximity_replacement", "pair_recovered": True,
                    "opposite_sides_valid": True, "cavity_sides": sides.tolist(),
                    "cavity_side_distances_px": signed.tolist(),
                    "cavity_side_extrapolated": bool(np.any(t < model["position_px"][0]) or
                                                       np.any(t > model["position_px"][-1])),
                    "pair_role": "proximity_replacement", "pair_completed_by_exclusion": False,
                    "side_rule_conflict": False, "shared_cusp_pair": False, "shared_cusp_id": None,
                    "uses_reinstated_cusp": True, "reinstated_cusp_ids": [spare["region_id"]]}
                pairs = [replacement if p["pair_id"] == old_pair["pair_id"] else deepcopy(p)
                         for p in anatomy["pairs"]]
                try:
                    validate_pair_set(pairs)
                except ValueError:
                    trial["status"] = "rejected_crossing_or_reused_cusp"
                    continue
                site = cusp_guided_sites(mask, [replacement], config)[0]
                trial["crossing_status"] = site["status"]
                if site["status"] != "available":
                    trial["status"] = "rejected_cavity_crossing"
                    continue
                site["component_id"] = int(component_labels[tuple(np.rint(site["center_rc"]).astype(int))])
                replacement.update(cavity_intersection_observed=True,
                                   cavity_intersection_length_px=site["width_px"])
                _, candidates = closest_ratio(isthmuses, [replacement], [site], config,
                                               "near-only", [ids])
                if not candidates:
                    trial["status"] = "rejected_isthmus_component"
                    continue
                candidate = min(candidates, key=measurement_order)
                near = candidate["within_near_tolerance"]
                trial.update(match_distance_px=candidate["match_distance_px"],
                             near_threshold_px=candidate["near_threshold_px"], within_near_tolerance=near)
                if not near and (not allow_relaxed or candidate["match_distance_px"] >= baseline_gap-1e-9):
                    trial["status"] = "rejected_proximity_or_no_improvement"
                    continue
                trial["status"] = "strict_near_candidate" if near else "closer_relaxed_candidate"
                options.append((candidate, pairs, site, spare["region_id"], dropped))
    if not options:
        audit["outcome"] = "no_qualifying_replacement"
        return None, audit
    chosen, pairs, site, reinstated, displaced = min(
        options, key=lambda option: (not option[0]["within_near_tolerance"],
                                     *measurement_order(option[0]), option[3], option[4]))
    new = deepcopy(anatomy)
    new.update(pairs=pairs, active_cusp_pairs=[p["region_ids"] for p in pairs],
               excluded_cusp_rescue_used=True, reinstated_cusp_ids=[reinstated],
               displaced_cusp_ids=[displaced],
               initial_excluded_region_ids=anatomy["excluded_region_ids"],
               initial_excluded_source_region_ids=anatomy["excluded_source_region_ids"])
    used = {i for pair in pairs for i in pair["region_ids"]}
    for region in new["regions"]:
        if region["region_id"] == reinstated:
            region.update(pairing_eligible=True, pairing_exclusion_reason="", reinstated_for_proximity=True)
        elif region["region_id"] == displaced:
            region.update(pairing_eligible=False, pairing_exclusion_reason="replaced_by_nearer_excluded_cusp")
    new["excluded_region_ids"] = [r["region_id"] for r in new["regions"] if not r["pairing_eligible"]]
    new["excluded_source_region_ids"] = [r["source_region_id"] for r in new["regions"] if not r["pairing_eligible"]]
    new["unpaired_region_ids"] = [r["region_id"] for r in new["regions"] if r["region_id"] not in used]
    audit.update(accepted=True, outcome="strict_proximity" if chosen["within_near_tolerance"] else "relaxed_proximity",
                 reinstated_cusp_id=reinstated, displaced_cusp_id=displaced,
                 new_pair=chosen["cusp_pair_label"], match_distance_px=chosen["match_distance_px"],
                 near_threshold_px=chosen["near_threshold_px"], within_near_tolerance=chosen["within_near_tolerance"])
    new["excluded_cusp_rescue_audit"] = audit
    new_sites = [site if old["pair_id"] == site["pair_id"] else deepcopy(old) for old in sites]
    return (new, new_sites), audit


def analyze_record(record, mask, cusp_mask, config, policy="nearest", enable_ratio_fallback=True):
    if tuple(record["mask_shape"]) != mask.shape or cusp_mask.shape != mask.shape:
        raise ValueError("Cavity, cusp and saved isthmus grids must have identical dimensions; no automatic resize.")
    result = {"filename": record["filename"], "status": "no_isthmus", "selected": None,
              "candidates": [], "isthmuses": record.get("isthmuses", []),
              "warnings": list(record.get("warnings", [])), "anatomy": {}, "crossings": []}
    result.update(pair_recovery_used=False, pair_recovery_audit=[])
    labels, _ = ndi.label(mask, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]
    cutoff = max(config.min_component_area, config.min_component_fraction * int(areas.max(initial=0)))
    kept = np.flatnonzero(areas >= cutoff) + 1
    clean = np.isin(labels, kept)
    result["clean_mask"] = clean
    padded = np.pad(clean, 1)
    signed = ndi.distance_transform_edt(padded) - ndi.distance_transform_edt(~padded)
    for neck in result["isthmuses"]:
        validate_isthmus(neck, mask.shape)
        center_id = int(labels[tuple(np.rint(neck["center_rc"]).astype(int))])
        if not center_id or center_id not in kept or center_id != neck["component_id"]:
            raise ValueError("Isthmus location/component does not match the current cavity mask; regenerate isthmus results.")
        endpoints = np.asarray(neck["endpoints_rc"])
        boundary_values = ndi.map_coordinates(signed, (endpoints + 1).T, order=1)
        samples = np.linspace(endpoints[0], endpoints[1], max(3, int(math.ceil(neck["width_px"] * 2)) + 1))
        inside_values = ndi.map_coordinates(signed, (samples + 1).T, order=1)
        if np.max(np.abs(boundary_values)) > 1.25 or np.min(inside_values) < -0.75:
            raise ValueError("Saved isthmus endpoints/segment no longer fit the cavity boundary; regenerate isthmus results.")
    anatomy = analyze_cusp_geometry(cusp_mask, clean)
    result["anatomy"] = anatomy
    if anatomy.get("disjoint_pair_completion_used"):
        result["pair_recovery_used"] = True
        result["pair_recovery_audit"] = [anatomy["disjoint_pair_completion_audit"]]
    result["warnings"] = sorted(set(result["warnings"] + anatomy["warnings"]))
    active_pairs = anatomy.get("active_cusp_pairs", ALLOWED_CUSP_PAIRS)
    sites = cusp_guided_sites(clean, anatomy["pairs"], config)
    for site in sites:
        if site["status"] == "available":
            site["component_id"] = int(labels[tuple(np.rint(site["center_rc"]).astype(int))])
    result["crossings"] = sites
    selected, candidates = closest_ratio(result["isthmuses"], anatomy["pairs"], sites, config, policy, active_pairs)
    result["baseline_pairing"] = {"pair_count": len(anatomy["pairs"]),
                                  "candidate_count": len(candidates),
                                  "warnings": list(anatomy["warnings"]),
                                  "crossing_statuses": [site["status"] for site in sites]}
    recovered, audit = rescue_excluded_cusp(anatomy, result["isthmuses"], clean, labels,
                                           sites, candidates, config, enable_ratio_fallback)
    if audit is not None:
        result["excluded_cusp_rescue_audit"] = audit
    if recovered is not None:
        anatomy, sites = recovered
        result.update(anatomy=anatomy, crossings=sites, pair_recovery_used=True)
        result["pair_recovery_audit"].append(audit)
        active_pairs = anatomy["active_cusp_pairs"]
        selected, candidates = closest_ratio(result["isthmuses"], anatomy["pairs"], sites,
                                             config, policy, active_pairs)
        if audit["outcome"] == "relaxed_proximity":
            # Keep the original near flag false; fallback records the expansion.
            selected = None
        result["warnings"].append("excluded_cusp_reinstated_for_closer_isthmus_match")
    result["ratio_fallback_used"] = False
    if selected is None and enable_ratio_fallback:
        selected, relaxed = fallback_ratio(result["isthmuses"], anatomy["pairs"], sites, config, candidates, active_pairs)
        if selected is not None:
            result["strict_candidates_before_fallback"] = candidates
            candidates = relaxed
            result["ratio_fallback_used"] = True
            result["warnings"].append("ratio_uses_relaxed_matching_fallback")
            if selected["crossing_checks_relaxed"]:
                result["warnings"].append("ratio_does_not_require_a_valid_cavity_crossing")
    result.update(selected=selected, candidates=candidates)
    if selected:
        review = bool(selected["ratio_is_fallback"] or selected["is_fallback"] or selected["pair_recovered"] or
                      selected["cavity_side_extrapolated"] or
                      not selected["within_near_tolerance"] or anatomy["warnings"])
        result["status"] = "ratio_requires_review" if review else "ratio_available"
        if selected["is_fallback"]:
            result["warnings"].append("ratio_uses_low_confidence_fallback_width")
        if not selected["within_near_tolerance"]:
            result["warnings"].append("selected_pair_is_far_from_isthmus")
        if selected["cavity_side_extrapolated"]:
            result["warnings"].append("cavity_side_assignment_extrapolated_beyond_observed_mask")
    elif not result["isthmuses"]:
        result["status"] = "no_isthmus"
    elif not anatomy["pairs"]:
        result["status"] = "no_valid_opposite_side_cusp_pair"
    elif not any(site["status"] == "available" for site in sites):
        result["status"] = "no_valid_cavity_crossing"
    elif not candidates:
        result["status"] = "no_crossing_in_isthmus_component"
    else:
        result["status"] = "no_pair_within_near_tolerance"
    return result


def measurement_row(measurement):
    row = {key: measurement.get(key) for key in MEASUREMENT_FIELDS}
    for key in ("isthmus_center_rc", "isthmus_endpoints_rc", "cusp_centers_rc", "cavity_crossing_endpoints_rc"):
        row[key] = json.dumps(measurement[key]) if measurement.get(key) is not None else ""
    return row


def summary_row(result):
    return {"filename": result["filename"], "status": result["status"],
            "n_isthmuses": len(result.get("isthmuses", [])),
            "n_cusp_centers": len(result.get("anatomy", {}).get("regions", [])),
            "n_pairing_cusp_centers": result.get("anatomy", {}).get("n_pairing_cusp_centers", 0),
            "n_excluded_cusp_centers": len(result.get("anatomy", {}).get("excluded_region_ids", [])),
            "excluded_cusp_ids": json.dumps(result.get("anatomy", {}).get("excluded_region_ids", [])),
            "excluded_cusp_rescue_used": result.get("anatomy", {}).get("excluded_cusp_rescue_used", False),
            "n_geometric_pairs": sum(not pair.get("pair_recovered", False)
                                     for pair in result.get("anatomy", {}).get("pairs", [])),
            "active_cusp_pairs": json.dumps(result.get("anatomy", {}).get("active_cusp_pairs", [])),
            "n_cusp_pairs": len(result.get("anatomy", {}).get("pairs", [])),
            "n_valid_combinations": len(result.get("candidates", [])),
            "pair_recovery_used": result.get("pair_recovery_used", False),
            "ratio_fallback_used": result.get("ratio_fallback_used", False),
            "n_recovered_pairs": sum(pair.get("pair_recovered", False)
                                     for pair in result.get("anatomy", {}).get("pairs", [])),
            **measurement_row(result.get("selected") or {}),
            "warnings": ";".join(result.get("warnings", [])), "error": result.get("error", "")}


def distance_rows(result):
    """Export both requested distances, including missing/rejected crossings."""
    pairs = {tuple(pair["region_ids"]): pair for pair in result.get("anatomy", {}).get("pairs", [])}
    sites = {site["pair_id"]: site for site in result.get("crossings", [])}
    selected = result.get("selected")
    rows = []
    active_pairs = result.get("anatomy", {}).get("active_cusp_pairs", ALLOWED_CUSP_PAIRS)
    for pair_id, ids in enumerate(active_pairs, 1):
        pair = pairs.get(tuple(ids))
        rows.append({"filename": result["filename"], "cusp_pair_id": pair_id,
                     "cusp_pair_label": f"C{ids[0]}-C{ids[1]}",
                     "cusp_region_ids": json.dumps(ids),
                     "pairing_switched_for_crossing": result.get("anatomy", {}).get("pairing_switched_for_crossing", False),
                     "opposite_sides_valid": pair.get("opposite_sides_valid", False) if pair else False,
                     "source_region_ids": json.dumps(pair.get("source_region_ids", [])) if pair else "",
                     "cavity_side_distances_px": json.dumps(pair.get("cavity_side_distances_px", [])) if pair else "",
                     "cavity_side_extrapolated": pair.get("cavity_side_extrapolated", False) if pair else False,
                     "cavity_intersection_observed": pair.get("cavity_intersection_observed", False) if pair else False,
                     "shared_cusp_pair": pair.get("shared_cusp_pair", False) if pair else False,
                     "shared_cusp_id": pair.get("shared_cusp_id") if pair else None,
                     "side_reference": pair.get("side_reference", "cavity_midline") if pair else "",
                     "pair_role": pair.get("pair_role", "opposite_side_pair") if pair else "",
                     "pair_completed_by_exclusion": pair.get("pair_completed_by_exclusion", False) if pair else False,
                     "side_rule_conflict": pair.get("side_rule_conflict", False) if pair else False,
                     "uses_reinstated_cusp": pair.get("uses_reinstated_cusp", False) if pair else False,
                     "reinstated_cusp_ids": json.dumps(pair.get("reinstated_cusp_ids", [])) if pair else "[]",
                     "intercuspal_distance_px": pair["intercusp_distance_px"] if pair else None,
                     "cusp_centers_rc": json.dumps(pair["centers_rc"]) if pair else "",
                     "crossing_status": sites.get(pair_id, {}).get("status", "missing_cusp_center" if not pair else "not_evaluated"),
                     "eligible_for_ratio": any(row["cusp_pair_id"] == pair_id and row["eligible"]
                                               for row in result.get("candidates", [])),
                     "selected": bool(selected and selected["cusp_pair_id"] == pair_id)})
        rows[-1]["selected_ratio_is_fallback"] = bool(rows[-1]["selected"] and selected.get("ratio_is_fallback"))
    return rows


def cusp_label_positions(regions, shape):
    """Place labels outside their core circles, away from neighboring centers.

    The old global-column split put a lower-right label beside the lower-left
    circle on rotated teeth. Use outward directions from the center cloud, then
    avoid neighboring circles and already placed labels. Coordinates stay in
    the mask grid; a leader explicitly identifies the intended center.
    """
    if not regions:
        return {}
    centers = np.asarray([region["center_rc"] for region in regions], float)
    radii = np.asarray([max(0., region["radius_px"]-.75) for region in regions])
    origin = centers.mean(axis=0)
    placements, occupied = {}, []
    for index, region in enumerate(regions):
        center = centers[index]
        outward = center-origin
        if np.linalg.norm(outward) < 1e-6:
            outward = np.array([-1., 0.])
        outward = outward/np.linalg.norm(outward)
        options = []
        for angle in (0, -20, 20, -40, 40, -70, 70, -100, 100, 180):
            radians = np.deg2rad(angle)
            rotation = np.array([[np.cos(radians), -np.sin(radians)],
                                 [np.sin(radians), np.cos(radians)]])
            for clearance in (10., 16., 23.):
                position = center+(rotation @ outward)*(radii[index]+clearance)
                if np.any(position < 7) or np.any(position > np.asarray(shape)-8):
                    continue
                distances = np.linalg.norm(centers-position, axis=1)
                penalty = float(np.sum(np.maximum(radii+7.-distances, 0.)**2))
                if occupied:
                    gaps = np.linalg.norm(np.asarray(occupied)-position, axis=1)
                    penalty += float(np.sum(np.maximum(15.-gaps, 0.)**2))
                # Prefer positions clearly belonging to this center, not another.
                if int(np.argmin(distances)) != index:
                    penalty += 1000.
                options.append((penalty, abs(angle), clearance, position))
        position = min(options, key=lambda x: x[:3])[3] if options else np.clip(
            center+outward*(radii[index]+10.), 7, np.asarray(shape)-8)
        placements[region["region_id"]] = position
        occupied.append(position)
    return placements


def save_overlay(result, image_path, output_path, mapping, cusp_mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    selected = result["selected"]
    size = None
    if image_path:
        with Image.open(image_path) as source:
            size = source.size
            axes[0].imshow(np.asarray(source.convert("RGB")))
    mapped = {"isthmuses": [dict(neck) for neck in result["isthmuses"]], "warnings": []}
    # Remove previously mapped coordinates so an unverified current image cannot reuse them.
    for neck in mapped["isthmuses"]:
        for key in ("image_center_rc", "image_endpoints_rc", "width_image_px"):
            neck.pop(key, None)
    map_to_image(mapped, result["clean_mask"].shape, size, mapping)
    result["image_mapping"] = mapped["image_mapping"]
    result["warnings"] = sorted(set(result["warnings"] + mapped["warnings"]))
    rgb = np.ones((*result["clean_mask"].shape, 3))
    rgb[result["clean_mask"]] = np.array([46, 175, 80]) / 255
    axes[1].imshow(rgb, interpolation="nearest")
    for neck in mapped["isthmuses"]:
        tag = ("F" if neck.get("is_fallback") else "I") + str(neck["isthmus_id"])
        for axis, key in ((axes[0], "image_endpoints_rc"), (axes[1], "endpoints_rc")):
            if key not in neck:
                continue
            points = np.asarray(neck[key])
            primary = selected is not None and neck["isthmus_id"] == selected["isthmus_id"]
            axis.plot(points[:, 1], points[:, 0], color="#e00000", lw=2.5 if primary else 1.5,
                      ls="--" if neck.get("is_fallback") else "-", marker="o", ms=3,
                      label=f"{tag}: {neck['width_px']:.2f} px" if axis is axes[1] else None)
            axis.annotate(tag, (points[:, 1].mean(), points[:, 0].mean()), xytext=(5, 5),
                          textcoords="offset points", color="#e00000")
    if selected and not selected.get("crossing_checks_relaxed") and selected["within_near_tolerance"] and "image_scale_rc" in mapped:
        points = (np.asarray(selected["cusp_centers_rc"]) + 0.5) * mapped["image_scale_rc"] - 0.5
        axes[0].plot(points[:, 1], points[:, 0], color="#4899dc", ls=":", marker="+", lw=1)
    axes[0].set_title("Tooth + matched measurements" if "image_scale_rc" in mapped else "Tooth (mapping unavailable)")
    axes[1].set_title("Predicted cavity\nRed: isthmus width")
    if mapped["isthmuses"]:
        axes[1].legend(loc="lower left", fontsize=8)
    cusp_rgb = np.ones((*cusp_mask.shape, 3))
    cusp_rgb[np.asarray(cusp_mask, dtype=bool)] = np.array([46, 175, 80]) / 255
    axes[2].imshow(cusp_rgb, interpolation="nearest")
    blue = "#0057d9"
    regions = result["anatomy"].get("regions", [])
    clear_labels = bool(result["anatomy"].get("disjoint_pair_completion_used") or
                        result["anatomy"].get("excluded_region_ids"))
    positions = cusp_label_positions(regions, cusp_mask.shape) if clear_labels else {}
    if clear_labels:
        result["cusp_label_placement"] = []
    mid_col = float(np.median([region["center_rc"][1] for region in regions])) if regions else 0.0
    for region in regions:
        row, col = region["center_rc"]
        excluded = not region.get("pairing_eligible", True)
        cusp_label = f"C{region['region_id']}" + ("*" if excluded else "")
        radius = max(0.0, region["radius_px"] - 0.75)
        axes[2].add_patch(Circle((col, row), radius, fill=False, edgecolor=blue, lw=1,
                                 linestyle="--" if excluded else "-"))
        axes[2].plot(col, row, marker="o", color=blue, ms=3, ls="None")
        if clear_labels:
            text_row, text_col = positions[region["region_id"]]
            label = axes[2].annotate(
                cusp_label, (col, row), xytext=(text_col, text_row),
                textcoords="data", ha="center", va="center", color=blue, fontsize=9,
                fontweight="bold", zorder=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .97, "pad": 1.3},
                arrowprops={"arrowstyle": "-", "color": blue, "lw": .7,
                            "shrinkA": 3, "shrinkB": 2, "connectionstyle": "arc3,rad=0"})
            label.set_gid(f"cusp-center-label-{region['region_id']}")
            result["cusp_label_placement"].append(
                {"cusp_id": region["region_id"], "center_rc": [row, col],
                 "text_rc": [float(text_row), float(text_col)], "leader_to_center": True})
        else:
            side = -1 if col < mid_col else 1
            axes[2].annotate(f"C{region['region_id']}", (col, row),
                             xytext=(col + side * (radius + 4), row - 3), textcoords="data",
                             ha="right" if side < 0 else "left", va="bottom", color=blue, fontsize=8,
                             bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    # Geometry display is independent of cavity-crossing/ratio eligibility.
    # The original-tooth panel retains its near-isthmus filter.
    allowed = {tuple(ids) for ids in result["anatomy"].get("active_cusp_pairs", ALLOWED_CUSP_PAIRS)}
    valid_pair_ids = {candidate["cusp_pair_id"] for candidate in result["candidates"] if candidate["eligible"]}
    for pair in result["anatomy"].get("pairs", []):
        if not pair.get("opposite_sides_valid", False) and not pair.get("pair_completed_by_exclusion"):
            continue
        primary = selected is not None and pair["pair_id"] == selected["cusp_pair_id"]
        if tuple(sorted(pair.get("region_ids", []))) not in allowed:
            continue
        points = np.asarray(pair["centers_rc"])
        primary = selected is not None and pair["pair_id"] == selected["cusp_pair_id"]
        distance = float(np.linalg.norm(points[1] - points[0]))
        label = f"{pair['pair_label']}: {distance:.2f} px" + (" (selected)" if primary else "")
        if primary and selected.get("ratio_is_fallback"):
            label += " (fallback)"
        elif pair.get("side_rule_conflict"):
            label += " (same side; display only)"
        elif pair["pair_id"] not in valid_pair_ids:
            label += " (display only)"
        axes[2].plot(points[:, 1], points[:, 0], color=blue, lw=2.2 if primary else 1.2,
                     ls=":" if primary and selected.get("ratio_is_fallback") else ("-" if primary else "--"), label=label)
        midpoint = points.mean(axis=0)
        axes[2].annotate(pair["pair_label"], (midpoint[1], midpoint[0]), xytext=(4, 5),
                         textcoords="offset points", color=blue, fontsize=8,
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    pair_labels = [pair["pair_label"] for pair in result["anatomy"].get("pairs", [])]
    title = "One pair per cusp" if result["anatomy"].get("disjoint_pair_completion_used") else "Opposite-side cusp pairs"
    axes[2].set_title(title + "\n" + (" and ".join(pair_labels) if pair_labels else "No valid opposite-side pair"))
    if result["anatomy"].get("pairs"):
        axes[2].legend(loc="lower left", fontsize=7)
    if selected:
        tag = ("F" if selected["is_fallback"] else "I") + str(selected["isthmus_id"])
        caption = (f"Selected: {tag} / {selected['cusp_pair_label']}    "
                   f"W/D = {selected['ratio_isthmus_to_intercuspal']:.4f}    "
                   f"D/W = {selected['ratio_intercuspal_to_isthmus']:.4f}    "
                   f"Location gap: {selected['match_distance_px']:.2f} px")
        flags = []
        excluded_ids = result["anatomy"].get("excluded_region_ids", [])
        if result["anatomy"].get("excluded_cusp_rescue_used"):
            reinstated = result["anatomy"]["reinstated_cusp_ids"]
            flags.append("Reinstated for proximity: " + ", ".join(f"C{i}" for i in reinstated))
            flags.append("Not paired: " + ", ".join(f"C{i}*" for i in excluded_ids))
        elif excluded_ids:
            flags.append("Excluded from pairing (smallest area): " + ", ".join(f"C{i}*" for i in excluded_ids))
        if result["anatomy"].get("disjoint_pair_completion_used"):
            flags.append("Each cusp used once; remaining two form the second pair")
            if any(p.get("side_rule_conflict") for p in result["anatomy"]["pairs"]):
                flags.append("Same-side complement: display only")
        if result["anatomy"].get("cusp_labels_reassigned"):
            flags.append("C1/C2: cavity bank A; C3/C4: bank B")
        if result["anatomy"].get("pairing_switched_for_crossing"):
            flags.append("Pairing changed to avoid crossing")
        if not selected["within_near_tolerance"]:
            flags.append("Distant match: review")
        if selected["is_fallback"]:
            flags.append("Fallback width: low confidence")
        if selected.get("ratio_is_fallback"):
            flags.append("Ratio fallback: relaxed matching (review)")
            if selected.get("crossing_checks_relaxed"):
                flags.append("Cavity crossing not required")
        unpaired = [i for i in result["anatomy"].get("unpaired_region_ids", []) if i not in excluded_ids]
        if unpaired:
            flags.append("Unpaired cores: " + ", ".join(f"C{item}" for item in unpaired) + " (review)")
        if flags:
            caption += "\n" + "; ".join(flags)
    else:
        caption = "Ratio unavailable: " + result["status"].replace("_", " ")
    fig.supxlabel(caption, fontsize=9)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(result["filename"])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_csv(path, rows, fieldnames=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_pairing_review(items, path):
    """Combined cavity/cusp context for auditing every pair in the batch."""
    import matplotlib.pyplot as plt

    columns = 5
    rows = math.ceil(len(items)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.0*rows), squeeze=False,
                             layout="constrained")
    for axis, (result, cusp_mask) in zip(axes.flat, items):
        cavity = result["clean_mask"]
        rgb = np.ones((*cavity.shape, 3))
        rgb[cavity] = [.78, .78, .78]
        rgb[cusp_mask] = np.array([46, 175, 80])/255
        axis.imshow(rgb, interpolation="nearest")
        anatomy = result["anatomy"]
        model = anatomy.get("cavity_side_model")
        if model:
            core_t = [r["cavity_longitudinal_position_px"] for r in anatomy["regions"]]
            lo, hi = min([*model["position_px"], *core_t]), max([*model["position_px"], *core_t])
            t = np.linspace(lo, hi, 180)
            centerline = (np.asarray(model["origin_rc"])+t[:, None]*model["axis_rc"]+
                          np.interp(t, model["position_px"], model["midpoint_px"])[:, None]*model["normal_rc"])
            axis.plot(centerline[:, 1], centerline[:, 0], color="#777777", ls=":", lw=.8)
        for pair in anatomy.get("pairs", []):
            points = np.asarray(pair["centers_rc"])
            axis.plot(points[:, 1], points[:, 0], color="#0057d9", lw=1.6,
                      ls="-" if pair["cavity_intersection_observed"] else "--", marker="o", ms=2.5)
        for region in anatomy.get("regions", []):
            r, c = region["center_rc"]
            label = f"C{region['region_id']}" + ("*" if not region.get("pairing_eligible", True) else "")
            axis.annotate(label, (c, r), xytext=(4, -6),
                          textcoords="offset points", fontsize=8, color="#003b92")
        points = np.argwhere(cavity | cusp_mask)
        if len(points):
            low, high = points.min(axis=0)-10, points.max(axis=0)+10
            axis.set_xlim(low[1], high[1])
            axis.set_ylim(high[0], low[0])
        axis.set_title(f"{result['filename'].replace('_mask.png', '')} · {len(anatomy.get('pairs', []))} pair(s)", fontsize=9)
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle("One-pair-per-cusp audit · *: not paired\nGray: cavity / dotted midline · Green: cusps · Blue: pairs (dashed if cavity intersection absent)", fontsize=11)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def load_saved_widths(args):
    """Read original widths, or recover their unchanged copy from prior results.

    The prior-result path is useful when the original width folder was moved.
    Verify both mask hashes before reusing coordinates; never redetect a width.
    Matching uses current defaults (and saved calibration) if config is absent.
    """
    if args.isthmus_json.exists() and args.previous_ratio_json is None:
        payload = json.loads(args.isthmus_json.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list) or "config" not in payload:
            raise ValueError("Expected an isthmus_disc details JSON with results and config.")
        return payload, args.isthmus_json, "original_isthmus_json"
    path = args.previous_ratio_json or Path("cus_ist_ratio_results_largest_four/details.json")
    previous = json.loads(path.read_text())
    records, calibration = [], []
    for old in previous["results"]:
        filename = old["filename"]
        if Path(filename).name != filename:
            raise ValueError("Expected mask basenames in previous results.")
        for key, folder in [("cavity", args.mask_dir), ("cusps", args.cusp_dir)]:
            digest = hashlib.sha256((folder/filename).read_bytes()).hexdigest()
            if digest != old.get("input_sha256", {}).get(key):
                raise ValueError(f"{filename}: {key} mask differs from saved measurements.")
        with Image.open(args.mask_dir/filename) as image:
            shape = [image.height, image.width]
        for width in old.get("isthmuses", []):
            if width.get("width_mm") is not None:
                calibration.append(width["width_mm"]/width["width_px"])
        records.append({"filename": filename, "mask_shape": shape,
                        "isthmuses": old.get("isthmuses", []), "warnings": []})
    config = asdict(Config())
    if calibration:
        if not np.allclose(calibration, calibration[0]):
            raise ValueError("Inconsistent saved pixel calibration.")
        config["pixel_size_mm"] = calibration[0]
    print(f"Reusing unchanged saved isthmuses from {path}; source masks verified.")
    return {"results": records, "config": previous.get("config", config),
            "algorithm_version": previous.get("isthmus_algorithm_version"),
            "original_isthmus_json_sha256": previous.get("original_isthmus_json_sha256") or previous.get("isthmus_json_sha256"),
            "config_source": "prior_config" if "config" in previous else "current_matching_defaults_saved_pixel_calibration"}, path, "widths_preserved_in_previous_ratio_results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isthmus-json", type=Path, default=Path("isthmus_disc_results_fallback/isthmus_details.json"))
    parser.add_argument("--previous-ratio-json", type=Path,
                        help="Reuse the exact saved widths embedded in an earlier ratio details.json; verify mask hashes.")
    parser.add_argument("--mask-dir", type=Path, default=Path("pred_O_M_masks_folder"))
    parser.add_argument("--cusp-dir", type=Path, default=Path("pred_O_M_cusp_molar_mask"))
    parser.add_argument("--image-dir", type=Path, default=Path("samples_stu_O"))
    parser.add_argument("--output-dir", type=Path, default=Path("cus_ist_ratio_results_cusp_rescue"))
    parser.add_argument("--pairing-policy", choices=("nearest", "near-only"), default="nearest")
    parser.add_argument("--image-mapping", choices=("same-size", "resize"), default="same-size")
    parser.add_argument("--glob", default="*_mask.png")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-ratio-fallback", action="store_true",
                        help="Keep blank ratios when the original matching checks fail.")
    args = parser.parse_args()
    for path in (args.mask_dir, args.cusp_dir):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    if args.output_dir.resolve() in {args.mask_dir.resolve(), args.cusp_dir.resolve(),
                                    args.image_dir.resolve(), args.isthmus_json.parent.resolve()}:
        parser.error("Use a separate output directory.")
    try:
        payload, width_source, width_source_method = load_saved_widths(args)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.output_dir.resolve() == width_source.parent.resolve():
        parser.error("Preserve previous measurements; choose a separate output directory.")
    known = {field.name for field in fields(Config)}
    config = Config(**{key: value for key, value in payload["config"].items() if key in known})
    if config.pixel_size_mm is not None and (not math.isfinite(config.pixel_size_mm) or config.pixel_size_mm <= 0):
        parser.error("Saved pixel calibration must be finite and positive.")
    records = [record for record in payload["results"] if fnmatch.fnmatch(record["filename"], args.glob)]
    if not records or len({record["filename"] for record in records}) != len(records):
        parser.error("No matching records or duplicate filenames in isthmus JSON.")
    threshold = payload.get("arguments", {}).get("threshold", 0.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, rows, candidates, distances, errors = [], [], [], [], 0
    review_items, center_rows = [], []
    for record in records:
        filename = record["filename"]
        try:
            if Path(filename).name != filename:
                raise ValueError("Expected a basename for each mask filename.")
            mask_path, cusp_path = args.mask_dir / filename, args.cusp_dir / filename
            if not mask_path.exists() or not cusp_path.exists():
                result = {"filename": filename, "status": "missing_cavity_or_cusp_mask",
                          "isthmuses": record.get("isthmuses", []), "warnings": [], "selected": None}
            elif record.get("status") == "error":
                result = {"filename": filename, "status": "source_isthmus_error", "selected": None,
                          "warnings": [record.get("error", "isthmus_processing_failed")]}
            else:
                cusp_mask = read_mask(cusp_path, threshold)
                result = analyze_record(record, read_mask(mask_path, threshold), cusp_mask,
                                        config, args.pairing_policy, not args.no_ratio_fallback)
                result["input_sha256"] = {"cavity": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                                          "cusps": hashlib.sha256(cusp_path.read_bytes()).hexdigest()}
                if not args.no_plots:
                    save_overlay(result, matching_image(mask_path, args.image_dir),
                                 args.output_dir / f"{mask_path.stem}_ratio.png", args.image_mapping, cusp_mask)
                    review_items.append((result, cusp_mask))
        except Exception as error:
            errors += 1
            result = {"filename": filename, "status": "error", "selected": None,
                      "error": f"{type(error).__name__}: {error}"}
        rows.append(summary_row(result))
        distances.extend(distance_rows(result))
        for region in result.get("anatomy", {}).get("regions", []):
            center_rows.append({"filename": filename, "cusp_id": region["region_id"],
                                "original_cusp_id": region.get("source_region_id"),
                                "center_row": region["center_rc"][0], "center_col": region["center_rc"][1],
                                "area_px": region["area_px"],
                                "pairing_eligible": region.get("pairing_eligible", True),
                                "pairing_exclusion_reason": region.get("pairing_exclusion_reason", ""),
                                "reinstated_for_proximity": region.get("reinstated_for_proximity", False),
                                "cavity_bank": region.get("cavity_side_name", "unavailable"),
                                "side_distance_px": region.get("cavity_side_distance_px"),
                                "side_extrapolated": region.get("cavity_side_extrapolated"),
                                "paired": region["region_id"] not in result["anatomy"].get("unpaired_region_ids", [])})
        for candidate in result.get("candidates", []):
            candidates.append({"filename": filename, **measurement_row(candidate),
                               "eligible": candidate["eligible"], "selected": candidate["selected"]})
        results.append(serializable(result))
        selected = result.get("selected")
        suffix = f" W/D={selected['ratio_isthmus_to_intercuspal']:.4f}" if selected else ""
        print(f"{filename}: {result['status']}{suffix}")
    write_csv(args.output_dir / "ratios.csv", rows)
    write_csv(args.output_dir / "intercuspal_distances.csv", distances)
    if center_rows:
        write_csv(args.output_dir / "cusp_centers.csv", center_rows)
    if review_items:
        save_pairing_review(review_items, args.output_dir/"cusp_review_grid.png")
    write_csv(args.output_dir / "ratio_candidates.csv", candidates,
              ["filename", *measurement_row({}), "eligible", "selected"])
    details = {"version": VERSION, "isthmus_algorithm_version": payload.get("algorithm_version"),
               "isthmus_json": str(width_source), "width_source_method": width_source_method,
               "isthmus_json_sha256": hashlib.sha256(width_source.read_bytes()).hexdigest(),
               "original_isthmus_json_sha256": payload.get("original_isthmus_json_sha256"),
               "config": asdict(config), "config_source": payload.get("config_source", "original_isthmus_config"),
               "pairing_policy": args.pairing_policy,
               "ratio_fallback_enabled": not args.no_ratio_fallback,
               "ratio_fallback": "Only for blank ratios: relax proximity, then cavity-crossing checks for finite opposite-bank segments. Never relax opposite-side validity or allow interior line crossings. Every cusp may participate in only one pair; same-side complements are display-only. Preserve original widths and flag extrapolated cavity sides.",
               "ratio_definition": "isthmus width / intercuspal center-to-center distance, both in mask pixels",
               "selection": "closest finite cavity-crossing/width segments; center distance breaks ties",
               "cusp_geometry": "Concavity separation and distance-peak watershed; original masks and saved measurements preserved.",
               "default_cusp_pairs": [list(ids) for ids in ALLOWED_CUSP_PAIRS],
               "crossing_alternative_cusp_pairs": [list(ids) for ids in ALTERNATE_CUSP_PAIRS],
               "crossing_policy": "Preserve complete disjoint opposite-bank pairs. With four cores, selecting a first pair fixes its complement as the remaining two. Reject crossing, touching or overlapping lines; never reuse a cusp. Export same-side complements for review but exclude them from ratio matching.",
               "label_convention": "Ordinary cases: C1/C2 bank A, C3/C4 bank B. Disjoint completion: original detection IDs C1-C4 restored, with bank membership recorded independently. No detected center is omitted.",
               "pair_recovery": "Start with the four largest cusp areas. If neither pair meets proximity, an excluded cusp can replace one paired core while the other pair is preserved. Require a valid same-component cavity crossing, opposite sides and noncrossing disjoint pairs. A distant replacement must improve the gap and is allowed only with an explicit ratio-fallback flag. Centers and saved widths are unchanged.",
               "coordinate_convention": "zero-based row, column in the cavity mask grid",
               "results": results}
    (args.output_dir / "details.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} samples to {args.output_dir}; {errors} processing errors.")
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
