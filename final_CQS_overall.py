#!/usr/bin/env python3
"""Calculate provisional Class II CQS using the latest saved student measurements.

Preserves previous scripts/results. Uses equal EFD weights, signed gingival
depth from the pulpal floor, and fixed weights without missing-value renormalization.
Run from the repository root: python3 final_CQS_overall.py
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys


VERSION = "1.1"
ROOT_RESULTS = Path("RESULTS-21-09-2026")
COMPONENTS = {
    "efd_occlusal": ("Occlusal EFD", .25, "#31688e"),
    "efd_proximal": ("Proximal EFD", .25, "#54a4bf"),
    "depth_occlusal": ("Occlusal depth", .10, "#3b8c67"),
    "depth_gingival": ("Pulpal-to-gingival depth", .10, "#8dc684"),
    "ratio": ("Isthmus / intercuspal ratio", .20, "#a584bf"),
    "regularity_occlusal": ("Occlusal floor regularity", .05, "#d69249"),
    "regularity_proximal": ("Proximal floor regularity", .05, "#efc180"),
}
DEFAULT_CONFIG = {
    "efd_column": "average_matching_score_pct",
    "occlusal_depth": {"min_mm": 1.5, "max_mm": 2.0, "tolerance_mm": .5},
    "gingival_depth": {"min_mm": .5, "max_mm": 1., "tolerance_mm": .5},
    "ratio": {"target": 1/3, "half_score_multiplier": 1.5},
    "regularity_scale_mm": {"occlusal": .5, "proximal": .5},
}
PRIMARY_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11402004/"
MINIMAL_COLUMNS = [
    "case_id", "Tooth", "CQS_overall", "CQS_status", "available_weight", "missing_components",
    "anatomical_review_required", "rubric_clinically_validated", "review_flags",
    "occlusal_average_matching_score_pct", "proximal_average_matching_score_pct",
    "OcclusalDepthMedian_mm", "GingivalFromPulpalMedian_mm", "ratio_isthmus_to_intercuspal",
    "OcclusalFloorResidualRMS_mm", "ProximalFloorResidualRMS_mm",
] + ["q_"+key for key in COMPONENTS] + ["points_"+key for key in COMPONENTS]
LEGACY_REFERENCE_COLUMNS = {
    "CQS_if_whole_floor_gingival_reference", "CQS_reference_shift_points",
    "GingivalFromPulpalWholeFloorMedian_mm", "GingivalFromPulpalLocalGlobalSensitivity_mm",
    "GingivalFromPulpalGeometryFlags", "GingivalFromPulpalObservations",
    "ProximalCrownReferencedDepthMedian_mm_not_scored",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def truth(value):
    return str(value).strip().lower() == "true"


def case_key(name):
    """Canonical numeric case ID; do not silently guess IDs from arbitrary text."""
    stem = Path(str(name)).stem
    match = re.fullmatch(r"[OP][_-](\d+)(?:_st)?(?:_mask)?", stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized case identifier: {name!r}")
    return int(match.group(1))


def read_table(path, id_column, required, view=None):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required+[id_column])-set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        result = {}
        for row in reader:
            key = case_key(row[id_column])
            if key in result:
                raise ValueError(f"{path}: duplicate canonical case ID {key}")
            if view and row.get("cavity_type") != view:
                raise ValueError(f"{path}: expected {view}, found {row.get('cavity_type')}")
            if view and row[id_column][0].upper() != view[0].upper():
                raise ValueError(f"{path}: incorrect view prefix in {row[id_column]}")
            result[key] = row
    if not result:
        raise ValueError(f"{path}: input table is empty")
    return result


def read_depth_table(path):
    """Accept the publication schema, retaining explicit support for older inputs."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        columns = csv.DictReader(stream).fieldnames or []
    publication = "GingivalMinusPulpalDepth_mm" in columns
    required = ["OcclusalDepthMedian_mm", "OcclusalFloorResidualRMS_mm",
                "ProximalFloorResidualRMS_mm", "OcclusalStatus", "ProximalStatus"]
    required += (["ProximalDepthMedian_mm", "GingivalMinusPulpalDepth_mm", "Error"] if publication
                 else ["GingivalFromPulpalMedian_mm", "GingivalFromPulpalStatus"])
    return read_table(path, "Tooth", required), ("publication" if publication else "local_pulpal_reference")


def normalize_depth_row(source):
    """Map the supplied floor-depth difference without recalculating its reference."""
    row = dict(source)
    if "GingivalMinusPulpalDepth_mm" not in row:
        return row
    pulpal, gingival, difference = (number(row.get(key)) for key in
        ("OcclusalDepthMedian_mm", "ProximalDepthMedian_mm", "GingivalMinusPulpalDepth_mm"))
    available = None not in (pulpal, gingival, difference)
    if available and not math.isclose(gingival-pulpal, difference, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{row.get('Tooth')}: gingival-minus-pulpal depth does not equal proximal minus occlusal medians")
    valid = (available and not row.get("Error", "").strip()
             and all(row.get(region+"Status", "").startswith("measured") for region in ("Occlusal", "Proximal")))
    # The existing export column is a compatibility alias for the new source metric.
    row["GingivalFromPulpalMedian_mm"] = difference if available else None
    row["GingivalFromPulpalStatus"] = "measured_requires_anatomical_review" if valid else "unresolved"
    row["GingivalFromPulpalError"] = row.get("Error", "")
    row["GingivalFromPulpalReferenceMethod"] = row.get("GingivalMinusPulpalMethod") or "difference_of_area_weighted_medians_same_crown_reference"
    # A local-plane sensitivity estimate cannot accompany a different measurement method.
    for key in ("GingivalFromPulpalWholeFloorMedian_mm", "GingivalFromPulpalLocalGlobalSensitivity_mm",
                "GingivalFromPulpalGeometryFlags", "GingivalFromPulpalObservations"):
        row.pop(key, None)
    return row


def validate_config(config):
    for name in ("occlusal_depth", "gingival_depth"):
        settings = config[name]
        lo, hi, tol = (number(settings[k]) for k in ("min_mm", "max_mm", "tolerance_mm"))
        if None in (lo, hi, tol) or lo > hi or tol <= 0:
            raise ValueError(f"Invalid {name} scoring band or tolerance")
    ratio = config["ratio"]
    if number(ratio["target"]) is None or ratio["target"] <= 0:
        raise ValueError("Ratio target must be positive and finite")
    if number(ratio["half_score_multiplier"]) is None or ratio["half_score_multiplier"] <= 1:
        raise ValueError("Ratio multiplier must be finite and greater than one")
    for value in config["regularity_scale_mm"].values():
        if number(value) is None or value <= 0:
            raise ValueError("Regularity scales must be positive and finite")
    if config["efd_column"] != "average_matching_score_pct":
        raise ValueError("This scorer requires average_matching_score_pct for both EFD inputs")
    if not math.isclose(sum(x[1] for x in COMPONENTS.values()), 1., abs_tol=1e-12):
        raise ValueError("Component weights must sum to one")


def efd_score(value):
    value = number(value)
    return value/100 if value is not None and 0 <= value <= 100 else None


def depth_score(value, settings):
    value = number(value)
    if value is None:
        return None
    distance = max(settings["min_mm"]-value, 0., value-settings["max_mm"])
    return max(0., 1.-distance/settings["tolerance_mm"])


def ratio_score(value, settings):
    value = number(value)
    if value is None or value <= 0:
        return None
    # Subtract logs to avoid overflow/underflow in value/target for valid floats.
    deviation = (math.log(value)-math.log(settings["target"]))/math.log(settings["half_score_multiplier"])
    return math.exp(-math.log(2.)*deviation**2)


def regularity_score(value, scale):
    value = number(value)
    return math.exp(-value/scale) if value is not None and value >= 0 else None


def aggregate(scores):
    """Missing components remain missing; never redistribute their weights."""
    missing = [key for key in COMPONENTS if scores.get(key) is None]
    points = {key: None if scores.get(key) is None else 10*settings[1]*scores[key]
              for key, settings in COMPONENTS.items()}
    known_sum = sum(value for value in points.values() if value is not None)
    weight = sum(COMPONENTS[key][1] for key in COMPONENTS if scores.get(key) is not None)
    return (known_sum if not missing else None), points, weight, missing


def make_row(key, sources, config):
    o, p, depth, ratio = (sources[name].get(key, {}) for name in ("occlusal_efd", "proximal_efd", "depth", "ratio"))
    depth = normalize_depth_row(depth)
    raw = {
        "occlusal_average_matching_score_pct": number(o.get(config["efd_column"])),
        "proximal_average_matching_score_pct": number(p.get(config["efd_column"])),
        "OcclusalDepthMedian_mm": number(depth.get("OcclusalDepthMedian_mm")),
        "GingivalFromPulpalMedian_mm": number(depth.get("GingivalFromPulpalMedian_mm")),
        "ratio_isthmus_to_intercuspal": number(ratio.get("ratio_isthmus_to_intercuspal")),
        "OcclusalFloorResidualRMS_mm": number(depth.get("OcclusalFloorResidualRMS_mm")),
        "ProximalFloorResidualRMS_mm": number(depth.get("ProximalFloorResidualRMS_mm")),
    }
    raw_keys = dict(zip(COMPONENTS, raw))
    scores = {
        "efd_occlusal": efd_score(raw[raw_keys["efd_occlusal"]]),
        "efd_proximal": efd_score(raw[raw_keys["efd_proximal"]]),
        "depth_occlusal": depth_score(raw[raw_keys["depth_occlusal"]], config["occlusal_depth"]),
        "depth_gingival": depth_score(raw[raw_keys["depth_gingival"]], config["gingival_depth"]),
        "ratio": ratio_score(raw[raw_keys["ratio"]], config["ratio"]),
        "regularity_occlusal": regularity_score(raw[raw_keys["regularity_occlusal"]], config["regularity_scale_mm"]["occlusal"]),
        "regularity_proximal": regularity_score(raw[raw_keys["regularity_proximal"]], config["regularity_scale_mm"]["proximal"]),
    }
    invalid = []
    # Finite stale values in failed source rows must not masquerade as measurements.
    for row, component in [(o, "efd_occlusal"), (p, "efd_proximal")]:
        if row and (row.get("status") not in {"OK", "REVIEW"} or row.get("error", "").strip()):
            scores[component] = None
            invalid.append(component+"_source_failed")
    for component, status_col, error_col in [
        ("depth_occlusal", "OcclusalStatus", "Error"),
        ("regularity_occlusal", "OcclusalStatus", "Error"),
        ("regularity_proximal", "ProximalStatus", "Error"),
        ("depth_gingival", "GingivalFromPulpalStatus", "GingivalFromPulpalError")]:
        if depth and (not depth.get(status_col, "").startswith("measured") or depth.get(error_col, "").strip()):
            scores[component] = None
            invalid.append(component+"_source_failed")
    if ratio:
        width, distance = number(ratio.get("isthmus_width_px")), number(ratio.get("intercuspal_distance_px"))
        if width is None or distance is None or width <= 0 or distance <= 0:
            scores["ratio"] = None
            invalid.append("invalid_selected_width_or_distance")
        elif raw["ratio_isthmus_to_intercuspal"] is None or not math.isclose(width/distance, raw["ratio_isthmus_to_intercuspal"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"O_{key}: selected ratio does not equal saved width / distance")
        if ratio.get("status") not in {"ratio_available", "ratio_requires_review"} or ratio.get("error", "").strip():
            scores["ratio"] = None
            invalid.append("ratio_source_failed")
        if (not truth(ratio.get("opposite_sides_valid")) or truth(ratio.get("shared_cusp_pair"))
                or truth(ratio.get("crossing_checks_relaxed"))):
            scores["ratio"] = None
            invalid.append("selected_ratio_pair_violates_required_geometry")
    total, points, weight, missing = aggregate(scores)
    review = list(invalid)
    for region, row in [("occlusal_efd", o), ("proximal_efd", p)]:
        if row.get("warnings"):
            review.append(region+": "+row["warnings"])
    if depth.get("GingivalFromPulpalGeometryFlags"):
        review.append("gingival_reference: "+depth["GingivalFromPulpalGeometryFlags"])
    if depth.get("GingivalFromPulpalObservations"):
        review.append("gingival_observation: "+depth["GingivalFromPulpalObservations"])
    if raw["GingivalFromPulpalMedian_mm"] is not None and raw["GingivalFromPulpalMedian_mm"] < 0:
        review.append("negative_gingival_median_retained_for_provisional_scoring")
    if truth(ratio.get("is_fallback")):
        review.append("isthmus_width_fallback")
    if truth(ratio.get("ratio_is_fallback")):
        review.append("ratio_fallback: "+ratio.get("ratio_fallback_stage", "unspecified"))
    if ratio and not truth(ratio.get("within_near_tolerance")):
        review.append("selected_pair_outside_preferred_proximity")
    if ratio.get("warnings"):
        review.append("ratio_observation: "+ratio["warnings"])
    for region in ("Occlusal", "Proximal"):
        tilt = number(depth.get(region+"FloorTilt_deg"))
        if tilt is not None and tilt > 45:
            review.append(region.lower()+"_steep_surface_verify_label_not_automatic_failure")
    global_depth = number(depth.get("GingivalFromPulpalWholeFloorMedian_mm"))
    alternative = None
    if total is not None and global_depth is not None:
        alternative_scores = dict(scores, depth_gingival=depth_score(global_depth, config["gingival_depth"]))
        alternative = aggregate(alternative_scores)[0]
    row = {
        "case_id": key, "Tooth": f"O_{key}", "CQS_overall": total,
        "CQS_status": "PROVISIONAL_FULL_CLASS_II" if total is not None else "INCOMPLETE_NO_RENORMALIZATION",
        "available_weight": weight, "missing_components": ";".join(missing),
        "anatomical_review_required": True, "rubric_clinically_validated": False,
        "review_flags": " | ".join(review), **raw,
        **{"q_"+k: v for k, v in scores.items()},
        **{"points_"+k: v for k, v in points.items()},
        "CQS_if_whole_floor_gingival_reference": alternative,
        "CQS_reference_shift_points": None if alternative is None else alternative-total,
        "GingivalFromPulpalWholeFloorMedian_mm": global_depth,
        "GingivalFromPulpalLocalGlobalSensitivity_mm": number(depth.get("GingivalFromPulpalLocalGlobalSensitivity_mm")),
        "GingivalFromPulpalGeometryFlags": depth.get("GingivalFromPulpalGeometryFlags", ""),
        "GingivalFromPulpalObservations": depth.get("GingivalFromPulpalObservations", ""),
        "GingivalFromPulpalReferenceMethod": depth.get("GingivalFromPulpalReferenceMethod", ""),
        "ProximalCrownReferencedDepthMedian_mm_not_scored": number(depth.get("ProximalDepthMedian_mm")),
        "ProximalDepthMedian_mm": number(depth.get("ProximalDepthMedian_mm")),
        "GingivalMinusPulpalDepth_mm": number(depth.get("GingivalMinusPulpalDepth_mm")),
        "depth_source_reference": depth.get("DepthReference", depth.get("ReferenceMethod", "")),
        "occlusal_sample": o.get("sample", ""), "proximal_sample": p.get("sample", ""),
        "occlusal_efd_status": o.get("status", ""), "proximal_efd_status": p.get("status", ""),
        "occlusal_efd_warnings": o.get("warnings", ""), "proximal_efd_warnings": p.get("warnings", ""),
        "source_depth_warnings": depth.get("Warnings", ""),
    }
    for col in ["filename", "status", "isthmus_id", "cusp_pair_label", "isthmus_width_px", "intercuspal_distance_px",
                "match_distance_px", "near_threshold_px", "within_near_tolerance", "is_fallback",
                "ratio_is_fallback", "ratio_fallback_stage", "opposite_sides_valid", "shared_cusp_pair",
                "crossing_checks_relaxed", "confidence_label", "warnings"]:
        row["ratio_source_"+col] = ratio.get(col, "")
    component_rows = [{"case_id": key, "Tooth": f"O_{key}", "component": component,
                       "label": settings[0], "weight": settings[1], "raw_metric": raw_keys[component],
                       "raw_value": raw[raw_keys[component]], "normalized_score": scores[component],
                       "points_out_of_10": points[component], "maximum_points": 10*settings[1]}
                      for component, settings in COMPONENTS.items()]
    return row, component_rows


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_minimal_csv(path, rows, schema_path=None):
    """Keep an existing compact table's chosen columns when refreshing results."""
    columns = list(MINIMAL_COLUMNS)
    if schema_path is not None:
        with Path(schema_path).open(newline="", encoding="utf-8-sig") as stream:
            columns = csv.DictReader(stream).fieldnames or []
    if (not columns or len(columns) != len(set(columns)) or set(columns)-set(rows[0])
            or not {"case_id", "Tooth", "CQS_overall"}.issubset(columns)):
        raise ValueError("Minimal CSV schema contains unsupported, duplicate or missing identifier/score columns")
    write_csv(path, [{key: row[key] for key in columns} for row in rows])
    return columns


def fmt(value, decimals=3):
    return "—" if value is None else f"{value:.{decimals}f}"


def plot_scores(rows, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cqs-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, max(7, len(rows)*.32+2.1)))
    y, left = np.arange(len(rows)), np.zeros(len(rows))
    for key, (label, weight, color) in COMPONENTS.items():
        values = np.array([row["points_"+key] or 0. for row in rows])
        ax.barh(y, values, left=left, height=.73, color=color, label=f"{label} ({weight:.0%})")
        left += values
    for i, row in enumerate(rows):
        ax.text(left[i]+.08, i, fmt(row["CQS_overall"], 2), va="center", fontsize=8)
    ax.set(yticks=y, yticklabels=[r["Tooth"] for r in rows], xlim=(0, 10.7), xticks=np.arange(11),
           xlabel="CQS contribution (points out of 10)")
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e9edf1", lw=.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.065), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Class II CQS · student preparations", x=.10, ha="left", fontsize=16, weight="bold", y=.985)
    fig.text(.10, .955, "Equal EFD weights · average reference matching · fixed seven-component scoring", fontsize=10)
    fig.text(.10, .018, "Scoring uses the supplied measurements. The provisional rubric and measured floor labels require review.", fontsize=8)
    fig.subplots_adjust(left=.10, right=.96, top=.93, bottom=.13)
    fig.savefig(output/"CQS_overall.png", dpi=600, facecolor="white")
    fig.savefig(output/"CQS_overall.pdf", facecolor="white")
    plt.close(fig)


def write_html(rows, config, output):
    body_rows = []
    for row in rows:
        pieces = []
        for key, (label, weight, color) in COMPONENTS.items():
            points = row["points_"+key]
            title = html.escape(f"{label}: {fmt(points)} / {10*weight:g} points")
            pieces.append(f'<span title="{title}" style="width:{(points or 0)*10:.8f}%;background:{color}"></span>')
        details = html.escape(row["review_flags"] or "No additional source flag; anatomical and rubric review still required.")
        sensitivity = ""
        if row.get("CQS_if_whole_floor_gingival_reference") is not None:
            sensitivity = (f"<p>Whole-floor gingival reference CQS: {fmt(row['CQS_if_whole_floor_gingival_reference'],2)}; "
                           f"change {fmt(row['CQS_reference_shift_points'])} points. "
                           "This is reference sensitivity, not a confidence interval.</p>")
        body_rows.append(f'''<tr data-case="{row['case_id']}" data-score="{row['CQS_overall'] if row['CQS_overall'] is not None else -1}">
<td><b>{row['Tooth']}</b></td><td><strong>{fmt(row['CQS_overall'],2)}</strong><div class="bar">{''.join(pieces)}</div></td>
<td>{fmt(row['occlusal_average_matching_score_pct'],2)}</td><td>{fmt(row['proximal_average_matching_score_pct'],2)}</td>
<td>{fmt(row['OcclusalDepthMedian_mm'])}</td><td>{fmt(row['GingivalFromPulpalMedian_mm'])}</td>
<td>{fmt(row['ratio_isthmus_to_intercuspal'])}</td><td>{fmt(row['OcclusalFloorResidualRMS_mm'])}</td><td>{fmt(row['ProximalFloorResidualRMS_mm'])}</td>
<td><details><summary>Review</summary>{details}{sensitivity}</details></td></tr>''')
    weights = "".join(f'<li><span style="color:{color}">■</span> {label}: <b>{weight:.0%}</b></li>'
                      for label, weight, color in COMPONENTS.values())
    score_values = [r["CQS_overall"] for r in rows if r["CQS_overall"] is not None]
    lo, hi = (min(score_values), max(score_values)) if score_values else (None, None)
    od, gd = config["occlusal_depth"], config["gingival_depth"]
    depth_definition = ("Pulpal-to-gingival depth uses the supplied difference between proximal gingival and occlusal pulpal "
                        "area-weighted median depths, measured from the same crown reference."
                        if config.get("depth_source_schema") == "publication" else
                        "Pulpal-to-gingival depth uses the supplied local pulpal-floor reference measurement.")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overall Class II CQS</title><style>
body{{font:15px/1.55 Arial,sans-serif;color:#243444;background:#f4f6f8;margin:0}}main{{max-width:1500px;margin:auto;padding:30px}}
h1{{font-size:32px;margin-bottom:5px}}section{{background:white;padding:22px;border:1px solid #dde3e9;border-radius:10px;margin:20px 0}}
a{{color:#28688e}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;border-bottom:1px solid #e4e9ed;padding:11px 9px;vertical-align:top}}
th{{background:#f0f4f7;white-space:nowrap}}.scroll{{overflow:auto}}.bar{{width:160px;height:12px;display:flex;background:#edf0f4;margin-top:6px}}
.bar span{{height:100%}}button,input{{padding:8px 11px;border:1px solid #c3ced7;border-radius:5px;background:white;margin:5px}}
details{{min-width:160px;max-width:340px}}summary{{cursor:pointer}}.weights{{display:flex;flex-wrap:wrap;gap:10px 22px;list-style:none;padding:0}}
.muted{{color:#607184}}.card{{font-size:21px}}img{{width:100%;max-width:1000px;display:block;margin:auto}}@media print{{button,input{{display:none}}main{{padding:0}}}}
</style><main><h1>Overall Class II CQS</h1><p class="muted">Student preparations · equal occlusal/proximal EFD weights · provisional scores out of 10</p>
<p><a href="CQS_overall_scores.csv">Overall CSV</a> · <a href="cqs_minimal.csv">Minimal CSV</a> · <a href="CQS_component_scores.csv">Component CSV</a> ·
<a href="CQS_overall.png">600-dpi PNG</a> · <a href="CQS_overall.pdf">Vector PDF</a> · <a href="scoring_config.json">Scoring parameters</a></p>
<section><div class="card">{len(score_values)}/{len(rows)} complete scores · Range {fmt(lo,2)}–{fmt(hi,2)} / 10</div>
<ul class="weights">{weights}</ul><p>CQS = sum of seven weighted contributions. Review flags do not add a penalty. Missing components are never silently reweighted.</p></section>
<section><h2>All cases</h2><p>Hover over a colored segment to see its contribution. Depths and floor residual RMS are in millimetres.</p>
<input id="filter" placeholder="Filter case, e.g. O_41" aria-label="Filter cases" oninput="filterRows()">
<button onclick="sortRows('case')">Case order</button><button onclick="sortRows('score')">Descending score</button>
<div class="scroll"><table><thead><tr><th>Case</th><th>CQS / 10</th><th>O EFD %</th><th>P EFD %</th><th>O depth</th><th>Gingival − pulpal depth</th>
<th>Width / distance</th><th>O floor RMS</th><th>P floor RMS</th><th>Measurement review</th></tr></thead><tbody id="cases">{''.join(body_rows)}</tbody></table></div></section>
<section><h2>Scoring definitions</h2><p>Both EFD scores use <code>average_matching_score_pct / 100</code>, without cohort rescaling.</p>
<p>Depth score = max(0, 1 − distance outside the full-credit band / tolerance).
Occlusal band: {od['min_mm']:g}–{od['max_mm']:g} mm, tolerance {od['tolerance_mm']:g} mm.
Pulpal-to-gingival band: {gd['min_mm']:g}–{gd['max_mm']:g} mm, tolerance {gd['tolerance_mm']:g} mm.</p>
<p>{depth_definition}</p>
<p>Ratio score = 2<sup>−[ln(r / {config['ratio']['target']:.8g}) / ln({config['ratio']['half_score_multiplier']:g})]²</sup>.
The selected local isthmus/cusp pair is used. Floor regularity score = exp(−RMS / scale), with scales
{config['regularity_scale_mm']['occlusal']:g} mm (occlusal) and {config['regularity_scale_mm']['proximal']:g} mm (proximal).
This measures floor planarity, not microscopic surface roughness.</p>
<p>The default gingival band (0.5–1.0 mm) follows the range in <a href="{PRIMARY_SOURCE}">Azhari et al., Table 1</a>;
the parameters applied to this run are listed above.
The continuous penalty is our adaptation; the full composite score has not been validated by that study or against faculty scores.</p>
<p>All raw student measurements, including negative gingival depths, are preserved. An unusual preparation is not automatically a detection error.
Negative values receive the specified depth penalty in this provisional calculation, while floor labels and source geometry remain subject to review.
No letter grades or pass/fail decisions are inferred.</p></section>
<section><h2>Component contributions</h2><img src="CQS_overall.png" alt="CQS component contribution chart for all cases"></section></main>
<script>
function filterRows(){{const text=document.getElementById('filter').value.toLowerCase();for(const row of document.querySelectorAll('#cases tr'))row.hidden=!row.cells[0].textContent.toLowerCase().includes(text);}}
function sortRows(key){{const body=document.getElementById('cases');const rows=Array.from(body.rows);rows.sort((a,b)=>key==='case'?Number(a.dataset.case)-Number(b.dataset.case):Number(b.dataset.score)-Number(a.dataset.score)||Number(a.dataset.case)-Number(b.dataset.case));for(const row of rows)body.appendChild(row);}}
</script></html>'''
    (output/"index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occlusal-efd-csv", type=Path, default=ROOT_RESULTS/"final_avg_efd_results/occlusal/efd_similarity_scores.csv")
    parser.add_argument("--proximal-efd-csv", type=Path, default=ROOT_RESULTS/"final_avg_efd_results/proximal/efd_similarity_scores.csv")
    parser.add_argument("--depth-csv", type=Path, default=ROOT_RESULTS/"publication_cavity_depth_results/depth_summary.csv")
    parser.add_argument("--ratio-csv", type=Path, default=ROOT_RESULTS/"cus_ist_ratio_results_cusp_rescue/ratios.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT_RESULTS/"CQS_overall_results")
    parser.add_argument("--minimal-schema-from", type=Path, help="Keep the column order of an existing minimal CSV")
    parser.add_argument("--gingival-depth-min", type=float, default=.5)
    parser.add_argument("--gingival-depth-max", type=float, default=1.)
    parser.add_argument("--gingival-depth-tol", type=float, default=.5)
    parser.add_argument("--occlusal-depth-min", type=float, default=1.5)
    parser.add_argument("--occlusal-depth-max", type=float, default=2.)
    parser.add_argument("--occlusal-depth-tol", type=float, default=.5)
    parser.add_argument("--ratio-target", type=float, default=1/3)
    parser.add_argument("--ratio-half-score-multiplier", type=float, default=1.5)
    parser.add_argument("--occlusal-regularity-scale", type=float, default=.5)
    parser.add_argument("--proximal-regularity-scale", type=float, default=.5)
    args = parser.parse_args()
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    for name in ("occlusal", "gingival"):
        config[name+"_depth"] = {"min_mm": getattr(args, name+"_depth_min"), "max_mm": getattr(args, name+"_depth_max"),
                                "tolerance_mm": getattr(args, name+"_depth_tol")}
    config["ratio"] = {"target": args.ratio_target, "half_score_multiplier": args.ratio_half_score_multiplier}
    config["regularity_scale_mm"] = {"occlusal": args.occlusal_regularity_scale, "proximal": args.proximal_regularity_scale}
    validate_config(config)
    output = args.out_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty output directory to preserve existing results")
    inputs = {"occlusal_efd": args.occlusal_efd_csv, "proximal_efd": args.proximal_efd_csv,
              "depth": args.depth_csv, "ratio": args.ratio_csv}
    protected = {str(p.resolve()): sha256(p) for p in inputs.values()}
    depth_table, depth_schema = read_depth_table(args.depth_csv)
    sources = {
        "occlusal_efd": read_table(args.occlusal_efd_csv, "sample", ["average_matching_score_pct", "status", "cavity_type"], "occlusal"),
        "proximal_efd": read_table(args.proximal_efd_csv, "sample", ["average_matching_score_pct", "status", "cavity_type"], "proximal"),
        "depth": depth_table,
        "ratio": read_table(args.ratio_csv, "filename", ["ratio_isthmus_to_intercuspal", "status", "isthmus_width_px", "intercuspal_distance_px",
                             "opposite_sides_valid", "shared_cusp_pair", "crossing_checks_relaxed"]),
    }
    keys = sorted(set().union(*(set(table) for table in sources.values())))
    rows, components = [], []
    for key in keys:
        row, detail = make_row(key, sources, config)
        if depth_schema == "publication":
            row = {name: value for name, value in row.items() if name not in LEGACY_REFERENCE_COLUMNS}
        rows.append(row)
        components.extend(detail)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"CQS_overall_scores.csv", rows)
    write_csv(output/"CQS_component_scores.csv", components)
    minimal_columns = write_minimal_csv(output/"cqs_minimal.csv", rows, args.minimal_schema_from)
    # Save all parameters and source definitions, rather than implicit defaults.
    metadata = {"version": VERSION, **config,
                "weights": {key: settings[1] for key, settings in COMPONENTS.items()},
                "depth_source_schema": depth_schema,
                "depth_columns": {"occlusal": "OcclusalDepthMedian_mm", "proximal":
                                  "GingivalMinusPulpalDepth_mm" if depth_schema == "publication" else "GingivalFromPulpalMedian_mm"},
                "depth_export_column": "GingivalFromPulpalMedian_mm",
                "depth_definition": ("ProximalDepthMedian_mm - OcclusalDepthMedian_mm" if depth_schema == "publication"
                                     else "supplied local pulpal-floor reference measurement"),
                "minimal_columns": minimal_columns,
                "regularity_columns": {r: r.title()+"FloorResidualRMS_mm" for r in ["occlusal", "proximal"]},
                "score_scale": [0, 10], "missing_components_renormalized": False,
                "review_flags_change_numeric_score": False, "rubric_status": "provisional_not_faculty_calibrated",
                "gingival_band_source": PRIMARY_SOURCE,
                "gingival_penalty_source": "Project continuous linear adaptation, not the published step rubric",
                "signed_negative_depth_policy": "Preserve raw signed value; apply provisional depth score; require review",
                "clinical_accuracy_validated": False}
    (output/"scoring_config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    plot_scores(rows, output)
    write_html(rows, metadata, output)
    # Verify exported sums and joins independently from file round-trips.
    exported = list(csv.DictReader((output/"CQS_overall_scores.csv").open()))
    maximum_error = 0.
    for row in exported:
        score = number(row["CQS_overall"])
        if score is not None:
            point_sum = sum(float(row["points_"+key]) for key in COMPONENTS)
            error = abs(score-point_sum)
            maximum_error = max(maximum_error, error)
            assert error < 1e-10 and 0 <= score <= 10+1e-10
            assert math.isclose(float(row["available_weight"]), 1.)
    duplicate_checks = []
    groups = {}
    for row in rows:
        signature = tuple(row["q_"+key] for key in COMPONENTS)
        groups.setdefault(signature, []).append(row)
    for group in groups.values():
        if len(group) > 1 and group[0]["CQS_overall"] is not None:
            error = max(r["CQS_overall"] for r in group)-min(r["CQS_overall"] for r in group)
            assert error < 1e-10
            duplicate_checks.append({"cases": [r["Tooth"] for r in group], "score_difference": error})
    unchanged = all(sha256(p) == digest for p, digest in protected.items())
    assert unchanged
    complete = [r for r in rows if r["CQS_overall"] is not None]
    validation = {"case_count": len(rows), "complete_scores": len(complete), "component_row_count": len(components),
                  "all_source_case_sets_equal": all(set(table) == set(keys) for table in sources.values()),
                  "case_ids": keys, "weight_sum": sum(x[1] for x in COMPONENTS.values()),
                  "efd_weights_equal": COMPONENTS["efd_occlusal"][1] == COMPONENTS["efd_proximal"][1],
                  "efd_source_column": config["efd_column"], "no_weight_renormalization": True,
                  "depth_source_schema": depth_schema, "depth_source_column": metadata["depth_columns"]["proximal"],
                  "minimal_column_count": len(minimal_columns),
                  "exported_component_sum_max_error": maximum_error, "input_files_unchanged": unchanged,
                  "identical_component_checks": duplicate_checks,
                  "negative_gingival_median_cases": [r["Tooth"] for r in rows if r["GingivalFromPulpalMedian_mm"] is not None and r["GingivalFromPulpalMedian_mm"] < 0],
                  "minimum_CQS": min((r["CQS_overall"] for r in complete), default=None),
                  "maximum_CQS": max((r["CQS_overall"] for r in complete), default=None),
                  "clinical_rubric_validated": False}
    (output/"validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False))
    (output/"run_manifest.json").write_text(json.dumps({"version": VERSION, "command": sys.argv,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(__file__),
        "inputs": {key: str(p.resolve()) for key, p in inputs.items()}, "input_sha256": protected,
        "input_files_unchanged": unchanged, "scoring_config_sha256": sha256(output/"scoring_config.json")}, indent=2))
    (output/"README.md").write_text(f'''# Overall Class II CQS

{len(complete)} complete provisional scores across {len(rows)} cases. Open `index.html`
for the sortable report, original measurements, component contributions and review notes.
`CQS_overall_scores.csv` contains the overall score, raw inputs, normalized components,
weighted points and source flags. `CQS_component_scores.csv` contains one row per
case/component. `cqs_minimal.csv` provides the compact results table.
PNG is 600 dpi; PDF is vector.

## Weights and measurements

Occlusal EFD 25%, proximal EFD 25%, occlusal depth 10%, pulpal-to-gingival
depth 10%, selected isthmus/intercuspal ratio 20%, occlusal floor regularity 5%,
proximal floor regularity 5%. Sum = 100%; maximum CQS = 10.

Both EFD values use `average_matching_score_pct / 100`, the existing average
over unique references. Best-reference `similarity_pct` is not used. The source
average is not rescaled to this cohort and duplicate specimens are not removed.

Depth source: `{args.depth_csv}` (schema: `{depth_schema}`).
Occlusal depth uses `OcclusalDepthMedian_mm`. Pulpal-to-gingival depth uses
`{metadata['depth_columns']['proximal']}`: {metadata['depth_definition']}.
The existing `GingivalFromPulpalMedian_mm` export column is retained as a
compatibility alias for that source measurement, including in the minimal CSV.
For publication inputs this is a difference of regional medians, rather than
a median distance from gingival samples to a local pulpal plane. The absolute
proximal median is included in the full table to verify the subtraction.
Floor regularity uses `OcclusalFloorResidualRMS_mm` and
`ProximalFloorResidualRMS_mm`, not the obsolete three-regional-maxima SD.
Ratio uses the saved selected local pair, checked against width/distance.

## Scoring functions

`q_efd = average_matching_score_pct / 100`.
`q_depth = max(0, 1 - distance(depth, [min,max]) / tolerance)`.
`q_ratio = 2 ** (-(ln(r/target)/ln(multiplier))**2)`.
`q_regularity = exp(-floor_RMS / scale)`.
`CQS = 10 * sum(weight * q)`.

All actual parameters are in `scoring_config.json`. Occlusal depth follows the
previous project recommendation. The default provisional 0.5–1.0 mm gingival band is
supported by Table 1 in
[Azhari et al. (2024)]({PRIMARY_SOURCE}). Our continuous linear penalty is
an adaptation, not their published categorical scoring function. The composite
weights, penalties and floor-regularity scales require faculty calibration.

## Student preparations and review

Source measurements are used as supplied. All finite
signed gingival depths are retained; negative values receive the specified
depth penalty in this provisional score and remain flagged. A negative or
unusual value can reflect preparation geometry, extrapolation or a label error;
the scorer does not decide which. Source flags never add a confidence penalty.
All scores are provisional; no clinical grades or pass/fail labels are assigned.

Missing/invalid components yield an explicit incomplete score and are not
renormalized. Case IDs are joined across all source tables without dropping
incomplete cases; duplicate IDs and incorrect source schemas raise errors.
Reference-sensitivity fields are exported only for the older local-plane input
schema; they are not applicable to the publication difference-of-medians method.

## Reproduction

`python3 final_CQS_overall.py --out-dir NEW_EMPTY_DIRECTORY`

Use `--help` for input CSV and rubric parameter overrides.
`python3 -m unittest discover -s tests -p test_final_CQS_overall.py`

Inputs are from `RESULTS-21-09-2026`; hashes and command are in
`run_manifest.json`. Existing `CQS_overall.py` and its old CSV are preserved.
''')
    print(f"Generated {len(complete)}/{len(rows)} complete provisional CQS scores: {output/'CQS_overall_scores.csv'}")
    print(f"Range: {validation['minimum_CQS']:.4f}–{validation['maximum_CQS']:.4f} / 10" if complete else "No complete scores")
    return 0 if len(complete) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
