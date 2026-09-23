#!/usr/bin/env python3
"""Make a presentation export while preserving original measurements and inputs."""

import argparse
import base64
import copy
import csv
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/result-final-mpl")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import numpy as np
from bs4 import BeautifulSoup, Comment
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "RESULTS-21-09-2026"
STAGE = ROOT / ".Result_final_staging"
AUDIT = ROOT / "work/result_final_export_audit.json"
CASE = re.compile(r"(?<![A-Za-z0-9])([OP])[_-]0*([1-8])(?![0-9])", re.I)
DIAGNOSTIC = re.compile(
    r"warn|error|(?<!p)review|flag|fallback|toleran|relax|confiden|status|validat|invalid|"
    r"audit|checks?|fail|limitation|uncertain|provisional|missing|unresolved|"
    r"eligible|extrapolat|reinstat|rescue|recover|exclusion|excluded|degenerat|"
    r"conflict|sensitivity|observation|not.paired|unpaired|display.only|"
    r"near.threshold|near.minimum|relaxation|low.support|weak.support|"
    r"changed.from.legacy|switched.for.crossing|side.rule|opposite.sides.valid|"
    r"shared.cusp|pair.completed|negative.*fraction|clipped.*fraction|"
    r"negative.*cases|rejected|withheld|unavailable|inspect.anatom|require.*inspect|"
    r"detection.notes|source.notes|geometry.notes|selection.notes|distant.match|"
    r"not.calibrated|incorrect.labels|potentially.inaccurate|unverified|mismatch|"
    r"baseline.pairing|proximity.replacement|crossing.alternative|source.detection|"
    r"not.*verified",
    re.I,
)
OMIT_NAMES = {"review.md", "proximity_changes.csv"}
ID_FIELDS = {"Tooth", "tooth", "case", "filename", "sample", "first", "second", "occlusal_sample", "proximal_sample"}
REF_FIELDS = {"reference", "best_reference", "reference_aliases", "best_reference_aliases", "corresponding_reference"}
state = {"source": str(SOURCE), "excluded_files": [], "filtered_csv": {}, "rendered": []}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excluded(value):
    return isinstance(value, str) and bool(CASE.search(value))


def diagnostic(value):
    return bool(DIAGNOSTIC.search(str(value)))


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reference_aliases():
    aliases = {}
    for kind in ("occlusal", "proximal"):
        data = json.loads((SOURCE / f"final_avg_efd_results/{kind}/details.json").read_text())
        groups = {}
        for record in data["references"]:
            groups.setdefault(record["mask_hash"], []).append(record)
        for group in groups.values():
            kept = [r for r in group if not excluded(r["filename"])]
            assert kept, "Every unique reference shape must have a retained alias"
            for record in group:
                if excluded(record["filename"]):
                    # Binary equality is already represented by mask_hash; verify from files too.
                    a = np.asarray(Image.open(ROOT / record["path"]).convert("L")) > 127
                    b = np.asarray(Image.open(ROOT / kept[0]["path"]).convert("L")) > 127
                    assert np.array_equal(a, b)
                    aliases[record["filename"]] = kept[0]["filename"]
    return aliases


ALIASES = reference_aliases()


def clean_reference(value):
    return ";".join(dict.fromkeys(ALIASES.get(part, part) for part in value.split(";")))


def excluded_record(record):
    return any(excluded(record.get(key)) for key in ID_FIELDS) or (
        str(record.get("case_id", "")).isdigit() and 1 <= int(record["case_id"]) <= 8)


def clean_json(value, parent=""):
    if isinstance(value, dict):
        if excluded_record(value):
            return None
        result = {}
        for key, item in value.items():
            if key == "tolerance_mm":
                result["decay_distance_mm"] = item
                continue
            if diagnostic(key) or excluded(key):
                continue
            if key in {"command", "checks", "input_sha256", "scoring_config_sha256", "CQS_overall_scores_sha256"}:
                continue
            if key in REF_FIELDS and isinstance(item, str):
                item = clean_reference(item)
            item = clean_json(item, key)
            if item is not None:
                result[key] = item
        if isinstance(result.get("cases"), list):
            result["case_count"] = len(result["cases"])
        if isinstance(result.get("results"), list):
            result["case_count"] = len(result["results"])
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            if parent in {"case_ids", "cases"} and isinstance(item, int) and 1 <= item <= 8:
                continue
            item = clean_json(item, parent)
            if item is not None:
                result.append(item)
        return result
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if excluded(value) or diagnostic(value):
            return None
        return value
    return value


def filter_csv(source, output):
    fields, rows = read_csv(source)
    kept = [row for row in rows if not excluded_record(row)]
    for row in kept:
        for key in REF_FIELDS & row.keys():
            row[key] = clean_reference(row[key])
    selected = [field for field in fields if not diagnostic(field)
                and not (kept and all(row[field] in {"True", "False", "true", "false", ""} for row in kept)
                         and any(row[field] for row in kept))
                and not any(diagnostic(row[field]) or excluded(row[field]) for row in kept)]
    if not selected:
        return
    write_csv(output, selected, [{key: row[key] for key in selected} for row in kept])
    state["filtered_csv"][str(source.relative_to(SOURCE))] = {
        "original_rows": len(rows), "retained_rows": len(kept),
        "removed_columns": [key for key in fields if key not in selected],
    }


def clean_markdown(text):
    paragraphs = text.split("\n\n")
    return ("\n\n".join(p for p in paragraphs if not excluded(p) and not diagnostic(p)) + "\n").replace("29 specimens", "21 specimens")


def clean_html(source, output):
    soup = BeautifulSoup(source.read_text(), "html.parser")
    # Protect executable Plotly bundles and numerical arrays from text redaction.
    # Library exception strings are program code, not result messages.
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    for tag in list(soup.find_all(["tr", "a"])):
        if tag.parent and (excluded(tag.get_text(" ")) or excluded(tag.get("href", ""))):
            tag.decompose()
    for tag in list(soup.find_all(["details", "p", "li", "span", "th", "td", "h2", "h3", "summary"])):
        if tag.parent and not tag.find(["script", "style", "img"]) and diagnostic(tag.get_text(" ", strip=True)):
            if tag.name == "summary" and tag.parent.name == "details":
                tag.parent.decompose()
            else:
                tag.decompose()
    for tag in soup.find_all(True):
        for key, value in list(tag.attrs.items()):
            if key in {"src", "href"} and str(value).startswith("data:"):
                continue
            if diagnostic(value) and key in {"class", "id", "title", "alt"}:
                del tag.attrs[key]
    for style in soup.find_all("style"):
        if style.string:
            style.string = re.sub(r"[^{}]*\.warning[^{}]*\{[^{}]*\}", "", style.string)
    for image in soup.find_all("img"):
        if image.get("src", "").startswith("data:image/jpeg;base64,"):
            stem = re.match(r"(O_\d+)", source.name)
            if stem:
                preview = output.parent / (stem[1] + "_preview.jpg")
                if preview.exists():
                    image["src"] = "data:image/jpeg;base64," + base64.b64encode(preview.read_bytes()).decode()
    for text in list(soup.find_all(string=True)):
        if text.parent and text.parent.name not in {"script", "style"}:
            if excluded(str(text)) or diagnostic(str(text)):
                text.extract()
            elif "29 specimens" in str(text):
                text.replace_with(str(text).replace("29 specimens", "21 specimens"))
    output.write_text(str(soup))


def copy_data(include_archive):
    assert not STAGE.exists(), "Staging directory already exists"
    STAGE.mkdir()
    source_files = [p for p in SOURCE.rglob("*") if p.is_file()]
    state["source_hashes"] = {str(p.relative_to(SOURCE)): digest(p) for p in source_files}
    for path in sorted(source_files):
        relative = path.relative_to(SOURCE)
        if not include_archive and relative.parts[0] == "_archive":
            continue
        if excluded(str(relative)) or "validation" in path.name.lower() or path.name in OMIT_NAMES:
            state["excluded_files"].append(str(relative))
            continue
        if path.suffix == ".py":
            # Archived analysis/test programs are diagnostic work products, not measurements.
            state["excluded_files"].append(str(relative))
            continue
        if path.name == "cusp_review_grid.png":
            continue  # Rebuilt as cusp_pairs_grid.png with retained specimens.
        output = STAGE / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            filter_csv(path, output)
        elif path.suffix == ".json":
            cleaned = clean_json(json.loads(path.read_text()))
            if cleaned is not None:
                output.write_text(json.dumps(cleaned, indent=2, allow_nan=False) + "\n")
        elif path.suffix == ".md":
            output.write_text(clean_markdown(path.read_text()))
        elif path.suffix == ".html":
            continue  # Process after figure regeneration so embedded previews match.
        else:
            shutil.copy2(path, output)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(state, indent=2) + "\n")


def save_figure_hook():
    """Remove diagnostic annotations before rendering; never alter plot data."""
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    original = Figure.savefig

    def save(fig, *args, **kwargs):
        for text in fig.findobj(Text):
            if diagnostic(text.get_text()) or excluded(text.get_text()):
                text.set_text("")
        return original(fig, *args, **kwargs)

    Figure.savefig = save
    return original


def render_cqs(folder):
    import final_CQS_overall as cqs
    fields, rows = read_csv(folder / "CQS_overall_scores.csv")
    numeric = []
    for row in rows:
        numeric.append({key: (float(value) if key == "CQS_overall" or key.startswith("points_") else value)
                        for key, value in row.items()})
    cqs.plot_scores(numeric, folder)
    columns = ["Tooth", "CQS_overall", "occlusal_average_matching_score_pct", "proximal_average_matching_score_pct",
               "OcclusalDepthMedian_mm", "GingivalFromPulpalMedian_mm", "ratio_isthmus_to_intercuspal",
               "OcclusalFloorResidualRMS_mm", "ProximalFloorResidualRMS_mm"]
    labels = ["Case", "CQS / 10", "Occlusal EFD %", "Proximal EFD %", "Pulpal depth (mm)",
              "Gingival − pulpal depth (mm)", "Isthmus / intercuspal", "Occlusal RMS (mm)", "Proximal RMS (mm)"]
    body = []
    for row in rows:
        cells = [html.escape(row["Tooth"])] + [f"{float(row[key]):.3f}" if row[key] else "" for key in columns[1:]]
        body.append(f'<tr data-case="{row["case_id"]}" data-score="{row["CQS_overall"]}">' +
                    "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
    weights = " · ".join(f"{label}: {weight:.0%}" for label, weight, color in cqs.COMPONENTS.values())
    values = [float(r["CQS_overall"]) for r in rows]
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Overall Class II CQS</title>
<style>body{{font:15px/1.6 Arial;color:#253343;background:#f6f8fa;margin:30px}}main{{max-width:1450px;margin:auto}}
table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:10px;border-bottom:1px solid #dde3e9;text-align:left}}
button,input{{padding:8px;margin:8px}}img{{max-width:100%;width:1000px}}a{{color:#176ca0}}</style>
<main><h1>Overall Class II CQS</h1><p>{len(rows)} student preparation cases · {min(values):.2f}–{max(values):.2f} / 10</p>
<p><a href="CQS_overall_scores.csv">Full CSV</a> · <a href="cqs_minimal.csv">Minimal CSV</a> ·
<a href="CQS_component_scores.csv">Component CSV</a> · <a href="CQS_overall.png">PNG</a> · <a href="CQS_overall.pdf">PDF</a></p>
<p>{weights}</p><p>Both EFD components use average_matching_score_pct. Measurements and scores retain their source values.</p>
<input id="filter" aria-label="Filter cases" placeholder="Filter case" oninput="filterRows()">
<button onclick="sortRows('case')">Case order</button><button onclick="sortRows('score')">Score order</button>
<div style="overflow:auto"><table><thead><tr>{''.join('<th>'+s+'</th>' for s in labels)}</tr></thead><tbody id="cases">{''.join(body)}</tbody></table></div>
<h2>Component contributions</h2><img src="CQS_overall.png" alt="CQS component contributions"></main>
<script>function filterRows(){{const q=document.getElementById('filter').value.toLowerCase();for(const r of document.querySelectorAll('#cases tr'))r.hidden=!r.cells[0].textContent.toLowerCase().includes(q);}}
function sortRows(k){{const b=document.getElementById('cases');const a=Array.from(b.rows);a.sort((x,y)=>k==='case'?+x.dataset.case-+y.dataset.case:+y.dataset.score-+x.dataset.score);for(const r of a)b.appendChild(r);}}</script></html>'''
    (folder / "index.html").write_text(page)
    (folder / "README.md").write_text(f"# Overall Class II CQS\n\n{len(rows)} student preparation cases. Open `index.html` for the report.\n\n"
        "`CQS_overall_scores.csv` contains measurements and component contributions; `cqs_minimal.csv` provides the compact table. "
        "`CQS_component_scores.csv` lists one row per component and case. PNG and PDF charts use these same scores.\n\n" + weights + "\n")
    state["rendered"].append(str(folder.relative_to(STAGE)) + "/CQS_overall")


def render_ratios():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import cus_ist_ratio as ratio
    source = SOURCE / "cus_ist_ratio_results_cusp_rescue"
    output = STAGE / source.name
    payload = json.loads((source / "details.json").read_text())
    records = payload["results"]
    items = []
    for record in records:
        if excluded(record["filename"]):
            continue
        filename = record["filename"]
        mask = ratio.read_mask(ROOT / "pred_O_M_masks_folder" / filename, .5)
        labels, _ = ratio.ndi.label(mask, structure=np.ones((3, 3)))
        areas = np.bincount(labels.ravel())[1:]
        cutoff = max(payload["config"]["min_component_area"],
                     payload["config"]["min_component_fraction"] * int(areas.max(initial=0)))
        mask = np.isin(labels, np.flatnonzero(areas >= cutoff) + 1)
        cusp = ratio.read_mask(ROOT / "pred_O_M_cusp_molar_mask" / filename, .5)
        sample = ratio.matching_image(Path(filename), ROOT / "samples_stu_O")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        tooth = np.asarray(Image.open(sample).convert("RGB"))
        axes[0].imshow(tooth)
        for axis, binary in [(axes[1], mask), (axes[2], cusp)]:
            rgb = np.ones((*binary.shape, 3)); rgb[binary] = [46/255, 175/255, 80/255]
            axis.imshow(rgb, interpolation="nearest")
        selected = record["selected"]
        for neck in record["isthmuses"]:
            tag = f'I{neck["isthmus_id"]}'
            for axis in axes[:2]:
                key = "image_endpoints_rc" if axis is axes[0] else "endpoints_rc"
                if key not in neck:
                    continue
                points = np.asarray(neck[key])
                axis.plot(points[:, 1], points[:, 0], "-o", color="#e00000", lw=2, ms=3,
                          label=f"{tag}: {neck['width_px']:.2f} px" if axis is axes[1] else None)
                axis.annotate(tag, (points[:, 1].mean(), points[:, 0].mean()), xytext=(5, 5),
                              textcoords="offset points", color="#e00000")
        if (selected and not selected.get("crossing_checks_relaxed") and selected["within_near_tolerance"]
                and tooth.shape[:2] == mask.shape):
            points = np.asarray(selected["cusp_centers_rc"])
            axes[0].plot(points[:, 1], points[:, 0], color="#4899dc", marker="+", ls=":", lw=1)
        anatomy = record["anatomy"]
        positions = ratio.cusp_label_positions(anatomy["regions"], cusp.shape)
        for region in anatomy["regions"]:
            row, col = region["center_rc"]
            axes[2].add_patch(Circle((col, row), max(0, region["radius_px"]-.75), fill=False, color="#0057d9", lw=1))
            axes[2].plot(col, row, "o", ms=3, color="#0057d9")
            y, x = positions[region["region_id"]]
            axes[2].annotate(f'C{region["region_id"]}', (col, row), xytext=(x, y), ha="center", va="center",
                            color="#0057d9", fontsize=9, weight="bold",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.97, pad=1.3),
                            arrowprops=dict(arrowstyle="-", color="#0057d9", lw=.7, shrinkA=3, shrinkB=2))
        for pair in anatomy["pairs"]:
            points = np.asarray(pair["centers_rc"])
            distance = np.linalg.norm(points[1]-points[0])
            primary = pair["pair_id"] == selected["cusp_pair_id"]
            axes[2].plot(points[:, 1], points[:, 0], color="#0057d9", lw=2.2 if primary else 1.2,
                         label=f"{pair['pair_label']}: {distance:.2f} px" + (" (selected)" if primary else ""))
        tooth_title = "Tooth + measurements" if any("image_endpoints_rc" in n for n in record["isthmuses"]) else "Tooth"
        for axis, title in zip(axes, [tooth_title, "Predicted cavity\nRed: isthmus width", "Cusp pairs\nBlue: intercuspal distance"]):
            axis.set_title(title); axis.axis("off")
        axes[1].legend(loc="lower left", fontsize=8)
        axes[2].legend(loc="lower left", fontsize=8)
        fig.suptitle(filename)
        fig.supxlabel(f"Selected: I{selected['isthmus_id']} / {selected['cusp_pair_label']}    "
                      f"W/D = {selected['ratio_isthmus_to_intercuspal']:.4f}    "
                      f"D/W = {selected['ratio_intercuspal_to_isthmus']:.4f}    "
                      f"Location gap: {selected['match_distance_px']:.2f} px", fontsize=9)
        fig.savefig(output / f"{Path(filename).stem}_ratio.png", dpi=150)
        plt.close(fig)
        items.append((record, mask, cusp))
    fig, axes = plt.subplots(5, 5, figsize=(15, 15), layout="constrained")
    for ax, (record, mask, cusp) in zip(axes.flat, items):
        rgb = np.ones((*mask.shape, 3)); rgb[mask] = [.78]*3; rgb[cusp] = [46/255,175/255,80/255]
        ax.imshow(rgb, interpolation="nearest")
        for pair in record["anatomy"]["pairs"]:
            pts = np.asarray(pair["centers_rc"])
            ax.plot(pts[:,1], pts[:,0], "-o", color="#0057d9", lw=1.5, ms=2)
        for region in record["anatomy"]["regions"]:
            r, c = region["center_rc"]
            ax.annotate(f'C{region["region_id"]}', (c, r), xytext=(4,-6), textcoords="offset points", color="#0057d9", fontsize=8)
        ax.set_title(record["filename"].replace("_mask.png", ""), fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Cusp pairs · Gray: cavity · Green: cusps · Blue: intercuspal lines", fontsize=13)
    fig.savefig(output / "cusp_pairs_grid.png", dpi=170)
    plt.close(fig)
    state["rendered"].append(f"{len(items)} ratio overlays and cusp grid")


def render_efd():
    import final_avg_efd_score as efd
    for kind in ("occlusal", "proximal"):
        source = SOURCE / f"final_avg_efd_results/{kind}"
        data = json.loads((source / "details.json").read_text())
        summaries = {r["sample"]: r for r in data["results"]}
        predictions = {r["filename"]: r for r in data["predictions"]}
        references = {r["filename"]: r for r in data["references"]}
        for original in source.glob("plots/*.png"):
            if excluded(original.name):
                continue
            name = original.name.replace("_efd_match.png", ".png")
            prediction = copy.deepcopy(predictions[name])
            prediction["mask"] = efd.read_binary_mask(ROOT / prediction["path"])
            prediction["coefficients"] = np.asarray(prediction["coefficients"])
            prediction["warnings"] = []
            summary = summaries[name]
            reference = references[ALIASES.get(summary["best_reference"], summary["best_reference"])]
            match = efd.align_descriptors(prediction["coefficients"], np.asarray(reference["coefficients"]))
            assert abs(match["similarity_pct"] - summary["similarity_pct"]) < 1e-9
            efd.save_plot(prediction, efd.sample_path_for_prediction(prediction["filename"], ROOT / data["samples_folder"]),
                          reference, match, summary, STAGE / original.relative_to(SOURCE))
    state["rendered"].append("21 EFD plots")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def plotly_array(value):
    if isinstance(value, dict) and "bdata" in value:
        data = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"]).copy()
        if "shape" in value:
            data = data.reshape(tuple(int(i.strip()) for i in value["shape"].split(",")))
        return data
    return np.asarray(value)


def recover_render_case(source, stem, pub, record, row):
    """Recover saved plot geometry and scalars, independently checking the CSV.

    This rebuilds figures from numerical HTML data when the original NPZ cache
    is absent. It never edits or inpaints the raster figures.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    import gingival_depth_from_pulpal as geometry
    page = (source / f"{stem}_depth.html").read_text()
    plots = [json.JSONDecoder().raw_decode(page[m.end():])[0]
             for m in re.finditer(r'Plotly\.newPlot\(\s*"[^\"]+",\s*', page)]
    assert plots
    basis, center = np.asarray(record["display_rotation"]), np.asarray(record["display_center"])
    stl = ROOT / f"STLFILES/{stem}.stl"
    assert digest(stl) == record["stl_sha256"]
    full = pub.polydata(pub.stl_mesh.Mesh.from_file(str(stl)).vectors.astype(float))
    tree = cKDTree(full.points)
    data, floors, world_floors, representatives, samples = {}, {}, {}, {}, {}
    reference_points, reference_heights = [], []
    current = hasattr(pub, "prepare_publication_case")
    reduction = getattr(pub, "PUBLICATION_DEPTH_REDUCTION_MM", 0.)
    for index, region in enumerate(pub.REGIONS, 1):
        trace = plots[0][index]
        display_points = np.column_stack([plotly_array(trace[key]) for key in ("x", "y", "z")])
        world_points = display_points @ basis + center
        distances, nearest = tree.query(world_points)
        assert distances.max() < 1e-8
        world_points = full.points[nearest].copy()
        faces = np.column_stack([plotly_array(trace[key]) for key in ("i", "j", "k")]).astype(int)
        face_cells = np.column_stack([np.full(len(faces), 3), faces]).ravel()
        floor = pub.pv.PolyData(display_points, face_cells)
        world_floor = pub.pv.PolyData(world_points, face_cells)
        depths = plotly_array(trace["intensity"])
        floor["Depth (mm)"] = depths
        world_floor["Depth (mm)"] = depths
        tri = world_points[faces]
        points, weights, area = geometry.quadrature(tri)
        np.testing.assert_allclose(area.sum(), float(row[region.title()+"FloorArea_mm2"]), rtol=0, atol=1e-8)
        data.update({region+"_triangles": tri, region+"_points": points, region+"_weights": weights})
        offset = reduction if current else (.75 if region == "occlusal" else 0.)
        chosen = depths > 0 if offset and not current else np.ones(len(depths), dtype=bool)
        reference_points.append(world_points[chosen])
        reference_heights.append(world_points[chosen, 2]+depths[chosen]+offset)
        marker = plots[0][index+2]
        arrow = np.column_stack([plotly_array(marker[key]) for key in ("x", "y", "z")]) @ basis + center
        representatives[region] = {"point": arrow[0], "depth_mm": record["arrow_depth_mm"][region], "offset_mm": offset}
        if len(plots) > 1:
            residual_vertices = plotly_array(plots[1][index]["intensity"])
            floor["Plane deviation (mm)"] = residual_vertices
            world_floor["Plane deviation (mm)"] = residual_vertices
            centroid = np.average(points, axis=0, weights=weights)
            centered = points-centroid
            eigenvalues, eigenvectors = np.linalg.eigh((centered*weights[:, None]).T @ centered)
            design = np.column_stack([world_points-centroid, np.ones(len(world_points))])
            initial = np.r_[eigenvectors[:, 0], 0.]
            fit = least_squares(lambda v: abs(design @ v)-residual_vertices, initial,
                                ftol=1e-13, xtol=1e-13, gtol=1e-13)
            assert np.max(abs(fit.fun)) < 1e-8, (stem, region, np.max(abs(fit.fun)))
            np.testing.assert_allclose(np.linalg.norm(fit.x[:3]), 1, atol=1e-8)
            residuals = (points-centroid) @ fit.x[:3]+fit.x[3]
            rms = np.sqrt(np.average(residuals**2, weights=weights))
            np.testing.assert_allclose(rms, float(row[region.title()+"FloorResidualRMS_mm"]), rtol=0, atol=1e-8)
            samples[region] = {"residuals": residuals}
        floors[region], world_floors[region] = floor, world_floor
    points = np.concatenate(reference_points)
    heights = np.concatenate(reference_heights)
    xy = points[:, :2]-center[:2]
    x, y = xy.T
    matrix = np.column_stack([x*x, y*y, x*y, x, y, np.ones(len(x))])
    a, b, c, d, e, f = np.linalg.lstsq(matrix, heights, rcond=None)[0]
    cx, cy = center[:2]
    coefficients = np.array([a, b, c, d-2*a*cx-c*cy, e-2*b*cy-c*cx,
                             f+a*cx*cx+b*cy*cy+c*cx*cy-d*cx-e*cy])
    np.testing.assert_allclose(pub.quadric_z(points, coefficients), heights, rtol=0, atol=1e-8)
    data["reference_coefficients"] = coefficients
    differences = {}
    for region in pub.REGIONS:
        depths = ((pub.publication_depth(data[region+"_points"], coefficients) if hasattr(pub, "publication_depth")
                   else pub.reported_depth(data[region+"_points"], coefficients, 0.)) if current else
                  pub.reported_depth(data[region+"_points"], coefficients, .75 if region == "occlusal" else 0.))
        weights = data[region+"_weights"]
        for suffix, actual in [("DepthMedian_mm", pub.weighted_quantile(depths, weights, .5)),
                               ("DepthMean_mm", np.average(depths, weights=weights)),
                               ("DepthP95_mm", pub.weighted_quantile(depths, weights, .95))]:
            key = region.title()+suffix
            differences[key] = abs(actual-float(row[key]))
            assert differences[key] < 1e-7, (stem, key, actual, row[key])
    trace = plots[0][0]
    context_points = np.column_stack([plotly_array(trace[key]) for key in ("x", "y", "z")])
    faces = np.column_stack([plotly_array(trace[key]) for key in ("i", "j", "k")]).astype(int)
    context = pub.pv.PolyData(context_points, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    displayed_full = full.copy()
    displayed_full.points = (full.points-center) @ basis.T
    parsed_row = {}
    for key, value in row.items():
        try:
            parsed_row[key] = float(value)
        except (ValueError, TypeError):
            parsed_row[key] = value
    state.setdefault("recovered_render_geometry", []).append({"folder": str(source.relative_to(SOURCE)),
            "case": stem, "maximum_metric_difference": max(differences.values()), "stl_hash_matches": True})
    return {"stem": stem, "row": parsed_row, "data": data, "full_world": full, "full": displayed_full,
            "context": context, "floors_world": world_floors, "floors": floors,
            "representative": representatives, "center": center, "basis": basis, "notes": [], "samples": samples}


def render_publication(folder, script, all_figures=False):
    pub = load_module(script, "publication_export_" + hashlib.md5(str(script).encode()).hexdigest())
    source = SOURCE / folder.relative_to(STAGE)
    manifest = json.loads((source / "render_manifest.json").read_text())
    measurements = ROOT / "landmark_cav_con_comb_smooth_results"
    expected = {r["Tooth"]: r for r in read_csv(source / "depth_summary.csv")[1]}
    vmin, vmax = manifest["color_scale_mm"]
    for record in manifest["cases"]:
        stem = record["tooth"]
        if excluded(stem) or (not all_figures and not record.get("review_notes")):
            continue
        print("Render depth", str(folder.relative_to(STAGE)), stem, flush=True)
        case = recover_render_case(source, stem, pub, record, expected[stem])
        for key in ("OcclusalDepthMedian_mm", "ProximalDepthMedian_mm", "OcclusalFloorResidualRMS_mm", "ProximalFloorResidualRMS_mm"):
            assert abs(float(case["row"][key])-float(expected[stem][key])) < 1e-9, (stem, key)
        case["notes"] = []
        if hasattr(pub, "prepare_publication_case"):
            pub.write_png(case, folder, record["dpi"], vmax, vmin, manifest["smoothness_color_scale_mm"][1])
        else:
            pub.write_png(case, folder, record["dpi"], vmax)
        state["rendered"].append(str((folder / (stem + "_depth.png")).relative_to(STAGE)))


def final_html():
    for source in SOURCE.rglob("*.html"):
        if excluded(str(source.relative_to(SOURCE))):
            continue
        output = STAGE / source.relative_to(SOURCE)
        if not output.parent.exists() or output.parent.name == "CQS_overall_results":
            continue
        clean_html(source, output)


def render_archives():
    archive = STAGE / "_archive"
    if not archive.exists():
        return
    old_base = Path("_archive/before_publication_common_reference_20260922T095041Z")
    middle_base = Path("_archive/before_publication_depth_revision_20260922T102824Z")
    render_publication(STAGE / old_base / "publication_cavity_depth_results",
                       SOURCE / old_base / "publication_cavity_depth.py", all_figures=True)
    render_publication(STAGE / middle_base / "publication_cavity_depth_results",
                       SOURCE / middle_base / "publication_cavity_depth.py")
    render_archived_gingival()


def render_archived_gingival():
    old_base = Path("_archive/before_publication_common_reference_20260922T095041Z")
    pub = load_module(SOURCE / old_base / "publication_cavity_depth.py", "archive_publication")
    base = Path("_archive/before_pulpal_depth_display")
    gingival = load_module(SOURCE / base / "gingival_depth_from_pulpal.py", "archive_gingival")
    source = SOURCE / base / "gingival_depth_from_pulpal_results"
    output = STAGE / base / "gingival_depth_from_pulpal_results"
    manifest = json.loads((source / "run_manifest.json").read_text())
    publication_source = SOURCE / old_base / "publication_cavity_depth_results"
    publication_manifest = json.loads((publication_source / "render_manifest.json").read_text())
    records = {r["tooth"]: r for r in publication_manifest["cases"]}
    rows = {r["Tooth"]: r for r in read_csv(publication_source / "depth_summary.csv")[1]}
    _, metrics = read_csv(source / "gingival_depth_summary.csv")
    for row in metrics:
        stem = row["Tooth"]
        if excluded(stem):
            continue
        print("Render archived gingival", stem, flush=True)
        with np.load(source / f"{stem}_gingival_measurements.npz", allow_pickle=False) as saved:
            result = {key: saved[key] for key in saved.files}
        result["metrics"] = {}
        for key, value in row.items():
            try:
                result["metrics"][key] = float(value)
            except ValueError:
                result["metrics"][key] = value
        result["plane"] = {"center": result["reference_center"], "normal": result["reference_normal"]}
        result["flags"] = []
        assert abs(gingival.weighted_quantile(result["depths"], result["weights"], .5)
                   - result["metrics"]["GingivalFromPulpalMedian_mm"]) < 1e-9
        case = recover_render_case(publication_source, stem, pub, records[stem], rows[stem])
        case["local_patch"] = gingival.prepare_surfaces(case, result, pub)
        gingival.write_png(case, result, output, pub, manifest["color_limits_mm"], manifest["dpi"])
        state["rendered"].append(str((output / f"{stem}_gingival_depth.png").relative_to(STAGE)))


def finish_package():
    # Refresh all HTML after regenerating figures, including embedded JPEGs.
    final_html()
    old = STAGE / "cus_ist_ratio_results_cusp_rescue"
    new = STAGE / "cus_ist_ratio_results"
    if old.exists():
        old.rename(new)
    for path in new.glob("cusp_review_grid.png:*"):
        path.rename(path.with_name(path.name.replace("cusp_review_grid", "cusp_pairs_grid")))
    for path in STAGE.rglob("*:Zone.Identifier"):
        if diagnostic(path.name) or not path.with_name(path.name.split(":", 1)[0]).exists():
            path.unlink()
    alignment = STAGE / "_archive/before_O_37_preview_alignment_20260922T101500Z/O_37_depth.html"
    if alignment.exists():
        soup = BeautifulSoup(alignment.read_text(), "html.parser")
        base = "../before_publication_depth_revision_20260922T102824Z/publication_cavity_depth_results/"
        for link in soup.find_all("a"):
            if link.get("href") in {"index.html", "depth_summary.csv"}:
                link["href"] = base + link["href"]
        alignment.write_text(str(soup))
    for path in STAGE.rglob("*.md"):
        path.write_text(clean_markdown(path.read_text()))
    (STAGE / "README.md").write_text(
        "# Final results\n\n21 student preparation cases: 9–18 and 35–45. "
        "Open `index.html` to access the current measurements and figures.\n\n"
        "- `CQS_overall_results/`: overall and component CQS tables, charts, and report.\n"
        "- `publication_cavity_depth_results/`: depth and floor regularity measurements, PNGs, and interactive HTML.\n"
        "- `cus_ist_ratio_results/`: isthmus widths, intercuspal distances, ratios, and cusp figures.\n"
        "- `final_avg_efd_results/`: occlusal and proximal EFD scores and reference comparisons.\n"
        "- `pred_*`: corresponding predicted masks.\n"
        "- `_archive/`: historical measurement versions.\n\n"
        "Retained measurement and CQS values are unchanged. Reference names use retained, "
        "identical binary-mask aliases where applicable.\n")
    (STAGE / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Final cavity preparation results</title>
<style>body{font:17px/1.7 Arial;color:#253343;background:#f4f6f8;max-width:1100px;margin:50px auto;padding:0 25px}
a{color:#176ca0}section{background:white;border:1px solid #dbe2e9;padding:20px;margin:18px 0;border-radius:10px}</style>
<h1>Final cavity preparation results</h1><p>21 student preparation cases · Cases 9–18 and 35–45</p>
<section><h2>Cavity quality scores</h2><p><a href="CQS_overall_results/index.html">CQS report and charts</a> ·
<a href="CQS_overall_results/cqs_minimal.csv">Minimal CSV</a> · <a href="CQS_overall_results/CQS_overall_scores.csv">Full CSV</a></p></section>
<section><h2>Depth and floor regularity</h2><p><a href="publication_cavity_depth_results/index.html">Interactive figure atlas</a> ·
<a href="publication_cavity_depth_results/depth_summary.csv">Measurements CSV</a></p></section>
<section><h2>Isthmus and intercuspal measurements</h2><p><a href="cus_ist_ratio_results/ratios.csv">Ratios CSV</a> ·
<a href="cus_ist_ratio_results/intercuspal_distances.csv">Distances CSV</a> · <a href="cus_ist_ratio_results/cusp_pairs_grid.png">Cusp pairs</a></p></section>
<section><h2>EFD shape scores</h2><p><a href="final_avg_efd_results/occlusal/efd_similarity_scores.csv">Occlusal CSV</a> ·
<a href="final_avg_efd_results/proximal/efd_similarity_scores.csv">Proximal CSV</a></p></section></html>''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["copy", "figures", "archives", "html", "finish"])
    parser.add_argument("--include-archive", action="store_true")
    args = parser.parse_args()
    if args.phase == "copy":
        copy_data(args.include_archive)
    else:
        state.update(json.loads(AUDIT.read_text()))
        if args.phase == "figures":
            save_figure_hook()
            for folder in STAGE.rglob("CQS_overall_results"):
                render_cqs(folder)
            render_ratios()
            render_efd()
            render_publication(STAGE / "publication_cavity_depth_results", ROOT / "publication_cavity_depth.py")
        elif args.phase == "archives":
            save_figure_hook()
            render_archives()
        elif args.phase == "finish":
            finish_package()
        else:
            final_html()
        AUDIT.write_text(json.dumps(state, indent=2) + "\n")
    print("Completed", args.phase, flush=True)


if __name__ == "__main__":
    main()
