#!/usr/bin/env python3
"""Relabel the presentation export without changing its measured values."""
import argparse
import base64
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/result-final-relabel-mpl")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")
ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "Result_final"
STAGE = FINAL / ".case_rename_stage"
TEMP = Path("/tmp/result_final_case_rename")
BEFORE = TEMP / "before"
OLD_IDS = list(range(9, 19)) + list(range(35, 46))
MAPPING = {old: new for new, old in enumerate(OLD_IDS, 1)}
CASE = re.compile(r"(?<![A-Za-z0-9])([OP])[_-]0*(\d+)(?:_st)?(?![0-9])")
CASE_FILE = re.compile(r"(?<![A-Za-z0-9])([OP])[_-]0*(\d+)(?:_st)?(?:_[A-Za-z][A-Za-z0-9_-]*)?\.(?:png|jpe?g|html|stl|npz|csv|json)", re.I)
REFERENCE_FIELDS = {"reference", "best_reference", "best_reference_aliases", "reference_aliases", "corresponding_reference"}
DROP_FIELDS = {"path", "source_path", "source_measurements", "protected_inputs", "inputs",
               "pred_folder", "ref_folder", "samples_folder", "depth_source", "isthmus_json",
               "png", "html", "complete_figure_aliases", "occlusal_sample", "proximal_sample",
               "ratio_source_filename"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def label(number):
    return f"MCL2_sample_{MAPPING[int(number)]}"


def names(text):
    return CASE.sub(lambda m: label(m[2]) if int(m[2]) in MAPPING else m[0], str(text))


def display(text):
    text = CASE_FILE.sub(lambda m: label(m[2]) if int(m[2]) in MAPPING else m[0], str(text))
    text = names(text)
    return text.replace("9–18 and 35–45", "1–21")


def case_number(text):
    m = CASE.search(str(text))
    return MAPPING[int(m[2])] if m and int(m[2]) in MAPPING else None


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reference_names():
    result = {}
    for kind in ("occlusal", "proximal"):
        data = json.loads((BEFORE / f"final_avg_efd_results/{kind}/details.json").read_text())
        records = sorted(data["references"], key=lambda row: int(CASE.search(row["filename"])[2]))
        for index, row in enumerate(records, 1):
            result[row["filename"]] = f"{kind.title()}_reference_{index}"
    return result


def reference_label(value):
    aliases = reference_names()
    return ";".join(dict.fromkeys(aliases.get(part, display(part)) for part in value.split(";")))


def snapshot():
    assert not STAGE.exists() and not (TEMP / "before_hashes.json").exists()
    TEMP.mkdir(exist_ok=True)
    # Snapshot existing files for comparison; model files can remain hardlinked.
    def copier(src, dst):
        if Path(src).suffix == ".h5":
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        return dst
    shutil.copytree(FINAL, BEFORE, copy_function=copier, dirs_exist_ok=True)
    shutil.copytree(BEFORE, STAGE, copy_function=copier)
    outside = {}
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if Path(folder) / d != FINAL]
        for name in files:
            p = Path(folder) / name
            st = p.stat()
            outside[str(p.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns]
    (TEMP / "outside.json").write_text(json.dumps(outside))
    hashes = {str(p.relative_to(BEFORE)): digest(p) for p in BEFORE.rglob("*") if p.is_file()}
    (TEMP / "before_hashes.json").write_text(json.dumps(hashes))
    print(f"Prepared {len(hashes)} files", flush=True)


def exporter():
    sys.path.insert(0, str(ROOT))
    import export_result_final as export
    export.STAGE = STAGE
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    original = Figure.savefig

    def save(fig, *args, **kwargs):
        for ax in fig.axes:
            if any(CASE.search(t.get_text()) for t in ax.get_yticklabels()):
                ax.set_yticks(ax.get_yticks(), labels=[display(t.get_text()) for t in ax.get_yticklabels()])
        for text in fig.findobj(Text):
            value = text.get_text()
            text.set_text("" if export.diagnostic(value) or export.excluded(value) else display(value))
        if args and Path(args[0]).name.startswith("CQS_overall"):
            fig.subplots_adjust(left=.245, top=.91)
            for text in fig.texts:
                if text.get_text().startswith("Equal EFD weights"):
                    text.set_position((.10, .945))
        if kwargs.get("metadata"):
            kwargs["metadata"] = {k: display(v) for k, v in kwargs["metadata"].items()}
        return original(fig, *args, **kwargs)

    Figure.savefig = save
    return export


def render(phase):
    export = exporter()
    if phase == "cqs":
        for folder in STAGE.rglob("CQS_overall_results"):
            export.render_cqs(folder)
    elif phase == "small":
        for folder in STAGE.rglob("CQS_overall_results"):
            export.render_cqs(folder)
        output = STAGE / "cus_ist_ratio_results_cusp_rescue"
        output.mkdir(exist_ok=True)
        export.render_ratios()
        for p in output.iterdir():
            shutil.move(str(p), str(STAGE / "cus_ist_ratio_results" / p.name))
        output.rmdir()
        import final_avg_efd_score as efd
        original = efd.save_plot
        def plot(prediction, sample_path, reference, match, summary, output):
            reference = {**reference, "filename": reference_label(reference["filename"])}
            return original(prediction, sample_path, reference, match, summary, output)
        efd.save_plot = plot
        export.render_efd()
    elif phase == "current":
        export.render_publication(STAGE / "publication_cavity_depth_results", ROOT / "publication_cavity_depth.py", True)
    elif phase in {"old", "middle"}:
        name = ("before_publication_common_reference_20260922T095041Z" if phase == "old"
                else "before_publication_depth_revision_20260922T102824Z")
        base = Path("_archive") / name
        export.render_publication(STAGE / base / "publication_cavity_depth_results", export.SOURCE / base / "publication_cavity_depth.py", True)
    elif phase == "gingival":
        export.render_archived_gingival()
    elif phase == "single":
        base = next(p for p in (STAGE / "_archive").iterdir() if "preview_alignment" in p.name)
        relative = base.relative_to(STAGE)
        src = export.SOURCE / relative
        pub = export.load_module(src / "publication_cavity_depth.py", "single_publication")
        manifest = json.loads((src / "render_manifest.json").read_text())
        stem = next(base.glob("*_depth.html")).name.removesuffix("_depth.html")
        record = next(r for r in manifest["cases"] if r["tooth"] == stem)
        rows = read_csv(export.SOURCE / "_archive/before_publication_depth_revision_20260922T102824Z/publication_cavity_depth_results/depth_summary.csv")[1]
        row = next(r for r in rows if r["Tooth"] == stem)
        case = export.recover_render_case(src, stem, pub, record, row)
        case["notes"] = []
        pub.write_png(case, base, record["dpi"], manifest["color_scale_mm"][1], manifest["color_scale_mm"][0], manifest["smoothness_color_scale_mm"][1])
    (TEMP / (phase + ".done")).write_text("complete\n")
    print("Rendered", phase, flush=True)


def convert_json(value, parent="", script_metadata=False):
    if isinstance(value, list):
        return [convert_json(v, parent, script_metadata) for v in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not script_metadata and key in DROP_FIELDS:
                continue
            if key == "case_id":
                result[key] = MAPPING[int(item)] if int(item) in MAPPING else item
            elif key in {"filename", "sample", "case", "Tooth", "tooth"} and isinstance(item, str) and case_number(item):
                if parent == "references":
                    result["reference_id"] = reference_label(item)
                else:
                    result["case_id"] = case_number(item)
                    result["Tooth"] = f"MCL2_sample_{case_number(item)}"
            elif key in REFERENCE_FIELDS and isinstance(item, str) and CASE.search(item):
                result[key] = reference_label(item)
            else:
                result[names(key)] = convert_json(item, key, script_metadata)
        return result
    if isinstance(value, str):
        return display(value)
    return value


def convert_csv(path):
    fields, rows = read_csv(path)
    script_metadata = "scripts" in path.relative_to(STAGE).parts
    identity = next((k for k in ["Tooth", "filename", "sample", "case"]
                     if k in fields and any(case_number(row[k]) for row in rows)), None)
    selected = fields if script_metadata else [k for k in fields if k not in DROP_FIELDS]
    output_fields = ["case_id", "Tooth"] + [k for k in selected if k not in {"case_id", "Tooth", identity}] if identity else selected
    converted = []
    for row in rows:
        new = {}
        for key in selected:
            value = row[key]
            if identity and key == identity:
                new["Tooth"] = display(value)
                new["case_id"] = str(case_number(value))
            elif key == "case_id":
                new[key] = str(MAPPING[int(value)])
            elif key in REFERENCE_FIELDS and CASE.search(value):
                new[key] = reference_label(value)
            elif path.name == "script_inventory.csv" and key == "path":
                new[key] = names(value)
            elif path.name == "script_inventory.csv" and key == "source" and CASE.search(value):
                new[key] = ""
            else:
                new[key] = display(value)
        converted.append(new)
    if identity:
        converted.sort(key=lambda row: (row.get("cavity_type", ""), int(row["case_id"])))
    write_csv(path, output_fields, converted)


def convert_html(path):
    from bs4 import BeautifulSoup, Comment
    soup = BeautifulSoup(path.read_text(), "html.parser")
    for node in list(soup.find_all(string=True)):
        if isinstance(node, Comment):
            node.extract()
        elif node.parent and node.parent.name == "script":
            text = str(node)
            if "Plotly.newPlot(" in text or len(text) < 500000:
                node.replace_with(names(text))
        elif node.parent and node.parent.name != "style":
            node.replace_with(display(str(node)))
    for tag in soup.find_all(True):
        for key, value in list(tag.attrs.items()):
            if not isinstance(value, str):
                continue
            if value.startswith("data:"):
                if tag.name == "img" and value.startswith("data:image/jpeg;base64,"):
                    match = re.match(r"(MCL2_sample_\d+)", path.name)
                    if match:
                        preview = path.parent / (match[1] + "_preview.jpg")
                        assert preview.exists()
                        tag[key] = "data:image/jpeg;base64," + base64.b64encode(preview.read_bytes()).decode()
            elif key == "data-case" and value.isdigit():
                tag[key] = str(MAPPING[int(value)])
            elif key in {"href", "src", "download"}:
                tag[key] = names(value)
            else:
                tag[key] = display(value)
    path.write_text(str(soup))


def convert_markdown(path):
    links = []
    def protect(match):
        links.append(names(match[1]))
        return f"](@@LINK{len(links)-1}@@)"
    text = re.sub(r"\]\(([^)]+)\)", protect, path.read_text())
    text = display(text)
    for i, target in enumerate(links):
        text = text.replace(f"@@LINK{i}@@", target)
    path.write_text(text)


def adapt_notebook():
    path = STAGE / "scripts/notebooks/latest06apr.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if cell.get("id") == "project-paths":
            text += '''

def exported_sample_id(stem):
    import re
    match = re.fullmatch(r"[OP]_(\\d+)(?:_st)?", stem)
    original_ids = list(range(9, 19)) + list(range(35, 46))
    if not match or int(match[1]) not in original_ids:
        return None
    return f"MCL2_sample_{original_ids.index(int(match[1])) + 1}"
'''
        if cell.get("metadata", {}).get("original_notebook_cell_index") in {66, 68, 83}:
            text = text.replace("    if base + '_mask.png' not in retained_mask_names:",
                                "    sample_id = exported_sample_id(base)\n    if sample_id is None or sample_id + '_mask.png' not in retained_mask_names:")
            text = text.replace("os.path.join(pred_folder, base + '_mask.png')", "os.path.join(pred_folder, sample_id + '_mask.png')")
            text = text.replace("    plt.show()", "    fig.suptitle(sample_id)\n    plt.show()")
        cell["source"] = text.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    sha = digest(path)
    provenance_path = path.parent / "notebook_sections.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["packaged_sha256"] = sha
    provenance["prediction_identifiers"] = "MCL2_sample_1 through MCL2_sample_21"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    details_path = STAGE / "scripts/model_performance/model_details.json"
    details = json.loads(details_path.read_text())
    for row in details:
        row["packaged_notebook_sha256"] = sha
    details_path.write_text(json.dumps(details, indent=2) + "\n")


def transform():
    assert all((TEMP / (p + ".done")).exists() for p in ["small", "current", "old", "middle", "gingival", "single"])
    # Keep every original artifact; omit storage sidecars and incidental renderer aliases.
    old_paths = {str(p.relative_to(BEFORE)) for p in BEFORE.rglob("*") if p.is_file()}
    for path in list(STAGE.rglob("*")):
        if path.is_file() and (path.name.endswith(":Zone.Identifier")
                or (path.name.endswith("_complete.png") and str(path.relative_to(STAGE)) not in old_paths)):
            path.unlink()
    for path in sorted(STAGE.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        new_name = names(path.name)
        if new_name != path.name:
            destination = path.with_name(new_name)
            assert not destination.exists(), destination
            path.rename(destination)
    for path in sorted(STAGE.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            convert_csv(path)
        elif path.suffix == ".json":
            value = convert_json(json.loads(path.read_text()), script_metadata="scripts" in path.relative_to(STAGE).parts)
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        elif path.suffix == ".md":
            convert_markdown(path)
        elif path.suffix == ".html":
            convert_html(path)
    adapt_notebook()
    readme = STAGE / "README.md"
    text = readme.read_text().replace("Reference names use retained, identical binary-mask aliases where applicable.",
                                     "Reference masks use neutral reference identifiers; source filenames are omitted from report content.")
    readme.write_text(text)
    # Preserve current tooling and record the changed packaged notebook checksum.
    tool = STAGE / "scripts/model_tools/rename_result_cases.py"
    shutil.copy2(Path(__file__).resolve(), tool)
    inventory_path = STAGE / "scripts/script_inventory.csv"
    fields, rows = read_csv(inventory_path)
    for row in rows:
        copied = STAGE / "scripts" / row["path"]
        assert copied.exists(), copied
        if row["sha256"] != digest(copied):
            row.update(sha256=digest(copied), bytes=str(copied.stat().st_size), copy_method="tailored_final_workflow")
    rows.append({"path": "model_tools/rename_result_cases.py", "source": "", "purpose": "Case relabeling and report regeneration",
                 "bytes": str(tool.stat().st_size), "sha256": digest(tool), "copy_method": "generated"})
    write_csv(inventory_path, fields, rows)
    (TEMP / "transform.done").write_text("complete\n")
    print("Relabeled filenames, tables, metadata, notebook and HTML", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["snapshot", "small", "cqs", "current", "old", "middle", "gingival", "single", "transform"])
    args = parser.parse_args()
    if args.phase == "snapshot":
        snapshot()
    elif args.phase == "transform":
        transform()
    else:
        render(args.phase)


if __name__ == "__main__":
    main()
