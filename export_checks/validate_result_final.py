#!/usr/bin/env python3
"""Independent checks for the filtered presentation export; audit stays outside it."""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import export_result_final as export

EXPECTED = set(range(9, 19)) | set(range(35, 46))
RESULTS = export.STAGE if export.STAGE.exists() else ROOT / "Result_final"
REPORT = ROOT / "work/result_final_validation.json"


def source_for(path):
    relative = str(path.relative_to(RESULTS)).replace("cus_ist_ratio_results/", "cus_ist_ratio_results_cusp_rescue/")
    return export.SOURCE / relative


def ocr_image(path):
    with Image.open(path) as im:
        im.thumbnail((2400, 2400))
        buffer = io.BytesIO()
        im.convert("RGB").save(buffer, format="PNG")
    process = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11"], input=buffer.getvalue(),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                             env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    return str(path.relative_to(RESULTS)), process.stdout.decode()


def main():
    audit = json.loads(export.AUDIT.read_text())
    checks = {"source_files_unchanged": True, "csv_files": [], "html_files": [],
              "image_files": [], "npz_files": [], "ocr_findings": []}
    for relative, expected_hash in audit["source_hashes"].items():
        assert export.digest(export.SOURCE / relative) == expected_hash, relative
    current_files = {str(p.relative_to(export.SOURCE)) for p in export.SOURCE.rglob("*") if p.is_file()}
    assert current_files == set(audit["source_hashes"])
    images = []
    for p in sorted(RESULTS.rglob("*")):
        if not p.is_file():
            continue
        relative = str(p.relative_to(RESULTS))
        assert not export.excluded(relative), relative
        assert not export.diagnostic(relative), relative
        source = source_for(p)
        if p.suffix in {".json", ".csv", ".md"}:
            text = p.read_text()
            assert not export.excluded(text), relative
            assert not export.diagnostic(text), (relative, export.DIAGNOSTIC.search(text).group())
        if p.suffix == ".json":
            json.loads(p.read_text())
        elif p.suffix == ".csv":
            fields, rows = export.read_csv(p)
            original_fields, originals = export.read_csv(source)
            expected_rows = [r for r in originals if not export.excluded_record(r)]
            assert len(rows) == len(expected_rows), relative
            for result, original in zip(rows, expected_rows):
                for key in fields:
                    expected = export.clean_reference(original[key]) if key in export.REF_FIELDS else original[key]
                    assert result[key] == expected, (relative, key)
            checks["csv_files"].append({"file": relative, "rows": len(rows), "columns": len(fields), "values_preserved": True})
        elif p.suffix == ".html":
            soup = BeautifulSoup(p.read_text(), "html.parser")
            for image in soup.find_all("img"):
                if image.get("src", "").startswith("data:image/jpeg;base64,"):
                    stem = re.match(r"O_\d+", p.name)[0]
                    assert base64.b64decode(image["src"].split(",", 1)[1]) == (p.parent / (stem+"_preview.jpg")).read_bytes(), relative
            for element in soup.find_all(["a", "img"]):
                target = element.get("href") if element.name == "a" else element.get("src")
                if not target or re.match(r"(?:https?:|data:|#|mailto:)", target):
                    continue
                assert (p.parent / unquote(target.split("#")[0])).exists(), (relative, target)
            source_scripts = None
            if source.exists() and p.parent.name != "CQS_overall_results":
                original_soup = BeautifulSoup(source.read_text(), "html.parser")
                source_scripts = [s.get_text() for s in original_soup.find_all("script")]
                assert [s.get_text() for s in soup.find_all("script")] == source_scripts, relative
            else:
                for script in soup.find_all("script"):
                    subprocess.run(["node", "--check"], input=script.get_text(), text=True, capture_output=True, check=True)
            for element in soup(["script", "style"]):
                element.decompose()
            text = soup.get_text(" ", strip=True)
            assert not export.excluded(text), relative
            assert not export.diagnostic(text), relative
            checks["html_files"].append(relative)
        elif p.suffix in {".png", ".jpg"}:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                metadata = str(im.info)
                assert not export.excluded(metadata), relative
                assert not export.diagnostic(metadata), relative
                size = im.size
                if source.exists() and p.name != "CQS_overall.png":
                    with Image.open(source) as old:
                        assert size == old.size, (relative, size, old.size)
            checks["image_files"].append(relative)
            if not p.parent.name.startswith("pred_"):
                images.append(p)
            elif source.exists():
                assert export.digest(p) == export.digest(source), relative
        elif p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as data:
                assert not any(export.diagnostic(key) or export.excluded(key) for key in data.files)
            assert export.digest(p) == export.digest(source), relative
            checks["npz_files"].append(relative)
        elif p.suffix == ".pdf":
            process = subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True, check=True)
            assert not export.excluded(process.stdout), relative
            assert not export.diagnostic(process.stdout), relative
    for relative, id_field in [("CQS_overall_results/CQS_overall_scores.csv", "case_id"),
                               ("CQS_overall_results/cqs_minimal.csv", "case_id"),
                               ("publication_cavity_depth_results/depth_summary.csv", "Tooth"),
                               ("final_avg_efd_results/occlusal/efd_similarity_scores.csv", "sample"),
                               ("final_avg_efd_results/proximal/efd_similarity_scores.csv", "sample"),
                               ("cus_ist_ratio_results/ratios.csv", "filename")]:
        rows = export.read_csv(RESULTS / relative)[1]
        numbers = {int(re.search(r"\d+", r[id_field])[0]) for r in rows}
        assert numbers == EXPECTED and len(rows) == 21, (relative, numbers)
    print(f"Structured, numerical, links, embedded images, and source checks passed; OCR of {len(images)} images", flush=True)
    ocr = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(ocr_image, p) for p in images]
        for index, future in enumerate(as_completed(futures), 1):
            name, text = future.result()
            ocr[name] = text
            # OCR may insert/remove underscores; detect simple written diagnostic words as well.
            hit = re.search(r"\b(warnings?|errors?|review|flags?|fallback|tolerance|confidence|unverified|relaxed)\b", text, re.I)
            if hit or export.excluded(text):
                checks["ocr_findings"].append({"file": name, "text": text})
            if index % 20 == 0:
                print(f"OCR {index}/{len(images)}", flush=True)
    checks.update(case_count=21, file_count=sum(p.is_file() for p in RESULTS.rglob("*")),
                  ocr_image_count=len(images), ocr=ocr,
                  output_hashes={str(p.relative_to(RESULTS)): export.digest(p)
                                 for p in RESULTS.rglob("*") if p.is_file()},
                  status="passed" if not checks["ocr_findings"] else "inspect_ocr_findings")
    REPORT.write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps({k:v for k,v in checks.items() if k not in {"csv_files", "html_files", "image_files", "npz_files", "ocr"}},indent=2),flush=True)


if __name__ == "__main__":
    main()
