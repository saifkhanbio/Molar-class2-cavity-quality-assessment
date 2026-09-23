#!/usr/bin/env python3
"""Compare checkpoint predictions with existing binary masks without writing masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("KERAS_HOME", "/tmp/result-final-keras")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=["occlusal", "proximal", "cusp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--input-cache", type=Path)
    parser.add_argument("--preprocessing", choices=["rgb_unit", "bgr_unit", "bgr_resnet50", "rgb_resnet50"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prefix = "P" if args.kind == "proximal" else "O"
    masks = {"occlusal": "pred_O_M_masks_folder", "proximal": "pred_P_M_masks_folder", "cusp": "pred_O_M_cusp_molar_mask"}[args.kind]
    cases = sorted((root / "Result_final" / masks).glob("*.png"))
    if not args.all:
        chosen = {f"{prefix}_11_mask.png", f"{prefix}_35_st_mask.png", f"{prefix}_43_st_mask.png"}
        cases = [p for p in cases if p.name in chosen]
    model = tf.keras.models.load_model(args.model, compile=False)
    cached = np.load(args.input_cache, allow_pickle=False) if args.input_cache else None
    variants = ["rgb_unit", "bgr_unit", "bgr_resnet50", "rgb_resnet50"] if args.kind == "occlusal" else ["rgb_unit"]
    if args.preprocessing:
        variants = [args.preprocessing]
    records = []
    for path in cases:
        filename = path.name.replace("_mask.png", ".png")
        sample = root / f"samples_stu_{prefix}" / filename
        if cached is None:
            import cv2
            bgr = cv2.resize(cv2.imread(str(sample)), (256, 256))
            expected = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127
        else:
            bgr = cached[filename+"_bgr"]
            expected = cached[path.name+"_"+args.kind]
        rgb = bgr[..., ::-1].copy()
        for method in variants:
            values = (rgb if method.startswith("rgb") else bgr).copy()
            values = (values / 255.0 if method.endswith("unit") else
                      tf.keras.applications.resnet50.preprocess_input(values.astype("float32")))
            probability = model(values[None, ...], training=False).numpy()[0, ..., 0]
            actual = probability > .5
            changed = int(np.count_nonzero(actual != expected))
            intersection = int((actual & expected).sum())
            union = int((actual | expected).sum())
            record = {"case": filename.removesuffix(".png"), "preprocessing": method,
                      "different_pixels": changed, "pixels": int(expected.size),
                      "mask_agreement_iou": intersection/union if union else 1.0}
            records.append(record)
            print(args.model, record, flush=True)
    payload = {"model": str(args.model), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
               "kind": args.kind, "tensorflow_version": tf.__version__, "input_shape": model.input_shape,
               "parameter_count": model.count_params(), "records": records}
    args.output.write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
