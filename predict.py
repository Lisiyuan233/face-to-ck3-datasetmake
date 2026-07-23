#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from PIL import Image
except ImportError as error:
    print(
        "PyTorch and Pillow are required; install requirements-train.txt.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error

from ck3_training.data import DualViewTransform
from ck3_training.model import FaceToCK3Model
from ck3_training.schema import TARGET_FAMILY, load_schema
from dna_normalizer import (
    apply_normalized_to_template,
    load_schema as load_raw_dna_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict normalized CK3 DNA from an already aligned face image."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--schema",
        default=Path("face_to_ck3_dataset_male_small/dna_schema.json"),
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--dna-output", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="use raw checkpoint weights instead of EMA weights",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.template and not args.dna_output:
        raise SystemExit("--dna-output is required when --template is provided")
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("schema", {}).get("target_family") != TARGET_FAMILY:
        raise SystemExit(
            "checkpoint uses the legacy color-prediction architecture; use a "
            "geometry-only checkpoint"
        )
    schema = load_schema(args.schema)
    saved_sha = checkpoint.get("schema", {}).get("schema_sha256")
    if saved_sha != schema.sha256:
        raise SystemExit(
            f"checkpoint schema {saved_sha} does not match {schema.sha256}"
        )
    config = checkpoint["config"]
    model_config = dict(config["model"])
    model_config["pretrained"] = False
    model = FaceToCK3Model(schema, model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if not args.raw_weights and checkpoint.get("ema", {}).get("shadow"):
        shadow = checkpoint["ema"]["shadow"]
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in shadow:
                    parameter.copy_(shadow[name].to(device))
    model.eval()

    transform = DualViewTransform(
        config["data"]["image_height"],
        config["data"]["image_width"],
        config["augmentation"],
        training=False,
        dual_view=bool(model_config["dual_view"]),
    )
    with Image.open(args.image) as source:
        geometry, reference = transform(source.convert("RGB"))
    with torch.inference_mode():
        outputs = model(
            geometry.unsqueeze(0).to(device), reference.unsqueeze(0).to(device)
        )

    classes = [0] * schema.categorical_dim
    confidence = [1.0] * schema.categorical_dim
    for index in schema.active_categorical_indices:
        probabilities = outputs["categorical_logits"][str(index)].softmax(dim=1)[0]
        classes[index] = int(probabilities.argmax().item())
        confidence[index] = float(probabilities.max().item())
    prediction = {
        "sample_id": args.image.stem,
        "signed": outputs["signed"][0].float().cpu().tolist(),
        "categorical_class": classes,
        "categorical_strength": outputs["categorical_strength"][0]
        .float()
        .cpu()
        .tolist(),
        "categorical_confidence": confidence,
        "schema_sha256": schema.sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.template:
        raw_schema = load_raw_dna_schema(args.schema)
        template = args.template.read_text(encoding="utf-8")
        dna = apply_normalized_to_template(template, prediction, raw_schema)
        args.dna_output.parent.mkdir(parents=True, exist_ok=True)
        args.dna_output.write_text(dna, encoding="utf-8")
    print(args.output)
    if args.dna_output:
        print(args.dna_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
