"""Reusable inference pipeline for the trained face-to-CK3 model.

The model predicts continuous facial targets only.  A complete CK3 DNA string
is produced by applying selected predictions to a caller-provided DNA template;
colors, categorical alleles, body genes, and low-confidence facial fields stay
template-controlled.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ck3_training.data import DualViewTransform
from ck3_training.model import FaceToCK3Model
from ck3_training.schema import CK3Schema, load_schema
from dna_normalizer import (
    QUOTED_FIELD_RE,
    apply_normalized_to_template,
    load_schema as load_dna_schema,
    normalize_record,
    parse_dna,
)


@dataclass(frozen=True)
class FieldQuality:
    family: str
    field: str
    baseline_mae: float
    model_mae: float
    improvement: float


@dataclass(frozen=True)
class DNAInferenceResult:
    dna: str
    normalized: dict[str, Any]
    predicted_fields: tuple[str, ...]
    preserved_fields: tuple[str, ...]
    field_details: tuple[dict[str, Any], ...]
    weight_source: str
    used_side_fallback: bool


def choose_device(choice: str) -> torch.device:
    if choice not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"unsupported device: {choice}")
    available = torch.cuda.is_available()
    if choice == "cuda" and not available:
        raise RuntimeError("已选择 CUDA，但当前 PyTorch 检测不到 CUDA")
    return torch.device("cuda" if available and choice != "cpu" else "cpu")


def load_field_quality(path: str | Path) -> dict[tuple[str, str], FieldQuality]:
    result: dict[tuple[str, str], FieldQuality] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            quality = FieldQuality(
                family=str(row["family"]),
                field=str(row["field"]),
                baseline_mae=float(row["baseline_mae"]),
                model_mae=float(row["model_mae"]),
                improvement=float(row["improvement"]),
            )
            result[(quality.family, quality.field)] = quality
    if not result:
        raise ValueError(f"字段质量表为空: {path}")
    return result


def center_crop_to_aspect(image: Image.Image, width: int, height: int) -> Image.Image:
    """Return a centered crop with the model input aspect ratio."""

    source = image.convert("RGB")
    target_ratio = float(width) / float(height)
    source_ratio = source.width / source.height
    if source_ratio > target_ratio:
        crop_width = max(1, round(source.height * target_ratio))
        left = (source.width - crop_width) // 2
        box = (left, 0, left + crop_width, source.height)
    else:
        crop_height = max(1, round(source.width / target_ratio))
        top = (source.height - crop_height) // 2
        box = (0, top, source.width, top + crop_height)
    return source.crop(box)


def crop_composite_views(
    image: Image.Image,
    *,
    front_crop: tuple[int, int, int, int],
    side_crop: tuple[int, int, int, int],
    expected_size: tuple[int, int] | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Crop the front/profile panels, scaling manifest coordinates if needed."""

    source = image.convert("RGB")
    scale_x = 1.0
    scale_y = 1.0
    if expected_size:
        expected_width, expected_height = expected_size
        if expected_width <= 0 or expected_height <= 0:
            raise ValueError("invalid expected composite size")
        scale_x = source.width / expected_width
        scale_y = source.height / expected_height

    def scaled(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        value = (
            round(left * scale_x),
            round(top * scale_y),
            round(right * scale_x),
            round(bottom * scale_y),
        )
        if (
            value[0] < 0
            or value[1] < 0
            or value[2] > source.width
            or value[3] > source.height
            or value[0] >= value[2]
            or value[1] >= value[3]
        ):
            raise ValueError(f"组合图裁剪区域越界: {value}, 图像={source.size}")
        return value

    return source.crop(scaled(front_crop)), source.crop(scaled(side_crop))


def prepare_input_views(
    front_path: str | Path,
    *,
    side_path: str | Path | None,
    composite: bool,
    front_crop: tuple[int, int, int, int],
    side_crop: tuple[int, int, int, int],
    expected_size: tuple[int, int] | None,
    model_size: tuple[int, int],
    allow_front_as_side: bool,
) -> tuple[Image.Image, Image.Image, bool]:
    with Image.open(front_path) as stream:
        front_source = stream.convert("RGB")
    if composite:
        front, side = crop_composite_views(
            front_source,
            front_crop=front_crop,
            side_crop=side_crop,
            expected_size=expected_size,
        )
        return front, side, False

    # A single file may already be an aligned training crop. Preserve it exactly
    # when its aspect ratio matches; otherwise use a centered aspect crop.
    front = center_crop_to_aspect(front_source, *model_size)
    if side_path:
        with Image.open(side_path) as stream:
            side_source = stream.convert("RGB")
        side = center_crop_to_aspect(side_source, *model_size)
        return front, side, False
    if not allow_front_as_side:
        raise ValueError("当前模型需要侧脸照片；请选择侧脸，或启用正脸替代")
    return front, front.copy(), True


class CK3Predictor:
    """Load one checkpoint and turn aligned image pairs into normalized targets."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        schema_path: str | Path,
        *,
        device: str = "auto",
        raw_weights: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.schema_path = Path(schema_path).resolve()
        self.device = choose_device(device)
        self.schema: CK3Schema = load_schema(self.schema_path)
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        saved_schema = checkpoint.get("schema", {})
        if saved_schema.get("target_family") != self.schema.target_family:
            raise RuntimeError("checkpoint 与 schema 的 target family 不一致")
        if saved_schema.get("schema_sha256") != self.schema.sha256:
            raise RuntimeError("checkpoint 与 schema 的 SHA-256 不一致")

        self.config = checkpoint["config"]
        model_config = dict(self.config["model"])
        model_config["pretrained"] = False
        self.model = FaceToCK3Model(self.schema, model_config)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.weight_source = "raw"
        if not raw_weights:
            shadow = checkpoint.get("ema", {}).get("shadow", {})
            parameters = dict(self.model.named_parameters())
            missing = sorted(set(parameters) - set(shadow))
            if not shadow:
                raise RuntimeError("checkpoint 没有 EMA 权重")
            if missing:
                raise RuntimeError("EMA 权重不完整: " + ", ".join(missing[:5]))
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(shadow[name].to(dtype=parameter.dtype))
            self.weight_source = "ema"
        del checkpoint
        self.model.to(self.device).eval()

        data_config = self.config["data"]
        self.model_size = (
            int(data_config["image_width"]),
            int(data_config["image_height"]),
        )
        self.transform = DualViewTransform(
            data_config["image_height"],
            data_config["image_width"],
            self.config["augmentation"],
            training=False,
            dual_view=bool(model_config["dual_view"]),
            geometry_map_config=model_config.get("geometry_branch"),
        )
        self.amp_mode = str(self.config["train"].get("amp", "none")).lower()

    def predict_normalized(
        self, front: Image.Image, side: Image.Image
    ) -> dict[str, Any]:
        if self.transform.geometry_map_enabled:
            geometry, reference, geometry_map = self.transform.with_geometry_map(front)
            side_tensor, _, side_geometry_map = self.transform.with_geometry_map(
                side, allow_horizontal_flip=False
            )
        else:
            geometry, reference = self.transform(front)
            side_tensor, _ = self.transform(side, allow_horizontal_flip=False)
            geometry_map = None
            side_geometry_map = None

        amp_enabled = self.device.type == "cuda" and self.amp_mode in {"bf16", "fp16"}
        amp_dtype = torch.bfloat16 if self.amp_mode == "bf16" else torch.float16
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = self.model(
                geometry.unsqueeze(0).to(self.device),
                reference.unsqueeze(0).to(self.device),
                side_tensor.unsqueeze(0).to(self.device),
                geometry_map.unsqueeze(0).to(self.device)
                if geometry_map is not None
                else None,
                side_geometry_map.unsqueeze(0).to(self.device)
                if side_geometry_map is not None
                else None,
            )
        return {
            "scalar": outputs["scalar"][0].float().cpu().tolist(),
            "signed": outputs["signed"][0].float().cpu().tolist(),
            "categorical_strength": outputs["categorical_strength"][0]
            .float()
            .cpu()
            .tolist(),
        }

    def close(self) -> None:
        """Release model/device memory held by this predictor."""

        self.model.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def build_dna_from_prediction(
    *,
    template_text: str,
    prediction: dict[str, Any],
    schema_path: str | Path,
    quality: dict[tuple[str, str], FieldQuality],
    minimum_improvement: float,
    weight_source: str,
    used_side_fallback: bool,
) -> DNAInferenceResult:
    """Apply only fields meeting the requested held-out-test improvement."""

    raw_schema = load_dna_schema(Path(schema_path))
    template_label = normalize_record(
        parse_dna(template_text),
        raw_schema,
        sample_id="inference_template",
        require_matching_pairs=False,
    )
    family_specs = (
        ("scalar", raw_schema.get("scalar_fields", ()), "scalar"),
        ("signed", raw_schema["signed_fields"], "signed"),
        (
            "strength",
            raw_schema["categorical_fields"],
            "categorical_strength",
        ),
    )
    selected: list[str] = []
    preserved: list[str] = []
    details: list[dict[str, Any]] = []
    merged = dict(template_label)
    merged["sample_id"] = "inference"
    for family, specs, key in family_specs:
        source_values = prediction[key]
        if len(source_values) != len(specs):
            raise ValueError(
                f"{key} 输出维度 {len(source_values)}，schema 需要 {len(specs)}"
            )
        values = list(template_label[key])
        for index, (spec, predicted) in enumerate(zip(specs, source_values)):
            field = str(spec["name"])
            field_quality = quality.get((family, field))
            use_prediction = (
                field_quality is not None
                and field_quality.improvement >= float(minimum_improvement)
            )
            if use_prediction:
                values[index] = float(predicted)
                selected.append(field)
            else:
                preserved.append(field)
            details.append(
                {
                    "family": family,
                    "field": field,
                    "source": "model" if use_prediction else "template",
                    "normalized_value": float(values[index]),
                    "raw_255_value": round(abs(float(values[index])) * 255),
                    "test_improvement": (
                        field_quality.improvement if field_quality else None
                    ),
                    "test_mae": field_quality.model_mae if field_quality else None,
                }
            )
        merged[key] = values

    # The class heads were not trained in the final run.  Preserve categorical
    # alleles (and colors) from the template while applying predicted strengths.
    merged["categorical_class"] = list(template_label["categorical_class"])
    if "colors" in template_label:
        merged["colors"] = list(template_label["colors"])
    candidate = apply_normalized_to_template(template_text, merged, raw_schema)
    candidate_genes = {
        match.group("key"): match.group(0)
        for match in QUOTED_FIELD_RE.finditer(candidate)
    }
    strength_values = {
        str(spec["name"]): round(
            min(1.0, max(0.0, float(value))) * 255
        )
        for spec, value in zip(
            raw_schema["categorical_fields"], merged["categorical_strength"]
        )
        if str(spec["name"]) in selected
    }
    selected_set = set(selected)

    def apply_selected(match: Any) -> str:
        field = match.group("key")
        if field not in selected_set:
            return match.group(0)
        if field in strength_values:
            # Class logits were not trained. Preserve each template allele even
            # for a heterozygous custom template and update strength only.
            value = strength_values[field]
            return (
                f'{match.group("indent")}{field}={{ "{match.group("allele1")}" '
                f'{value} "{match.group("allele2")}" {value} }}'
            )
        return candidate_genes[field]

    dna = QUOTED_FIELD_RE.sub(apply_selected, template_text)
    return DNAInferenceResult(
        dna=dna,
        normalized=merged,
        predicted_fields=tuple(selected),
        preserved_fields=tuple(preserved),
        field_details=tuple(details),
        weight_source=weight_source,
        used_side_fallback=used_side_fallback,
    )


def load_preprocessing_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    crops = value.get("crops", {})
    if "front" not in crops or "side" not in crops:
        raise ValueError("预处理 manifest 缺少正面/侧面裁剪信息")
    return value
