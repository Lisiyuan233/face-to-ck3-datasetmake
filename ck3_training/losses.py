from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .schema import CK3Schema


def _balanced_class_weights(
    counts: tuple[int, ...], minimum: float, maximum: float
) -> torch.Tensor:
    values = torch.tensor(counts, dtype=torch.float64)
    weights = values.clamp_min(1.0).rsqrt()
    weights = weights / weights.mean()
    return weights.clamp(minimum, maximum).to(dtype=torch.float32)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_field_profiles(
    schema: CK3Schema, config: dict[str, Any]
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    names = {
        "scalar": list(schema.scalar_fields),
        "signed": list(schema.signed_fields),
        "strength": [field.name for field in schema.categorical_fields],
        "class": [field.name for field in schema.categorical_fields],
    }
    values = {family: [1.0] * len(field_names) for family, field_names in names.items()}
    decisions: dict[str, dict[str, Any]] = {}
    decision_path = config.get("field_weights_path")
    if decision_path:
        decisions = json.loads(Path(decision_path).read_text(encoding="utf-8"))
        if not isinstance(decisions, dict):
            raise ValueError("field weight profile must be a JSON object")

    texture: dict[str, float] = {}
    texture_path = config.get("texture_metrics_path")
    if texture_path:
        with Path(texture_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                field_name = str(row["field"])
                weight = _optional_float(row.get("recommended_texture_weight"))
                if weight is not None:
                    texture[field_name] = min(1.0, max(0.0, weight))
    texture_blend = float(config.get("texture_weight_blend", 0.0))

    for family, field_names in names.items():
        for index, field_name in enumerate(field_names):
            decision = decisions.get(field_name, {})
            if family == "scalar":
                selected = _optional_float(decision.get("merged_head_weight"))
            elif family == "class":
                selected = _optional_float(decision.get("source_head_weight"))
            elif family == "signed":
                selected = _optional_float(decision.get("source_head_weight"))
            else:
                selected = _optional_float(decision.get("recommended_weight"))
            if selected is None:
                selected = _optional_float(decision.get("recommended_weight"))
            base_weight = 1.0 if selected is None else max(0.0, selected)
            texture_weight = texture.get(field_name, 1.0)
            values[family][index] = base_weight * (
                (1.0 - texture_blend) + texture_blend * texture_weight
            )

    metadata = {
        "field_weights_path": str(decision_path) if decision_path else None,
        "texture_metrics_path": str(texture_path) if texture_path else None,
        "texture_weight_blend": texture_blend,
        "weights": {
            family: dict(zip(names[family], family_values))
            for family, family_values in values.items()
        },
    }
    return values, metadata


class MultitaskLoss(nn.Module):
    def __init__(
        self,
        schema: CK3Schema,
        config: dict[str, Any],
        categorical_class_counts: list[list[int]],
    ) -> None:
        super().__init__()
        self.schema = schema
        self.beta = float(config["smooth_l1_beta"])
        self.weights = {
            "scalar": float(config.get("scalar_weight", 1.0)),
            "signed": float(config["signed_weight"]),
            "class": float(config["class_weight"]),
            "strength": float(config["strength_weight"]),
            "consistency": float(config["consistency_weight"]),
        }
        self.class_label_smoothing = float(config["class_label_smoothing"])
        fallback_visibility = float(config["class_visibility_threshold"])
        if bool(config.get("use_schema_visibility_thresholds", False)):
            visibility_thresholds = schema.categorical_visibility_thresholds(
                fallback_visibility
            )
        else:
            visibility_thresholds = (fallback_visibility,) * schema.categorical_dim
        self.register_buffer(
            "class_visibility_thresholds",
            torch.tensor(visibility_thresholds, dtype=torch.float32),
        )
        self.reference_only_fields = set(
            config.get("reference_only_categorical_fields", ())
        )
        self.consistency_excluded_fields = set(
            config.get("consistency_excluded_categorical_fields", ())
        )
        known_fields = {field.name for field in schema.categorical_fields}
        unknown_fields = (
            self.reference_only_fields | self.consistency_excluded_fields
        ) - known_fields
        if unknown_fields:
            raise ValueError(
                "unknown categorical fields in loss config: "
                + ", ".join(sorted(unknown_fields))
            )

        profile_values, self._profile_metadata = _load_field_profiles(schema, config)
        for family, values in profile_values.items():
            self.register_buffer(
                f"{family}_field_weights",
                torch.tensor(values, dtype=torch.float32),
            )
        self._profile_metadata["categorical_visibility_thresholds"] = dict(
            zip(
                (field.name for field in schema.categorical_fields),
                visibility_thresholds,
            )
        )

        if len(categorical_class_counts) != schema.categorical_dim:
            raise ValueError("train label statistics do not match categorical fields")
        for index in schema.active_categorical_indices:
            field = schema.categorical_fields[index]
            counts = tuple(int(value) for value in categorical_class_counts[index])
            if len(counts) != len(field.classes):
                raise ValueError(f"invalid train class counts for {field.name}")
            self.register_buffer(
                f"class_weights_{index}",
                _balanced_class_weights(
                    counts,
                    float(config["class_weight_min"]),
                    float(config["class_weight_max"]),
                ),
            )

    def profile_metadata(self) -> dict[str, Any]:
        return self._profile_metadata

    def observable_thresholds(self) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in self.class_visibility_thresholds.detach().cpu().tolist()
        )

    def _field_mean_smooth_l1(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        field_weights: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape[1] == 0:
            return prediction.sum() * 0.0
        loss = F.smooth_l1_loss(
            prediction, target, beta=self.beta, reduction="none"
        ).mean(dim=0)
        weights = field_weights.to(dtype=loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(
            torch.finfo(loss.dtype).eps
        )

    @staticmethod
    def _observable_weighted_mean(
        loss: torch.Tensor, visibility: torch.Tensor, observable: torch.Tensor
    ) -> torch.Tensor:
        weights = visibility * observable.to(visibility.dtype)
        denominator = weights.sum()
        return (loss * weights).sum() / denominator.clamp_min(
            torch.finfo(weights.dtype).eps
        )

    @staticmethod
    def _weighted_task_mean(
        values: Sequence[tuple[torch.Tensor, torch.Tensor]], zero: torch.Tensor
    ) -> torch.Tensor:
        if not values:
            return zero
        numerator = zero
        denominator = zero
        for value, weight in values:
            numerator = numerator + value * weight
            denominator = denominator + weight
        return numerator / denominator.clamp_min(torch.finfo(zero.dtype).eps)

    def forward(
        self, outputs: dict[str, Any], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = outputs["signed"].sum() * 0.0
        if self.schema.scalar_dim:
            scalar = self._field_mean_smooth_l1(
                outputs["scalar"], targets["scalar"], self.scalar_field_weights
            )
        else:
            scalar = zero
        signed = self._field_mean_smooth_l1(
            outputs["signed"], targets["signed"], self.signed_field_weights
        )
        strength = self._field_mean_smooth_l1(
            outputs["categorical_strength"],
            targets["categorical_strength"],
            self.strength_field_weights,
        )
        class_losses: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index in self.schema.active_categorical_indices:
            field_weight = self.class_field_weights[index]
            if float(field_weight.detach()) <= 0:
                continue
            field = self.schema.categorical_fields[index]
            prediction_outputs = outputs
            if field.name in self.reference_only_fields and "reference" in outputs:
                prediction_outputs = outputs["reference"]
            logits = prediction_outputs["categorical_logits"][str(index)]
            class_target = targets["categorical_class"][:, index]
            visibility = targets["categorical_strength"][:, index]
            observable = visibility >= self.class_visibility_thresholds[index]
            if not bool(observable.any()):
                continue
            class_weight = getattr(self, f"class_weights_{index}")
            sample_loss = F.cross_entropy(
                logits,
                class_target,
                weight=class_weight,
                reduction="none",
                label_smoothing=self.class_label_smoothing,
            )
            class_losses.append(
                (
                    self._observable_weighted_mean(
                        sample_loss, visibility, observable
                    ),
                    field_weight,
                )
            )
        class_loss = self._weighted_task_mean(class_losses, zero)

        consistency = zero
        if self.weights["consistency"] > 0 and "reference" in outputs:
            reference = outputs["reference"]
            continuous_terms = []
            if self.schema.scalar_dim:
                continuous_terms.append(
                    self._field_mean_smooth_l1(
                        outputs["scalar"],
                        reference["scalar"],
                        self.scalar_field_weights,
                    )
                )
            continuous_terms.extend(
                (
                    self._field_mean_smooth_l1(
                        outputs["signed"],
                        reference["signed"],
                        self.signed_field_weights,
                    ),
                    self._field_mean_smooth_l1(
                        outputs["categorical_strength"],
                        reference["categorical_strength"],
                        self.strength_field_weights,
                    ),
                )
            )
            continuous_consistency = torch.stack(continuous_terms).mean()
            class_consistency: list[tuple[torch.Tensor, torch.Tensor]] = []
            for index in self.schema.active_categorical_indices:
                field = self.schema.categorical_fields[index]
                field_weight = self.class_field_weights[index]
                if (
                    field.name in self.consistency_excluded_fields
                    or float(field_weight.detach()) <= 0
                ):
                    continue
                key = str(index)
                primary_log = F.log_softmax(
                    outputs["categorical_logits"][key], dim=1
                )
                reference_log = F.log_softmax(
                    reference["categorical_logits"][key], dim=1
                )
                primary_probability = primary_log.exp()
                reference_probability = reference_log.exp()
                symmetric_kl = 0.5 * (
                    (primary_probability * (primary_log - reference_log)).sum(dim=1)
                    + (reference_probability * (reference_log - primary_log)).sum(dim=1)
                )
                visibility = targets["categorical_strength"][:, index]
                observable = visibility >= self.class_visibility_thresholds[index]
                if not bool(observable.any()):
                    continue
                class_consistency.append(
                    (
                        self._observable_weighted_mean(
                            symmetric_kl, visibility, observable
                        ),
                        field_weight,
                    )
                )
            categorical_consistency = self._weighted_task_mean(
                class_consistency, zero
            )
            consistency = 0.5 * (
                continuous_consistency + categorical_consistency
            )

        components = {
            "scalar": scalar,
            "signed": signed,
            "class": class_loss,
            "strength": strength,
            "consistency": consistency,
        }
        total = sum(self.weights[key] * value for key, value in components.items())
        components["total"] = total
        return total, components
