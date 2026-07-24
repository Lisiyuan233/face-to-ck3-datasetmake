from __future__ import annotations

from typing import Any

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
            "signed": float(config["signed_weight"]),
            "class": float(config["class_weight"]),
            "strength": float(config["strength_weight"]),
            "consistency": float(config["consistency_weight"]),
        }
        self.class_label_smoothing = float(config["class_label_smoothing"])
        self.class_visibility_threshold = float(
            config["class_visibility_threshold"]
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
        if len(categorical_class_counts) != schema.categorical_dim:
            raise ValueError("train label statistics do not match categorical fields")
        for index in schema.active_categorical_indices:
            field = schema.categorical_fields[index]
            counts = tuple(int(value) for value in categorical_class_counts[index])
            if len(counts) != len(field.classes) or sum(counts) <= 0:
                raise ValueError(f"invalid train class counts for {field.name}")
            self.register_buffer(
                f"class_weights_{index}",
                _balanced_class_weights(
                    counts,
                    float(config["class_weight_min"]),
                    float(config["class_weight_max"]),
                ),
            )

    def _field_mean_smooth_l1(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(
            prediction, target, beta=self.beta, reduction="none"
        )
        return loss.mean(dim=0).mean()

    @staticmethod
    def _observable_weighted_mean(
        loss: torch.Tensor, visibility: torch.Tensor, observable: torch.Tensor
    ) -> torch.Tensor:
        weights = visibility * observable.to(visibility.dtype)
        denominator = weights.sum()
        return (loss * weights).sum() / denominator.clamp_min(
            torch.finfo(weights.dtype).eps
        )

    def forward(
        self, outputs: dict[str, Any], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        signed = self._field_mean_smooth_l1(outputs["signed"], targets["signed"])
        strength = self._field_mean_smooth_l1(
            outputs["categorical_strength"], targets["categorical_strength"]
        )
        class_losses = []
        for index in self.schema.active_categorical_indices:
            field = self.schema.categorical_fields[index]
            prediction_outputs = outputs
            if (
                field.name in self.reference_only_fields
                and "reference" in outputs
            ):
                prediction_outputs = outputs["reference"]
            logits = prediction_outputs["categorical_logits"][str(index)]
            class_target = targets["categorical_class"][:, index]
            visibility = targets["categorical_strength"][:, index]
            observable = visibility >= self.class_visibility_threshold
            class_weight = getattr(self, f"class_weights_{index}")
            sample_loss = F.cross_entropy(
                logits,
                class_target,
                weight=class_weight,
                reduction="none",
                label_smoothing=self.class_label_smoothing,
            )
            class_losses.append(
                self._observable_weighted_mean(
                    sample_loss, visibility, observable
                )
            )
        class_loss = torch.stack(class_losses).mean()

        consistency = signed.new_zeros(())
        if (
            self.weights["consistency"] > 0
            and "reference" in outputs
        ):
            reference = outputs["reference"]
            continuous_consistency = 0.5 * (
                F.smooth_l1_loss(
                    outputs["signed"], reference["signed"], beta=self.beta
                )
                + F.smooth_l1_loss(
                    outputs["categorical_strength"],
                    reference["categorical_strength"],
                    beta=self.beta,
                )
            )
            class_consistency = []
            for index in self.schema.active_categorical_indices:
                field = self.schema.categorical_fields[index]
                if field.name in self.consistency_excluded_fields:
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
                observable = visibility >= self.class_visibility_threshold
                class_consistency.append(
                    self._observable_weighted_mean(
                        symmetric_kl, visibility, observable
                    )
                )
            categorical_consistency = (
                torch.stack(class_consistency).mean()
                if class_consistency
                else continuous_consistency.new_zeros(())
            )
            consistency = 0.5 * (
                continuous_consistency + categorical_consistency
            )

        components = {
            "signed": signed,
            "class": class_loss,
            "strength": strength,
            "consistency": consistency,
        }
        total = sum(self.weights[key] * value for key, value in components.items())
        components["total"] = total
        return total, components
