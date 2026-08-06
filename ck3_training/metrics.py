from __future__ import annotations

from typing import Any, Sequence

import torch

from .schema import CK3Schema


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    numerator = numerator.to(dtype=torch.float64)
    denominator = denominator.to(dtype=torch.float64)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(numerator),
    )


def _macro_f1(confusion: torch.Tensor) -> float:
    confusion = confusion.to(dtype=torch.float64)
    true_positive = confusion.diag()
    precision = _safe_divide(true_positive, confusion.sum(dim=0))
    recall = _safe_divide(true_positive, confusion.sum(dim=1))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    present = confusion.sum(dim=1) > 0
    return float(f1[present].mean().item()) if present.any() else 0.0


class MetricAccumulator:
    def __init__(
        self,
        schema: CK3Schema,
        device: torch.device,
        observable_threshold: float | Sequence[float] = 0.10,
    ) -> None:
        self.schema = schema
        self.device = device
        if isinstance(observable_threshold, (int, float)):
            thresholds = [float(observable_threshold)] * schema.categorical_dim
        else:
            thresholds = [float(value) for value in observable_threshold]
            if len(thresholds) != schema.categorical_dim:
                raise ValueError("observable thresholds do not match categorical fields")
        self.observable_thresholds = torch.tensor(
            thresholds, dtype=torch.float32, device=device
        )
        self.sample_count = torch.zeros((), dtype=torch.float64, device=device)
        self.scalar_abs = torch.zeros(
            schema.scalar_dim, dtype=torch.float64, device=device
        )
        self.signed_abs = torch.zeros(schema.signed_dim, dtype=torch.float64, device=device)
        self.strength_abs = torch.zeros(
            schema.categorical_dim, dtype=torch.float64, device=device
        )
        self.confusion = {
            index: torch.zeros(
                (len(schema.categorical_fields[index].classes),) * 2,
                dtype=torch.int64,
                device=device,
            )
            for index in schema.active_categorical_indices
        }
        self.observable_confusion = {
            index: torch.zeros_like(matrix)
            for index, matrix in self.confusion.items()
        }
        self.race_abs = torch.zeros(17, dtype=torch.float64, device=device)
        self.race_count = torch.zeros(17, dtype=torch.float64, device=device)
        self.loss_sums: dict[str, torch.Tensor] = {}
        self.loss_batches = torch.zeros((), dtype=torch.float64, device=device)
        self.geometry_gate_sum = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        self.geometry_gate_count = torch.zeros(
            (), dtype=torch.float64, device=device
        )

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, Any],
        targets: dict[str, torch.Tensor],
        components: dict[str, torch.Tensor],
    ) -> None:
        batch_size = targets["signed"].shape[0]
        self.sample_count += batch_size
        scalar_error = None
        if self.schema.scalar_dim:
            scalar_error = (outputs["scalar"] - targets["scalar"]).abs()
            self.scalar_abs += scalar_error.sum(dim=0, dtype=torch.float64)
        signed_error = (outputs["signed"] - targets["signed"]).abs()
        strength_error = (
            outputs["categorical_strength"] - targets["categorical_strength"]
        ).abs()
        self.signed_abs += signed_error.sum(dim=0, dtype=torch.float64)
        self.strength_abs += strength_error.sum(dim=0, dtype=torch.float64)
        if "geometry_gate_mean" in outputs:
            gate = outputs["geometry_gate_mean"]
            self.geometry_gate_sum += gate.sum(dtype=torch.float64)
            self.geometry_gate_count += gate.numel()

        for index in self.schema.active_categorical_indices:
            logits = outputs["categorical_logits"][str(index)]
            prediction = logits.argmax(dim=1)
            target = targets["categorical_class"][:, index]
            class_count = self.confusion[index].shape[0]
            flattened = target * class_count + prediction
            counts = torch.bincount(flattened, minlength=class_count * class_count)
            self.confusion[index] += counts.reshape(class_count, class_count)
            observable = (
                targets["categorical_strength"][:, index]
                >= self.observable_thresholds[index]
            )
            if observable.any():
                observable_flattened = (
                    target[observable] * class_count + prediction[observable]
                )
                observable_counts = torch.bincount(
                    observable_flattened,
                    minlength=class_count * class_count,
                )
                self.observable_confusion[index] += observable_counts.reshape(
                    class_count, class_count
                )

        race_group = targets["race_group"]
        valid = (race_group >= 0) & (race_group < len(self.race_abs))
        if valid.any():
            per_sample = signed_error.mean(dim=1) + strength_error.mean(dim=1)
            if scalar_error is not None:
                per_sample = per_sample + scalar_error.mean(dim=1)
            self.race_abs.scatter_add_(0, race_group[valid], per_sample[valid].double())
            self.race_count.scatter_add_(
                0,
                race_group[valid],
                torch.ones_like(race_group[valid], dtype=torch.float64),
            )

        for key, value in components.items():
            if key not in self.loss_sums:
                self.loss_sums[key] = torch.zeros(
                    (), dtype=torch.float64, device=self.device
                )
            self.loss_sums[key] += value.detach().double()
        self.loss_batches += 1

    def compute(self) -> dict[str, Any]:
        count = self.sample_count.clamp_min(1)
        scalar_fields = (self.scalar_abs / count).cpu()
        signed_fields = (self.signed_abs / count).cpu()
        strength_fields = (self.strength_abs / count).cpu()

        categorical = []
        accuracies = []
        macro_f1_values = []
        observable_accuracies = []
        observable_macro_f1_values = []
        for index in self.schema.active_categorical_indices:
            matrix = self.confusion[index]
            observable_matrix = self.observable_confusion[index]
            accuracy = float(
                _safe_divide(matrix.diag().sum(), matrix.sum()).item()
            )
            macro_f1 = _macro_f1(matrix)
            observable_accuracy = float(
                _safe_divide(
                    observable_matrix.diag().sum(),
                    observable_matrix.sum(),
                ).item()
            )
            observable_macro_f1 = _macro_f1(observable_matrix)
            categorical.append(
                {
                    "index": index,
                    "name": self.schema.categorical_fields[index].name,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "observable_accuracy": observable_accuracy,
                    "observable_macro_f1": observable_macro_f1,
                    "observable_count": int(observable_matrix.sum().item()),
                    "visibility_threshold": float(
                        self.observable_thresholds[index].item()
                    ),
                    "confusion_matrix": matrix.cpu().tolist(),
                    "observable_confusion_matrix": observable_matrix.cpu().tolist(),
                }
            )
            accuracies.append(accuracy)
            macro_f1_values.append(macro_f1)
            if observable_matrix.sum() > 0:
                observable_accuracies.append(observable_accuracy)
                observable_macro_f1_values.append(observable_macro_f1)

        race_groups = []
        for index in range(len(self.race_abs)):
            if self.race_count[index] > 0:
                race_groups.append(
                    {
                        "race_group": index,
                        "count": int(self.race_count[index].item()),
                        "composite_error": float(
                            (self.race_abs[index] / self.race_count[index]).item()
                        ),
                    }
                )

        loss_batches = self.loss_batches.clamp_min(1)
        losses = {
            key: float((value / loss_batches).item())
            for key, value in self.loss_sums.items()
        }
        result = {
            "sample_count": int(self.sample_count.item()),
            "loss": losses,
            "signed_mae": float(signed_fields.mean().item()),
            "signed_mae_raw255": float(signed_fields.mean().item() * 255.0),
            "signed_mae_by_field": signed_fields.tolist(),
            "strength_mae": float(strength_fields.mean().item()),
            "strength_mae_raw255": float(strength_fields.mean().item() * 255.0),
            "strength_mae_by_field": strength_fields.tolist(),
            "categorical_accuracy": sum(accuracies) / max(1, len(accuracies)),
            "categorical_macro_f1": sum(macro_f1_values)
            / max(1, len(macro_f1_values)),
            "categorical_observable_accuracy": sum(observable_accuracies)
            / max(1, len(observable_accuracies)),
            "categorical_observable_macro_f1": sum(observable_macro_f1_values)
            / max(1, len(observable_macro_f1_values)),
            "categorical_fields": categorical,
            "race_groups": race_groups,
        }
        if self.schema.scalar_dim:
            result.update(
                {
                    "scalar_mae": float(scalar_fields.mean().item()),
                    "scalar_mae_raw255": float(
                        scalar_fields.mean().item() * 255.0
                    ),
                    "scalar_mae_by_field": scalar_fields.tolist(),
                }
            )
        if self.geometry_gate_count > 0:
            result["geometry_gate_mean"] = float(
                (
                    self.geometry_gate_sum / self.geometry_gate_count
                ).item()
            )
        return result


def selection_score(
    metrics: dict[str, Any], weights: dict[str, Any] | None = None
) -> float:
    """Lower is better; all terms are normalized validation quantities."""
    weights = weights or {
        "scalar_mae_weight": 0.0,
        "signed_mae_weight": 0.40,
        "strength_mae_weight": 0.25,
        "categorical_error_weight": 0.35,
    }
    total_weight = sum(float(value) for value in weights.values())
    categorical_f1 = float(
        metrics.get(
            "categorical_observable_macro_f1",
            metrics["categorical_macro_f1"],
        )
    )
    return (
        float(weights.get("scalar_mae_weight", 0.0))
        * float(metrics.get("scalar_mae", 0.0))
        + float(weights["signed_mae_weight"]) * float(metrics["signed_mae"])
        + float(weights["strength_mae_weight"]) * float(metrics["strength_mae"])
        + float(weights["categorical_error_weight"])
        * (1.0 - categorical_f1)
    ) / total_weight
