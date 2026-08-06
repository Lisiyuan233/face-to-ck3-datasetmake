from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


TARGET_FAMILY = "geometry_only_v1"
IDENTIFIABILITY_TARGET_FAMILY = "geometry_identifiability_v2"


@dataclass(frozen=True)
class CategoricalField:
    name: str
    classes: tuple[str, ...]
    class_counts: tuple[int, ...]
    visibility_threshold: float | None = None
    prediction_strategy: str = "independent_prediction"

    @property
    def is_constant(self) -> bool:
        return len(self.classes) == 1


@dataclass(frozen=True)
class CK3Schema:
    path: Path
    sha256: str
    version: int
    sample_count: int
    scalar_fields: tuple[str, ...]
    signed_fields: tuple[str, ...]
    categorical_fields: tuple[CategoricalField, ...]
    source_schema_path: Path | None = None
    source_schema_sha256: str | None = None
    source_signed_fields: tuple[str, ...] = ()
    source_categorical_fields: tuple[CategoricalField, ...] = ()
    scalar_source_indices: tuple[int, ...] = ()
    signed_source_indices: tuple[int, ...] = ()
    categorical_source_indices: tuple[int, ...] = ()
    categorical_class_maps: tuple[tuple[int, ...], ...] = ()

    @property
    def target_family(self) -> str:
        return (
            IDENTIFIABILITY_TARGET_FAMILY
            if self.version == 2
            else TARGET_FAMILY
        )

    @property
    def active_categorical_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, field in enumerate(self.categorical_fields)
            if not field.is_constant
        )

    @property
    def scalar_dim(self) -> int:
        return len(self.scalar_fields)

    @property
    def signed_dim(self) -> int:
        return len(self.signed_fields)

    @property
    def categorical_dim(self) -> int:
        return len(self.categorical_fields)

    def validate_label(self, label: dict[str, Any]) -> None:
        expected = {
            "scalar": self.scalar_dim,
            "signed": self.signed_dim,
            "categorical_class": self.categorical_dim,
            "categorical_strength": self.categorical_dim,
        }
        for key, length in expected.items():
            if key == "scalar" and self.scalar_dim == 0 and key not in label:
                continue
            values = label.get(key)
            if not isinstance(values, list) or len(values) != length:
                actual = len(values) if isinstance(values, list) else type(values).__name__
                raise ValueError(f"label {key!r} has {actual}, expected {length}")

        if self.scalar_dim and not all(
            0.0 <= float(value) <= 1.0 for value in label["scalar"]
        ):
            raise ValueError("scalar label is outside [0, 1]")
        if not all(-1.0 <= float(value) <= 1.0 for value in label["signed"]):
            raise ValueError("signed label is outside [-1, 1]")
        if not all(
            0.0 <= float(value) <= 1.0
            for value in label["categorical_strength"]
        ):
            raise ValueError("categorical_strength label is outside [0, 1]")
        for index, (class_id, field) in enumerate(
            zip(label["categorical_class"], self.categorical_fields)
        ):
            if not 0 <= int(class_id) < len(field.classes):
                raise ValueError(
                    f"categorical field {index} ({field.name}) has invalid class {class_id}"
                )

    def adapt_label(self, label: dict[str, Any]) -> dict[str, Any]:
        """Return a label in this schema, adapting a legacy v1 shard label.

        Identifiability v2 deliberately reuses existing tar shards.  Thirty
        signed source targets are converted to allele-invariant scalar
        strengths with ``abs(signed)``; retained signed and categorical targets
        are selected by field name.
        """

        try:
            self.validate_label(label)
            if self.scalar_dim == 0 or "scalar" in label:
                return label
        except ValueError:
            if self.version != 2:
                raise

        if self.version != 2 or not self.source_signed_fields:
            self.validate_label(label)
            return label
        source_signed = label.get("signed")
        source_classes = label.get("categorical_class")
        source_strengths = label.get("categorical_strength")
        if not isinstance(source_signed, list) or len(source_signed) != len(
            self.source_signed_fields
        ):
            raise ValueError("legacy label signed targets do not match source schema")
        if not isinstance(source_classes, list) or not isinstance(source_strengths, list):
            raise ValueError("legacy label is missing categorical targets")
        if len(source_classes) != len(self.source_categorical_fields) or len(
            source_strengths
        ) != len(self.source_categorical_fields):
            raise ValueError("legacy categorical targets do not match source schema")

        adapted = dict(label)
        adapted["scalar"] = [
            abs(float(source_signed[index])) for index in self.scalar_source_indices
        ]
        adapted["signed"] = [
            float(source_signed[index]) for index in self.signed_source_indices
        ]
        adapted_classes: list[int] = []
        adapted_strengths: list[float] = []
        for source_index, class_map in zip(
            self.categorical_source_indices, self.categorical_class_maps
        ):
            source_class_id = int(source_classes[source_index])
            if not 0 <= source_class_id < len(class_map):
                raise ValueError("legacy categorical class id is outside source schema")
            adapted_classes.append(class_map[source_class_id])
            adapted_strengths.append(float(source_strengths[source_index]))
        adapted["categorical_class"] = adapted_classes
        adapted["categorical_strength"] = adapted_strengths
        self.validate_label(adapted)
        return adapted

    def accepts_statistics_schema(self, sha256: str) -> bool:
        return sha256 == self.sha256 or (
            self.source_schema_sha256 is not None
            and sha256 == self.source_schema_sha256
        )

    def adapt_class_counts(
        self, counts: Sequence[Sequence[int]]
    ) -> list[list[int]]:
        if len(counts) == self.categorical_dim:
            direct = [list(map(int, values)) for values in counts]
            if all(
                len(values) == len(field.classes)
                for values, field in zip(direct, self.categorical_fields)
            ):
                return direct
        if not self.source_categorical_fields or len(counts) != len(
            self.source_categorical_fields
        ):
            raise ValueError("categorical statistics do not match schema")
        adapted: list[list[int]] = []
        for target, source_index, class_map in zip(
            self.categorical_fields,
            self.categorical_source_indices,
            self.categorical_class_maps,
        ):
            source = self.source_categorical_fields[source_index]
            source_counts = counts[source_index]
            if len(source_counts) != len(source.classes):
                raise ValueError(f"invalid source class counts for {target.name}")
            target_counts = [0] * len(target.classes)
            for target_index, value in zip(class_map, source_counts):
                target_counts[target_index] += int(value)
            adapted.append(target_counts)
        return adapted

    def categorical_visibility_thresholds(self, fallback: float) -> tuple[float, ...]:
        return tuple(
            float(fallback)
            if field.visibility_threshold is None
            else float(field.visibility_threshold)
            for field in self.categorical_fields
        )

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "schema_path": str(self.path),
            "schema_sha256": self.sha256,
            "schema_version": self.version,
            "source_schema_sha256": self.source_schema_sha256,
            "scalar_fields": list(self.scalar_fields),
            "signed_fields": list(self.signed_fields),
            "categorical_fields": [
                {"name": field.name, "classes": list(field.classes)}
                for field in self.categorical_fields
            ],
            "target_family": self.target_family,
        }


def _field_name(spec: dict[str, Any]) -> str:
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("schema field is missing a name")
    return name


def _visibility_threshold(spec: dict[str, Any]) -> float | None:
    raw = spec.get("recommended_visibility_threshold")
    if raw is None:
        return None
    value = float(raw) / 255.0
    if value < 0:
        raise ValueError("recommended visibility threshold must be >= 0")
    return value


def _categorical_fields(data: dict[str, Any]) -> tuple[CategoricalField, ...]:
    result = []
    for spec in data["categorical_fields"]:
        classes = tuple(str(value) for value in spec["classes"])
        counts = tuple(int(value) for value in spec["class_counts"])
        if not classes or len(classes) != len(counts):
            raise ValueError(f"invalid categorical schema for {_field_name(spec)}")
        result.append(
            CategoricalField(
                name=_field_name(spec),
                classes=classes,
                class_counts=counts,
                visibility_threshold=_visibility_threshold(spec),
                prediction_strategy=str(
                    spec.get("prediction_strategy", "independent_prediction")
                ),
            )
        )
    return tuple(result)


def _resolve_source_path(schema_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (schema_path.parent / candidate).resolve()


def load_schema(path: str | Path) -> CK3Schema:
    schema_path = Path(path).resolve()
    raw = schema_path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    version = int(data.get("schema_version", 0))
    if version not in {1, 2}:
        raise ValueError(f"unsupported schema_version={version!r}")

    source_path: Path | None = None
    source_sha256: str | None = None
    source_signed: tuple[str, ...] = ()
    source_categorical: tuple[CategoricalField, ...] = ()
    sample_count = data.get("sample_count")
    if version == 2:
        raw_source_path = data.get("source_schema_path")
        if not isinstance(raw_source_path, str) or not raw_source_path:
            raise ValueError("schema v2 requires source_schema_path")
        source_path = _resolve_source_path(schema_path, raw_source_path)
        source_raw = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_raw).hexdigest()
        expected_source_sha256 = data.get("source_schema_sha256")
        if expected_source_sha256 != source_sha256:
            raise ValueError("schema v2 source_schema_sha256 does not match source file")
        source_data = json.loads(source_raw.decode("utf-8"))
        if source_data.get("schema_version") != 1:
            raise ValueError("schema v2 source must be schema version 1")
        source_signed = tuple(
            _field_name(spec) for spec in source_data["signed_fields"]
        )
        source_categorical = _categorical_fields(source_data)
        sample_count = source_data["sample_count"] if sample_count is None else sample_count

    scalar_fields = tuple(
        _field_name(spec) for spec in data.get("scalar_fields", ())
    )
    signed_fields = tuple(_field_name(spec) for spec in data["signed_fields"])
    categorical = _categorical_fields(data)
    scalar_source_indices: tuple[int, ...] = ()
    signed_source_indices: tuple[int, ...] = ()
    categorical_source_indices: tuple[int, ...] = ()
    categorical_class_maps: tuple[tuple[int, ...], ...] = ()
    if version == 2:
        source_signed_index = {
            name: index for index, name in enumerate(source_signed)
        }
        scalar_source_indices = tuple(
            source_signed_index[name] for name in scalar_fields
        )
        signed_source_indices = tuple(
            source_signed_index[name] for name in signed_fields
        )
        source_categorical_index = {
            field.name: index for index, field in enumerate(source_categorical)
        }
        categorical_source_indices = tuple(
            source_categorical_index[field.name] for field in categorical
        )
        categorical_class_maps = tuple(
            tuple(
                target.classes.index(class_name)
                for class_name in source_categorical[source_index].classes
            )
            for target, source_index in zip(
                categorical, categorical_source_indices
            )
        )

    schema = CK3Schema(
        path=schema_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        version=version,
        sample_count=int(sample_count),
        scalar_fields=scalar_fields,
        signed_fields=signed_fields,
        categorical_fields=categorical,
        source_schema_path=source_path,
        source_schema_sha256=source_sha256,
        source_signed_fields=source_signed,
        source_categorical_fields=source_categorical,
        scalar_source_indices=scalar_source_indices,
        signed_source_indices=signed_source_indices,
        categorical_source_indices=categorical_source_indices,
        categorical_class_maps=categorical_class_maps,
    )
    if schema.signed_dim == 0 or schema.categorical_dim == 0:
        raise ValueError("schema contains an empty target family")
    if version == 1 and schema.scalar_dim:
        raise ValueError("schema v1 cannot contain scalar_fields")
    if version == 2:
        source_names = set(source_signed)
        continuous_names = set(schema.scalar_fields) | set(schema.signed_fields)
        if len(continuous_names) != schema.scalar_dim + schema.signed_dim:
            raise ValueError("schema v2 scalar/signed fields overlap")
        if not continuous_names.issubset(source_names):
            raise ValueError("schema v2 contains a field absent from source signed fields")
        source_categorical_names = {field.name for field in source_categorical}
        if not {field.name for field in categorical}.issubset(source_categorical_names):
            raise ValueError("schema v2 categorical field is absent from source schema")
    return schema
