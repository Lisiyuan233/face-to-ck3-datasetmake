from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CategoricalField:
    name: str
    classes: tuple[str, ...]
    class_counts: tuple[int, ...]

    @property
    def is_constant(self) -> bool:
        return len(self.classes) == 1


@dataclass(frozen=True)
class CK3Schema:
    path: Path
    sha256: str
    sample_count: int
    signed_fields: tuple[str, ...]
    categorical_fields: tuple[CategoricalField, ...]
    color_fields: tuple[str, ...]

    @property
    def active_categorical_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, field in enumerate(self.categorical_fields)
            if not field.is_constant
        )

    @property
    def signed_dim(self) -> int:
        return len(self.signed_fields)

    @property
    def categorical_dim(self) -> int:
        return len(self.categorical_fields)

    @property
    def color_dim(self) -> int:
        return len(self.color_fields) * 2

    def validate_label(self, label: dict[str, Any]) -> None:
        expected = {
            "signed": self.signed_dim,
            "categorical_class": self.categorical_dim,
            "categorical_strength": self.categorical_dim,
            "colors": self.color_dim,
        }
        for key, length in expected.items():
            values = label.get(key)
            if not isinstance(values, list) or len(values) != length:
                actual = len(values) if isinstance(values, list) else type(values).__name__
                raise ValueError(f"label {key!r} has {actual}, expected {length}")

        if not all(-1.0 <= float(value) <= 1.0 for value in label["signed"]):
            raise ValueError("signed label is outside [-1, 1]")
        for key in ("categorical_strength", "colors"):
            if not all(0.0 <= float(value) <= 1.0 for value in label[key]):
                raise ValueError(f"{key} label is outside [0, 1]")
        for index, (class_id, field) in enumerate(
            zip(label["categorical_class"], self.categorical_fields)
        ):
            if not 0 <= int(class_id) < len(field.classes):
                raise ValueError(
                    f"categorical field {index} ({field.name}) has invalid class {class_id}"
                )

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "schema_path": str(self.path),
            "schema_sha256": self.sha256,
            "signed_fields": list(self.signed_fields),
            "categorical_fields": [
                {"name": field.name, "classes": list(field.classes)}
                for field in self.categorical_fields
            ],
            "color_fields": list(self.color_fields),
        }


def _field_name(spec: dict[str, Any]) -> str:
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("schema field is missing a name")
    return name


def load_schema(path: str | Path) -> CK3Schema:
    schema_path = Path(path).resolve()
    raw = schema_path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version={data.get('schema_version')!r}")

    categorical = []
    for spec in data["categorical_fields"]:
        classes = tuple(str(value) for value in spec["classes"])
        counts = tuple(int(value) for value in spec["class_counts"])
        if not classes or len(classes) != len(counts):
            raise ValueError(f"invalid categorical schema for {_field_name(spec)}")
        categorical.append(
            CategoricalField(
                name=_field_name(spec),
                classes=classes,
                class_counts=counts,
            )
        )

    schema = CK3Schema(
        path=schema_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        sample_count=int(data["sample_count"]),
        signed_fields=tuple(_field_name(spec) for spec in data["signed_fields"]),
        categorical_fields=tuple(categorical),
        color_fields=tuple(_field_name(spec) for spec in data["color_fields"]),
    )
    if schema.signed_dim == 0 or schema.categorical_dim == 0 or schema.color_dim == 0:
        raise ValueError("schema contains an empty target family")
    return schema
