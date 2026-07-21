from __future__ import annotations

import io
import json
import math
import random
import tarfile
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .schema import CK3Schema


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _uniform_jitter(image: Image.Image, rng: random.Random, brightness: float,
                    contrast: float, saturation: float) -> Image.Image:
    operations = []
    if brightness > 0:
        operations.append(
            lambda value: TF.adjust_brightness(
                value, rng.uniform(1.0 - brightness, 1.0 + brightness)
            )
        )
    if contrast > 0:
        operations.append(
            lambda value: TF.adjust_contrast(
                value, rng.uniform(1.0 - contrast, 1.0 + contrast)
            )
        )
    if saturation > 0:
        operations.append(
            lambda value: TF.adjust_saturation(
                value, rng.uniform(1.0 - saturation, 1.0 + saturation)
            )
        )
    rng.shuffle(operations)
    for operation in operations:
        image = operation(image)
    return image


class DualViewTransform:
    """Build geometry-robust and color-preserving views from one aligned face."""

    def __init__(
        self,
        height: int,
        width: int,
        augmentation: dict[str, Any],
        training: bool,
        dual_view: bool,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.augmentation = augmentation
        self.training = training
        self.dual_view = dual_view

    def _geometry(self, image: Image.Image, rng: random.Random) -> Image.Image:
        image = TF.resize(
            image,
            [self.height, self.width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if not self.training:
            return image

        if rng.random() < float(self.augmentation["horizontal_flip"]):
            image = TF.hflip(image)
        angle = rng.uniform(
            -float(self.augmentation["rotation_degrees"]),
            float(self.augmentation["rotation_degrees"]),
        )
        translate_fraction = float(self.augmentation["translate_fraction"])
        translate = [
            round(rng.uniform(-translate_fraction, translate_fraction) * self.width),
            round(rng.uniform(-translate_fraction, translate_fraction) * self.height),
        ]
        scale = rng.uniform(
            float(self.augmentation["scale_min"]),
            float(self.augmentation["scale_max"]),
        )
        return TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=[34, 36, 40],
        )

    def _to_normalized_tensor(self, image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random
        aligned = self._geometry(image, rng)
        if not self.training:
            tensor = self._to_normalized_tensor(aligned)
            return tensor, tensor

        geometry = _uniform_jitter(
            aligned.copy(),
            rng,
            float(self.augmentation["geometry_brightness"]),
            float(self.augmentation["geometry_contrast"]),
            float(self.augmentation["geometry_saturation"]),
        )
        if rng.random() < float(self.augmentation["geometry_grayscale"]):
            geometry = TF.to_grayscale(geometry, num_output_channels=3)
        if rng.random() < float(self.augmentation["blur_probability"]):
            sigma = rng.uniform(0.1, 1.0)
            geometry = TF.gaussian_blur(
                geometry, kernel_size=[3, 3], sigma=[sigma, sigma]
            )

        color = _uniform_jitter(
            aligned.copy(),
            rng,
            float(self.augmentation["color_brightness"]),
            float(self.augmentation["color_contrast"]),
            float(self.augmentation["color_saturation"]),
        )

        geometry_tensor = TF.to_tensor(geometry)
        noise_std = float(self.augmentation["noise_std"])
        if noise_std > 0:
            geometry_tensor = (
                geometry_tensor + torch.randn_like(geometry_tensor) * noise_std
            ).clamp_(0.0, 1.0)
        if rng.random() < float(self.augmentation["upper_occlusion_probability"]):
            min_height = max(2, round(self.height * 0.04))
            max_height = max(min_height, round(self.height * 0.14))
            occlusion_height = rng.randint(min_height, max_height)
            max_top = max(0, round(self.height * 0.22) - occlusion_height)
            top = rng.randint(0, max_top) if max_top else 0
            left = rng.randint(0, max(0, round(self.width * 0.15)))
            right = rng.randint(
                max(left + 1, round(self.width * 0.85)), self.width
            )
            fill = geometry_tensor.mean(dim=(1, 2), keepdim=True)
            geometry_tensor[:, top : top + occlusion_height, left:right] = fill

        geometry_tensor = TF.normalize(geometry_tensor, IMAGENET_MEAN, IMAGENET_STD)
        color_tensor = self._to_normalized_tensor(color)
        if not self.dual_view:
            color_tensor = geometry_tensor
        return geometry_tensor, color_tensor


class TarShardDataset(IterableDataset):
    """Dependency-light reader for the repository's WebDataset-compatible tar files."""

    def __init__(
        self,
        shards: list[Path],
        schema: CK3Schema,
        transform: DualViewTransform,
        *,
        training: bool,
        repeat: bool,
        shuffle_buffer: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        if not shards:
            raise ValueError("no tar shards were found")
        self.shards = tuple(Path(path) for path in shards)
        self.schema = schema
        self.transform = transform
        self.training = training
        self.repeat = repeat
        self.shuffle_buffer = max(1, int(shuffle_buffer))
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assigned_shards(self) -> tuple[list[Path], random.Random]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers_per_rank = worker.num_workers if worker else 1
        global_worker = self.rank * workers_per_rank + worker_id
        total_workers = self.world_size * workers_per_rank
        rng = random.Random(
            self.seed + self.epoch * 1_000_003 + global_worker * 10_007
        )
        shards = list(self.shards)
        if self.training:
            rng.shuffle(shards)
        assigned = shards[global_worker::total_workers]
        if not assigned:
            raise RuntimeError(
                f"worker {global_worker}/{total_workers} has no shard; "
                "reduce DataLoader workers or distributed world size"
            )
        return assigned, rng

    def _read_shard(self, path: Path) -> Iterator[dict[str, Any]]:
        pending: dict[str, dict[str, bytes]] = {}
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name)
                suffix = name.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".json"}:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot extract {member.name} from {path}")
                key = name.stem
                parts = pending.setdefault(key, {})
                parts["image" if suffix in {".jpg", ".jpeg"} else "json"] = stream.read()
                if "image" not in parts or "json" not in parts:
                    continue
                yield {
                    "key": key,
                    "source": str(path),
                    "image_bytes": parts["image"],
                    "json_bytes": parts["json"],
                }
                pending.pop(key, None)
        if pending:
            missing = ", ".join(sorted(pending)[:5])
            raise RuntimeError(f"unpaired members in {path}: {missing}")

    def _decode(self, raw: dict[str, Any]) -> dict[str, Any]:
        try:
            label = json.loads(raw["json_bytes"].decode("utf-8"))
            self.schema.validate_label(label)
            with Image.open(io.BytesIO(raw["image_bytes"])) as decoded:
                image = decoded.convert("RGB")
            geometry, color = self.transform(image)
            return {
                "sample_id": str(label["sample_id"]),
                "geometry_view": geometry,
                "color_view": color,
                "signed": torch.tensor(label["signed"], dtype=torch.float32),
                "categorical_class": torch.tensor(
                    label["categorical_class"], dtype=torch.long
                ),
                "categorical_strength": torch.tensor(
                    label["categorical_strength"], dtype=torch.float32
                ),
                "colors": torch.tensor(label["colors"], dtype=torch.float32),
                "race_group": torch.tensor(
                    int(label.get("race_group", -1)), dtype=torch.long
                ),
            }
        except Exception as error:
            raise RuntimeError(
                f"invalid sample {raw['key']} in {raw['source']}: {error}"
            ) from error

    def _source(self, assigned: list[Path], rng: random.Random) -> Iterator[dict[str, Any]]:
        while True:
            order = list(assigned)
            if self.training:
                rng.shuffle(order)
            for shard in order:
                yield from self._read_shard(shard)
            if not self.repeat:
                break

    def __iter__(self) -> Iterator[dict[str, Any]]:
        assigned, rng = self._assigned_shards()
        source = self._source(assigned, rng)
        if not self.training or self.shuffle_buffer <= 1:
            for raw in source:
                yield self._decode(raw)
            return

        buffer: list[dict[str, Any]] = []
        for sample in source:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(sample)
                continue
            index = rng.randrange(len(buffer))
            yield self._decode(buffer[index])
            buffer[index] = sample
        while buffer:
            yield self._decode(buffer.pop(rng.randrange(len(buffer))))


def discover_shards(data_root: str | Path, split: str) -> list[Path]:
    split_dir = Path(data_root) / split
    shards = sorted(split_dir.glob(f"{split}-*.tar"))
    partials = list(split_dir.glob("*.partial"))
    if partials:
        raise RuntimeError(f"partial shards exist in {split_dir}: {partials[:3]}")
    if not shards:
        raise FileNotFoundError(f"no {split} shards found in {split_dir}")
    return shards


def manifest_counts(path: str | Path) -> dict[str, int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: int(value) for key, value in data["split"]["counts"].items()}
