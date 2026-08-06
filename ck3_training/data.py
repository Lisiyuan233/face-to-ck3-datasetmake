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
from torch.nn import functional as F
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .sampling import stable_fraction_includes
from .schema import CK3Schema


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_geometry_map(
    image: Image.Image,
    *,
    grid_height: int,
    grid_width: int,
    foreground_margin: float,
    foreground_softness: float,
) -> torch.Tensor:
    """Build a compact, photometry-resistant foreground/edge representation."""
    if int(grid_height) < 8 or int(grid_width) < 8:
        raise ValueError("geometry map grid dimensions must be at least 8")
    if float(foreground_margin) < 0:
        raise ValueError("foreground_margin must be >= 0")
    if float(foreground_softness) <= 0:
        raise ValueError("foreground_softness must be > 0")

    rgb = TF.to_tensor(image.convert("RGB"))
    _, height, width = rgb.shape
    patch_height = max(2, round(height * 0.08))
    patch_width = max(2, round(width * 0.12))
    top_corners = torch.cat(
        (
            rgb[:, :patch_height, :patch_width].reshape(3, -1),
            rgb[:, :patch_height, width - patch_width :].reshape(3, -1),
        ),
        dim=1,
    )
    background = top_corners.mean(dim=1)[:, None, None]
    corner_distance = torch.linalg.vector_norm(
        top_corners - background.flatten(1), dim=0
    )
    background_variation = (
        corner_distance.mean() + 2.0 * corner_distance.std()
    )
    threshold = (background_variation + float(foreground_margin)).clamp(
        min=0.03, max=0.45
    )
    distance = torch.linalg.vector_norm(rgb - background, dim=0)
    foreground = torch.sigmoid(
        (distance - threshold) / float(foreground_softness)
    )

    gray = (
        0.2989 * rgb[0]
        + 0.5870 * rgb[1]
        + 0.1140 * rgb[2]
    )
    horizontal = torch.zeros_like(gray)
    vertical = torch.zeros_like(gray)
    horizontal[:, :-1] = (gray[:, 1:] - gray[:, :-1]).abs()
    vertical[:-1, :] = (gray[1:, :] - gray[:-1, :]).abs()
    edge = ((horizontal + vertical) * 2.0).clamp_(0.0, 1.0)
    edge = edge * (0.25 + 0.75 * foreground)

    maps = torch.stack((foreground, edge), dim=0).unsqueeze(0)
    return F.adaptive_avg_pool2d(
        maps, (int(grid_height), int(grid_width))
    ).squeeze(0)


def _uniform_jitter(
    image: Image.Image,
    rng: random.Random,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float = 0.0,
) -> Image.Image:
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
    if hue > 0:
        operations.append(
            lambda value: TF.adjust_hue(value, rng.uniform(-hue, hue))
        )
    rng.shuffle(operations)
    for operation in operations:
        image = operation(image)
    return image


def normalize_exposure(
    image: Image.Image,
    *,
    target_mean: float,
    target_std: float,
    min_gain: float,
    max_gain: float,
) -> Image.Image:
    """Normalize luminance drift while retaining relative RGB differences."""

    tensor = TF.to_tensor(image.convert("RGB"))
    luminance = 0.2989 * tensor[0] + 0.5870 * tensor[1] + 0.1140 * tensor[2]
    source_mean = luminance.mean()
    source_std = luminance.std().clamp_min(1e-4)
    gain = (float(target_std) / source_std).clamp(
        min=float(min_gain), max=float(max_gain)
    )
    offset = float(target_mean) - source_mean * gain
    normalized = (tensor * gain + offset).clamp_(0.0, 1.0)
    return TF.to_pil_image(normalized)


class DualViewTransform:
    """Build strong and weak appearance views of the same aligned face."""

    def __init__(
        self,
        height: int,
        width: int,
        augmentation: dict[str, Any],
        training: bool,
        dual_view: bool,
        geometry_map_config: dict[str, Any] | None = None,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.augmentation = augmentation
        self.training = training
        self.dual_view = dual_view
        self.geometry_map_config = dict(geometry_map_config or {})
        self.geometry_map_enabled = bool(
            self.geometry_map_config.get("enabled", False)
        )

    def _geometry(
        self,
        image: Image.Image,
        rng: random.Random,
        allow_horizontal_flip: bool,
    ) -> Image.Image:
        image = TF.resize(
            image,
            [self.height, self.width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        exposure = self.augmentation.get("exposure_normalization", {})
        if bool(exposure.get("enabled", False)):
            image = normalize_exposure(
                image,
                target_mean=float(exposure["target_mean"]),
                target_std=float(exposure["target_std"]),
                min_gain=float(exposure["min_gain"]),
                max_gain=float(exposure["max_gain"]),
            )
        if not self.training:
            return image

        if (
            allow_horizontal_flip
            and rng.random() < float(self.augmentation["horizontal_flip"])
        ):
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

    def _views_from_aligned(
        self, aligned: Image.Image, rng: random.Random
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            tensor = self._to_normalized_tensor(aligned)
            return tensor, tensor

        geometry = _uniform_jitter(
            aligned.copy(),
            rng,
            float(self.augmentation["geometry_brightness"]),
            float(self.augmentation["geometry_contrast"]),
            float(self.augmentation["geometry_saturation"]),
            float(self.augmentation["geometry_hue"]),
        )
        if rng.random() < float(self.augmentation["geometry_grayscale"]):
            geometry = TF.to_grayscale(geometry, num_output_channels=3)
        if rng.random() < float(self.augmentation["blur_probability"]):
            sigma = rng.uniform(0.1, 1.0)
            geometry = TF.gaussian_blur(
                geometry, kernel_size=[3, 3], sigma=[sigma, sigma]
            )

        reference = _uniform_jitter(
            aligned.copy(),
            rng,
            float(self.augmentation["reference_brightness"]),
            float(self.augmentation["reference_contrast"]),
            float(self.augmentation["reference_saturation"]),
            float(self.augmentation["reference_hue"]),
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
        reference_tensor = self._to_normalized_tensor(reference)
        if not self.dual_view:
            reference_tensor = geometry_tensor
        return geometry_tensor, reference_tensor

    def __call__(
        self, image: Image.Image, *, allow_horizontal_flip: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random
        aligned = self._geometry(image, rng, allow_horizontal_flip)
        return self._views_from_aligned(aligned, rng)

    def with_geometry_map(
        self, image: Image.Image, *, allow_horizontal_flip: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.geometry_map_enabled:
            raise RuntimeError("geometry map generation is not enabled")
        rng = random
        aligned = self._geometry(image, rng, allow_horizontal_flip)
        geometry, reference = self._views_from_aligned(aligned, rng)
        geometry_map = build_geometry_map(
            aligned,
            grid_height=int(self.geometry_map_config["grid_height"]),
            grid_width=int(self.geometry_map_config["grid_width"]),
            foreground_margin=float(
                self.geometry_map_config["foreground_margin"]
            ),
            foreground_softness=float(
                self.geometry_map_config["foreground_softness"]
            ),
        )
        return geometry, reference, geometry_map


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
        sample_fraction: float = 1.0,
        require_side_view: bool = False,
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
        self.sample_fraction = float(sample_fraction)
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        self.require_side_view = bool(require_side_view)
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
                filename = name.name
                lower = filename.lower()
                if lower.endswith(".front.jpg"):
                    key = filename[: -len(".front.jpg")]
                    part = "front_image"
                elif lower.endswith(".front.jpeg"):
                    key = filename[: -len(".front.jpeg")]
                    part = "front_image"
                elif lower.endswith(".side.jpg"):
                    key = filename[: -len(".side.jpg")]
                    part = "side_image"
                elif lower.endswith(".side.jpeg"):
                    key = filename[: -len(".side.jpeg")]
                    part = "side_image"
                elif lower.endswith((".jpg", ".jpeg")):
                    key = name.stem
                    part = "front_image"
                elif lower.endswith(".json"):
                    key = name.stem
                    part = "json"
                else:
                    continue
                if not stable_fraction_includes(
                    key, self.sample_fraction, self.seed
                ):
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot extract {member.name} from {path}")
                parts = pending.setdefault(key, {})
                parts[part] = stream.read()
                required = {"front_image", "json"}
                if self.require_side_view:
                    required.add("side_image")
                if not required.issubset(parts):
                    continue
                yield {
                    "key": key,
                    "source": str(path),
                    "front_image_bytes": parts["front_image"],
                    "side_image_bytes": parts.get("side_image"),
                    "json_bytes": parts["json"],
                }
                pending.pop(key, None)
        if pending:
            missing = ", ".join(sorted(pending)[:5])
            raise RuntimeError(f"unpaired members in {path}: {missing}")

    def _decode(self, raw: dict[str, Any]) -> dict[str, Any]:
        try:
            source_label = json.loads(raw["json_bytes"].decode("utf-8"))
            label = self.schema.adapt_label(source_label)
            self.schema.validate_label(label)
            with Image.open(io.BytesIO(raw["front_image_bytes"])) as decoded:
                front_image = decoded.convert("RGB")
            if self.transform.geometry_map_enabled:
                geometry, reference, geometry_map = (
                    self.transform.with_geometry_map(front_image)
                )
            else:
                geometry, reference = self.transform(front_image)
                geometry_map = None
            sample = {
                "sample_id": str(label["sample_id"]),
                "geometry_view": geometry,
                "reference_view": reference,
                "scalar": torch.tensor(
                    label.get("scalar", []), dtype=torch.float32
                ),
                "signed": torch.tensor(label["signed"], dtype=torch.float32),
                "categorical_class": torch.tensor(
                    label["categorical_class"], dtype=torch.long
                ),
                "categorical_strength": torch.tensor(
                    label["categorical_strength"], dtype=torch.float32
                ),
                "race_group": torch.tensor(
                    int(label.get("race_group", -1)), dtype=torch.long
                ),
            }
            if geometry_map is not None:
                sample["geometry_map"] = geometry_map
            if raw["side_image_bytes"] is not None:
                with Image.open(io.BytesIO(raw["side_image_bytes"])) as decoded:
                    side_image = decoded.convert("RGB")
                if self.transform.geometry_map_enabled:
                    side, _, side_geometry_map = (
                        self.transform.with_geometry_map(
                            side_image, allow_horizontal_flip=False
                        )
                    )
                    sample["side_geometry_map"] = side_geometry_map
                else:
                    side, _ = self.transform(
                        side_image, allow_horizontal_flip=False
                    )
                sample["side_view"] = side
            return sample
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
