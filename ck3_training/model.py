from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .schema import CK3Schema


class GeometryMapEncoder(nn.Module):
    """Encode low-resolution silhouette/edge maps without a second image backbone."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(128, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
        )

    def forward(self, geometry_map: torch.Tensor) -> torch.Tensor:
        if geometry_map.ndim != 4:
            raise ValueError(
                "geometry_map must have shape [batch, channels, height, width]"
            )
        if geometry_map.shape[1] != self.input_channels:
            raise ValueError(
                f"geometry_map has {geometry_map.shape[1]} channels; "
                f"expected {self.input_channels}"
            )
        return self.projection(self.features(geometry_map))


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    try:
        from torchvision.models import (
            ConvNeXt_Tiny_Weights,
            ResNet18_Weights,
            convnext_tiny,
            resnet18,
        )
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required; install requirements-train.txt"
        ) from error

    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        try:
            backbone = convnext_tiny(weights=weights)
        except Exception as error:
            if pretrained:
                raise RuntimeError(
                    "unable to load ConvNeXt-Tiny pretrained weights; place the weights "
                    "in the torch cache or set model.pretrained=false"
                ) from error
            raise
        feature_dim = int(backbone.classifier[2].in_features)
        backbone.classifier[2] = nn.Identity()
        return backbone, feature_dim

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            backbone = resnet18(weights=weights)
        except Exception as error:
            if pretrained:
                raise RuntimeError(
                    "unable to load ResNet-18 pretrained weights; place the weights "
                    "in the torch cache or set model.pretrained=false"
                ) from error
            raise
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    raise ValueError(f"unsupported backbone: {name}")


class FaceToCK3Model(nn.Module):
    def __init__(self, schema: CK3Schema, config: dict[str, Any]) -> None:
        super().__init__()
        self.schema = schema
        self.dual_view = bool(config.get("dual_view", True))
        self.use_side_view = bool(config.get("side_view", False))
        geometry_config = config.get("geometry_branch", {})
        self.use_geometry_branch = bool(geometry_config.get("enabled", False))
        self.backbone, feature_dim = _build_backbone(
            str(config["backbone"]), bool(config.get("pretrained", True))
        )
        dropout = float(config.get("dropout", 0.1))
        self.feature_dropout = nn.Dropout(dropout)
        self.signed_head = nn.Linear(feature_dim, schema.signed_dim)
        self.strength_head = nn.Linear(feature_dim, schema.categorical_dim)
        self.categorical_heads = nn.ModuleDict(
            {
                str(index): nn.Linear(feature_dim, len(schema.categorical_fields[index].classes))
                for index in schema.active_categorical_indices
            }
        )
        if self.use_side_view:
            self.side_projection = nn.Linear(feature_dim, feature_dim)
            self.side_gate = nn.Linear(feature_dim * 2, feature_dim)
            self.fusion_norm = nn.LayerNorm(feature_dim)
        if self.use_geometry_branch:
            input_channels = 4 if self.use_side_view else 2
            self.geometry_encoder = GeometryMapEncoder(
                input_channels=input_channels,
                hidden_dim=int(geometry_config.get("hidden_dim", 256)),
                output_dim=feature_dim,
                dropout=float(geometry_config.get("dropout", dropout)),
            )
            self.geometry_gate = nn.Linear(feature_dim * 2, feature_dim)
            self.geometry_fusion_norm = nn.LayerNorm(feature_dim)
            self.geometry_gate_bias = float(geometry_config.get("gate_bias", -2.0))
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        modules = [self.signed_head, self.strength_head]
        modules.extend(self.categorical_heads.values())
        if self.use_side_view:
            modules.extend([self.side_projection, self.side_gate])
        if self.use_geometry_branch:
            modules.append(self.geometry_gate)
        for module in modules:
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        if self.use_side_view:
            nn.init.constant_(self.side_gate.bias, -1.0)
        if self.use_geometry_branch:
            nn.init.constant_(self.geometry_gate.bias, self.geometry_gate_bias)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.feature_dropout(features)

    def _geometry_outputs(self, features: torch.Tensor) -> dict[str, Any]:
        return {
            "signed": torch.tanh(self.signed_head(features)),
            "categorical_strength": torch.sigmoid(self.strength_head(features)),
            "categorical_logits": {
                key: head(features) for key, head in self.categorical_heads.items()
            },
        }

    def _fuse_views(
        self, front_features: torch.Tensor, side_features: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.sigmoid(
            self.side_gate(torch.cat((front_features, side_features), dim=1))
        )
        side_delta = self.side_projection(side_features)
        return self.fusion_norm(front_features + gate * side_delta)

    def _encode_geometry(
        self,
        geometry_map: torch.Tensor | None,
        side_geometry_map: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.use_geometry_branch:
            return None
        if geometry_map is None:
            raise ValueError("geometry branch requires geometry_map")
        maps = [geometry_map]
        if self.use_side_view:
            if side_geometry_map is None:
                raise ValueError(
                    "multi-view geometry branch requires side_geometry_map"
                )
            if geometry_map.shape[0] != side_geometry_map.shape[0]:
                raise ValueError(
                    "front and side geometry maps use different batches"
                )
            if geometry_map.shape[2:] != side_geometry_map.shape[2:]:
                raise ValueError(
                    "front and side geometry maps use different grids"
                )
            maps.append(side_geometry_map)
        return self.geometry_encoder(torch.cat(maps, dim=1))

    def _fuse_geometry(
        self,
        visual_features: torch.Tensor,
        geometry_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if geometry_features is None:
            return visual_features, None
        gate = torch.sigmoid(
            self.geometry_gate(
                torch.cat((visual_features, geometry_features), dim=1)
            )
        )
        fused = self.geometry_fusion_norm(
            visual_features + gate * geometry_features
        )
        return fused, gate.mean(dim=1)

    def forward(
        self,
        geometry_view: torch.Tensor,
        reference_view: torch.Tensor | None = None,
        side_view: torch.Tensor | None = None,
        geometry_map: torch.Tensor | None = None,
        side_geometry_map: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        geometry_features = self.encode(geometry_view)
        geometry_map_features = self._encode_geometry(
            geometry_map, side_geometry_map
        )
        side_features = None
        if self.use_side_view:
            if side_view is None:
                raise ValueError("multi-view model requires side_view")
            side_features = self.encode(side_view)
            geometry_features = self._fuse_views(
                geometry_features, side_features
            )
        geometry_features, geometry_gate_mean = self._fuse_geometry(
            geometry_features, geometry_map_features
        )
        outputs = self._geometry_outputs(geometry_features)
        if geometry_gate_mean is not None:
            outputs["geometry_gate_mean"] = geometry_gate_mean
        if self.dual_view:
            if reference_view is None:
                raise ValueError("dual-view model requires reference_view")
            reference_features = self.encode(reference_view)
            if side_features is not None:
                reference_features = self._fuse_views(
                    reference_features, side_features
                )
            reference_features, reference_gate_mean = self._fuse_geometry(
                reference_features, geometry_map_features
            )
            outputs["reference"] = self._geometry_outputs(reference_features)
            if reference_gate_mean is not None:
                outputs["reference_geometry_gate_mean"] = (
                    reference_gate_mean
                )
        return outputs

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    def parameter_groups(
        self, backbone_lr: float, head_lr: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        groups = {
            "backbone_decay": [],
            "backbone_no_decay": [],
            "heads_decay": [],
            "heads_no_decay": [],
        }
        for name, parameter in self.named_parameters():
            family = "backbone" if id(parameter) in backbone_ids else "heads"
            decay = not (parameter.ndim <= 1 or name.endswith(".bias"))
            groups[f"{family}_{'decay' if decay else 'no_decay'}"].append(parameter)
        result = []
        for name, parameters in groups.items():
            if not parameters:
                continue
            family = "backbone" if name.startswith("backbone") else "heads"
            result.append(
                {
                    "params": parameters,
                    "lr": float(backbone_lr if family == "backbone" else head_lr),
                    "weight_decay": float(weight_decay if name.endswith("_decay") else 0.0),
                    "name": name,
                }
            )
        return result
