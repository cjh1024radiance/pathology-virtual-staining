from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent_stain_model_torch import AgentStainModelConfig, CrossModalFeatureAlignModule, DecisionSignalEncoder
from .diagnosis_aware_loss_torch import DiagnosisAwareWeightedLossTorch


@dataclass
class VirtualStainResUNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    attention_input_channels: int = 1
    use_attention_conditioned_input: bool = True
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    agent_channels: int = 4
    attention_dropout: float = 0.0
    use_residual_gate: bool = True
    clinical_loss_weight: float = 0.45
    use_he_aux_head: bool = True
    he_aux_weight: float = 0.20
    focus_reconstruction_weight: float = 0.75
    background_reconstruction_weight: float = 0.25
    focus_gamma: float = 1.25
    output_activation: str = "sigmoid"

    def encoder_channels(self) -> Tuple[int, ...]:
        return tuple(self.base_channels * scale for scale in self.channel_multipliers)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()
        self.block = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.downsample(x))


class AttentionConditionedStem(nn.Module):
    def __init__(self, rgb_channels: int, attention_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(rgb_channels + attention_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(attention_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat([rgb, attention], dim=1))
        gate = self.gate(attention)
        return fused * (1.0 + gate)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.block = nn.Sequential(
            ResidualConvBlock(out_channels + skip_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class GlobalContextBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 16)
        self.branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        x = self.branch(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class LocalDetailBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class DualBranchFusionHead(nn.Module):
    def __init__(self, decoder_channels: int, global_channels: int, local_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = decoder_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels + global_channels + local_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, decoder_feat: torch.Tensor, global_feat: torch.Tensor, local_feat: torch.Tensor, activation: str) -> torch.Tensor:
        if global_feat.shape[-2:] != decoder_feat.shape[-2:]:
            global_feat = F.interpolate(global_feat, size=decoder_feat.shape[-2:], mode="bilinear", align_corners=False)
        if local_feat.shape[-2:] != decoder_feat.shape[-2:]:
            local_feat = F.interpolate(local_feat, size=decoder_feat.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([decoder_feat, global_feat, local_feat], dim=1))
        return apply_output_activation(self.out(fused), activation)


class HEAuxiliaryHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        he = self.head(x)
        return {
            "hematoxylin": he[:, 0:1],
            "eosin": he[:, 1:2],
        }


class AttentionConditionedResUNet(nn.Module):
    """
    Mainstream stain-transfer backbone with light but meaningful innovations:
    1. Agent attention maps are injected through cross-modal alignment modules.
    2. A dual-branch representation separates global tissue color and local nuclear / membrane detail.
    3. The decoder predicts both RGB stain output and auxiliary H/E maps.
    """

    def __init__(self, config: Optional[VirtualStainResUNetConfig] = None) -> None:
        super().__init__()
        self.config = config or VirtualStainResUNetConfig()
        channels = self.config.encoder_channels()
        stem_channels = channels[0]

        self.attention_stem = AttentionConditionedStem(
            rgb_channels=self.config.in_channels,
            attention_channels=self.config.attention_input_channels,
            out_channels=stem_channels,
        )
        self.global_branch = GlobalContextBranch(stem_channels, stem_channels)
        self.local_branch = LocalDetailBranch(stem_channels, stem_channels)

        self.encoder_stages = nn.ModuleList()
        in_channels = stem_channels
        for index, out_channels in enumerate(channels):
            self.encoder_stages.append(EncoderStage(in_channels, out_channels, downsample=index > 0))
            in_channels = out_channels

        self.bottleneck = nn.Sequential(
            ResidualConvBlock(channels[-1], channels[-1]),
            ResidualConvBlock(channels[-1], channels[-1]),
        )

        self.decision_encoder = DecisionSignalEncoder(
            AgentStainModelConfig(
                agent_channels=self.config.agent_channels,
                encoder_channels=channels,
                attention_dropout=self.config.attention_dropout,
                use_residual_gate=self.config.use_residual_gate,
            )
        )
        self.align_modules = nn.ModuleList(
            [
                CrossModalFeatureAlignModule(
                    feature_channels=channel,
                    attention_dropout=self.config.attention_dropout,
                    use_residual_gate=self.config.use_residual_gate,
                )
                for channel in channels
            ]
        )

        reversed_channels = list(reversed(channels))
        self.decoder_stages = nn.ModuleList()
        for index in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[index]
            skip_ch = reversed_channels[index + 1]
            out_ch = reversed_channels[index + 1]
            self.decoder_stages.append(DecoderStage(in_ch, skip_ch, out_ch))

        self.rgb_head = DualBranchFusionHead(
            decoder_channels=channels[0],
            global_channels=channels[0],
            local_channels=channels[0],
            out_channels=self.config.out_channels,
        )
        self.he_head = HEAuxiliaryHead(channels[0]) if self.config.use_he_aux_head else None

    def forward(self, rgb: torch.Tensor, decision_maps: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected rgb tensor [B, {self.config.in_channels}, H, W].")
        if decision_maps.ndim != 4 or decision_maps.shape[1] != self.config.agent_channels:
            raise ValueError(f"Expected decision_maps tensor [B, {self.config.agent_channels}, H, W].")

        attention_input = decision_maps[:, : self.config.attention_input_channels]
        if not self.config.use_attention_conditioned_input:
            attention_input = torch.zeros_like(attention_input)
        stem_feat = self.attention_stem(rgb, attention_input)
        global_feat = self.global_branch(stem_feat, rgb.shape[-2:])
        local_feat = self.local_branch(stem_feat)

        encoder_features: List[torch.Tensor] = []
        x = stem_feat
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)

        decision_features = self.decision_encoder(decision_maps)
        conditioned_skips: List[torch.Tensor] = []
        conditioned_encoder: List[torch.Tensor] = []
        for encoder_feat, decision_feat, align in zip(encoder_features, decision_features, self.align_modules):
            enc_out, skip_out = align(encoder_feat, encoder_feat, decision_feat)
            conditioned_encoder.append(enc_out)
            conditioned_skips.append(skip_out)

        x = self.bottleneck(conditioned_encoder[-1])
        decoder_skips = list(reversed(conditioned_skips[:-1]))
        for index, decoder in enumerate(self.decoder_stages):
            x = decoder(x, decoder_skips[index])

        rgb_out = self.rgb_head(x, global_feat, local_feat, self.config.output_activation)
        if self.he_head is not None:
            he_out = self.he_head(x)
        else:
            zeros = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
            he_out = {
                "hematoxylin": zeros,
                "eosin": zeros,
            }
        return {
            "rgb": rgb_out,
            "hematoxylin": he_out["hematoxylin"],
            "eosin": he_out["eosin"],
        }


class VirtualStainCompositeLoss(nn.Module):
    """
    Composite objective for the final training setup:
    - diagnosis-aware RGB reconstruction
    - H/E auxiliary supervision
    - lesion-focused reconstruction
    - clinical structure preservation
    """

    def __init__(self, config: Optional[VirtualStainResUNetConfig] = None) -> None:
        super().__init__()
        self.config = config or VirtualStainResUNetConfig()
        self.diagnosis_loss = DiagnosisAwareWeightedLossTorch()
        self.diagnosis_loss.config.alpha_clinical = self.config.clinical_loss_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_rgb: torch.Tensor,
        decision_maps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        focus_target, focus_weight = self._build_focus_supervision(decision_maps)
        rgb_loss = self.diagnosis_loss(outputs["rgb"], target_rgb)
        focus_reconstruction_loss = self._weighted_rgb_l1(outputs["rgb"], target_rgb, focus_weight)
        if self.config.use_he_aux_head and self.config.he_aux_weight > 0:
            he_target = rgb_to_he_maps(target_rgb)
            he_pred = torch.cat([outputs["hematoxylin"], outputs["eosin"]], dim=1)
            he_loss = F.l1_loss(he_pred, he_target)
        else:
            he_loss = torch.zeros((), dtype=target_rgb.dtype, device=target_rgb.device)

        total_loss = (
            rgb_loss["total_loss"]
            + self.config.focus_reconstruction_weight * focus_reconstruction_loss
            + self.config.he_aux_weight * he_loss
        )

        result = dict(rgb_loss)
        result["focus_reconstruction_loss"] = focus_reconstruction_loss
        result["he_aux_loss"] = he_loss
        result["focus_region_mean"] = focus_target.mean()
        result["focus_weight_mean"] = focus_weight.mean()
        result["total_loss"] = total_loss
        return result

    def _build_focus_supervision(self, decision_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention = decision_maps[:, 0:1].clamp(0.0, 1.0)
        highres = decision_maps[:, 1:2].clamp(0.0, 1.0)
        tissue = decision_maps[:, 2:3].clamp(0.0, 1.0)
        confidence = decision_maps[:, 3:4].clamp(0.0, 1.0)
        focus = torch.maximum(attention, 0.85 * highres)
        focus = torch.maximum(focus, attention * confidence)
        focus = focus.pow(self.config.focus_gamma).clamp(0.0, 1.0) * tissue
        weight = self.config.background_reconstruction_weight + (
            self.config.focus_reconstruction_weight - self.config.background_reconstruction_weight
        ) * focus
        return focus, weight

    @staticmethod
    def _weighted_rgb_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        diff = torch.abs(pred - target).mean(dim=1, keepdim=True)
        return (diff * weight).sum() / weight.sum().clamp(min=eps)


def rgb_to_he_maps(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected RGB tensor [B, 3, H, W].")
    rgb = rgb.float()
    if rgb.max() > 1.0 or rgb.min() < 0.0:
        rgb = rgb / 255.0
    rgb = rgb.clamp(eps, 1.0)
    od = -torch.log(rgb)
    matrix = torch.tensor(
        [
            [0.650, 0.072, 0.268],
            [0.704, 0.990, 0.570],
            [0.286, 0.105, 0.776],
        ],
        dtype=rgb.dtype,
        device=rgb.device,
    )
    matrix = matrix / (torch.linalg.norm(matrix, dim=0, keepdim=True) + eps)
    inv_t = torch.inverse(matrix.t())
    batch, _, height, width = od.shape
    flat = od.permute(0, 2, 3, 1).reshape(-1, 3)
    concentrations = flat @ inv_t
    concentrations = concentrations.clamp(min=0.0).reshape(batch, height, width, 3).permute(0, 3, 1, 2)
    he = concentrations[:, 0:2]
    return normalize_batch_maps(he)


def normalize_batch_maps(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    batch = tensor.shape[0]
    flat = tensor.reshape(batch, tensor.shape[1], -1)
    low = torch.quantile(flat, 0.01, dim=2, keepdim=True)
    high = torch.quantile(flat, 0.99, dim=2, keepdim=True)
    norm = (flat - low) / (high - low).clamp(min=eps)
    return norm.clamp(0.0, 1.0).reshape_as(tensor)


def apply_output_activation(tensor: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(tensor)
    if activation == "tanh":
        return (torch.tanh(tensor) + 1.0) / 2.0
    if activation == "identity":
        return tensor
    raise ValueError(f"Unsupported output activation: {activation}")


__all__ = [
    "VirtualStainResUNetConfig",
    "AttentionConditionedResUNet",
    "DiagnosisPreservingVirtualHENetwork",
    "AgentGuidedDiagnosisAwareResUNet",
    "VirtualStainCompositeLoss",
    "HEAuxiliaryHead",
    "rgb_to_he_maps",
    "build_decision_maps",
]


DiagnosisPreservingVirtualHENetwork = AttentionConditionedResUNet
AgentGuidedDiagnosisAwareResUNet = AttentionConditionedResUNet


def build_decision_maps(
    attention_map: torch.Tensor,
    highres_mask: Optional[torch.Tensor] = None,
    tissue_map: Optional[torch.Tensor] = None,
    lesion_confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Packs agent-side outputs into the 4-channel decision tensor expected by the model.

    Channel convention:
    0. continuous attention / importance
    1. high-resolution routing mask
    2. tissue mask
    3. lesion confidence
    """
    if attention_map.ndim != 4 or attention_map.shape[1] != 1:
        raise ValueError("attention_map must have shape [B, 1, H, W].")
    base = attention_map.float()
    highres = base if highres_mask is None else highres_mask.float()
    tissue = torch.ones_like(base) if tissue_map is None else tissue_map.float()
    confidence = base if lesion_confidence is None else lesion_confidence.float()
    return torch.cat([base, highres, tissue, confidence], dim=1).clamp(0.0, 1.0)
