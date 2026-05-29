from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AgentStainModelConfig:
    agent_channels: int = 4
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)
    attention_dropout: float = 0.0
    use_residual_gate: bool = True


class DecisionSignalEncoder(nn.Module):
    """
    Encodes agent decision maps into multi-scale priors.
    Input channels can include:
    - importance map
    - high-resolution routing mask
    - tissue / ROI map
    - optional lesion confidence map
    """

    def __init__(self, config: AgentStainModelConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = config.agent_channels
        for out_channels in config.encoder_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.GELU(),
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(layers)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, decision_maps: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        x = decision_maps
        for index, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if index < len(self.blocks) - 1:
                x = self.pool(x)
        return features


class CrossModalFeatureAlignModule(nn.Module):
    """
    Injects agent decision features into stain-transfer encoder and skip features.
    This is the core cross-modal alignment module for the paper's innovation point.
    """

    def __init__(self, feature_channels: int, attention_dropout: float = 0.0, use_residual_gate: bool = True) -> None:
        super().__init__()
        self.use_residual_gate = use_residual_gate
        self.agent_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.encoder_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(attention_dropout),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_norm = nn.InstanceNorm2d(feature_channels)

    def forward(self, encoder_feat: torch.Tensor, skip_feat: torch.Tensor, agent_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if agent_feat.shape[-2:] != encoder_feat.shape[-2:]:
            agent_feat = F.interpolate(agent_feat, size=encoder_feat.shape[-2:], mode="bilinear", align_corners=False)
        if skip_feat.shape[-2:] != encoder_feat.shape[-2:]:
            skip_feat = F.interpolate(skip_feat, size=encoder_feat.shape[-2:], mode="bilinear", align_corners=False)

        agent_embed = self.agent_proj(agent_feat)
        encoder_embed = self.encoder_proj(encoder_feat)
        skip_embed = self.skip_proj(skip_feat)

        attention = self.attention_conv(torch.cat([encoder_embed, skip_embed, agent_embed], dim=1))
        conditioned_encoder = encoder_feat + attention * agent_embed if self.use_residual_gate else attention * agent_embed
        conditioned_skip = skip_feat + attention * agent_embed if self.use_residual_gate else attention * agent_embed
        conditioned_encoder = self.out_norm(conditioned_encoder)
        conditioned_skip = self.out_norm(conditioned_skip)
        return conditioned_encoder, conditioned_skip


__all__ = [
    "AgentStainModelConfig",
    "DecisionSignalEncoder",
    "CrossModalFeatureAlignModule",
]
