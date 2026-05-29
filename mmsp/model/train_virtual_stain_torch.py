from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .virtual_stain_resunet_torch import (
    AttentionConditionedResUNet,
    VirtualStainCompositeLoss,
    VirtualStainResUNetConfig,
    build_decision_maps,
)


@dataclass
class TrainStepConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0


class VirtualStainTrainer(nn.Module):
    """
    Minimal training scaffold for:
    unstained patch + agent attention -> diagnosis-preserving virtual HE.

    Expected batch keys:
    - input_rgb: [B, 3, H, W]
    - target_rgb: [B, 3, H, W]
    - attention_map: [B, 1, H, W]
    Optional:
    - highres_mask: [B, 1, H, W]
    - tissue_map: [B, 1, H, W]
    - lesion_confidence: [B, 1, H, W]
    """

    def __init__(
        self,
        model_config: VirtualStainResUNetConfig | None = None,
        train_config: TrainStepConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config or VirtualStainResUNetConfig()
        self.train_config = train_config or TrainStepConfig()
        self.model = AttentionConditionedResUNet(self.model_config)
        self.loss_fn = VirtualStainCompositeLoss(self.model_config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        decision_maps = build_decision_maps(
            attention_map=batch["attention_map"],
            highres_mask=batch.get("highres_mask"),
            tissue_map=batch.get("tissue_map"),
            lesion_confidence=batch.get("lesion_confidence"),
        )
        outputs = self.model(batch["input_rgb"], decision_maps)
        loss_dict = self.loss_fn(outputs, batch["target_rgb"], decision_maps)
        loss_dict["lesion_region_ssim"] = self._lesion_region_ssim(outputs["rgb"], batch["target_rgb"], decision_maps).detach()
        return outputs, loss_dict

    def configure_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss_dict = self.forward(batch)
        loss_dict["total_loss"].backward()
        if self.train_config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.train_config.grad_clip_norm)
        optimizer.step()
        return {key: float(value.detach().cpu()) for key, value in loss_dict.items()}

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.eval()
        _, loss_dict = self.forward(batch)
        return {key: float(value.detach().cpu()) for key, value in loss_dict.items()}

    @staticmethod
    def _lesion_region_ssim(pred: torch.Tensor, target: torch.Tensor, decision_maps: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        focus = torch.maximum(decision_maps[:, 0:1], 0.85 * decision_maps[:, 1:2]).clamp(0.0, 1.0)
        gray_pred = pred.mean(dim=1, keepdim=True)
        gray_target = target.mean(dim=1, keepdim=True)

        def blur(x: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        mu_x = blur(gray_pred)
        mu_y = blur(gray_target)
        sigma_x = blur(gray_pred * gray_pred) - mu_x * mu_x
        sigma_y = blur(gray_target * gray_target) - mu_y * mu_y
        sigma_xy = blur(gray_pred * gray_target) - mu_x * mu_y
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-6
        )
        weight = 0.25 + 0.75 * focus
        return (ssim_map * weight).sum() / weight.sum().clamp(min=1e-6)


__all__ = [
    "TrainStepConfig",
    "VirtualStainTrainer",
]
