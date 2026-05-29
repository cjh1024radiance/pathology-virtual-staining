from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiagnosisAwareTorchConfig:
    nucleus_weight: float = 16.0
    membrane_weight: float = 12.0
    gland_weight: float = 10.0
    background_weight: float = 1.0
    alpha_l1: float = 1.0
    alpha_clinical: float = 0.45
    eps: float = 1e-6


class DiagnosticRegionMaskExtractorTorch(nn.Module):
    """
    Torch version of diagnostic region prior extraction.

    Later this module can be replaced by a pretrained pathology segmentation
    network that outputs nuclei / membrane / gland masks. For now it provides a
    differentiable heuristic fallback from RGB tensors.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._validate(rgb)
        gray = rgb.mean(dim=1, keepdim=True)
        tissue_mask = (gray < 0.95).float()

        hematoxylin, eosin = self._he_deconvolution(rgb)
        nucleus_signal = self._normalize_map(hematoxylin)
        eosin_signal = self._normalize_map(eosin)

        nucleus_thr = self._masked_quantile(nucleus_signal, tissue_mask, 0.72)
        nucleus_mask = ((nucleus_signal >= nucleus_thr) * tissue_mask).clamp(0.0, 1.0)

        membrane_mask = self._build_membrane_mask(rgb, tissue_mask)
        gland_mask = self._build_gland_mask(gray, eosin_signal, tissue_mask, nucleus_mask)
        background_mask = 1.0 - tissue_mask

        return {
            "nucleus_mask": nucleus_mask,
            "membrane_mask": membrane_mask,
            "gland_mask": gland_mask,
            "background_mask": background_mask,
            "tissue_mask": tissue_mask,
            "hematoxylin": hematoxylin,
            "eosin": eosin,
        }

    def _he_deconvolution(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = rgb.clamp(self.eps, 1.0)
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
        matrix = matrix / (torch.linalg.norm(matrix, dim=0, keepdim=True) + self.eps)
        inv_t = torch.inverse(matrix.t())

        batch, channels, height, width = od.shape
        flat = od.permute(0, 2, 3, 1).reshape(-1, 3)
        concentrations = flat @ inv_t
        concentrations = concentrations.clamp(min=0.0)
        concentrations = concentrations.reshape(batch, height, width, 3).permute(0, 3, 1, 2)
        hematoxylin = concentrations[:, 0:1]
        eosin = concentrations[:, 1:2]
        return hematoxylin, eosin

    def _build_membrane_mask(self, rgb: torch.Tensor, tissue_mask: torch.Tensor) -> torch.Tensor:
        gray = rgb.mean(dim=1, keepdim=True)
        edge = self._local_edge(gray)
        edge_norm = self._normalize_map(edge)
        red_bias = rgb[:, 0:1] - rgb[:, 1:2]
        red_norm = self._normalize_map(red_bias)
        membrane_score = 0.6 * edge_norm + 0.4 * red_norm
        return ((membrane_score > 0.58).float() * tissue_mask).clamp(0.0, 1.0)

    def _build_gland_mask(
        self,
        gray: torch.Tensor,
        eosin_signal: torch.Tensor,
        tissue_mask: torch.Tensor,
        nucleus_mask: torch.Tensor,
    ) -> torch.Tensor:
        bright_tissue = self._normalize_map(gray)
        eosin_thr = self._masked_quantile(eosin_signal, tissue_mask, 0.55)
        lumen_candidates = ((bright_tissue > 0.68).float() * tissue_mask * (1.0 - nucleus_mask)).clamp(0.0, 1.0)
        eosin_candidates = (eosin_signal >= eosin_thr).float()
        gland = lumen_candidates * eosin_candidates
        return self._majority_filter(gland)

    def _local_edge(self, gray: torch.Tensor) -> torch.Tensor:
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=gray.dtype,
            device=gray.device,
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=gray.dtype,
            device=gray.device,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kernel_x, padding=1)
        gy = F.conv2d(gray, kernel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def _normalize_map(self, tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.shape[0]
        flat = tensor.reshape(batch, -1)
        low = torch.quantile(flat, 0.01, dim=1, keepdim=True)
        high = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        denom = (high - low).clamp(min=self.eps)
        normalized = (flat - low) / denom
        return normalized.clamp(0.0, 1.0).reshape_as(tensor)

    def _masked_quantile(self, tensor: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        outputs = []
        batch = tensor.shape[0]
        for idx in range(batch):
            values = tensor[idx][mask[idx] > 0.5]
            if values.numel() == 0:
                outputs.append(torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device))
            else:
                outputs.append(torch.quantile(values, q))
        return torch.stack(outputs).view(batch, 1, 1, 1)

    def _majority_filter(self, mask: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
        return (pooled >= (5.0 / 9.0)).float()

    @staticmethod
    def _validate(rgb: torch.Tensor) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Expected tensor shape [B, 3, H, W].")


class DynamicRegionWeightMapTorch(nn.Module):
    def __init__(self, config: DiagnosisAwareTorchConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        weight = torch.full_like(masks["background_mask"], self.config.background_weight)
        weight = torch.maximum(weight, masks["gland_mask"] * self.config.gland_weight + (1.0 - masks["gland_mask"]) * weight)
        weight = torch.maximum(weight, masks["membrane_mask"] * self.config.membrane_weight + (1.0 - masks["membrane_mask"]) * weight)
        weight = torch.maximum(weight, masks["nucleus_mask"] * self.config.nucleus_weight + (1.0 - masks["nucleus_mask"]) * weight)
        return weight


class ClinicalFeatureConsistencyTorch(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.extractor = DiagnosticRegionMaskExtractorTorch(eps=eps)
        self.eps = eps

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        pred_masks: Optional[Dict[str, torch.Tensor]] = None,
        target_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if pred_masks is None:
            with torch.no_grad():
                pred_masks = self.extractor(pred_rgb)
        if target_masks is None:
            with torch.no_grad():
                target_masks = self.extractor(target_rgb)

        pred_feat = self._extract_features(pred_rgb, pred_masks, grad_safe=True)
        target_feat = self._extract_features(target_rgb, target_masks, grad_safe=False)
        diffs = [torch.abs(pred_feat[key] - target_feat[key]).mean() for key in pred_feat]
        return torch.stack(diffs).mean()

    def _extract_features(self, rgb: torch.Tensor, masks: Dict[str, torch.Tensor], grad_safe: bool) -> Dict[str, torch.Tensor]:
        hematoxylin, eosin = self.extractor._he_deconvolution(rgb)
        tissue = masks["tissue_mask"]
        tissue_area = tissue.sum(dim=(1, 2, 3)).clamp(min=self.eps)
        membrane_response = self.extractor._local_edge(rgb.mean(dim=1, keepdim=True))

        return {
            "nucleus_density": masks["nucleus_mask"].sum(dim=(1, 2, 3)) / tissue_area,
            "hematoxylin_intensity_mean": (self._normalize_map_grad_safe(hematoxylin) * tissue).sum(dim=(1, 2, 3)) / tissue_area,
            "eosin_intensity_mean": (self._normalize_map_grad_safe(eosin) * tissue).sum(dim=(1, 2, 3)) / tissue_area,
            "membrane_response_mean": (membrane_response * tissue).sum(dim=(1, 2, 3)) / tissue_area,
            "gland_area_ratio": masks["gland_mask"].sum(dim=(1, 2, 3)) / tissue_area,
        }

    def _normalize_map_grad_safe(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.flatten(start_dim=2)
        low = flat.amin(dim=2, keepdim=True)
        high = flat.amax(dim=2, keepdim=True)
        norm = (flat - low) / (high - low).clamp(min=self.eps)
        return norm.clamp(0.0, 1.0).reshape_as(tensor)


class DiagnosisAwareWeightedLossTorch(nn.Module):
    def __init__(self, config: Optional[DiagnosisAwareTorchConfig] = None) -> None:
        super().__init__()
        self.config = config or DiagnosisAwareTorchConfig()
        self.mask_extractor = DiagnosticRegionMaskExtractorTorch(eps=self.config.eps)
        self.weight_builder = DynamicRegionWeightMapTorch(self.config)
        self.clinical_consistency = ClinicalFeatureConsistencyTorch(eps=self.config.eps)

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_rgb = self._prepare(pred_rgb)
        target_rgb = self._prepare(target_rgb)
        if pred_rgb.shape != target_rgb.shape:
            raise ValueError("pred_rgb and target_rgb must have identical shape.")

        with torch.no_grad():
            target_masks = self.mask_extractor(target_rgb)
            pred_masks = self.mask_extractor(pred_rgb)
        weight_map = self.weight_builder(target_masks)

        weighted_l1 = self._weighted_l1(pred_rgb, target_rgb, weight_map)
        clinical_consistency_loss = self.clinical_consistency(pred_rgb, target_rgb, pred_masks, target_masks)

        total_loss = (
            self.config.alpha_l1 * weighted_l1
            + self.config.alpha_clinical * clinical_consistency_loss
        )

        return {
            "total_loss": total_loss,
            "weighted_l1": weighted_l1,
            "clinical_consistency_loss": clinical_consistency_loss,
        }

    def _weighted_l1(self, pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target).mean(dim=1, keepdim=True)
        return (diff * weight_map).sum() / weight_map.sum().clamp(min=self.config.eps)

    @staticmethod
    def _prepare(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("Expected [B, 3, H, W] tensor.")
        if tensor.max() > 1.0 or tensor.min() < 0.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0).float()


__all__ = [
    "DiagnosisAwareTorchConfig",
    "DiagnosticRegionMaskExtractorTorch",
    "DynamicRegionWeightMapTorch",
    "ClinicalFeatureConsistencyTorch",
    "DiagnosisAwareWeightedLossTorch",
]
