import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from core import SimpleStainTransfer, load_rgb_image, save_image


@dataclass
class DiagnosisAwareLossConfig:
    nucleus_weight: float = 16.0
    membrane_weight: float = 12.0
    gland_weight: float = 10.0
    background_weight: float = 1.0
    alpha_l1: float = 1.0
    alpha_mse: float = 0.5
    alpha_ssim: float = 0.6
    alpha_dists_proxy: float = 0.8
    alpha_clinical: float = 1.2
    tile_size: int = 64


@dataclass
class RegionMasks:
    nucleus_mask: np.ndarray
    membrane_mask: np.ndarray
    gland_mask: np.ndarray
    background_mask: np.ndarray


@dataclass
class ClinicalFeatureSet:
    nucleus_density: float
    hematoxylin_intensity_mean: float
    eosin_intensity_mean: float
    membrane_response_mean: float
    gland_area_ratio: float


class DiagnosticRegionMaskExtractor:
    """
    Region prior extractor that can later be replaced by a pretrained pathology
    segmentation model. For now it provides a deterministic fallback that
    approximates nuclei, membranes and gland-like luminal structures.
    """

    def __init__(self) -> None:
        self.he_visualizer = SimpleStainTransfer()

    def extract(self, rgb: np.ndarray) -> RegionMasks:
        hematoxylin, eosin = self.he_visualizer.deconvolve_he(rgb)
        nucleus_signal = self.he_visualizer.normalize_channel(hematoxylin)
        eosin_signal = self.he_visualizer.normalize_channel(eosin)

        gray = np.mean(rgb.astype(np.float32), axis=2)
        tissue_mask = gray < 242.0

        nucleus_mask = (nucleus_signal > np.percentile(nucleus_signal[tissue_mask], 72.0)) & tissue_mask if np.any(tissue_mask) else np.zeros_like(gray, dtype=bool)
        membrane_mask = self._build_membrane_mask(rgb, tissue_mask)
        gland_mask = self._build_gland_mask(gray, eosin_signal, tissue_mask, nucleus_mask)
        background_mask = ~tissue_mask

        return RegionMasks(
            nucleus_mask=nucleus_mask,
            membrane_mask=membrane_mask,
            gland_mask=gland_mask,
            background_mask=background_mask,
        )

    @staticmethod
    def _build_membrane_mask(rgb: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        gray = np.mean(rgb.astype(np.float32), axis=2)
        edge = DiagnosticRegionMaskExtractor._local_edge(gray)
        edge_norm = DiagnosticRegionMaskExtractor._normalize(edge)
        red_bias = rgb[..., 0].astype(np.float32) - rgb[..., 1].astype(np.float32)
        red_norm = DiagnosticRegionMaskExtractor._normalize(red_bias)
        membrane = ((0.6 * edge_norm + 0.4 * red_norm) > 0.58) & tissue_mask
        return membrane

    @staticmethod
    def _build_gland_mask(
        gray: np.ndarray,
        eosin_signal: np.ndarray,
        tissue_mask: np.ndarray,
        nucleus_mask: np.ndarray,
    ) -> np.ndarray:
        bright_tissue = DiagnosticRegionMaskExtractor._normalize(gray)
        lumen_candidates = (bright_tissue > 0.68) & tissue_mask & (~nucleus_mask)
        eosin_candidates = eosin_signal > np.percentile(eosin_signal[tissue_mask], 55.0) if np.any(tissue_mask) else np.zeros_like(gray, dtype=bool)
        gland_mask = lumen_candidates & eosin_candidates
        gland_mask = DiagnosticRegionMaskExtractor._majority_filter(gland_mask, threshold=5)
        return gland_mask

    @staticmethod
    def _local_edge(gray: np.ndarray) -> np.ndarray:
        padded = np.pad(gray, 1, mode="edge")
        gx = padded[1:-1, 2:] - padded[1:-1, :-2]
        gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
        return np.sqrt(gx * gx + gy * gy)

    @staticmethod
    def _normalize(channel: np.ndarray) -> np.ndarray:
        low = np.percentile(channel, 1.0)
        high = np.percentile(channel, 99.0)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.zeros_like(channel, dtype=np.float32)
        return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _majority_filter(mask: np.ndarray, threshold: int = 5) -> np.ndarray:
        padded = np.pad(mask.astype(np.uint8), 1, mode="edge")
        total = np.zeros_like(mask, dtype=np.uint8)
        for dy in range(3):
            for dx in range(3):
                total += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return total >= threshold


class DynamicRegionWeightMap:
    def __init__(self, config: DiagnosisAwareLossConfig) -> None:
        self.config = config

    def build(self, masks: RegionMasks) -> np.ndarray:
        weight = np.full(masks.background_mask.shape, self.config.background_weight, dtype=np.float32)
        weight[masks.gland_mask] = np.maximum(weight[masks.gland_mask], self.config.gland_weight)
        weight[masks.membrane_mask] = np.maximum(weight[masks.membrane_mask], self.config.membrane_weight)
        weight[masks.nucleus_mask] = np.maximum(weight[masks.nucleus_mask], self.config.nucleus_weight)
        return weight


class ClinicalFeatureConsistency:
    def __init__(self) -> None:
        self.he_visualizer = SimpleStainTransfer()

    def extract(self, rgb: np.ndarray, masks: Optional[RegionMasks] = None) -> ClinicalFeatureSet:
        hematoxylin, eosin = self.he_visualizer.deconvolve_he(rgb)
        nucleus_signal = self.he_visualizer.normalize_channel(hematoxylin)
        eosin_signal = self.he_visualizer.normalize_channel(eosin)

        if masks is None:
            masks = DiagnosticRegionMaskExtractor().extract(rgb)

        tissue = ~(masks.background_mask)
        tissue_area = float(np.count_nonzero(tissue)) if np.any(tissue) else 1.0
        membrane_response = DiagnosticRegionMaskExtractor._local_edge(np.mean(rgb.astype(np.float32), axis=2))

        return ClinicalFeatureSet(
            nucleus_density=float(np.count_nonzero(masks.nucleus_mask) / tissue_area),
            hematoxylin_intensity_mean=float(nucleus_signal[tissue].mean()) if np.any(tissue) else 0.0,
            eosin_intensity_mean=float(eosin_signal[tissue].mean()) if np.any(tissue) else 0.0,
            membrane_response_mean=float(membrane_response[tissue].mean()) if np.any(tissue) else 0.0,
            gland_area_ratio=float(np.count_nonzero(masks.gland_mask) / tissue_area),
        )

    def loss(self, pred_rgb: np.ndarray, target_rgb: np.ndarray, pred_masks: RegionMasks, target_masks: RegionMasks) -> float:
        pred = self.extract(pred_rgb, pred_masks)
        target = self.extract(target_rgb, target_masks)
        diffs = [
            abs(pred.nucleus_density - target.nucleus_density),
            abs(pred.hematoxylin_intensity_mean - target.hematoxylin_intensity_mean),
            abs(pred.eosin_intensity_mean - target.eosin_intensity_mean),
            abs(pred.membrane_response_mean - target.membrane_response_mean) / 32.0,
            abs(pred.gland_area_ratio - target.gland_area_ratio),
        ]
        return float(np.mean(diffs))


class DiagnosisAwareWeightedLoss:
    def __init__(self, config: Optional[DiagnosisAwareLossConfig] = None) -> None:
        self.config = config or DiagnosisAwareLossConfig()
        self.mask_extractor = DiagnosticRegionMaskExtractor()
        self.weight_builder = DynamicRegionWeightMap(self.config)
        self.clinical_consistency = ClinicalFeatureConsistency()

    def forward(self, pred_rgb: np.ndarray, target_rgb: np.ndarray) -> Dict[str, float]:
        pred = pred_rgb.astype(np.float32)
        target = target_rgb.astype(np.float32)
        if pred.shape != target.shape:
            raise ValueError("pred_rgb and target_rgb must have identical shape.")

        target_masks = self.mask_extractor.extract(target_rgb)
        pred_masks = self.mask_extractor.extract(pred_rgb)
        return self.forward_with_masks(pred_rgb, target_rgb, pred_masks=pred_masks, target_masks=target_masks)

    def forward_with_masks(
        self,
        pred_rgb: np.ndarray,
        target_rgb: np.ndarray,
        pred_masks: Optional[RegionMasks] = None,
        target_masks: Optional[RegionMasks] = None,
    ) -> Dict[str, float]:
        pred = pred_rgb.astype(np.float32)
        target = target_rgb.astype(np.float32)
        if pred.shape != target.shape:
            raise ValueError("pred_rgb and target_rgb must have identical shape.")

        if target_masks is None:
            target_masks = self.mask_extractor.extract(target_rgb)
        if pred_masks is None:
            pred_masks = self.mask_extractor.extract(pred_rgb)
        weight_map = self.weight_builder.build(target_masks)

        weighted_l1 = self._weighted_l1(pred, target, weight_map)
        weighted_mse = self._weighted_mse(pred, target, weight_map)
        weighted_ssim = 1.0 - self._weighted_ssim(pred, target, weight_map)
        dists_proxy = self._dists_proxy_loss(pred, target)
        clinical = self.clinical_consistency.loss(pred_rgb, target_rgb, pred_masks, target_masks)

        total = (
            self.config.alpha_l1 * weighted_l1
            + self.config.alpha_mse * weighted_mse
            + self.config.alpha_ssim * weighted_ssim
            + self.config.alpha_dists_proxy * dists_proxy
            + self.config.alpha_clinical * clinical
        )
        return {
            "total_loss": float(total),
            "weighted_l1": float(weighted_l1),
            "weighted_mse": float(weighted_mse),
            "weighted_ssim_loss": float(weighted_ssim),
            "dists_proxy_loss": float(dists_proxy),
            "clinical_consistency_loss": float(clinical),
            "mean_region_weight": float(weight_map.mean()),
            "nucleus_weight_coverage": float(np.mean(target_masks.nucleus_mask.astype(np.float32))),
            "membrane_weight_coverage": float(np.mean(target_masks.membrane_mask.astype(np.float32))),
            "gland_weight_coverage": float(np.mean(target_masks.gland_mask.astype(np.float32))),
        }

    @staticmethod
    def merge_region_masks(base_masks: RegionMasks, prior_masks: RegionMasks) -> RegionMasks:
        return RegionMasks(
            nucleus_mask=np.logical_or(base_masks.nucleus_mask, prior_masks.nucleus_mask),
            membrane_mask=np.logical_or(base_masks.membrane_mask, prior_masks.membrane_mask),
            gland_mask=np.logical_or(base_masks.gland_mask, prior_masks.gland_mask),
            background_mask=np.logical_and(base_masks.background_mask, prior_masks.background_mask),
        )

    @staticmethod
    def _weighted_l1(pred: np.ndarray, target: np.ndarray, weight_map: np.ndarray) -> float:
        diff = np.abs(pred - target).mean(axis=2)
        return float(np.sum(diff * weight_map) / np.sum(weight_map))

    @staticmethod
    def _weighted_mse(pred: np.ndarray, target: np.ndarray, weight_map: np.ndarray) -> float:
        diff = np.square(pred - target).mean(axis=2)
        return float(np.sum(diff * weight_map) / np.sum(weight_map))

    def _weighted_ssim(self, pred: np.ndarray, target: np.ndarray, weight_map: np.ndarray) -> float:
        pred_gray = np.mean(pred, axis=2)
        target_gray = np.mean(target, axis=2)
        pred_blur = self._gaussian_blur(pred_gray, radius=1.2)
        target_blur = self._gaussian_blur(target_gray, radius=1.2)

        mu_x = pred_blur
        mu_y = target_blur
        sigma_x = self._gaussian_blur(pred_gray * pred_gray, radius=1.2) - mu_x * mu_x
        sigma_y = self._gaussian_blur(target_gray * target_gray, radius=1.2) - mu_y * mu_y
        sigma_xy = self._gaussian_blur(pred_gray * target_gray, radius=1.2) - mu_x * mu_y

        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-6)
        return float(np.sum(ssim_map * weight_map) / np.sum(weight_map))

    def _dists_proxy_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        scales = [1, 2, 4]
        values = []
        for scale in scales:
            pred_scaled = self._resize(pred, scale)
            target_scaled = self._resize(target, scale)
            pred_feat = self._feature_stack(pred_scaled)
            target_feat = self._feature_stack(target_scaled)
            values.append(float(np.mean(np.abs(pred_feat - target_feat))))
        return float(np.mean(values))

    def _feature_stack(self, rgb: np.ndarray) -> np.ndarray:
        gray = np.mean(rgb, axis=2)
        blur = self._gaussian_blur(gray, radius=1.5)
        edge = DiagnosticRegionMaskExtractor._local_edge(gray)
        texture = np.abs(gray - blur)
        stack = np.stack(
            [
                self._normalize(gray),
                self._normalize(blur),
                self._normalize(edge),
                self._normalize(texture),
            ],
            axis=-1,
        )
        return stack

    @staticmethod
    def _resize(rgb: np.ndarray, scale: int) -> np.ndarray:
        if scale == 1:
            return rgb
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        width, height = image.size
        new_size = (max(1, width // scale), max(1, height // scale))
        return np.array(image.resize(new_size, Image.Resampling.LANCZOS), dtype=np.float32)

    @staticmethod
    def _gaussian_blur(gray: np.ndarray, radius: float) -> np.ndarray:
        image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(blurred, dtype=np.float32)

    @staticmethod
    def _normalize(channel: np.ndarray) -> np.ndarray:
        low = np.percentile(channel, 1.0)
        high = np.percentile(channel, 99.0)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.zeros_like(channel, dtype=np.float32)
        return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)


def save_mask_visualizations(output_dir: str, masks: RegionMasks, weight_map: np.ndarray) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mapping = {
        "nucleus_mask.png": masks.nucleus_mask,
        "membrane_mask.png": masks.membrane_mask,
        "gland_mask.png": masks.gland_mask,
        "background_mask.png": masks.background_mask,
    }
    for name, mask in mapping.items():
        rgb = np.stack([mask.astype(np.uint8) * 255] * 3, axis=-1)
        save_image(rgb, os.path.join(output_dir, name))

    weight_norm = DiagnosisAwareWeightedLoss._normalize(weight_map)
    weight_rgb = np.stack(
        [
            np.clip(255.0 * weight_norm, 0, 255),
            np.clip(255.0 * (1.0 - weight_norm), 0, 255),
            np.full_like(weight_norm, 80.0),
        ],
        axis=-1,
    ).astype(np.uint8)
    save_image(weight_rgb, os.path.join(output_dir, "diagnostic_weight_map.png"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnosis-aware weighted loss demo for virtual staining.")
    parser.add_argument("--pred", type=str, required=True, help="Predicted virtual stain image.")
    parser.add_argument("--target", type=str, required=True, help="Target pathology image.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for loss report and masks.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pred_rgb = load_rgb_image(args.pred)
    target_rgb = load_rgb_image(args.target)
    if pred_rgb.shape != target_rgb.shape:
        target_image = Image.fromarray(target_rgb).resize((pred_rgb.shape[1], pred_rgb.shape[0]), Image.Resampling.LANCZOS)
        target_rgb = np.array(target_image)

    loss_fn = DiagnosisAwareWeightedLoss()
    target_masks = loss_fn.mask_extractor.extract(target_rgb)
    weight_map = loss_fn.weight_builder.build(target_masks)
    report = loss_fn.forward(pred_rgb=pred_rgb, target_rgb=target_rgb)

    os.makedirs(args.output_dir, exist_ok=True)
    save_mask_visualizations(args.output_dir, target_masks, weight_map)
    report_path = os.path.join(args.output_dir, "diagnosis_aware_loss_report.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"pred={args.pred}")
    print(f"target={args.target}")
    print(f"total_loss={report['total_loss']:.6f}")
    print(f"clinical_consistency_loss={report['clinical_consistency_loss']:.6f}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
