import argparse
import json
import os
import sys
import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_client import ApiConfig, OpenAIVisionClient
from core import SlideReader, SimpleStainTransfer, UnpairedVirtualHEBaseline, load_medical_config, load_rgb_image, save_image
from diagnosis_aware_loss import DiagnosisAwareWeightedLoss, RegionMasks, save_mask_visualizations
from model.virtual_stain_resunet_torch import AttentionConditionedResUNet, VirtualStainResUNetConfig, build_decision_maps
from test.pathology_report import write_pathology_report


INFERENCE_ABLATION_DESCRIPTIONS = {
    "full": "Full",
    "wo_morphology_aware_prompt": "w/o morphology-aware prompt",
    "wo_local_roi_api_review": "w/o local ROI API review",
    "wo_local_refinement": "w/o local refinement",
    "wo_routing_constraint": "w/o routing constraint",
}


def normalize_endpoint(raw: str) -> str:
    if not raw:
        return ""
    url = raw.rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def resize_map_like(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape_hw
    if arr.shape[:2] == (target_h, target_w):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    image = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    resized = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32) / 255.0


def resolve_workspace_path(raw_path: str) -> str:
    if not raw_path:
        return ""
    normalized = raw_path.strip()
    if len(normalized) >= 2 and normalized[1] == ":":
        candidate = Path(normalized)
        if candidate.exists():
            return str(candidate)
        return ""
    candidate = Path(normalized)
    if candidate.is_absolute():
        return str(candidate)
    return str((WORKSPACE_ROOT / candidate).resolve())


@dataclass
class PathologyConfig:
    preview_max_side: int = 1536
    grid_size: int = 5
    top_k: int = 5
    navigation_rounds: int = 6
    score_threshold: float = 0.28
    uncertainty_low: float = 0.18
    uncertainty_high: float = 0.85
    highres_patch_side: int = 896
    stain_mode: str = "virtual_he"
    fusion_strength: float = 1.0
    api_enabled: bool = True
    api_endpoint: str = ""
    api_key: str = ""
    api_model: str = "gpt-4o"
    api_timeout: int = 180
    api_prompt_hint: str = ""
    use_morphology_aware_prompt: bool = True
    use_local_roi_api_review: bool = True
    use_local_refinement: bool = True
    use_routing_constraint: bool = True
    model_path: str = ""
    model_device: str = "auto"


@dataclass
class NavigationROI:
    tile_id: int
    rel_coordinate: Tuple[float, float, float, float]
    stage1_score: float
    stage2_score: float
    final_score: float
    lesion_confidence: float
    selected_highres: bool
    uncertainty_flag: bool
    explanation: str
    suspicion_type: str = "unspecified"
    suggested_magnification: str = "10x"
    local_boxes: List[Dict[str, float]] = field(default_factory=list)
    visited: bool = False
    visited_round: int = 0
    stage_trace: List[str] = field(default_factory=list)


@dataclass
class StageArtifact:
    name: str
    path: str
    description: str


@dataclass
class PathologyResult:
    input_path: str
    target_path: Optional[str]
    overview_risk_heatmap_path: str
    roi_overlay_path: str
    routing_hard_mask_path: str
    routing_soft_attention_path: str
    lowres_stain_path: str
    fused_stain_path: str
    fused_roi_overlay_path: str
    diagnostic_priors_overlay_path: Optional[str]
    artifacts: List[StageArtifact] = field(default_factory=list)
    rois: List[NavigationROI] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnosis_loss: Dict[str, float] = field(default_factory=dict)
    api_usage: Dict[str, object] = field(default_factory=dict)
    llm_overview: Dict[str, object] = field(default_factory=dict)
    report_path: Optional[str] = None


class PathologyNavigationAgent:
    """
    Innovation 1:
    Evidence-seeking navigation agent.
    Stage 1: low-magnification overview screening.
    Stage 2: local review for high-score candidates.
    Stage 3: uncertain regions trigger an extra external-model review.
    """

    def __init__(self, config: PathologyConfig, client: OpenAIVisionClient) -> None:
        self.config = config
        self.client = client
        self.last_overview_context: Dict[str, object] = {}

    def analyze(self, input_path: str, preview_rgb: np.ndarray, output_dir: str) -> Tuple[List[NavigationROI], np.ndarray, str]:
        rois = self._stage1_overview(preview_rgb)
        reader = SlideReader(input_path, max_side=self.config.highres_patch_side)
        visited_ids: set[int] = set()
        max_rounds = min(self.config.navigation_rounds, len(rois))
        for nav_round in range(1, max_rounds + 1):
            candidate = self._select_next_candidate(rois, visited_ids)
            if candidate is None:
                break
            visited_ids.add(candidate.tile_id)
            candidate.visited = True
            candidate.visited_round = nav_round
            candidate.stage_trace.append(f"navigation_round={nav_round}")

            if candidate.stage1_score < max(self.config.score_threshold * 0.75, 0.20):
                candidate.stage_trace.append("stage2_skipped_low_stage1_score")
                candidate.stage2_score = candidate.stage1_score
                candidate.final_score = candidate.stage1_score
                self._update_candidate_scores(rois, candidate, visited_ids)
                continue

            patch_rgb = reader.load_region(candidate.rel_coordinate)
            local_score, local_explanation, local_type = self._stage2_local_review(patch_rgb)
            candidate.stage2_score = local_score
            candidate.lesion_confidence = max(candidate.lesion_confidence, local_score)
            candidate.explanation = local_explanation
            candidate.suspicion_type = local_type or candidate.suspicion_type
            candidate.stage_trace.append(f"stage2_local_review={local_score:.3f}")
            candidate.uncertainty_flag = self.config.uncertainty_low <= local_score <= self.config.uncertainty_high

            should_review = (
                self.config.api_enabled
                and self.client.enabled()
                and self.config.use_local_roi_api_review
                and (candidate.uncertainty_flag or candidate.stage2_score >= self.config.score_threshold)
            )
            if should_review:
                external_score, metadata = self._stage3_external_review(patch_rgb, candidate)
                if external_score is not None:
                    # increase external review weight to emphasize API/local review
                    candidate.final_score = 0.20 * candidate.stage1_score + 0.20 * candidate.stage2_score + 0.60 * external_score
                    candidate.lesion_confidence = max(candidate.lesion_confidence, external_score)
                    candidate.explanation = str(metadata.get("explanation", candidate.explanation))
                    candidate.suspicion_type = str(metadata.get("suspicion_type", candidate.suspicion_type))
                    candidate.suggested_magnification = str(metadata.get("next_magnification", candidate.suggested_magnification))
                    candidate.local_boxes = list(metadata.get("boxes", [])) if isinstance(metadata.get("boxes", []), list) else []
                    candidate.stage_trace.append(f"stage3_external_review={external_score:.3f}")
                    candidate.stage_trace.append("api_review_used=true")
                else:
                    candidate.final_score = 0.40 * candidate.stage1_score + 0.60 * candidate.stage2_score
                    candidate.stage_trace.append("stage3_external_review_failed")
                    candidate.stage_trace.append("api_review_used=true")
            else:
                candidate.final_score = 0.40 * candidate.stage1_score + 0.60 * candidate.stage2_score
                if candidate.uncertainty_flag:
                    candidate.stage_trace.append("stage3_external_review_unavailable")
                else:
                    candidate.stage_trace.append("stage3_not_needed")

            self._update_candidate_scores(rois, candidate, visited_ids)

        for roi in rois:
            roi.selected_highres = False
        visited_ranked = sorted([roi for roi in rois if roi.visited], key=lambda item: item.final_score, reverse=True)
        if visited_ranked:
            top_scores = [float(roi.final_score) for roi in visited_ranked[: max(3, min(len(visited_ranked), self.config.top_k + 1))]]
            score_anchor = float(np.mean(top_scores)) if top_scores else float(visited_ranked[0].final_score)
            dynamic_threshold = max(
                0.20,
                min(
                    self.config.score_threshold,
                    0.62 * float(visited_ranked[0].final_score) + 0.20 * score_anchor,
                ),
            )
            selected_rois = [roi for roi in visited_ranked if roi.final_score >= dynamic_threshold][: self.config.top_k]
            min_keep = min(2, self.config.top_k, len(visited_ranked))
            if len(selected_rois) < min_keep:
                selected_rois = visited_ranked[:min_keep]
            for roi in selected_rois:
                roi.selected_highres = True
        rois.sort(key=lambda item: item.tile_id)
        overview_heatmap = self._build_risk_heatmap(preview_rgb, rois)
        heatmap_path = os.path.join(output_dir, "overview_risk_heatmap.png")
        save_image(overview_heatmap, heatmap_path)
        return rois, overview_heatmap, heatmap_path

    @staticmethod
    def _select_next_candidate(rois: List[NavigationROI], visited_ids: set[int]) -> Optional[NavigationROI]:
        remaining = [roi for roi in rois if roi.tile_id not in visited_ids]
        if not remaining:
            return None
        return max(remaining, key=lambda item: item.final_score)

    def _stage1_overview(self, preview_rgb: np.ndarray) -> List[NavigationROI]:
        h, w = preview_rgb.shape[:2]
        tile_h = max(1, h // self.config.grid_size)
        tile_w = max(1, w // self.config.grid_size)
        rois: List[NavigationROI] = []
        tile_id = 1
        for row in range(self.config.grid_size):
            for col in range(self.config.grid_size):
                y1 = row * tile_h
                x1 = col * tile_w
                y2 = h if row == self.config.grid_size - 1 else min(h, (row + 1) * tile_h)
                x2 = w if col == self.config.grid_size - 1 else min(w, (col + 1) * tile_w)
                patch = preview_rgb[y1:y2, x1:x2]
                gray = patch.astype(np.float32).mean(axis=2)
                tissue = gray < 240.0
                tissue_ratio = float(tissue.mean())
                if np.any(tissue):
                    darkness = float(np.clip((205.0 - gray[tissue].mean()) / 110.0, 0.0, 1.0))
                    contrast = float(np.clip(gray[tissue].std() / 55.0, 0.0, 1.0))
                    blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
                    detail = float(np.clip(np.abs(gray - blur)[tissue].mean() / 28.0, 0.0, 1.0))
                else:
                    darkness = 0.0
                    contrast = 0.0
                    detail = 0.0
                focus_map = self._breast_lesion_focus_map(patch)
                focus_mean = float(focus_map.mean())
                focus_peak = float(focus_map.max())
                stage1 = 0.22 * darkness + 0.20 * contrast + 0.18 * detail + 0.15 * tissue_ratio + 0.15 * focus_mean + 0.10 * focus_peak
                rois.append(
                    NavigationROI(
                        tile_id=tile_id,
                        rel_coordinate=(x1 / w, y1 / h, x2 / w, y2 / h),
                        stage1_score=stage1,
                        stage2_score=stage1,
                        final_score=stage1,
                        lesion_confidence=stage1,
                        selected_highres=False,
                        uncertainty_flag=False,
                        explanation="Overview heuristic screening",
                        suspicion_type="overview suspicious tissue",
                        suggested_magnification="10x",
                        stage_trace=[f"stage1_overview={stage1:.3f}"],
                    )
                )
                tile_id += 1

        if self.config.api_enabled and self.client.enabled():
            annotated = self._annotated_grid(preview_rgb, rois)
            response = self.client.score_tiles(annotated, [roi.tile_id for roi in rois], self.config.top_k)
            if response:
                self.last_overview_context = {
                    "overall_assessment": response.get("overall_assessment", ""),
                    "focus_recommendation": response.get("focus_recommendation", ""),
                    "summary": response.get("summary", ""),
                    "suspicious_regions": response.get("suspicious_regions", []),
                }
                ranking = {
                    int(item["tile_id"]): item
                    for item in response.get("ranked_tiles", [])
                    if isinstance(item, dict) and "tile_id" in item
                }
                for roi in rois:
                    item = ranking.get(roi.tile_id)
                    if item is None:
                        continue
                    api_score = float(np.clip(float(item.get("importance", roi.stage1_score)), 0.0, 1.0))
                    # Increase API influence to amplify score dynamic range
                    roi.stage1_score = 0.30 * roi.stage1_score + 0.70 * api_score
                    roi.stage2_score = roi.stage1_score
                    roi.final_score = roi.stage1_score
                    roi.lesion_confidence = float(np.clip(float(item.get("lesion_confidence", roi.stage1_score)), 0.0, 1.0))
                    roi.suspicion_type = str(item.get("suspicion_type", roi.suspicion_type))
                    roi.suggested_magnification = str(item.get("next_magnification", roi.suggested_magnification))
                    roi.explanation = str(item.get("reason", item.get("explanation", roi.explanation)))
                    roi.stage_trace.append(f"stage1_api_ranking={api_score:.3f}")
        return rois

    def _stage2_local_review(self, patch_rgb: np.ndarray) -> Tuple[float, str, str]:
        gray = patch_rgb.astype(np.float32).mean(axis=2)
        tissue = gray < 242.0
        if not np.any(tissue):
            # Raise baseline for low-tissue tiles so they are not all treated as empty
            return 0.12, "Low tissue content (baseline boosted)", "low tissue"
        blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32)
        detail = np.abs(gray - blur)
        detail_score = float(np.clip(detail[tissue].mean() / 30.0, 0.0, 1.0))
        dense_score = float(np.clip((220.0 - gray[tissue].mean()) / 110.0, 0.0, 1.0))
        edge_score = float(np.clip(gray[tissue].std() / 60.0, 0.0, 1.0))
        score = 0.40 * dense_score + 0.35 * detail_score + 0.25 * edge_score
        if dense_score >= max(detail_score, edge_score):
            suspicion_type = "hypercellular / nuclear crowding"
        elif detail_score >= edge_score:
            suspicion_type = "architectural distortion"
        else:
            suspicion_type = "high contrast atypical region"
        explanation = f"{suspicion_type}; density={dense_score:.2f}, detail={detail_score:.2f}, contrast={edge_score:.2f}"
        return score, explanation, suspicion_type

    def _stage3_external_review(self, patch_rgb: np.ndarray, roi: NavigationROI) -> Tuple[Optional[float], Dict[str, str]]:
        response = self.client.localize_roi(patch_rgb)
        if not response:
            return None, {"explanation": roi.explanation, "suspicion_type": roi.suspicion_type, "next_magnification": roi.suggested_magnification}
        boxes = response.get("boxes", [])
        if not isinstance(boxes, list) or not boxes:
            fallback_boxes = self._infer_local_focus_boxes(patch_rgb, roi.stage2_score)
            summary = str(response.get("summary", roi.explanation))
            return roi.stage2_score, {
                "explanation": summary,
                "suspicion_type": str(response.get("suspicion_type", roi.suspicion_type)),
                "next_magnification": str(response.get("next_magnification", roi.suggested_magnification)),
                "boxes": fallback_boxes,
            }
        refined_boxes = self._refine_local_boxes(patch_rgb, boxes)
        if refined_boxes:
            boxes = self._deduplicate_boxes(refined_boxes)
        confs = [float(np.clip(float(box.get("confidence", roi.stage2_score)), 0.0, 1.0)) for box in boxes if isinstance(box, dict)]
        if not confs:
            fallback_boxes = self._infer_local_focus_boxes(patch_rgb, roi.stage2_score)
            return roi.stage2_score, {
                "explanation": str(response.get("summary", roi.explanation)),
                "suspicion_type": str(response.get("suspicion_type", roi.suspicion_type)),
                "next_magnification": str(response.get("next_magnification", roi.suggested_magnification)),
                "boxes": fallback_boxes,
            }
        return float(sum(confs) / len(confs)), {
            "explanation": str(response.get("reason", response.get("summary", roi.explanation))),
            "suspicion_type": str(response.get("suspicion_type", roi.suspicion_type)),
            "next_magnification": str(response.get("next_magnification", roi.suggested_magnification)),
            "boxes": [box for box in boxes if isinstance(box, dict)],
        }

    @staticmethod
    def _refine_local_boxes(patch_rgb: np.ndarray, boxes: List[Dict[str, object]]) -> List[Dict[str, float]]:
        h, w = patch_rgb.shape[:2]
        gray = patch_rgb.astype(np.float32).mean(axis=2)
        tissue = (gray < 248.0).astype(np.float32)
        dark = np.clip((232.0 - gray) / 92.0, 0.0, 1.0)
        fine = np.abs(gray - np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32))
        coarse = np.abs(gray - np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=6)), dtype=np.float32))
        focus = PathologyNavigationAgent._breast_lesion_focus_map(patch_rgb)
        refined: List[Dict[str, float]] = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            try:
                x1 = int(float(box.get("x1", 0.0)) * w)
                y1 = int(float(box.get("y1", 0.0)) * h)
                x2 = int(float(box.get("x2", 1.0)) * w)
                y2 = int(float(box.get("y2", 1.0)) * h)
                confidence = float(np.clip(float(box.get("confidence", 0.7)), 0.0, 1.0))
            except Exception:
                continue
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            local = focus[y1:y2, x1:x2]
            if local.size == 0:
                continue
            threshold = np.percentile(local, 48.0)
            ys, xs = np.where(local >= threshold)
            if ys.size < 8 or xs.size < 8:
                rx1, ry1, rx2, ry2 = PathologyNavigationAgent._expand_box_with_context(x1, y1, x2, y2, w, h, context_ratio=0.12)
                refined.append(
                    {
                        "x1": float(rx1 / w),
                        "y1": float(ry1 / h),
                        "x2": float(rx2 / w),
                        "y2": float(ry2 / h),
                        "confidence": confidence,
                        "label": str(box.get("label", "suspicious lesion")),
                    }
                )
                continue
            rx1 = x1 + int(xs.min())
            ry1 = y1 + int(ys.min())
            rx2 = x1 + int(xs.max()) + 1
            ry2 = y1 + int(ys.max()) + 1
            rx1, ry1, rx2, ry2 = PathologyNavigationAgent._expand_box_with_context(
                rx1,
                ry1,
                rx2,
                ry2,
                w,
                h,
                context_ratio=0.16,
                min_ratio=0.18,
            )
            refined.append(
                {
                    "x1": float(rx1 / w),
                    "y1": float(ry1 / h),
                    "x2": float(rx2 / w),
                    "y2": float(ry2 / h),
                    "confidence": confidence,
                    "label": str(box.get("label", "suspicious lesion")),
                }
            )
        return refined

    @staticmethod
    def _infer_local_focus_boxes(patch_rgb: np.ndarray, confidence: float, max_boxes: int = 2) -> List[Dict[str, float]]:
        focus = PathologyNavigationAgent._breast_lesion_focus_map(patch_rgb)
        h, w = focus.shape
        if h < 8 or w < 8 or focus.max() <= 0:
            return []
        threshold = max(0.30, float(np.percentile(focus[focus > 0], 68.0))) if np.any(focus > 0) else 1.0
        mask = (focus >= threshold).astype(np.uint8)
        boxes: List[Dict[str, float]] = []
        for _ in range(max_boxes):
            ys, xs = np.where(mask > 0)
            if ys.size < max(24, int(0.004 * h * w)):
                break
            scores = focus[ys, xs]
            order = int(np.argmax(scores))
            seed_y = int(ys[order])
            seed_x = int(xs[order])
            x1 = max(0, seed_x - int(0.22 * w))
            y1 = max(0, seed_y - int(0.22 * h))
            x2 = min(w, seed_x + int(0.22 * w))
            y2 = min(h, seed_y + int(0.22 * h))
            local = focus[y1:y2, x1:x2]
            local_mask = local >= max(0.22, float(np.percentile(local[local > 0], 42.0)) if np.any(local > 0) else 1.0)
            lys, lxs = np.where(local_mask)
            if lys.size >= 12 and lxs.size >= 12:
                bx1 = x1 + int(lxs.min())
                by1 = y1 + int(lys.min())
                bx2 = x1 + int(lxs.max()) + 1
                by2 = y1 + int(lys.max()) + 1
            else:
                bx1, by1, bx2, by2 = x1, y1, x2, y2
            bx1, by1, bx2, by2 = PathologyNavigationAgent._expand_box_with_context(
                bx1,
                by1,
                bx2,
                by2,
                w,
                h,
                context_ratio=0.18,
                min_ratio=0.20,
            )
            boxes.append(
                {
                    "x1": float(bx1 / w),
                    "y1": float(by1 / h),
                    "x2": float(bx2 / w),
                    "y2": float(by2 / h),
                    "confidence": float(np.clip(confidence, 0.35, 0.85)),
                    "label": "local breast lesion candidate",
                    "reason": "dense disordered tissue focus",
                }
            )
            clear_pad_x = max(6, int(0.08 * w))
            clear_pad_y = max(6, int(0.08 * h))
            mask[max(0, by1 - clear_pad_y):min(h, by2 + clear_pad_y), max(0, bx1 - clear_pad_x):min(w, bx2 + clear_pad_x)] = 0
        return PathologyNavigationAgent._deduplicate_boxes(boxes)

    @staticmethod
    def _expand_box_with_context(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
        context_ratio: float = 0.14,
        min_ratio: float = 0.16,
    ) -> Tuple[int, int, int, int]:
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        target_w = max(bw + 2 * int(context_ratio * bw), int(min_ratio * width))
        target_h = max(bh + 2 * int(context_ratio * bh), int(min_ratio * height))
        nx1 = max(0, int(round(cx - target_w / 2.0)))
        ny1 = max(0, int(round(cy - target_h / 2.0)))
        nx2 = min(width, max(nx1 + 1, int(round(cx + target_w / 2.0))))
        ny2 = min(height, max(ny1 + 1, int(round(cy + target_h / 2.0))))
        nx1 = max(0, min(nx1, width - 1))
        ny1 = max(0, min(ny1, height - 1))
        nx2 = max(nx1 + 1, min(nx2, width))
        ny2 = max(ny1 + 1, min(ny2, height))
        return nx1, ny1, nx2, ny2

    @staticmethod
    def _box_iou(box_a: Dict[str, float], box_b: Dict[str, float]) -> float:
        ax1, ay1, ax2, ay2 = float(box_a["x1"]), float(box_a["y1"]), float(box_a["x2"]), float(box_a["y2"])
        bx1, by1, bx2, by2 = float(box_b["x1"]), float(box_b["y1"]), float(box_b["x2"]), float(box_b["y2"])
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _deduplicate_boxes(boxes: List[Dict[str, float]], iou_threshold: float = 0.38, max_keep: int = 2) -> List[Dict[str, float]]:
        kept: List[Dict[str, float]] = []
        ordered = sorted(boxes, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        for box in ordered:
            if any(PathologyNavigationAgent._box_iou(box, prev) >= iou_threshold for prev in kept):
                continue
            kept.append(box)
            if len(kept) >= max_keep:
                break
        return kept

    @staticmethod
    def _breast_lesion_focus_map(patch_rgb: np.ndarray) -> np.ndarray:
        gray = patch_rgb.astype(np.float32).mean(axis=2)
        tissue = (gray < 248.0).astype(np.float32)
        dark = np.clip((232.0 - gray) / 92.0, 0.0, 1.0)
        fine_blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
        coarse_blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=8)), dtype=np.float32)
        fine_detail = resize_map_like(np.abs(gray - fine_blur) / (np.abs(gray - fine_blur).max() + 1e-6), gray.shape)
        architecture_disorder = resize_map_like(np.abs(gray - coarse_blur) / (np.abs(gray - coarse_blur).max() + 1e-6), gray.shape)
        local_mean = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.float32)
        density_difference = np.clip((local_mean - gray) / 80.0, 0.0, 1.0)
        focus = np.clip(
            0.34 * dark
            + 0.26 * fine_detail
            + 0.25 * architecture_disorder
            + 0.15 * density_difference,
            0.0,
            1.0,
        ) * tissue
        focus = np.array(
            Image.fromarray((focus * 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1)),
            dtype=np.float32,
        ) / 255.0
        focus[focus < 0.12] = 0.0
        return np.clip(focus * tissue, 0.0, 1.0)

    def _update_candidate_scores(self, rois: List[NavigationROI], anchor_roi: NavigationROI, visited_ids: set[int]) -> None:
        ax1, ay1, ax2, ay2 = anchor_roi.rel_coordinate
        acx = 0.5 * (ax1 + ax2)
        acy = 0.5 * (ay1 + ay2)
        for roi in rois:
            if roi.tile_id in visited_ids:
                continue
            x1, y1, x2, y2 = roi.rel_coordinate
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            dist = float(np.sqrt((cx - acx) ** 2 + (cy - acy) ** 2))
            diversity_penalty = np.clip(1.0 - dist / 0.35, 0.0, 1.0)
            evidence_boost = 0.08 if roi.suspicion_type == anchor_roi.suspicion_type else 0.0
            updated = 0.82 * roi.final_score + 0.18 * roi.stage1_score - 0.10 * diversity_penalty + evidence_boost
            roi.final_score = float(np.clip(updated, 0.0, 1.0))
            roi.stage_trace.append(f"round_update_from={anchor_roi.tile_id}")

    @staticmethod
    def _annotated_grid(preview_rgb: np.ndarray, rois: List[NavigationROI]) -> np.ndarray:
        image = Image.fromarray(preview_rgb.copy())
        draw = ImageDraw.Draw(image)
        w, h = image.size
        for roi in rois:
            x1 = int(roi.rel_coordinate[0] * w)
            y1 = int(roi.rel_coordinate[1] * h)
            x2 = int(roi.rel_coordinate[2] * w)
            y2 = int(roi.rel_coordinate[3] * h)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=2)
            draw.rectangle([x1 + 4, y1 + 4, x1 + 42, y1 + 24], fill=(0, 0, 0))
            draw.text((x1 + 8, y1 + 6), str(roi.tile_id), fill=(255, 255, 255))
        return np.array(image)

    def _build_risk_heatmap(self, preview_rgb: np.ndarray, rois: List[NavigationROI]) -> np.ndarray:
        h, w = preview_rgb.shape[:2]
        risk = np.zeros((h, w), dtype=np.float32)
        for roi in rois:
            x1 = int(roi.rel_coordinate[0] * w)
            y1 = int(roi.rel_coordinate[1] * h)
            x2 = int(roi.rel_coordinate[2] * w)
            y2 = int(roi.rel_coordinate[3] * h)
            risk[y1:y2, x1:x2] = np.maximum(risk[y1:y2, x1:x2], roi.final_score)
        risk = np.array(Image.fromarray((risk * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=16)), dtype=np.float32) / 255.0
        overlay = preview_rgb.astype(np.float32).copy()
        heat = np.stack([255.0 * risk, 80.0 * (1.0 - np.abs(risk - 0.5) * 2.0), 40.0 * (1.0 - risk)], axis=2)
        overlay = 0.60 * overlay + 0.40 * heat
        return np.clip(overlay, 0, 255).astype(np.uint8)


class RegionRoutingModule:
    """
    Innovation 2:
    Agent decision -> stain scheduling.
    Produces:
    - hard routing mask for high-quality stain regions
    - soft attention map for continuous guidance
    """

    def __init__(self, config: PathologyConfig) -> None:
        self.config = config

    def build(self, preview_rgb: np.ndarray, rois: List[NavigationROI]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = preview_rgb.shape[:2]
        hard = np.zeros((h, w), dtype=np.float32)
        soft = np.zeros((h, w), dtype=np.float32)
        tissue = (preview_rgb.astype(np.float32).mean(axis=2) < 245.0).astype(np.float32)
        for roi in rois:
            x1 = int(roi.rel_coordinate[0] * w)
            y1 = int(roi.rel_coordinate[1] * h)
            x2 = int(roi.rel_coordinate[2] * w)
            y2 = int(roi.rel_coordinate[3] * h)
            if not roi.selected_highres:
                continue
            roi_h = max(1, y2 - y1)
            roi_w = max(1, x2 - x1)
            roi_preview = preview_rgb[y1:y2, x1:x2]
            if not self.config.use_routing_constraint:
                hard[y1:y2, x1:x2] = np.maximum(hard[y1:y2, x1:x2], tissue[y1:y2, x1:x2] * max(0.55, roi.lesion_confidence))
                soft[y1:y2, x1:x2] = np.maximum(soft[y1:y2, x1:x2], tissue[y1:y2, x1:x2] * max(0.50, roi.final_score))
                continue
            if roi.local_boxes:
                seed_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
                for box in roi.local_boxes:
                    try:
                        bx1 = x1 + int(float(box.get("x1", 0.0)) * roi_w)
                        by1 = y1 + int(float(box.get("y1", 0.0)) * roi_h)
                        bx2 = x1 + int(float(box.get("x2", 1.0)) * roi_w)
                        by2 = y1 + int(float(box.get("y2", 1.0)) * roi_h)
                    except Exception:
                        continue
                    bx1 = max(x1, min(bx1, x2 - 1))
                    by1 = max(y1, min(by1, y2 - 1))
                    bx2 = max(bx1 + 1, min(bx2, x2))
                    by2 = max(by1 + 1, min(by2, y2))
                    confidence = float(np.clip(float(box.get("confidence", roi.lesion_confidence)), 0.0, 1.0))
                    seed_mask[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = np.maximum(
                        seed_mask[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1], confidence
                    )
                refined_mask = self._refine_preview_region_mask(roi_preview, seed_mask, enabled=self.config.use_local_refinement)
                hard[y1:y2, x1:x2] = np.maximum(hard[y1:y2, x1:x2], refined_mask)
                soft_seed = np.clip(0.55 * roi.final_score + 0.45 * refined_mask, 0.0, 1.0)
                soft[y1:y2, x1:x2] = np.maximum(soft[y1:y2, x1:x2], refined_mask * soft_seed)
            else:
                shrink = 0.08
                sx1 = x1 + int(roi_w * shrink)
                sy1 = y1 + int(roi_h * shrink)
                sx2 = max(sx1 + 1, x2 - int(roi_w * shrink))
                sy2 = max(sy1 + 1, y2 - int(roi_h * shrink))
                seed_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
                seed_mask[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = roi.lesion_confidence
                refined_mask = self._refine_preview_region_mask(roi_preview, seed_mask, enabled=self.config.use_local_refinement)
                hard[y1:y2, x1:x2] = np.maximum(hard[y1:y2, x1:x2], refined_mask)
                soft[y1:y2, x1:x2] = np.maximum(soft[y1:y2, x1:x2], refined_mask * roi.final_score)
        hard = np.array(Image.fromarray((hard * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=5)), dtype=np.float32) / 255.0
        soft = np.array(Image.fromarray((soft * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=14)), dtype=np.float32) / 255.0
        hard = np.clip(hard * tissue, 0.0, 1.0)
        soft = np.clip(soft * tissue, 0.0, 1.0)
        # Reduce pruning thresholds so small but meaningful attention is retained
        hard[hard < 0.015] = 0.0
        soft[soft < 0.005] = 0.0
        return hard, soft

    @staticmethod
    def _refine_preview_region_mask(roi_rgb: np.ndarray, seed_mask: np.ndarray, enabled: bool = True) -> np.ndarray:
        if roi_rgb.size == 0 or seed_mask.size == 0:
            return seed_mask
        if not enabled:
            gray = roi_rgb.astype(np.float32).mean(axis=2)
            tissue = (gray < 248.0).astype(np.float32)
            seed = np.array(
                Image.fromarray((np.clip(seed_mask, 0.0, 1.0) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4)),
                dtype=np.float32,
            ) / 255.0
            seed = np.maximum(seed, 0.82 * np.clip(seed_mask, 0.0, 1.0))
            seed[seed < 0.03] = 0.0
            return np.clip(seed * tissue, 0.0, 1.0)
        gray = roi_rgb.astype(np.float32).mean(axis=2)
        tissue = (gray < 248.0).astype(np.float32)
        focus = PathologyNavigationAgent._breast_lesion_focus_map(roi_rgb)
        if seed_mask.max() <= 0:
            return np.zeros_like(seed_mask, dtype=np.float32)
        support = np.array(
            Image.fromarray((seed_mask > 0.02).astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(radius=6)),
            dtype=np.float32,
        ) / 255.0
        support = (support > 0.015).astype(np.float32) * tissue
        threshold = np.percentile(focus[support > 0], 35.0) if np.any(support > 0) else np.percentile(focus, 60.0)
        refined = np.where((focus >= threshold) & (support > 0), np.maximum(0.78 * seed_mask, focus), 0.0).astype(np.float32)
        refined = np.array(
            Image.fromarray((np.clip(refined, 0.0, 1.0) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3)),
            dtype=np.float32,
        ) / 255.0
        if float((refined > 0.08).mean()) < 0.035:
            refined = np.maximum(refined, support * max(0.62, float(seed_mask.max()) * 0.82))
        refined[refined < 0.04] = 0.0
        return np.clip(refined * tissue, 0.0, 1.0)


class DiagnosisPriorProvider:
    def __init__(self, client: OpenAIVisionClient) -> None:
        self.client = client
        self.loss_module = DiagnosisAwareWeightedLoss()

    def build_target_masks(self, target_rgb: np.ndarray) -> RegionMasks:
        base = self.loss_module.mask_extractor.extract(target_rgb)
        if not self.client.enabled():
            return base
        response = self.client.diagnostic_priors(target_rgb)
        if not response:
            return base
        prior = RegionMasks(
            nucleus_mask=self._boxes_to_mask(response.get("nuclei", []), target_rgb.shape[:2]),
            membrane_mask=self._boxes_to_mask(response.get("membrane", []), target_rgb.shape[:2]),
            gland_mask=self._boxes_to_mask(response.get("gland", []), target_rgb.shape[:2]),
            background_mask=np.zeros(target_rgb.shape[:2], dtype=bool),
        )
        return self.loss_module.merge_region_masks(base, prior)

    def build_prior_overlay(self, rgb: np.ndarray, masks: RegionMasks) -> np.ndarray:
        overlay = rgb.astype(np.float32).copy()
        overlay[masks.nucleus_mask] = 0.55 * overlay[masks.nucleus_mask] + 0.45 * np.array([90.0, 60.0, 190.0], dtype=np.float32)
        overlay[masks.membrane_mask] = 0.55 * overlay[masks.membrane_mask] + 0.45 * np.array([220.0, 70.0, 70.0], dtype=np.float32)
        overlay[masks.gland_mask] = 0.55 * overlay[masks.gland_mask] + 0.45 * np.array([70.0, 180.0, 210.0], dtype=np.float32)
        return np.clip(overlay, 0, 255).astype(np.uint8)

    @staticmethod
    def _boxes_to_mask(boxes: object, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        if not isinstance(boxes, list):
            return mask
        for box in boxes:
            if not isinstance(box, dict):
                continue
            try:
                x1 = int(float(box["x1"]) * w)
                y1 = int(float(box["y1"]) * h)
                x2 = int(float(box["x2"]) * w)
                y2 = int(float(box["y2"]) * h)
            except Exception:
                continue
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            mask[y1:y2, x1:x2] = True
        return mask


class DiagnosisPreservingVirtualStainModule:
    """
    Inference-side stain scheduler.
    Keeps the project logic:
    agent locates and routes, stain module focuses on selected regions.
    """

    def __init__(self, config: PathologyConfig, client: OpenAIVisionClient) -> None:
        self.config = config
        self.client = client
        self.he_visualizer = SimpleStainTransfer()
        self.virtual_he = UnpairedVirtualHEBaseline()
        self.torch_model: Optional[AttentionConditionedResUNet] = None
        self.torch_device: Optional[torch.device] = None
        self.model_load_status: str = "not_requested"
        self._load_torch_model()

    def reload_model(self) -> str:
        """Force reload the torch model from `self.config.model_path`. Returns the load status."""
        self._load_torch_model(force=True)
        return self.model_load_status

    def run(
        self,
        input_path: str,
        preview_rgb: np.ndarray,
        rois: List[NavigationROI],
        hard_mask: np.ndarray,
        soft_attention: np.ndarray,
        output_dir: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[StageArtifact]]:
        lowres = self._stain_preview(preview_rgb)
        fused = lowres.astype(np.float32).copy()
        reader = SlideReader(input_path, max_side=self.config.highres_patch_side)
        h, w = preview_rgb.shape[:2]
        artifacts: List[StageArtifact] = []

        for roi in rois:
            if not roi.selected_highres:
                continue
            patch_rgb = reader.load_region(roi.rel_coordinate)
            x1 = int(roi.rel_coordinate[0] * w)
            y1 = int(roi.rel_coordinate[1] * h)
            x2 = int(roi.rel_coordinate[2] * w)
            y2 = int(roi.rel_coordinate[3] * h)
            local_hard = hard_mask[y1:y2, x1:x2]
            local_soft = soft_attention[y1:y2, x1:x2]
            patch_stain = self._stain_image(patch_rgb, local_soft, local_hard, roi)
            target_w = max(1, x2 - x1)
            target_h = max(1, y2 - y1)
            patch_resized = np.array(Image.fromarray(patch_stain).resize((target_w, target_h), Image.Resampling.LANCZOS), dtype=np.float32)
            feather = self._feather_mask(target_h, target_w)
            local_weight = np.clip(np.maximum(local_hard, local_soft), 0.0, 1.0)
            weight = np.clip(self.config.fusion_strength * feather * local_weight, 0.0, 1.0)
            fused[y1:y2, x1:x2] = (1.0 - weight[..., None]) * fused[y1:y2, x1:x2] + weight[..., None] * patch_resized

            tile_dir = os.path.join(output_dir, f"roi_{roi.tile_id:02d}")
            os.makedirs(tile_dir, exist_ok=True)
            patch_path = os.path.join(tile_dir, "patch.png")
            stain_path = os.path.join(tile_dir, "stain.png")
            weight_path = os.path.join(tile_dir, "routing_weight.png")
            save_image(patch_rgb, patch_path)
            save_image(patch_stain, stain_path)
            save_image(np.repeat((weight * 255.0).round().astype(np.uint8)[..., None], 3, axis=2), weight_path)
            artifacts.extend(
                [
                    StageArtifact(f"roi_{roi.tile_id:02d}_patch", patch_path, "Selected ROI patch"),
                    StageArtifact(f"roi_{roi.tile_id:02d}_stain", stain_path, "High-quality local stain result"),
                    StageArtifact(f"roi_{roi.tile_id:02d}_weight", weight_path, "Local routing weight"),
                ]
            )

        return lowres, np.clip(fused, 0, 255).astype(np.uint8), artifacts

    def _stain_preview(self, rgb: np.ndarray) -> np.ndarray:
        if self.config.stain_mode == "he_visualize":
            return self.he_visualizer.transform(rgb)
        return rgb.copy()

    def _load_torch_model(self, force: bool = False) -> None:
        if not self.config.model_path:
            self.model_load_status = "no_model_path"
            return
        # Avoid reloading repeatedly unless forced
        if not force and self.torch_model is not None:
            return
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            print(f"warning=trained_model_not_found path={model_path}")
            self.model_load_status = "model_not_found"
            return
        device = self._resolve_device(self.config.model_device)
        checkpoint = torch.load(str(model_path), map_location=device)
        model_config_raw = checkpoint.get("model_config", {})
        if isinstance(model_config_raw, dict):
            valid_keys = set(inspect.signature(VirtualStainResUNetConfig).parameters.keys())
            filtered_model_config = {key: value for key, value in model_config_raw.items() if key in valid_keys}
            removed_keys = sorted(set(model_config_raw.keys()) - valid_keys)
            if removed_keys:
                print(f"warning=ignored_legacy_model_config_keys keys={removed_keys}")
            model_config = VirtualStainResUNetConfig(**filtered_model_config)
        else:
            model_config = VirtualStainResUNetConfig()
        model = AttentionConditionedResUNet(model_config).to(device)
        raw_state = checkpoint["model_state_dict"]
        cleaned_state: Dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            if key.startswith("model."):
                cleaned_state[key[len("model."):]] = value
            elif key.startswith("loss_fn."):
                continue
            else:
                cleaned_state[key] = value
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if unexpected:
            print(f"warning=unexpected_model_keys count={len(unexpected)}")
        if missing:
            print(f"warning=missing_model_keys count={len(missing)}")
        total_keys = max(1, len(model.state_dict()))
        missing_ratio = len(missing) / float(total_keys)
        if missing_ratio > 0.12:
            print(
                f"warning=incompatible_checkpoint_retrain_recommended "
                f"missing_ratio={missing_ratio:.3f}"
            )
            self.model_load_status = f"partially_loaded_missing_ratio={missing_ratio:.3f}"
        else:
            self.model_load_status = "loaded"
        model.eval()
        self.torch_model = model
        self.torch_device = device
        print(f"loaded_trained_virtual_stain_model={model_path} device={device}")

    def _stain_image(self, rgb: np.ndarray, soft_attention: np.ndarray, hard_mask: np.ndarray, roi: NavigationROI) -> np.ndarray:
        # If model is not loaded but a checkpoint exists now, try to reload once.
        if (self.torch_model is None or self.torch_device is None) and self.config.model_path:
            try:
                model_path = Path(self.config.model_path)
                if model_path.exists():
                    print(f"info=found_model_now_attempt_reload path={model_path}")
                    # force reload to pick up a newly placed best.pt
                    self._load_torch_model(force=True)
            except Exception as exc:
                print(f"warning=model_reload_attempt_failed error={exc}")

        if self.torch_model is not None and self.torch_device is not None:
            return self._stain_with_model(rgb, soft_attention, hard_mask, roi)
        # Fallback: if no torch model, return preview-based stain (or raw rgb depending on mode)
        return self._stain_preview(rgb)

    def _stain_with_model(self, rgb: np.ndarray, soft_attention: np.ndarray, hard_mask: np.ndarray, roi: NavigationROI) -> np.ndarray:
        assert self.torch_model is not None
        assert self.torch_device is not None
        patch_h, patch_w = rgb.shape[:2]
        attention = resize_map_like(soft_attention, (patch_h, patch_w))
        highres = resize_map_like(hard_mask, (patch_h, patch_w))
        tissue = (rgb.astype(np.float32).mean(axis=2) < 248.0).astype(np.float32)
        confidence = np.full((patch_h, patch_w), float(np.clip(roi.lesion_confidence, 0.0, 1.0)), dtype=np.float32)

        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        attention_tensor = torch.from_numpy(attention[None, None, ...]).float()
        highres_tensor = torch.from_numpy(highres[None, None, ...]).float()
        tissue_tensor = torch.from_numpy(tissue[None, None, ...]).float()
        confidence_tensor = torch.from_numpy(confidence[None, None, ...]).float()

        rgb_tensor = rgb_tensor.to(self.torch_device)
        attention_tensor = attention_tensor.to(self.torch_device)
        highres_tensor = highres_tensor.to(self.torch_device)
        tissue_tensor = tissue_tensor.to(self.torch_device)
        confidence_tensor = confidence_tensor.to(self.torch_device)

        decision_maps = build_decision_maps(
            attention_map=attention_tensor,
            highres_mask=highres_tensor,
            tissue_map=tissue_tensor,
            lesion_confidence=confidence_tensor,
        )
        with torch.no_grad():
            outputs = self.torch_model(rgb_tensor, decision_maps)
            pred = outputs["rgb"][0].detach().float().cpu().clamp(0.0, 1.0).numpy()
        pred_rgb = (pred.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        blend_weight = np.clip((0.85 * attention + 1.25 * highres) * tissue, 0.0, 1.0)
        blend_weight = np.array(
            Image.fromarray((blend_weight * 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=5)),
            dtype=np.float32,
        ) / 255.0
        blend_weight = np.maximum(blend_weight, 0.75 * highres * tissue)
        blend_weight[blend_weight < 0.015] = 0.0
        fused = (1.0 - blend_weight[..., None]) * rgb.astype(np.float32) + blend_weight[..., None] * pred_rgb.astype(np.float32)
        fused_clipped = np.clip(fused, 0.0, 255.0).astype(np.uint8)

        # Diagnostic prints to help debug missing staining
        try:
            att_mean = float(attention.mean())
            highres_mean = float(highres.mean())
            blend_mean = float(blend_weight.mean())
            blend_nonzero_frac = float((blend_weight > 0.025).mean())
            rgb_mean = float(rgb.astype(np.float32).mean())
            pred_mean = float(pred_rgb.astype(np.float32).mean())
            fused_mean = float(fused_clipped.astype(np.float32).mean())
            print(
                f"debug=stain_tile tile_id={getattr(roi, 'tile_id', 'NA')} model_status={self.model_load_status} "
                f"att_mean={att_mean:.4f} highres_mean={highres_mean:.4f} blend_mean={blend_mean:.4f} "
                f"blend_nonzero_frac={blend_nonzero_frac:.4f} rgb_mean={rgb_mean:.1f} pred_mean={pred_mean:.1f} fused_mean={fused_mean:.1f}"
            )
        except Exception:
            pass

        return fused_clipped

    @staticmethod
    def _resolve_device(raw: str) -> torch.device:
        if raw == "cpu":
            return torch.device("cpu")
        if raw == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _feather_mask(h: int, w: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = max(1.0, (w - 1) / 2.0)
        cy = max(1.0, (h - 1) / 2.0)
        edge = np.maximum(np.abs((xx - cx) / cx), np.abs((yy - cy) / cy))
        return np.exp(-np.power(edge / 0.9, 4)).astype(np.float32)


class PathologyEnd2EndSystem:
    def __init__(self, config: PathologyConfig) -> None:
        self.config = config
        self.client = OpenAIVisionClient(
            ApiConfig(
                config.api_endpoint,
                config.api_key,
                config.api_model,
                timeout=config.api_timeout,
                prompt_hint=config.api_prompt_hint,
                use_morphology_aware_prompt=config.use_morphology_aware_prompt,
            )
        )
        self.navigator = PathologyNavigationAgent(config, self.client)
        self.router = RegionRoutingModule(config)
        self.stainer = DiagnosisPreservingVirtualStainModule(config, self.client)
        self.prior_provider = DiagnosisPriorProvider(self.client)
        self.loss_module = DiagnosisAwareWeightedLoss()

    def run(self, input_path: str, output_dir: str, target_path: Optional[str] = None) -> PathologyResult:
        os.makedirs(output_dir, exist_ok=True)
        reader = SlideReader(input_path, max_side=self.config.preview_max_side)
        preview = reader.load_preview()

        rois, risk_heatmap, risk_heatmap_path = self.navigator.analyze(input_path, preview, output_dir)
        hard_mask, soft_attention = self.router.build(preview, rois)
        hard_mask_path = os.path.join(output_dir, "routing_hard_mask.png")
        soft_attention_path = os.path.join(output_dir, "routing_soft_attention.png")
        save_image(np.repeat((hard_mask * 255.0).round().astype(np.uint8)[..., None], 3, axis=2), hard_mask_path)
        save_image(np.repeat((soft_attention * 255.0).round().astype(np.uint8)[..., None], 3, axis=2), soft_attention_path)

        lowres_stain, fused_stain, artifacts = self.stainer.run(input_path, preview, rois, hard_mask, soft_attention, output_dir)
        lowres_stain_path = os.path.join(output_dir, "lowres_stain.png")
        fused_stain_path = os.path.join(output_dir, "fused_stain.png")
        roi_overlay_path = os.path.join(output_dir, "roi_overlay.png")
        fused_roi_overlay_path = os.path.join(output_dir, "fused_roi_overlay.png")
        save_image(lowres_stain, lowres_stain_path)
        save_image(fused_stain, fused_stain_path)
        roi_overlay = self._build_roi_overlay(preview, rois)
        save_image(roi_overlay, roi_overlay_path)
        fused_overlay = self._build_roi_overlay(fused_stain, rois)
        save_image(fused_overlay, fused_roi_overlay_path)

        metrics = self._metrics(rois, hard_mask, soft_attention)
        diagnosis_loss: Dict[str, float] = {}
        prior_overlay_path: Optional[str] = None
        if target_path:
            target_rgb = load_rgb_image(target_path)
            if target_rgb.shape[:2] != fused_stain.shape[:2]:
                target_rgb = np.array(
                    Image.fromarray(target_rgb).resize((fused_stain.shape[1], fused_stain.shape[0]), Image.Resampling.LANCZOS)
                )
            target_masks = self.prior_provider.build_target_masks(target_rgb)
            diagnosis_loss = self.loss_module.forward_with_masks(fused_stain, target_rgb, target_masks=target_masks)
            loss_dir = os.path.join(output_dir, "diagnosis")
            os.makedirs(loss_dir, exist_ok=True)
            weight_map = self.loss_module.weight_builder.build(target_masks)
            save_mask_visualizations(loss_dir, target_masks, weight_map)
            prior_overlay_path = os.path.join(loss_dir, "diagnostic_priors_overlay.png")
            prior_overlay = self.prior_provider.build_prior_overlay(target_rgb, target_masks)
            save_image(prior_overlay, prior_overlay_path)
            with open(os.path.join(loss_dir, "loss.json"), "w", encoding="utf-8") as file:
                json.dump(diagnosis_loss, file, ensure_ascii=False, indent=2)

        result = PathologyResult(
            input_path=input_path,
            target_path=target_path,
            overview_risk_heatmap_path=risk_heatmap_path,
            roi_overlay_path=roi_overlay_path,
            routing_hard_mask_path=hard_mask_path,
            routing_soft_attention_path=soft_attention_path,
            lowres_stain_path=lowres_stain_path,
            fused_stain_path=fused_stain_path,
            fused_roi_overlay_path=fused_roi_overlay_path,
            diagnostic_priors_overlay_path=prior_overlay_path,
            artifacts=[StageArtifact("overview_risk_heatmap", risk_heatmap_path, "Overview risk heatmap")] + artifacts,
            rois=rois,
            metrics=metrics,
            diagnosis_loss=diagnosis_loss,
            api_usage={**self.client.usage_dict(), "model_load_status": self.stainer.model_load_status},
            llm_overview=self.navigator.last_overview_context,
        )
        with open(os.path.join(output_dir, "pathology_result.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "input_path": result.input_path,
                    "target_path": result.target_path,
                    "overview_risk_heatmap_path": result.overview_risk_heatmap_path,
                    "roi_overlay_path": result.roi_overlay_path,
                    "routing_hard_mask_path": result.routing_hard_mask_path,
                    "routing_soft_attention_path": result.routing_soft_attention_path,
                    "lowres_stain_path": result.lowres_stain_path,
                    "fused_stain_path": result.fused_stain_path,
                    "fused_roi_overlay_path": result.fused_roi_overlay_path,
                    "diagnostic_priors_overlay_path": result.diagnostic_priors_overlay_path,
                    "artifacts": [asdict(item) for item in result.artifacts],
                    "rois": [asdict(item) for item in result.rois],
                    "metrics": result.metrics,
                    "diagnosis_loss": result.diagnosis_loss,
                    "api_usage": result.api_usage,
                    "llm_overview": result.llm_overview,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        report_path = write_pathology_report(os.path.join(output_dir, "pathology_result.json"))
        result.report_path = report_path
        return result

    @staticmethod
    def _build_roi_overlay(preview_rgb: np.ndarray, rois: List[NavigationROI]) -> np.ndarray:
        image = Image.fromarray(preview_rgb.copy())
        draw = ImageDraw.Draw(image)
        w, h = image.size
        for roi in rois:
            x1 = int(roi.rel_coordinate[0] * w)
            y1 = int(roi.rel_coordinate[1] * h)
            x2 = int(roi.rel_coordinate[2] * w)
            y2 = int(roi.rel_coordinate[3] * h)
            color = (255, 80, 64) if roi.selected_highres else (255, 210, 90)
            width = 3 if roi.selected_highres else 1
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label = f"{roi.tile_id}:{roi.final_score:.2f}"
            if roi.visited_round > 0:
                label += f"/r{roi.visited_round}"
            draw.rectangle([x1 + 4, y1 + 4, min(x2, x1 + 96), min(y2, y1 + 24)], fill=(0, 0, 0))
            draw.text((x1 + 8, y1 + 6), label, fill=(255, 255, 255))
            if roi.selected_highres and roi.local_boxes:
                roi_w = max(1, x2 - x1)
                roi_h = max(1, y2 - y1)
                for box in roi.local_boxes:
                    try:
                        bx1 = x1 + int(float(box.get("x1", 0.0)) * roi_w)
                        by1 = y1 + int(float(box.get("y1", 0.0)) * roi_h)
                        bx2 = x1 + int(float(box.get("x2", 1.0)) * roi_w)
                        by2 = y1 + int(float(box.get("y2", 1.0)) * roi_h)
                    except Exception:
                        continue
                    bx1 = max(x1, min(bx1, x2 - 1))
                    by1 = max(y1, min(by1, y2 - 1))
                    bx2 = max(bx1 + 1, min(bx2, x2))
                    by2 = max(by1 + 1, min(by2, y2))
                    draw.rectangle([bx1, by1, bx2, by2], outline=(64, 255, 255), width=2)
        return np.array(image)

    def _metrics(self, rois: List[NavigationROI], hard_mask: np.ndarray, soft_attention: np.ndarray) -> Dict[str, float]:
        selected = sum(1 for roi in rois if roi.selected_highres)
        total = max(1, len(rois))
        visited = max(1, sum(1 for roi in rois if roi.visited))
        selected_ratio = selected / float(visited)
        compute_reduction = 1.0 - selected_ratio
        selected_scores = [float(roi.final_score) for roi in rois if roi.selected_highres]
        selected_confidences = [float(roi.lesion_confidence) for roi in rois if roi.selected_highres]
        return {
            "total_rois": float(total),
            "visited_rois": float(visited),
            "selected_highres_rois": float(selected),
            "selected_ratio": selected_ratio,
            "selected_score_mean": float(np.mean(selected_scores)) if selected_scores else 0.0,
            "selected_confidence_mean": float(np.mean(selected_confidences)) if selected_confidences else 0.0,
            "estimated_compute_reduction_percent": compute_reduction * 100.0,
            "hard_mask_coverage": float(hard_mask.mean()),
            "soft_attention_mean": float(soft_attention.mean()),
            "api_total_calls": float(self.client.usage.total_calls),
            "api_failed_calls": float(self.client.usage.failed_calls),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight closed-loop pathology pipeline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--mode", default="virtual_he", choices=["virtual_he", "he_visualize"])
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.44)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--ablation-mode", default="full", choices=sorted(INFERENCE_ABLATION_DESCRIPTIONS.keys()))
    parser.add_argument("--prompt-hint", default=None)
    parser.add_argument("--disable-api", action="store_true")
    return parser


def make_default_config() -> PathologyConfig:
    cfg = load_medical_config()
    return PathologyConfig(
        api_endpoint=normalize_endpoint(str(cfg.get("llm_api_url", ""))),
        api_key=str(cfg.get("llm_api_key", "")),
        api_model=str(cfg.get("vlm_model", "gpt-4o")),
        api_timeout=int(cfg.get("request_timeout", 180)),
        api_prompt_hint=str(cfg.get("api_prompt_hint", "")),
        model_path=resolve_workspace_path(str(cfg.get("virtual_stain_model_path", ""))),
        model_device=str(cfg.get("virtual_stain_model_device", "auto")),
    )


def apply_inference_ablation(config: PathologyConfig, mode: str) -> None:
    if mode not in INFERENCE_ABLATION_DESCRIPTIONS:
        raise ValueError(f"Unsupported inference ablation mode: {mode}")
    if mode == "full":
        return
    if mode == "wo_morphology_aware_prompt":
        config.use_morphology_aware_prompt = False
        return
    if mode == "wo_local_roi_api_review":
        config.use_local_roi_api_review = False
        return
    if mode == "wo_local_refinement":
        config.use_local_refinement = False
        return
    if mode == "wo_routing_constraint":
        config.use_routing_constraint = False
        return


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = make_default_config()
    config.grid_size = args.grid
    config.top_k = args.top_k
    config.score_threshold = args.threshold
    config.stain_mode = args.mode
    config.api_enabled = not args.disable_api
    apply_inference_ablation(config, args.ablation_mode)
    if args.model_path:
        config.model_path = resolve_workspace_path(args.model_path)
    if args.prompt_hint:
        config.api_prompt_hint = args.prompt_hint
    system = PathologyEnd2EndSystem(config)
    result = system.run(args.input, args.output_dir, args.target)
    print(f"input={result.input_path}")
    print(f"overview_risk_heatmap={result.overview_risk_heatmap_path}")
    print(f"roi_overlay={result.roi_overlay_path}")
    print(f"routing_hard_mask={result.routing_hard_mask_path}")
    print(f"routing_soft_attention={result.routing_soft_attention_path}")
    print(f"fused_stain={result.fused_stain_path}")
    if result.report_path:
        print(f"report={result.report_path}")
    print(f"api_total_calls={result.api_usage.get('total_calls', 0)}")
    print(f"api_score_tiles_calls={result.api_usage.get('score_tiles_calls', 0)}")
    print(f"api_localize_roi_calls={result.api_usage.get('localize_roi_calls', 0)}")
    print(f"api_diagnostic_priors_calls={result.api_usage.get('diagnostic_priors_calls', 0)}")
    print(f"api_failed_calls={result.api_usage.get('failed_calls', 0)}")
    print(f"model_load_status={result.api_usage.get('model_load_status', 'unknown')}")
    if result.diagnostic_priors_overlay_path:
        print(f"diagnostic_priors_overlay={result.diagnostic_priors_overlay_path}")
    if result.diagnosis_loss:
        print(f"diagnosis_total_loss={result.diagnosis_loss['total_loss']:.6f}")


if __name__ == "__main__":
    main()
