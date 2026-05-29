import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.pathology_e2e import (
    INFERENCE_ABLATION_DESCRIPTIONS,
    PathologyEnd2EndSystem,
    apply_inference_ablation,
    make_default_config,
    resolve_workspace_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch pathology inference + compact ROI metric evaluation.")
    parser.add_argument("--manifest", default=None, help="JSON file with case list and optional annotations.")
    parser.add_argument("--dataset-root", default=None, help="Dataset root to auto-discover *_unstained.svs and *_he.svs pairs.")
    parser.add_argument("--output-dir", required=True, help="Directory to store per-case results and summary files.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N discovered cases.")
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional explicit case_id list. Only these cases will be processed.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K selected ROIs used for ROI metrics.")
    parser.add_argument("--iou-threshold", type=float, default=0.15, help="IoU threshold for ROI match.")
    parser.add_argument("--threshold", type=float, default=0.44, help="Inference score threshold.")
    parser.add_argument("--grid", type=int, default=5, help="Overview grid size.")
    parser.add_argument("--disable-api", action="store_true")
    parser.add_argument("--prompt-hint", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--fast", action="store_true", help="Enable fast mode: lower preview resolution and fewer navigation rounds.")
    parser.add_argument("--ablation-mode", default="full", choices=sorted(INFERENCE_ABLATION_DESCRIPTIONS.keys()))
    parser.add_argument(
        "--use-target-paths",
        action="store_true",
        help="Use target_path from manifest for image-level staining metrics. "
             "Do not enable this when target_path points to full-resolution .svs files.",
    )
    return parser


def load_manifest(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, dict):
        cases = raw.get("cases", [])
    elif isinstance(raw, list):
        cases = raw
    else:
        raise ValueError("Manifest must be a list or an object containing a 'cases' list.")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest contains no cases.")
    return [dict(item) for item in cases if isinstance(item, dict)]


def auto_discover_cases(dataset_root: str) -> List[Dict[str, object]]:
    root = Path(resolve_workspace_path(dataset_root))
    if not root.exists():
        raise ValueError(f"dataset_root does not exist: {dataset_root}")
    cases: List[Dict[str, object]] = []
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        case_id = case_dir.name
        unstained = case_dir / f"{case_id}_unstained.svs"
        he = case_dir / f"{case_id}_he.svs"
        if unstained.exists():
            cases.append(
                {
                    "case_id": case_id,
                    "input_path": str(unstained),
                    "target_path": "",
                    "roi_boxes": [],
                    "lesion_mask_path": "",
                }
            )
    if not cases:
        raise ValueError(f"No *_unstained.svs cases found under {root}")
    return cases


def normalized_box_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def roi_metrics(
    pred_boxes: Sequence[Dict[str, float]],
    gt_boxes: Sequence[Dict[str, float]],
    top_k: int,
    iou_threshold: float,
) -> Dict[str, float]:
    preds = list(pred_boxes)[:top_k]
    gts = list(gt_boxes)
    if not gts:
        return {
            "roi_recall_at_k": 0.0,
            "roi_precision_at_k": 0.0,
            "roi_mean_best_iou": 0.0,
        }

    matched_gt = 0
    best_ious: List[float] = []
    for gt in gts:
        best_iou = max((normalized_box_iou(pred, gt) for pred in preds), default=0.0)
        best_ious.append(best_iou)
        if best_iou >= iou_threshold:
            matched_gt += 1

    matched_pred = 0
    for pred in preds:
        best_iou = max((normalized_box_iou(pred, gt) for gt in gts), default=0.0)
        if best_iou >= iou_threshold:
            matched_pred += 1

    denom_pred = max(1, len(preds))
    return {
        "roi_recall_at_k": matched_gt / float(len(gts)),
        "roi_precision_at_k": matched_pred / float(denom_pred),
        "roi_mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
    }


def load_binary_mask(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return (np.array(image, dtype=np.uint8) >= 128).astype(np.uint8)


def masked_bbox_from_boxes(boxes: Sequence[Dict[str, float]], shape_hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    h, w = shape_hw
    xs1 = [int(float(box["x1"]) * w) for box in boxes]
    ys1 = [int(float(box["y1"]) * h) for box in boxes]
    xs2 = [int(float(box["x2"]) * w) for box in boxes]
    ys2 = [int(float(box["y2"]) * h) for box in boxes]
    x1, y1 = max(0, min(xs1)), max(0, min(ys1))
    x2, y2 = min(w, max(xs2)), min(h, max(ys2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def grayscale_ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    blur_a = np.array(Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    blur_b = np.array(Image.fromarray(np.clip(b * 255.0, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = blur_a
    mu_y = blur_b
    sigma_x = np.array(Image.fromarray(np.clip((a * a) * 255.0, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0 - mu_x * mu_x
    sigma_y = np.array(Image.fromarray(np.clip((b * b) * 255.0, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0 - mu_y * mu_y
    sigma_xy = np.array(Image.fromarray(np.clip((a * b) * 255.0, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0 - mu_x * mu_y
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-6)
    return float(np.mean(ssim_map))


def lesion_region_ssim(result_json: Dict[str, object], case: Dict[str, object], gt_mask_resolved: Optional[str]) -> Optional[float]:
    target_path = case.get("target_path")
    if not isinstance(target_path, str) or not target_path.strip():
        return None
    target_resolved = resolve_workspace_path(target_path)
    if not target_resolved or not os.path.exists(target_resolved):
        return None

    fused = np.array(Image.open(result_json["fused_stain_path"]).convert("RGB"))
    target = np.array(Image.open(target_resolved).convert("RGB").resize((fused.shape[1], fused.shape[0]), Image.Resampling.LANCZOS))
    if gt_mask_resolved and os.path.exists(gt_mask_resolved):
        gt_mask = load_binary_mask(gt_mask_resolved, fused.shape[:2])
        bbox = bbox_from_mask(gt_mask)
    else:
        gt_boxes = case.get("roi_boxes", [])
        bbox = masked_bbox_from_boxes(gt_boxes, fused.shape[:2]) if isinstance(gt_boxes, list) else None
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    fused_crop = np.mean(fused[y1:y2, x1:x2].astype(np.float32), axis=2)
    target_crop = np.mean(target[y1:y2, x1:x2].astype(np.float32), axis=2)
    if fused_crop.size == 0 or target_crop.size == 0:
        return None
    return grayscale_ssim(fused_crop, target_crop)


def mask_metrics(pred_mask_path: str, gt_mask_path: str) -> Dict[str, float]:
    pred = load_binary_mask(pred_mask_path, Image.open(pred_mask_path).convert("L").size[::-1])
    gt = load_binary_mask(gt_mask_path, pred.shape[:2])
    inter = float(np.logical_and(pred > 0, gt > 0).sum())
    pred_sum = float((pred > 0).sum())
    gt_sum = float((gt > 0).sum())
    union = float(np.logical_or(pred > 0, gt > 0).sum())
    dice = (2.0 * inter) / max(pred_sum + gt_sum, 1.0)
    iou = inter / max(union, 1.0)
    return {
        "lesion_dice": dice,
        "lesion_iou": iou,
    }


def selected_roi_boxes(rois: Sequence[Dict[str, object]]) -> List[Dict[str, float]]:
    selected = [roi for roi in rois if bool(roi.get("selected_highres", False))]
    selected.sort(key=lambda item: float(item.get("final_score", 0.0)), reverse=True)
    boxes: List[Dict[str, float]] = []
    for roi in selected:
        rel = roi.get("rel_coordinate", [0.0, 0.0, 1.0, 1.0])
        if isinstance(rel, (list, tuple)) and len(rel) == 4:
            boxes.append(
                {
                    "x1": float(rel[0]),
                    "y1": float(rel[1]),
                    "x2": float(rel[2]),
                    "y2": float(rel[3]),
                }
            )
    return boxes


def summarize_case(
    case: Dict[str, object],
    result_json_path: str,
    top_k: int,
    iou_threshold: float,
) -> Dict[str, object]:
    with open(result_json_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    rois = list(result.get("rois", []))
    metrics = dict(result.get("metrics", {}))
    diagnosis_loss = dict(result.get("diagnosis_loss", {}))
    row: Dict[str, object] = {
        "case_id": str(case.get("case_id", Path(str(case.get("input_path", ""))).stem)),
        "input_path": str(case.get("input_path", "")),
        "selected_highres_ratio": float(metrics.get("selected_ratio", 0.0)),
        "hard_mask_coverage": float(metrics.get("hard_mask_coverage", 0.0)),
        "soft_attention_mean": float(metrics.get("soft_attention_mean", 0.0)),
        "has_roi_annotation": 0.0,
        "has_mask_annotation": 0.0,
        "case_success": 1.0,
    }

    if "weighted_l1" in diagnosis_loss:
        row["weighted_l1"] = float(diagnosis_loss["weighted_l1"])
    if "clinical_consistency_loss" in diagnosis_loss:
        row["clinical_consistency_loss"] = float(diagnosis_loss["clinical_consistency_loss"])

    gt_boxes = case.get("roi_boxes", [])
    if isinstance(gt_boxes, list) and gt_boxes:
        row["has_roi_annotation"] = 1.0
        row.update(roi_metrics(selected_roi_boxes(rois), gt_boxes, top_k=top_k, iou_threshold=iou_threshold))

    # Auxiliary and validation metrics when no human annotation available
    # roi_count: number of ROIs produced by the agent
    row["roi_count"] = float(len(rois))
    # empty/background ROI ratio: fraction of ROIs with very low lesion confidence
    if rois:
        empty_count = sum(1 for roi in rois if float(getattr(roi, "lesion_confidence", 0.0)) < 0.08)
        row["empty_background_roi_ratio"] = float(empty_count) / float(len(rois))
    else:
        row["empty_background_roi_ratio"] = 0.0

    gt_mask_resolved: Optional[str] = None
    gt_mask_path = case.get("lesion_mask_path")
    if isinstance(gt_mask_path, str) and gt_mask_path.strip():
        gt_mask_resolved = resolve_workspace_path(gt_mask_path)
        if gt_mask_resolved and os.path.exists(gt_mask_resolved):
            row["has_mask_annotation"] = 1.0
            row.update(mask_metrics(result["routing_hard_mask_path"], gt_mask_resolved))

    lesion_ssim = lesion_region_ssim(result, case, gt_mask_resolved)
    if lesion_ssim is not None:
        row["lesion_region_ssim"] = float(lesion_ssim)

    # Valid ROI Ratio: fraction of selected ROIs that hit tissue in the routing hard mask
    try:
        routing_hard_mask_path = result.get("routing_hard_mask_path")
        selected_boxes = selected_roi_boxes(rois)
        valid = 0
        total = len(selected_boxes)
        if routing_hard_mask_path and total > 0 and os.path.exists(routing_hard_mask_path):
            mask_img = Image.open(routing_hard_mask_path).convert("L")
            mask_shape = mask_img.size[::-1]
            mask = load_binary_mask(routing_hard_mask_path, mask_shape)
            h, w = mask_shape
            for box in selected_boxes:
                x1 = int(max(0, min(w - 1, round(box["x1"] * w))))
                y1 = int(max(0, min(h - 1, round(box["y1"] * h))))
                x2 = int(max(x1 + 1, min(w, round(box["x2"] * w))))
                y2 = int(max(y1 + 1, min(h, round(box["y2"] * h))))
                if x2 <= x1 or y2 <= y1:
                    continue
                region = mask[y1:y2, x1:x2]
                if region.size > 0 and float((region > 0).sum()) > 0.0:
                    valid += 1
        row["valid_roi_ratio"] = float(valid) / float(total) if total > 0 else 0.0
    except Exception:
        row["valid_roi_ratio"] = 0.0

    return row


def summarize_failed_case(case: Dict[str, object], error: Exception) -> Dict[str, object]:
    case_id = str(case.get("case_id", Path(str(case.get("input_path", ""))).stem))
    return {
        "case_id": case_id,
        "input_path": str(case.get("input_path", "")),
        "case_success": 0.0,
        "skip_reason": str(error),
        "has_roi_annotation": 0.0,
        "has_mask_annotation": 0.0,
    }


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"case_id", "input_path"}
        }
    )
    summary: Dict[str, float] = {"num_cases": float(len(rows))}
    summary["roi_annotation_cases"] = float(sum(1 for row in rows if float(row.get("has_roi_annotation", 0.0)) > 0.5))
    summary["mask_annotation_cases"] = float(sum(1 for row in rows if float(row.get("has_mask_annotation", 0.0)) > 0.5))
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def filter_cases(
    cases: Sequence[Dict[str, object]],
    case_ids: Optional[Sequence[str]],
    limit: Optional[int],
) -> List[Dict[str, object]]:
    filtered = list(cases)
    if case_ids:
        wanted = {case_id.strip() for case_id in case_ids if case_id and case_id.strip()}
        filtered = [case for case in filtered if str(case.get("case_id", "")).strip() in wanted]
    if limit is not None and limit > 0:
        filtered = filtered[:limit]
    if not filtered:
        raise ValueError("No cases left after applying --case-ids/--limit filters.")
    return filtered


def write_outputs(output_dir: Path, rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    summary = aggregate_rows(rows)
    csv_path = output_dir / "batch_metrics.csv"
    json_path = output_dir / "batch_metrics.json"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "cases": rows}, file, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    if args.manifest:
        cases = load_manifest(args.manifest)
    elif args.dataset_root:
        cases = auto_discover_cases(args.dataset_root)
    else:
        raise ValueError("Please provide either --manifest or --dataset-root.")
    cases = filter_cases(cases, args.case_ids, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    try:
        for index, case in enumerate(cases, start=1):
            input_path = resolve_workspace_path(str(case.get("input_path", "")))
            if not input_path:
                raise ValueError(f"Case {index} is missing a valid input_path.")
            case_id = str(case.get("case_id", Path(input_path).stem))
            case_output_dir = output_dir / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                config = make_default_config()
                config.grid_size = args.grid
                config.top_k = args.top_k
                config.score_threshold = args.threshold
                if args.fast:
                    # reduce expensive steps for quick local testing
                    config.preview_max_side = 1024
                    config.navigation_rounds = 3
                    config.highres_patch_side = 512
                config.api_enabled = not args.disable_api
                apply_inference_ablation(config, args.ablation_mode)
                if args.prompt_hint:
                    config.api_prompt_hint = args.prompt_hint
                if args.model_path:
                    config.model_path = resolve_workspace_path(args.model_path)

                system = PathologyEnd2EndSystem(config)
                target_path = case.get("target_path")
                target_resolved = None
                if args.use_target_paths and isinstance(target_path, str) and target_path.strip():
                    candidate_target = resolve_workspace_path(str(target_path))
                    if candidate_target and os.path.exists(candidate_target):
                        suffix = Path(candidate_target).suffix.lower()
                        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                            target_resolved = candidate_target
                system.run(input_path, str(case_output_dir), target_resolved)
                row = summarize_case(case, os.path.join(case_output_dir, "pathology_result.json"), top_k=args.top_k, iou_threshold=args.iou_threshold)
                rows.append(row)
            except Exception as exc:
                failure_row = summarize_failed_case(case, exc)
                rows.append(failure_row)
                with open(case_output_dir / "failure.json", "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "case_id": case_id,
                            "input_path": input_path,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"warning=skip_case case_id={case_id} reason={type(exc).__name__} error={exc}")
            summary = write_outputs(output_dir, rows)
            recall_text = "NA"
            row_for_print = rows[-1]
            if "roi_recall_at_k" in row_for_print:
                recall_text = f"{float(row_for_print['roi_recall_at_k']):.3f}"
            print(
                f"case={index}/{len(cases)} id={case_id} "
                f"mode={args.ablation_mode} "
                f"selected_ratio={float(row_for_print.get('selected_highres_ratio', 0.0)):.3f} "
                f"recall_at_k={recall_text}"
            )
    except KeyboardInterrupt:
        summary = write_outputs(output_dir, rows) if rows else {"num_cases": 0.0}
        print("\ninfo=interrupted_partial_results_saved")
        print("summary=" + json.dumps(summary, ensure_ascii=False))
        return

    summary = write_outputs(output_dir, rows)
    print(f"batch_metrics_csv={output_dir / 'batch_metrics.csv'}")
    print(f"batch_metrics_json={output_dir / 'batch_metrics.json'}")
    print("summary=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
