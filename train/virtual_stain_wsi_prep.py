import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import SlideReader, openslide, save_image


@dataclass
class WSIPrepareConfig:
    dataset_dir: str = "data/private_dataset"
    prepared_dir: str = "outputs/train/prepared_patches"
    preview_max_side: int = 1536
    region_max_side: int = 768
    export_patch_size: int = 512
    grid_size: int = 6
    top_k_per_slide: int = 16
    min_tissue_ratio: float = 0.08
    preview_align_search_radius: int = 36
    preview_align_step: int = 12
    patch_align_search_radius: int = 20
    patch_align_step: int = 4
    patch_expand_ratio: float = 0.18
    pair_quality_threshold: float = 0.48
    min_pair_tissue_overlap: float = 0.12
    min_pair_center_overlap: float = 0.08
    max_pair_tissue_ratio_gap: float = 0.38
    min_pair_gray_correlation: float = 0.08
    overwrite_prepared: bool = False
    max_cases: int = 0


def prepare_paired_dataset(config: WSIPrepareConfig) -> Dict[str, object]:
    prepared_root = Path(config.prepared_dir)
    input_dir = prepared_root / "train_input"
    target_dir = prepared_root / "train_target"
    attention_dir = prepared_root / "train_attention"
    metadata_path = prepared_root / "metadata.json"

    if prepared_root.exists() and not config.overwrite_prepared:
        if has_prepared_pairs(input_dir, target_dir):
            return {
                "prepared_dir": str(prepared_root),
                "train_input_dir": str(input_dir),
                "train_target_dir": str(target_dir),
                "train_attention_dir": str(attention_dir),
                "metadata_path": str(metadata_path),
                "reused_existing": True,
            }

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(attention_dir, exist_ok=True)

    clear_directory(input_dir)
    clear_directory(target_dir)
    clear_directory(attention_dir)

    cases = discover_cases(config.dataset_dir)
    if cases and openslide is None:
        raise RuntimeError(
            "OpenSlide is not available in the current environment, so .svs files cannot be read. "
            "Please install libopenslide and openslide-python (or python-openslide) first."
        )
    if config.max_cases > 0:
        cases = cases[: config.max_cases]
    metadata_cases: List[Dict[str, object]] = []
    skipped_cases: List[Dict[str, str]] = []

    valid_case_count = 0
    for case_index, case in enumerate(cases, start=1):
        case_start = time.time()
        try:
            unstained_reader = SlideReader(case["unstained"], max_side=config.preview_max_side)
            he_reader = SlideReader(case["he"], max_side=config.preview_max_side)
            unstained_preview = unstained_reader.load_preview()
            he_preview = he_reader.load_preview()
        except Exception as exc:
            skipped_cases.append(
                {
                    "case_id": case["case_id"],
                    "unstained_path": case["unstained"],
                    "he_path": case["he"],
                    "reason": f"preview_open_failed: {exc}",
                }
            )
            print(
                f"warning=skip_case case_id={case['case_id']} "
                f"reason=preview_open_failed unstained={case['unstained']} he={case['he']} error={exc}"
            )
            continue
        coarse_dx, coarse_dy, coarse_score = normalize_translation_result(
            estimate_preview_translation(
                unstained_preview,
                he_preview,
                search_radius=config.preview_align_search_radius,
                step=config.preview_align_step,
            )
        )
        rel_regions = select_regions_from_preview(
            unstained_preview,
            grid_size=config.grid_size,
            top_k=config.top_k_per_slide,
            min_tissue_ratio=config.min_tissue_ratio,
        )
        exported, pair_failed, strict_stats = export_case_patches(
            case=case,
            unstained_reader=unstained_reader,
            he_reader=he_reader,
            rel_regions=rel_regions,
            coarse_dx=coarse_dx,
            coarse_dy=coarse_dy,
            coarse_score=coarse_score,
            preview_shape=unstained_preview.shape[:2],
            config=config,
            input_dir=input_dir,
            target_dir=target_dir,
            attention_dir=attention_dir,
        )

        if not pair_failed and not exported:
            relaxed_regions = select_regions_from_preview(
                unstained_preview,
                grid_size=max(8, config.grid_size),
                top_k=min(config.top_k_per_slide, 12),
                min_tissue_ratio=max(0.02, config.min_tissue_ratio * 0.5),
            )
            relaxed_config = WSIPrepareConfig(
                **{
                    **asdict(config),
                    "pair_quality_threshold": max(0.42, config.pair_quality_threshold - 0.10),
                    "min_pair_center_overlap": max(0.08, config.min_pair_center_overlap - 0.05),
                    "min_pair_gray_correlation": max(0.05, config.min_pair_gray_correlation - 0.10),
                    "preview_align_search_radius": max(24, config.preview_align_search_radius - 8),
                    "patch_align_search_radius": max(14, config.patch_align_search_radius - 6),
                    "patch_align_step": max(4, config.patch_align_step),
                }
            )
            exported, pair_failed, relaxed_stats = export_case_patches(
                case=case,
                unstained_reader=unstained_reader,
                he_reader=he_reader,
                rel_regions=relaxed_regions,
                coarse_dx=coarse_dx,
                coarse_dy=coarse_dy,
                coarse_score=coarse_score,
                preview_shape=unstained_preview.shape[:2],
                config=relaxed_config,
                input_dir=input_dir,
                target_dir=target_dir,
                attention_dir=attention_dir,
            )
            if exported:
                strict_stats = relaxed_stats
                strict_stats["mode"] = "relaxed"
        else:
            strict_stats["mode"] = "strict"

        if pair_failed or not exported:
            for item in exported:
                safe_unlink(Path(item["input_path"]))
                safe_unlink(Path(item["target_path"]))
                safe_unlink(Path(item["attention_path"]))
            skipped_cases.append(
                {
                    "case_id": case["case_id"],
                    "unstained_path": case["unstained"],
                    "he_path": case["he"],
                    "reason": "pair_unusable_after_patch_extraction" if pair_failed else "no_valid_patches_exported",
                }
            )
            print(
                f"warning=skip_case case_id={case['case_id']} "
                f"reason={'pair_unusable_after_patch_extraction' if pair_failed else 'no_valid_patches_exported'}"
            )
            continue

        valid_case_count += 1
        metadata_cases.append(
            {
                "case_id": case["case_id"],
                "unstained_path": case["unstained"],
                "he_path": case["he"],
                "preview_shape": list(unstained_preview.shape),
                "exported_patches": exported,
            }
        )
        print(
            f"case={case_index}/{len(cases)} id={case['case_id']} "
            f"mode={strict_stats.get('mode', 'strict')} "
            f"cand={strict_stats['candidate_regions']} tissue={strict_stats['tissue_pass_regions']} "
            f"align={strict_stats['alignment_pass_regions']} keep={len(exported)} "
            f"time={time.time() - case_start:.1f}s"
        )

    metadata = {
        "config": asdict(config),
        "num_cases": len(metadata_cases),
        "num_skipped_cases": len(skipped_cases),
        "cases": metadata_cases,
        "skipped_cases": skipped_cases,
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    total_exported = sum(len(case["exported_patches"]) for case in metadata_cases)
    print(
        f"prepared_cases={valid_case_count} "
        f"skipped_cases={len(skipped_cases)} "
        f"exported_patches={total_exported}"
    )
    if total_exported == 0:
        raise RuntimeError(
            "No valid training patches were exported. Check metadata.json and whether the .svs files are readable by openslide."
        )

    return {
        "prepared_dir": str(prepared_root),
        "train_input_dir": str(input_dir),
        "train_target_dir": str(target_dir),
        "train_attention_dir": str(attention_dir),
        "metadata_path": str(metadata_path),
        "reused_existing": False,
    }


def export_case_patches(
    case: Dict[str, str],
    unstained_reader: SlideReader,
    he_reader: SlideReader,
    rel_regions: List[Tuple[float, float, float, float]],
    coarse_dx: int,
    coarse_dy: int,
    coarse_score: float,
    preview_shape: Tuple[int, int],
    config: WSIPrepareConfig,
    input_dir: Path,
    target_dir: Path,
    attention_dir: Path,
) -> Tuple[List[Dict[str, object]], bool, Dict[str, int]]:
    exported: List[Dict[str, object]] = []
    pair_failed = False
    stats = {
        "candidate_regions": len(rel_regions),
        "loaded_regions": 0,
        "tissue_pass_regions": 0,
        "alignment_attempts": 0,
        "alignment_pass_regions": 0,
        "quality_pass_regions": 0,
    }
    for idx, rel in enumerate(rel_regions, start=1):
        projected_rel = project_rel_coordinate(
            rel,
            preview_shape=preview_shape,
            coarse_dx=coarse_dx,
            coarse_dy=coarse_dy,
        )
        try:
            unstained_patch = unstained_reader.load_region(rel)
            stats["loaded_regions"] += 1
            if patch_tissue_ratio(unstained_patch) < max(0.03, config.min_tissue_ratio * 0.75):
                continue
            stats["tissue_pass_regions"] += 1
            stats["alignment_attempts"] += 1
            he_patch, fine_dx, fine_dy, pair_quality, pair_metrics = extract_aligned_he_patch(
                he_reader=he_reader,
                unstained_patch=unstained_patch,
                projected_rel=projected_rel,
                expand_ratio=config.patch_expand_ratio,
                region_max_side=config.region_max_side,
                export_patch_size=config.export_patch_size,
                search_radius=config.patch_align_search_radius,
                step=config.patch_align_step,
                min_tissue_overlap=config.min_pair_tissue_overlap,
                min_center_overlap=config.min_pair_center_overlap,
                max_tissue_ratio_gap=config.max_pair_tissue_ratio_gap,
                min_gray_correlation=config.min_pair_gray_correlation,
            )
        except Exception as exc:
            pair_failed = True
            print(
                f"warning=skip_case case_id={case['case_id']} patch_index={idx} "
                f"reason=region_extract_failed error={exc}"
            )
            break
        stats["alignment_pass_regions"] += 1
        unstained_patch = resize_square(unstained_patch, config.export_patch_size)
        if pair_quality < config.pair_quality_threshold:
            continue
        stats["quality_pass_regions"] += 1
        attention = build_attention_map(unstained_patch)

        patch_id = f"{case['case_id']}_{idx:02d}"
        input_path = input_dir / f"{patch_id}_unstained.png"
        target_path = target_dir / f"{patch_id}_he.png"
        attention_path = attention_dir / f"{patch_id}_attention.png"
        save_image(unstained_patch, str(input_path))
        save_image(he_patch, str(target_path))
        save_image(np.repeat((attention * 255.0).round().astype(np.uint8)[..., None], 3, axis=2), str(attention_path))
        exported.append(
            {
                "patch_id": patch_id,
                "relative_coordinate": [float(v) for v in rel],
                "projected_he_relative_coordinate": [float(v) for v in projected_rel],
                "coarse_preview_shift_xy": [int(coarse_dx), int(coarse_dy)],
                "coarse_preview_score": float(coarse_score),
                "fine_patch_shift_xy": [int(fine_dx), int(fine_dy)],
                "pair_quality": float(pair_quality),
                "pair_metrics": pair_metrics,
                "input_path": str(input_path),
                "target_path": str(target_path),
                "attention_path": str(attention_path),
            }
        )
    return exported, pair_failed, stats


def discover_cases(dataset_dir: str) -> List[Dict[str, str]]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    cases: List[Dict[str, str]] = []
    for case_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        case_id = case_dir.name
        unstained = case_dir / f"{case_id}_unstained.svs"
        he = case_dir / f"{case_id}_he.svs"
        if unstained.exists() and he.exists():
            cases.append({"case_id": case_id, "unstained": str(unstained), "he": str(he)})
    if not cases:
        raise RuntimeError("No paired *_unstained.svs / *_he.svs cases found.")
    print(f"discovered_cases={len(cases)} dataset_dir={dataset_dir}")
    return cases


def select_regions_from_preview(
    preview_rgb: np.ndarray,
    grid_size: int,
    top_k: int,
    min_tissue_ratio: float,
) -> List[Tuple[float, float, float, float]]:
    h, w = preview_rgb.shape[:2]
    tile_h = max(1, h // grid_size)
    tile_w = max(1, w // grid_size)
    scored: List[Tuple[float, Tuple[float, float, float, float]]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * tile_h
            x1 = col * tile_w
            y2 = h if row == grid_size - 1 else min(h, (row + 1) * tile_h)
            x2 = w if col == grid_size - 1 else min(w, (col + 1) * tile_w)
            patch = preview_rgb[y1:y2, x1:x2]
            score, tissue_ratio = score_patch(patch)
            if tissue_ratio < min_tissue_ratio:
                continue
            scored.append((score, (x1 / w, y1 / h, x2 / w, y2 / h)))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored[:top_k]]
    if not selected and scored:
        selected = [scored[0][1]]
    return selected


def score_patch(patch: np.ndarray) -> Tuple[float, float]:
    gray = patch.astype(np.float32).mean(axis=2)
    tissue = gray < 240.0
    tissue_ratio = float(tissue.mean())
    if not np.any(tissue):
        return 0.0, tissue_ratio
    h, w = gray.shape
    cy1 = int(h * 0.25)
    cy2 = int(h * 0.75)
    cx1 = int(w * 0.25)
    cx2 = int(w * 0.75)
    center_gray = gray[cy1:cy2, cx1:cx2]
    center_tissue = center_gray < 240.0
    darkness = float(np.clip((210.0 - gray[tissue].mean()) / 120.0, 0.0, 1.0))
    blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
    detail = float(np.clip(np.abs(gray - blur)[tissue].mean() / 28.0, 0.0, 1.0))
    contrast = float(np.clip(gray[tissue].std() / 55.0, 0.0, 1.0))
    if np.any(center_tissue):
        center_blur = np.array(
            Image.fromarray(np.clip(center_gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2)),
            dtype=np.float32,
        )
        center_darkness = float(np.clip((212.0 - center_gray[center_tissue].mean()) / 112.0, 0.0, 1.0))
        center_detail = float(np.clip(np.abs(center_gray - center_blur)[center_tissue].mean() / 24.0, 0.0, 1.0))
        center_tissue_ratio = float(center_tissue.mean())
    else:
        center_darkness = 0.0
        center_detail = 0.0
        center_tissue_ratio = 0.0
    score = (
        0.28 * darkness
        + 0.28 * detail
        + 0.14 * contrast
        + 0.15 * center_darkness
        + 0.10 * center_detail
        + 0.05 * center_tissue_ratio
    )
    return score, tissue_ratio


def estimate_preview_translation(
    unstained_preview: np.ndarray,
    he_preview: np.ndarray,
    search_radius: int,
    step: int,
) -> Tuple[int, int, float]:
    src = to_structure_map(unstained_preview)
    tgt = to_structure_map(he_preview)
    return normalize_translation_result(
        multi_scale_translation_search(src, tgt, search_radius=search_radius, base_step=step)
    )


def normalize_translation_result(result: object) -> Tuple[int, int, float]:
    if isinstance(result, tuple):
        if len(result) >= 3:
            return int(result[0]), int(result[1]), float(result[2])
        if len(result) == 2:
            return int(result[0]), int(result[1]), 0.0
        if len(result) == 1:
            return int(result[0]), 0, 0.0
    if isinstance(result, list):
        if len(result) >= 3:
            return int(result[0]), int(result[1]), float(result[2])
        if len(result) == 2:
            return int(result[0]), int(result[1]), 0.0
        if len(result) == 1:
            return int(result[0]), 0, 0.0
    if isinstance(result, (int, float)):
        return int(result), 0, 0.0
    return 0, 0, 0.0


def project_rel_coordinate(
    rel: Tuple[float, float, float, float],
    preview_shape: Tuple[int, int],
    coarse_dx: int,
    coarse_dy: int,
) -> Tuple[float, float, float, float]:
    h, w = preview_shape
    x1 = rel[0] * w + coarse_dx
    y1 = rel[1] * h + coarse_dy
    x2 = rel[2] * w + coarse_dx
    y2 = rel[3] * h + coarse_dy
    x1 = np.clip(x1 / w, 0.0, 1.0)
    y1 = np.clip(y1 / h, 0.0, 1.0)
    x2 = np.clip(x2 / w, x1 + 1.0 / max(w, 2), 1.0)
    y2 = np.clip(y2 / h, y1 + 1.0 / max(h, 2), 1.0)
    return float(x1), float(y1), float(x2), float(y2)


def extract_aligned_he_patch(
    he_reader: SlideReader,
    unstained_patch: np.ndarray,
    projected_rel: Tuple[float, float, float, float],
    expand_ratio: float,
    region_max_side: int,
    export_patch_size: int,
    search_radius: int,
    step: int,
    min_tissue_overlap: float,
    min_center_overlap: float,
    max_tissue_ratio_gap: float,
    min_gray_correlation: float,
) -> Tuple[np.ndarray, int, int, float, Dict[str, float]]:
    expanded_rel = expand_rel_coordinate(projected_rel, expand_ratio)
    expanded_patch = he_reader.load_region(expanded_rel)
    expanded_patch = resize_square(expanded_patch, export_patch_size + 4 * search_radius)
    unstained_patch = resize_square(unstained_patch, export_patch_size)
    aligned_patch, dx, dy, quality, metrics = align_patch_with_local_search(
        source_patch=unstained_patch,
        target_expanded_patch=expanded_patch,
        crop_size=export_patch_size,
        search_radius=2 * search_radius,
        step=step,
        min_tissue_overlap=min_tissue_overlap,
        min_center_overlap=min_center_overlap,
        max_tissue_ratio_gap=max_tissue_ratio_gap,
        min_gray_correlation=min_gray_correlation,
    )
    return aligned_patch, dx, dy, quality, metrics


def expand_rel_coordinate(rel: Tuple[float, float, float, float], expand_ratio: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = rel
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * (1.0 + 2.0 * expand_ratio)
    h = (y2 - y1) * (1.0 + 2.0 * expand_ratio)
    nx1 = max(0.0, cx - 0.5 * w)
    ny1 = max(0.0, cy - 0.5 * h)
    nx2 = min(1.0, cx + 0.5 * w)
    ny2 = min(1.0, cy + 0.5 * h)
    return float(nx1), float(ny1), float(nx2), float(ny2)


def align_patch_with_local_search(
    source_patch: np.ndarray,
    target_expanded_patch: np.ndarray,
    crop_size: int,
    search_radius: int,
    step: int,
    min_tissue_overlap: float,
    min_center_overlap: float,
    max_tissue_ratio_gap: float,
    min_gray_correlation: float,
) -> Tuple[np.ndarray, int, int, float, Dict[str, float]]:
    src_map = to_structure_map(source_patch)
    tgt_map = to_structure_map(target_expanded_patch)
    center = search_radius
    best_dx = 0
    best_dy = 0
    best_score = -1.0
    best_crop = crop_center(target_expanded_patch, crop_size, center, center)
    best_metrics: Dict[str, float] = {}
    search_steps = sorted({max(step, 1), max(step * 2, 2)}, reverse=True)
    current_dx = 0
    current_dy = 0
    for local_step in search_steps:
        local_radius = max(local_step * 2, search_radius if local_step == search_steps[0] else local_step * 2)
        dy_values = range(max(-search_radius, current_dy - local_radius), min(search_radius, current_dy + local_radius) + 1, local_step)
        dx_values = range(max(-search_radius, current_dx - local_radius), min(search_radius, current_dx + local_radius) + 1, local_step)
        for dy in dy_values:
            for dx in dx_values:
                crop_rgb = crop_center(target_expanded_patch, crop_size, center + dx, center + dy)
                crop_map = crop_center(tgt_map, crop_size, center + dx, center + dy)
                if crop_map.shape[0] != crop_size or crop_map.shape[1] != crop_size:
                    continue
                score, metrics = local_patch_similarity(
                    src_map,
                    crop_map,
                    source_rgb=source_patch,
                    target_rgb=crop_rgb,
                    min_tissue_overlap=min_tissue_overlap,
                    min_center_overlap=min_center_overlap,
                    max_tissue_ratio_gap=max_tissue_ratio_gap,
                    min_gray_correlation=min_gray_correlation,
                )
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy
                    current_dx = dx
                    current_dy = dy
                    best_crop = crop_rgb
                    best_metrics = metrics
    if not best_metrics:
        best_metrics = {
            "structure_similarity": 0.0,
            "edge_similarity": 0.0,
            "gray_correlation": 0.0,
            "tissue_overlap": 0.0,
            "center_overlap": 0.0,
            "tissue_ratio_gap": 1.0,
        }
    return best_crop, int(best_dx), int(best_dy), float(best_score), best_metrics


def crop_center(arr: np.ndarray, crop_size: int, offset_x: int, offset_y: int) -> np.ndarray:
    h, w = arr.shape[:2]
    x1 = int(np.clip(offset_x, 0, max(0, w - crop_size)))
    y1 = int(np.clip(offset_y, 0, max(0, h - crop_size)))
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    return arr[y1:y2, x1:x2]


def to_structure_map(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.float32).mean(axis=2)
    blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
    detail = normalize_map(np.abs(gray - blur))
    gray_norm = 1.0 - normalize_map(gray)
    tissue = (gray < 245.0).astype(np.float32)
    return np.clip((0.55 * gray_norm + 0.45 * detail) * tissue, 0.0, 1.0).astype(np.float32)


def multi_scale_translation_search(
    src: np.ndarray,
    tgt: np.ndarray,
    search_radius: int,
    base_step: int,
) -> Tuple[int, int, float]:
    scales = [0.5, 1.0]
    best_dx = 0
    best_dy = 0
    best_score = -1.0
    for idx, scale in enumerate(scales):
        src_scaled = resize_map(src, scale)
        tgt_scaled = resize_map(tgt, scale)
        local_radius = max(4, int(round(search_radius * scale)))
        local_step = max(1, int(round(base_step * scale)))
        if idx == 0:
            dx_range = range(-local_radius, local_radius + 1, local_step)
            dy_range = range(-local_radius, local_radius + 1, local_step)
        else:
            seed_dx = int(round(best_dx / scales[idx - 1] * scale))
            seed_dy = int(round(best_dy / scales[idx - 1] * scale))
            dx_range = range(seed_dx - local_step * 2, seed_dx + local_step * 2 + 1, local_step)
            dy_range = range(seed_dy - local_step * 2, seed_dy + local_step * 2 + 1, local_step)
        for dy in dy_range:
            for dx in dx_range:
                score = shifted_similarity(src_scaled, tgt_scaled, dx, dy)
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy
    return int(best_dx), int(best_dy), float(best_score)


def resize_map(arr: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return arr
    h, w = arr.shape[:2]
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))
    image = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    return np.array(image.resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def shifted_similarity(src: np.ndarray, tgt: np.ndarray, dx: int, dy: int) -> float:
    h, w = src.shape
    sx1 = max(0, -dx)
    sy1 = max(0, -dy)
    tx1 = max(0, dx)
    ty1 = max(0, dy)
    width = min(w - sx1, w - tx1)
    height = min(h - sy1, h - ty1)
    if width <= 8 or height <= 8:
        return -1.0
    src_crop = src[sy1:sy1 + height, sx1:sx1 + width]
    tgt_crop = tgt[ty1:ty1 + height, tx1:tx1 + width]
    score, _ = local_patch_similarity(src_crop, tgt_crop)
    return float(score)


def patch_tissue_ratio(rgb: np.ndarray) -> float:
    gray = rgb.astype(np.float32).mean(axis=2)
    return float((gray < 242.0).mean())


def local_patch_similarity(
    src: np.ndarray,
    tgt: np.ndarray,
    source_rgb: Optional[np.ndarray] = None,
    target_rgb: Optional[np.ndarray] = None,
    min_tissue_overlap: float = 0.08,
    min_center_overlap: float = 0.06,
    max_tissue_ratio_gap: float = 0.40,
    min_gray_correlation: float = -1.0,
) -> Tuple[float, Dict[str, float]]:
    if src.shape != tgt.shape:
        return -1.0, {}
    src_tissue = src > 0.05
    tgt_tissue = tgt > 0.05
    overlap = (src_tissue & tgt_tissue).astype(np.float32)
    overlap_ratio = float(overlap.mean())
    h, w = src.shape
    cy1 = int(h * 0.25)
    cy2 = int(h * 0.75)
    cx1 = int(w * 0.25)
    cx2 = int(w * 0.75)
    center_overlap = float(overlap[cy1:cy2, cx1:cx2].mean()) if cy2 > cy1 and cx2 > cx1 else overlap_ratio
    if overlap_ratio < min_tissue_overlap:
        return -1.0, {
            "structure_similarity": 0.0,
            "edge_similarity": 0.0,
            "gray_correlation": 0.0,
            "tissue_overlap": overlap_ratio,
            "center_overlap": center_overlap,
            "tissue_ratio_gap": 1.0,
        }
    if center_overlap < min_center_overlap:
        return -1.0, {
            "structure_similarity": 0.0,
            "edge_similarity": 0.0,
            "gray_correlation": 0.0,
            "tissue_overlap": overlap_ratio,
            "center_overlap": center_overlap,
            "tissue_ratio_gap": 1.0,
        }
    src_tissue_ratio = float(src_tissue.mean())
    tgt_tissue_ratio = float(tgt_tissue.mean())
    tissue_ratio_gap = abs(src_tissue_ratio - tgt_tissue_ratio)
    if tissue_ratio_gap > max_tissue_ratio_gap:
        return -1.0, {
            "structure_similarity": 0.0,
            "edge_similarity": 0.0,
            "gray_correlation": 0.0,
            "tissue_overlap": overlap_ratio,
            "center_overlap": center_overlap,
            "tissue_ratio_gap": tissue_ratio_gap,
        }
    diff = np.abs(src - tgt)
    mad = float((diff * overlap).sum() / (overlap.sum() + 1e-6))
    structure_corr = float(np.mean((src * tgt) * overlap) / (np.sqrt(np.mean((src * src) * overlap) * np.mean((tgt * tgt) * overlap)) + 1e-6))
    center_src = src[cy1:cy2, cx1:cx2]
    center_tgt = tgt[cy1:cy2, cx1:cx2]
    center_mask = overlap[cy1:cy2, cx1:cx2]
    if center_mask.size > 0 and center_mask.mean() > 0:
        center_structure = float(
            np.mean((center_src * center_tgt) * center_mask)
            / (
                np.sqrt(
                    np.mean((center_src * center_src) * center_mask)
                    * np.mean((center_tgt * center_tgt) * center_mask)
                )
                + 1e-6
            )
        )
    else:
        center_structure = 0.0

    edge_similarity = 0.0
    gray_corr = 0.0
    if source_rgb is not None and target_rgb is not None:
        src_gray = np.mean(source_rgb.astype(np.float32), axis=2)
        tgt_gray = np.mean(target_rgb.astype(np.float32), axis=2)
        src_edge = normalize_map(np.abs(src_gray - gaussian_blur_map(src_gray, radius=2)))
        tgt_edge = normalize_map(np.abs(tgt_gray - gaussian_blur_map(tgt_gray, radius=2)))
        edge_similarity = float(np.mean((src_edge * tgt_edge) * overlap) / (np.sqrt(np.mean((src_edge * src_edge) * overlap) * np.mean((tgt_edge * tgt_edge) * overlap)) + 1e-6))
        gray_corr = correlation_coefficient(src_gray, tgt_gray, mask=overlap > 0)
        if gray_corr < min_gray_correlation:
            return -1.0, {
                "structure_similarity": float(np.clip(structure_corr, 0.0, 1.0)),
                "edge_similarity": float(np.clip(edge_similarity, 0.0, 1.0)),
                "gray_correlation": float(gray_corr),
                "tissue_overlap": overlap_ratio,
                "center_overlap": center_overlap,
                "tissue_ratio_gap": tissue_ratio_gap,
            }

    score = (
        0.28 * (1.0 - np.clip(mad, 0.0, 1.0))
        + 0.20 * np.clip(structure_corr, 0.0, 1.0)
        + 0.18 * np.clip(center_structure, 0.0, 1.0)
        + 0.12 * np.clip(edge_similarity, 0.0, 1.0)
        + 0.10 * overlap_ratio
        + 0.07 * np.clip(gray_corr, 0.0, 1.0)
        + 0.05 * center_overlap
    )
    metrics = {
        "structure_similarity": float(np.clip(structure_corr, 0.0, 1.0)),
        "center_structure_similarity": float(np.clip(center_structure, 0.0, 1.0)),
        "edge_similarity": float(np.clip(edge_similarity, 0.0, 1.0)),
        "gray_correlation": float(gray_corr),
        "tissue_overlap": overlap_ratio,
        "center_overlap": center_overlap,
        "tissue_ratio_gap": tissue_ratio_gap,
    }
    return float(np.clip(score, 0.0, 1.0)), metrics


def build_attention_map(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.float32).mean(axis=2)
    tissue = (gray < 248.0).astype(np.float32)
    dark = np.clip((232.0 - gray) / 92.0, 0.0, 1.0)
    blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
    detail = normalize_map(np.abs(gray - blur))
    local_contrast = normalize_map(np.abs(gray - gaussian_blur_map(gray, radius=6)))
    base_attention = np.clip(0.45 * dark + 0.35 * detail + 0.20 * local_contrast, 0.0, 1.0) * tissue
    tissue_values = base_attention[tissue > 0]
    dynamic_threshold = float(np.percentile(tissue_values, 72.0)) if tissue_values.size > 16 else 0.30
    base_attention[base_attention < max(0.30, dynamic_threshold)] = 0.0
    attention = np.array(
        Image.fromarray((base_attention * 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4)),
        dtype=np.float32,
    ) / 255.0
    attention = np.clip(attention * tissue, 0.0, 1.0)
    attention[attention < 0.16] = 0.0
    return np.clip(attention, 0.0, 1.0)


def gaussian_blur_map(channel: np.ndarray, radius: float) -> np.ndarray:
    return np.array(Image.fromarray(np.clip(channel, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)


def correlation_coefficient(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 16:
        return 0.0
    av = a[mask].astype(np.float32)
    bv = b[mask].astype(np.float32)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = float(np.sqrt((av * av).mean() * (bv * bv).mean()) + 1e-6)
    return float((av * bv).mean() / denom)


def normalize_map(channel: np.ndarray) -> np.ndarray:
    low = np.percentile(channel, 1.0)
    high = np.percentile(channel, 99.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)


def resize_square(rgb: np.ndarray, patch_size: int) -> np.ndarray:
    return np.array(Image.fromarray(rgb).resize((patch_size, patch_size), Image.Resampling.LANCZOS))


def clear_directory(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def has_prepared_pairs(input_dir: Path, target_dir: Path) -> bool:
    if not input_dir.exists() or not target_dir.exists():
        return False
    input_count = sum(1 for path in input_dir.iterdir() if path.is_file())
    target_count = sum(1 for path in target_dir.iterdir() if path.is_file())
    return input_count > 0 and target_count > 0


__all__ = [
    "WSIPrepareConfig",
    "prepare_paired_dataset",
]
