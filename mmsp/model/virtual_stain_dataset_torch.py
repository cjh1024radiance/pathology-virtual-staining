import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class VirtualStainDatasetConfig:
    patch_size: int = 256
    random_crop: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = True
    normalize_to_unit: bool = True
    use_generated_attention_if_missing: bool = True


def list_image_files(directory: str) -> List[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])


def normalize_stem(stem: str) -> str:
    normalized = stem.lower()
    suffixes = [
        "_unstained",
        "_he",
        "_virtual_he",
        "_attention",
        "_mask",
        "_target",
        "_source",
        "_input",
        "_label",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
    return normalized


def build_pair_map(input_dir: str, target_dir: str, attention_dir: Optional[str] = None) -> List[Dict[str, Optional[str]]]:
    input_files = list_image_files(input_dir)
    target_files = list_image_files(target_dir)
    attention_lookup: Dict[str, str] = {}
    if attention_dir:
        attention_lookup = {normalize_stem(path.stem): str(path) for path in list_image_files(attention_dir)}

    target_lookup = {normalize_stem(path.stem): str(path) for path in target_files}
    pairs: List[Dict[str, Optional[str]]] = []
    missing: List[str] = []
    for input_path in input_files:
        key = normalize_stem(input_path.stem)
        target_path = target_lookup.get(key)
        if target_path is None:
            missing.append(input_path.name)
            continue
        pairs.append(
            {
                "id": key,
                "case_id": extract_case_id(key),
                "input_path": str(input_path),
                "target_path": target_path,
                "attention_path": attention_lookup.get(key),
            }
        )
    if not pairs:
        raise RuntimeError("No paired images found between input_dir and target_dir.")
    return pairs


def extract_case_id(identifier: str) -> str:
    match = re.match(r"^(.*)_([0-9]+)$", identifier)
    if match:
        return match.group(1)
    return identifier


def split_pairs_by_case(
    pairs: Sequence[Dict[str, Optional[str]]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    items = list(pairs)
    case_to_pairs: Dict[str, List[Dict[str, Optional[str]]]] = {}
    for item in items:
        case_id = str(item.get("case_id") or extract_case_id(str(item["id"])))
        case_to_pairs.setdefault(case_id, []).append(item)

    case_ids = list(case_to_pairs.keys())
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    num_cases = len(case_ids)
    if num_cases == 1:
        only = case_to_pairs[case_ids[0]]
        return only, [], []

    train_count = max(1, int(round(num_cases * train_ratio)))
    val_count = int(round(num_cases * val_ratio))
    test_count = num_cases - train_count - val_count

    if num_cases >= 3:
        val_count = max(1, val_count)
        test_count = max(1, test_count)
        train_count = max(1, num_cases - val_count - test_count)

    while train_count + val_count + test_count > num_cases:
        if train_count >= val_count and train_count >= test_count and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    train_case_ids = set(case_ids[:train_count])
    val_case_ids = set(case_ids[train_count : train_count + val_count])
    test_case_ids = set(case_ids[train_count + val_count : train_count + val_count + test_count])

    train_pairs = [item for case_id in train_case_ids for item in case_to_pairs[case_id]]
    val_pairs = [item for case_id in val_case_ids for item in case_to_pairs[case_id]]
    test_pairs = [item for case_id in test_case_ids for item in case_to_pairs[case_id]]
    return train_pairs, val_pairs, test_pairs


class PairedVirtualStainDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Dict[str, Optional[str]]],
        config: Optional[VirtualStainDatasetConfig] = None,
    ) -> None:
        self.pairs = list(pairs)
        self.config = config or VirtualStainDatasetConfig()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        item = self.pairs[index]
        input_rgb = self._load_rgb(item["input_path"])
        target_rgb = self._load_rgb(item["target_path"])
        attention = self._load_attention(item.get("attention_path"), input_rgb)

        input_rgb, target_rgb, attention = self._joint_resize(input_rgb, target_rgb, attention)
        input_rgb, target_rgb, attention = self._joint_crop(input_rgb, target_rgb, attention)
        input_rgb, target_rgb, attention = self._joint_augment(input_rgb, target_rgb, attention)

        tissue_map = self._build_tissue_map(input_rgb)
        highres_mask = (attention > 0.45).astype(np.float32)
        lesion_confidence = attention.copy()

        return {
            "id": str(item["id"]),
            "input_rgb": self._to_tensor(input_rgb),
            "target_rgb": self._to_tensor(target_rgb),
            "attention_map": self._to_tensor(attention[..., None]),
            "highres_mask": self._to_tensor(highres_mask[..., None]),
            "tissue_map": self._to_tensor(tissue_map[..., None]),
            "lesion_confidence": self._to_tensor(lesion_confidence[..., None]),
        }

    def _load_rgb(self, path: Optional[str]) -> np.ndarray:
        if not path:
            raise ValueError("RGB path is missing.")
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))

    def _load_attention(self, path: Optional[str], input_rgb: np.ndarray) -> np.ndarray:
        if path:
            with Image.open(path) as image:
                arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
                return np.clip(arr, 0.0, 1.0)
        if not self.config.use_generated_attention_if_missing:
            return np.ones(input_rgb.shape[:2], dtype=np.float32)
        return self._generate_attention(input_rgb)

    def _joint_resize(self, input_rgb: np.ndarray, target_rgb: np.ndarray, attention: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        patch = self.config.patch_size
        min_h = min(input_rgb.shape[0], target_rgb.shape[0], attention.shape[0])
        min_w = min(input_rgb.shape[1], target_rgb.shape[1], attention.shape[1])
        input_rgb = input_rgb[:min_h, :min_w]
        target_rgb = target_rgb[:min_h, :min_w]
        attention = attention[:min_h, :min_w]

        if min(min_h, min_w) >= patch:
            return input_rgb, target_rgb, attention

        scale = patch / float(min(min_h, min_w))
        new_w = max(patch, int(round(min_w * scale)))
        new_h = max(patch, int(round(min_h * scale)))
        input_rgb = np.array(Image.fromarray(input_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS))
        target_rgb = np.array(Image.fromarray(target_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS))
        attention = np.array(Image.fromarray((attention * 255).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        return input_rgb, target_rgb, attention

    def _joint_crop(self, input_rgb: np.ndarray, target_rgb: np.ndarray, attention: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        patch = self.config.patch_size
        h, w = input_rgb.shape[:2]
        if h == patch and w == patch:
            return input_rgb, target_rgb, attention
        if self.config.random_crop:
            top = random.randint(0, h - patch)
            left = random.randint(0, w - patch)
        else:
            top = max(0, (h - patch) // 2)
            left = max(0, (w - patch) // 2)
        bottom = top + patch
        right = left + patch
        return input_rgb[top:bottom, left:right], target_rgb[top:bottom, left:right], attention[top:bottom, left:right]

    def _joint_augment(self, input_rgb: np.ndarray, target_rgb: np.ndarray, attention: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.horizontal_flip and random.random() < 0.5:
            input_rgb = np.ascontiguousarray(np.flip(input_rgb, axis=1))
            target_rgb = np.ascontiguousarray(np.flip(target_rgb, axis=1))
            attention = np.ascontiguousarray(np.flip(attention, axis=1))
        if self.config.vertical_flip and random.random() < 0.5:
            input_rgb = np.ascontiguousarray(np.flip(input_rgb, axis=0))
            target_rgb = np.ascontiguousarray(np.flip(target_rgb, axis=0))
            attention = np.ascontiguousarray(np.flip(attention, axis=0))
        return input_rgb, target_rgb, attention

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array.transpose(2, 0, 1) if array.ndim == 3 else array[None, ...]).float()
        if self.config.normalize_to_unit:
            tensor = tensor / 255.0 if tensor.max() > 1.0 else tensor
        return tensor.contiguous()

    @staticmethod
    def _build_tissue_map(rgb: np.ndarray) -> np.ndarray:
        gray = rgb.astype(np.float32).mean(axis=2)
        return (gray < 0.95 * 255.0).astype(np.float32)

    @staticmethod
    def _generate_attention(rgb: np.ndarray) -> np.ndarray:
        gray = rgb.astype(np.float32).mean(axis=2)
        gray_norm = 1.0 - normalize_map(gray)
        blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
        detail = normalize_map(np.abs(gray - blur))
        tissue = (gray < 248.0).astype(np.float32)
        attention = np.clip(0.65 * gray_norm + 0.35 * detail, 0.0, 1.0) * tissue
        return attention.astype(np.float32)


def normalize_map(channel: np.ndarray) -> np.ndarray:
    low = np.percentile(channel, 1.0)
    high = np.percentile(channel, 99.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)


__all__ = [
    "VirtualStainDatasetConfig",
    "PairedVirtualStainDataset",
    "build_pair_map",
    "extract_case_id",
    "split_pairs_by_case",
]
