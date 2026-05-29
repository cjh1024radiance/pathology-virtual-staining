import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import openslide  # type: ignore
except ImportError:  # pragma: no cover
    openslide = None


def load_medical_config() -> Dict[str, object]:
    config_path = Path(__file__).parent / "config" / "config.json"
    config: Dict[str, object] = {}
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
                if isinstance(loaded, dict):
                    config.update(loaded)
    except Exception:
        config = {}

    env_overrides = {
        "llm_api_url": os.getenv("PATHOLOGY_LLM_API_URL", "").strip(),
        "llm_api_key": os.getenv("PATHOLOGY_LLM_API_KEY", "").strip(),
        "vlm_model": os.getenv("PATHOLOGY_VLM_MODEL", "").strip(),
        "api_prompt_hint": os.getenv("PATHOLOGY_API_PROMPT_HINT", "").strip(),
        "virtual_stain_model_path": os.getenv("PATHOLOGY_VIRTUAL_STAIN_MODEL_PATH", "").strip(),
        "virtual_stain_model_device": os.getenv("PATHOLOGY_VIRTUAL_STAIN_MODEL_DEVICE", "").strip(),
    }
    for key, value in env_overrides.items():
        if value:
            config[key] = value
    return config


def save_image(rgb: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def load_rgb_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"))


class SlideReader:
    def __init__(self, slide_path: str, max_side: int = 2048) -> None:
        self.slide_path = slide_path
        self.max_side = max_side
        self.is_wsi = Path(slide_path).suffix.lower() in {".svs", ".tif", ".tiff"}

    def load_preview(self) -> np.ndarray:
        if self.is_wsi:
            return self._load_wsi_preview()
        return self._resize_long_side(load_rgb_image(self.slide_path), self.max_side)

    def load_region(self, rel_coordinate: Sequence[float]) -> np.ndarray:
        if self.is_wsi:
            return self._load_wsi_region(rel_coordinate)
        rgb = load_rgb_image(self.slide_path)
        h, w = rgb.shape[:2]
        x1 = int(rel_coordinate[0] * w)
        y1 = int(rel_coordinate[1] * h)
        x2 = max(x1 + 1, int(rel_coordinate[2] * w))
        y2 = max(y1 + 1, int(rel_coordinate[3] * h))
        return self._resize_long_side(rgb[y1:y2, x1:x2], self.max_side)

    def _load_wsi_preview(self) -> np.ndarray:
        if openslide is None:
            raise ImportError("openslide is required to read .svs previews.")
        try:
            slide = openslide.OpenSlide(self.slide_path)
            try:
                width0, height0 = slide.level_dimensions[0]
                target_downsample = max(width0, height0) / float(self.max_side)
                level = 0
                for idx, downsample in enumerate(slide.level_downsamples):
                    if float(downsample) <= target_downsample:
                        level = idx
                    else:
                        break
                image = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
                return self._resize_long_side(np.array(image), self.max_side)
            finally:
                slide.close()
        except Exception as exc:
            fallback = self._load_pil_preview()
            if fallback is not None:
                return fallback
            raise exc

    def _load_wsi_region(self, rel_coordinate: Sequence[float]) -> np.ndarray:
        if openslide is None:
            raise ImportError("openslide is required to read .svs regions.")
        try:
            slide = openslide.OpenSlide(self.slide_path)
            try:
                width0, height0 = slide.level_dimensions[0]
                x1 = int(rel_coordinate[0] * width0)
                y1 = int(rel_coordinate[1] * height0)
                x2 = int(rel_coordinate[2] * width0)
                y2 = int(rel_coordinate[3] * height0)
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                target_downsample = max(width, height) / float(self.max_side)
                level = 0
                for idx, downsample in enumerate(slide.level_downsamples):
                    if float(downsample) <= target_downsample:
                        level = idx
                    else:
                        break
                downsample = float(slide.level_downsamples[level])
                read_w = max(1, int(round(width / downsample)))
                read_h = max(1, int(round(height / downsample)))
                image = slide.read_region((x1, y1), level, (read_w, read_h)).convert("RGB")
                return self._resize_long_side(np.array(image), self.max_side)
            finally:
                slide.close()
        except Exception as exc:
            fallback = self._load_pil_region(rel_coordinate)
            if fallback is not None:
                return fallback
            raise exc

    def _load_pil_preview(self) -> np.ndarray | None:
        try:
            with Image.open(self.slide_path) as image:
                rgb = np.array(image.convert("RGB"))
            return self._resize_long_side(rgb, self.max_side)
        except Exception:
            return None

    def _load_pil_region(self, rel_coordinate: Sequence[float]) -> np.ndarray | None:
        try:
            with Image.open(self.slide_path) as image:
                width, height = image.size
                if max(width, height) > 12000:
                    return None
                rgb = np.array(image.convert("RGB"))
        except Exception:
            return None
        h, w = rgb.shape[:2]
        x1 = int(rel_coordinate[0] * w)
        y1 = int(rel_coordinate[1] * h)
        x2 = max(x1 + 1, int(rel_coordinate[2] * w))
        y2 = max(y1 + 1, int(rel_coordinate[3] * h))
        return self._resize_long_side(rgb[y1:y2, x1:x2], self.max_side)

    @staticmethod
    def _resize_long_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
        h, w = rgb.shape[:2]
        long_side = max(h, w)
        if long_side <= max_side:
            return rgb
        scale = max_side / float(long_side)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return np.array(Image.fromarray(rgb).resize(new_size, Image.Resampling.LANCZOS))


class SimpleStainTransfer:
    def __init__(self) -> None:
        matrix = np.array(
            [
                [0.650, 0.072, 0.268],
                [0.704, 0.990, 0.570],
                [0.286, 0.105, 0.776],
            ],
            dtype=np.float32,
        )
        matrix = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-6)
        self.stain_matrix_inv_t = np.linalg.inv(matrix.T)

    def deconvolve_he(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rgb = np.clip(rgb.astype(np.float32), 1.0, 255.0)
        od = -np.log(rgb / 255.0)
        flat = od.reshape(-1, 3)
        concentrations = flat @ self.stain_matrix_inv_t
        concentrations = np.clip(concentrations, 0.0, None)
        return concentrations[:, 0].reshape(rgb.shape[:2]), concentrations[:, 1].reshape(rgb.shape[:2])

    @staticmethod
    def normalize_channel(channel: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
        low = np.percentile(channel, low_q)
        high = np.percentile(channel, high_q)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.zeros_like(channel, dtype=np.float32)
        return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)

    def transform(self, rgb: np.ndarray) -> np.ndarray:
        hematoxylin, eosin = self.deconvolve_he(rgb)
        nucleus = self.normalize_channel(hematoxylin)
        eos = self.normalize_channel(eosin)
        background = np.array([248.0, 244.0, 246.0], dtype=np.float32)
        ht = np.array([88.0, 72.0, 164.0], dtype=np.float32)
        et = np.array([244.0, 164.0, 182.0], dtype=np.float32)
        visual = background[None, None, :].copy()
        visual = visual * (1.0 - 0.55 * nucleus[..., None])
        visual += ht[None, None, :] * (0.75 * nucleus[..., None])
        visual += et[None, None, :] * (0.35 * eos[..., None])
        return np.clip(visual, 0, 255).astype(np.uint8)


class UnpairedVirtualHEBaseline:
    def __init__(self) -> None:
        self.tissue_color = np.array([250.0, 238.0, 242.0], dtype=np.float32)
        self.eosin_color = np.array([224.0, 170.0, 186.0], dtype=np.float32)
        self.nucleus_color = np.array([92.0, 70.0, 152.0], dtype=np.float32)
        self.target_means = np.array([210.0, 172.0, 196.0], dtype=np.float32)
        self.target_stds = np.array([28.0, 24.0, 30.0], dtype=np.float32)

    def transform(self, rgb: np.ndarray) -> np.ndarray:
        gray = np.mean(rgb.astype(np.float32), axis=2)
        tissue_mask = (gray < 248.0).astype(np.float32)
        gray_norm = 1.0 - self._normalize_map(gray)
        blur = np.array(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
        detail = self._normalize_map(np.abs(gray - blur))
        nucleus = np.clip(0.65 * gray_norm + 0.35 * detail, 0.0, 1.0) * tissue_mask
        eosin = np.clip((1.0 - 0.55 * nucleus) * (0.45 + 0.55 * gray_norm), 0.0, 1.0) * tissue_mask
        background = np.full((*gray.shape, 3), 255.0, dtype=np.float32)
        tissue = self.tissue_color[None, None, :]
        eos_map = self.eosin_color[None, None, :] * eosin[..., None]
        nuc_map = self.nucleus_color[None, None, :] * np.power(nucleus[..., None], 1.15)
        residual = np.clip(1.0 - 0.72 * nucleus - 0.38 * eosin, 0.0, 1.0)
        out = background * (1.0 - tissue_mask[..., None]) + tissue * residual[..., None] * tissue_mask[..., None]
        out += eos_map
        out = out * (1.0 - 0.22 * nucleus[..., None]) + nuc_map
        out = self._match_canonical(out.astype(np.float32), tissue_mask > 0)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _normalize_map(channel: np.ndarray) -> np.ndarray:
        low = np.percentile(channel, 1.0)
        high = np.percentile(channel, 99.0)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.zeros_like(channel, dtype=np.float32)
        return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)

    def _match_canonical(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return rgb
        out = rgb.copy()
        for idx in range(3):
            vals = out[..., idx][mask]
            mean = float(vals.mean())
            std = float(vals.std() + 1e-6)
            out[..., idx][mask] = ((vals - mean) / std) * self.target_stds[idx] + self.target_means[idx]
        return out
