import base64
import json
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from PIL import Image


@dataclass
class ApiConfig:
    endpoint: str
    api_key: str
    model: str
    timeout: int = 120
    prompt_hint: str = ""
    use_morphology_aware_prompt: bool = True


@dataclass
class ApiUsageStats:
    total_calls: int = 0
    score_tiles_calls: int = 0
    localize_roi_calls: int = 0
    diagnostic_priors_calls: int = 0
    failed_calls: int = 0
    last_error: str = ""
    call_trace: List[str] = field(default_factory=list)


class OpenAIVisionClient:
    def __init__(self, config: ApiConfig) -> None:
        self.config = config
        self.usage = ApiUsageStats()

    def enabled(self) -> bool:
        return bool(self.config.endpoint and self.config.api_key and self.config.model)

    def score_tiles(self, overview_rgb: np.ndarray, tile_ids: List[int], top_k: int) -> Optional[Dict[str, object]]:
        if not self.enabled():
            return None
        self.usage.total_calls += 1
        self.usage.score_tiles_calls += 1
        self.usage.call_trace.append(f"score_tiles(top_k={top_k}, num_tiles={len(tile_ids)})")
        prompt = self._overview_prompt(tile_ids, top_k, morphology_aware=self.config.use_morphology_aware_prompt)
        return self._chat_json(self._augment_prompt(prompt), overview_rgb)

    def localize_roi(self, patch_rgb: np.ndarray) -> Optional[Dict[str, object]]:
        if not self.enabled():
            return None
        self.usage.total_calls += 1
        self.usage.localize_roi_calls += 1
        self.usage.call_trace.append("localize_roi()")
        prompt = self._local_prompt(morphology_aware=self.config.use_morphology_aware_prompt)
        return self._chat_json(self._augment_prompt(prompt), patch_rgb)

    def diagnostic_priors(self, patch_rgb: np.ndarray) -> Optional[Dict[str, object]]:
        if not self.enabled():
            return None
        self.usage.total_calls += 1
        self.usage.diagnostic_priors_calls += 1
        self.usage.call_trace.append("diagnostic_priors()")
        prompt = (
            "You are a pathology structural prior model. "
            "Identify rough diagnostic priors in this patch for nuclei, membrane, and gland regions. "
            "Return JSON only in this format: "
            '{"nuclei":[{"x1":0.1,"y1":0.1,"x2":0.3,"y2":0.3,"confidence":0.8}],'
            '"membrane":[{"x1":0.2,"y1":0.2,"x2":0.5,"y2":0.5,"confidence":0.7}],'
            '"gland":[{"x1":0.4,"y1":0.4,"x2":0.8,"y2":0.8,"confidence":0.75}],'
            '"summary":"short summary"}. '
            "Use normalized coordinates in [0, 1]."
        )
        return self._chat_json(self._augment_prompt(prompt), patch_rgb)

    def _augment_prompt(self, prompt: str) -> str:
        hint = self.config.prompt_hint.strip()
        if not hint:
            return prompt
        return f"{prompt}\nAdditional user instruction: {hint}"

    @staticmethod
    def _overview_prompt(tile_ids: List[int], top_k: int, morphology_aware: bool = True) -> str:
        if morphology_aware:
            return (
                "You are a pathology navigation model. "
                "Analyze this unstained breast pathology overview image with tile indices and rank only the most suspicious lesion candidate tiles for focused virtual H&E staining. "
                "Prioritize breast cancer morphology: dense atypical epithelial cell clusters, disrupted duct or gland architecture, irregular nests or cords, infiltrative borders into stroma, and tissue texture that is disordered compared with surrounding normal breast tissue. "
                "Be selective, but do not be overly conservative: choose 4 to 6 lesion-centered candidate tiles when supported by morphology, instead of returning too few tiny suspicious regions. "
                "Avoid blank background, adipose-only regions, folds, dust, and broad low-information stroma. "
                "Return JSON only in this format: "
                '{"overall_assessment":"normal/possibly abnormal/clearly abnormal",'
                '"focus_recommendation":"short recommendation",'
                '"suspicious_regions":[{"position":"left upper / right upper / left lower / right lower / center","suspicion_level":"low/medium/high","reason":"short reason"}],'
                '"ranked_tiles":[{"tile_id":3,"importance":0.82,"highres":true,"lesion_confidence":0.77,"suspicion_type":"nuclear atypia","reason":"short reason","next_magnification":"20x","explanation":"short explanation"}],'
                f'"top_k":{top_k},"summary":"short summary"}}. '
                f"tile_id must be chosen only from this list: {tile_ids}."
            )
        return (
            "You are a pathology navigation model. "
            "Analyze this unstained pathology overview image with tile indices and rank several suspicious candidate tiles for follow-up review. "
            "Focus on tissue regions that appear more abnormal or informative than surrounding regions. "
            "Avoid blank background and obvious artifacts. "
            "Return JSON only in this format: "
            '{"overall_assessment":"normal/possibly abnormal/clearly abnormal",'
            '"focus_recommendation":"short recommendation",'
            '"suspicious_regions":[{"position":"left upper / right upper / left lower / right lower / center","suspicion_level":"low/medium/high","reason":"short reason"}],'
            '"ranked_tiles":[{"tile_id":3,"importance":0.82,"highres":true,"lesion_confidence":0.77,"suspicion_type":"abnormal tissue","reason":"short reason","next_magnification":"20x","explanation":"short explanation"}],'
            f'"top_k":{top_k},"summary":"short summary"}}. '
            f"tile_id must be chosen only from this list: {tile_ids}."
        )

    @staticmethod
    def _local_prompt(morphology_aware: bool = True) -> str:
        if morphology_aware:
            return (
                "You are a pathology lesion localization model. "
                "Find the main suspicious breast cancer lesion tissue region in this local unstained pathology patch and summarize the main evidence type. "
                "Prioritize morphology consistent with tumor: cellular crowding, dark/dense nuclei-like texture, duct or gland destruction, irregular nests or cords, and infiltrative growth into stroma. "
                "Return one to three boxes only. Each box should cover a visible suspicious tissue region with enough surrounding context for virtual staining and pathology review. "
                "Do not return tiny point-like boxes, blank background, adipose-only tissue, tissue folds, dust, or very broad nonspecific tissue areas. "
                "Return JSON only in this format: "
                '{"boxes":[{"x1":0.10,"y1":0.12,"x2":0.42,"y2":0.48,"confidence":0.81,"label":"suspicious lesion","reason":"short explanation"}],'
                '"summary":"short summary","local_assessment":"low/medium/high suspicion","suspicion_type":"nuclear atypia","next_magnification":"20x","reason":"short reason","focus_recommendation":"short recommendation"}. '
                "Use normalized coordinates in [0, 1]."
            )
        return (
            "You are a pathology ROI review model. "
            "Find one to three suspicious tissue subregions in this unstained local pathology patch. "
            "Return boxes that cover the most informative abnormal-looking tissue with enough surrounding context. "
            "Avoid blank background and obvious artifacts. "
            "Return JSON only in this format: "
            '{"boxes":[{"x1":0.10,"y1":0.12,"x2":0.42,"y2":0.48,"confidence":0.81,"label":"suspicious tissue","reason":"short explanation"}],'
            '"summary":"short summary","local_assessment":"low/medium/high suspicion","suspicion_type":"abnormal tissue","next_magnification":"20x","reason":"short reason","focus_recommendation":"short recommendation"}. '
            "Use normalized coordinates in [0, 1]."
        )

    def _chat_json(self, prompt: str, image_rgb: np.ndarray) -> Optional[Dict[str, object]]:
        try:
            content = self._chat_with_image(prompt, image_rgb)
        except Exception as exc:
            self.usage.failed_calls += 1
            self.usage.last_error = str(exc)
            return None
        return self._extract_json(content)

    def usage_dict(self) -> Dict[str, object]:
        return asdict(self.usage)

    def _chat_with_image(self, prompt: str, image_rgb: np.ndarray) -> str:
        image_b64 = self._image_to_base64(image_rgb)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }
        payload = {
            "model": self.config.model,
            "temperature": 0.1,
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
        }
        req = urllib_request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API HTTP error {exc.code}: {body[:300]}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"API connection error: {exc}") from exc
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _image_to_base64(image_rgb: np.ndarray, max_side: int = 1200, quality: int = 85) -> str:
        image = Image.fromarray(image_rgb)
        width, height = image.size
        long_side = max(width, height)
        if long_side > max_side:
            scale = max_side / float(long_side)
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, object]]:
        if not content:
            return None
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
