"""Client minimal pour la Meta Model API (Muse Spark, SAM 3.1, Muse Voice Transcribe) et Muse Glimmer.

Références :
- base URL et compatibilité SDK OpenAI : https://dev.meta.ai/docs/overview
- SAM 3.1 : https://dev.meta.ai/docs/sam/segmenting
- transcription de fichier : https://dev.meta.ai/docs/speech-to-text
"""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

from .config import Settings, get_settings
from .sam_format import Box, parse_boxes


@dataclass
class Usage:
    """Compteurs pour le calcul du coût (tokens, images SAM, secondes d'audio)."""
    input_tokens: int = 0
    output_tokens: int = 0
    sam_images: int = 0
    audio_seconds: float = 0.0
    calls: Dict[str, int] = field(default_factory=dict)

    def add_call(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1


def image_data_url(image: bytes) -> str:
    """Encode une image en data URL base64 (format accepté par `input_image.image_url`)."""
    from PIL import Image

    fmt = (Image.open(io.BytesIO(image)).format or "PNG").lower()
    mime = "jpeg" if fmt in ("jpg", "jpeg", "mpo") else fmt
    return f"data:image/{mime};base64," + base64.b64encode(image).decode()


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Récupère le premier objet JSON d'une réponse (tolère ```json ... ``` et du texte autour)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise ValueError("Pas de JSON dans la réponse : %r" % text[:200])
        return json.loads(m.group(0))


class MetaClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        if not self.s.meta_api_key:
            raise RuntimeError("MODEL_API_KEY manquant : ajoute-le dans le fichier .env")
        from openai import OpenAI

        self.client = OpenAI(base_url=self.s.meta_base_url, api_key=self.s.meta_api_key)
        self.usage = Usage()

    # ------------------------------------------------------------------ modèles
    def list_models(self) -> List[str]:
        return sorted(m.id for m in self.client.models.list())

    # ------------------------------------------------------------------ Muse Spark
    def spark(self, prompt: str, images: Sequence[bytes] = (), model: Optional[str] = None,
              instructions: Optional[str] = None, max_output_tokens: int = 1200) -> str:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for img in images:
            content.append({"type": "input_image", "image_url": image_data_url(img)})
        kwargs: Dict[str, Any] = {
            "model": model or self.s.spark_model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
        }
        if instructions:
            kwargs["instructions"] = instructions
        resp = self.client.responses.create(**kwargs)
        self.usage.add_call("spark")
        if getattr(resp, "usage", None):
            self.usage.input_tokens += getattr(resp.usage, "input_tokens", 0) or 0
            self.usage.output_tokens += getattr(resp.usage, "output_tokens", 0) or 0
        return resp.output_text

    def spark_json(self, prompt: str, images: Sequence[bytes] = (), **kwargs) -> Dict[str, Any]:
        text = self.spark(prompt + "\n\nRéponds uniquement avec un objet JSON valide.", images, **kwargs)
        return extract_json(text)

    # ------------------------------------------------------------------ SAM 3.1
    def segment(self, image: bytes, phrase: str, include_confidence: bool = True) -> List[Box]:
        """Boîtes des objets qui correspondent à `phrase` (un groupe nominal court : "QR code", "logo")."""
        stream_kwargs: Dict[str, Any] = {
            "model": self.s.sam_model,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {"type": "input_image", "image_url": image_data_url(image)},
                ],
            }],
        }
        if include_confidence:
            stream_kwargs["metadata"] = {"include_confidence": "true"}
        chunks: List[str] = []
        # Flux brut d'événements (plus tolérant que le helper .stream()) : on ne garde que le texte.
        for event in self.client.responses.create(stream=True, **stream_kwargs):
            if getattr(event, "type", "") == "response.output_text.delta":
                chunks.append(event.delta)
        self.usage.add_call("sam")
        self.usage.sam_images += 1
        return parse_boxes("".join(chunks))

    # ------------------------------------------------------------------ Voice Transcribe
    def transcribe(self, wav: bytes, language_bias: Sequence[str] = ("French",),
                   session_id: str = "reflex") -> str:
        """Transcrit un WAV mono PCM 16 bits 16/24 kHz (voir reflex.extract.to_wav)."""
        request_part = {
            "mode": "PUSH_TO_TALK",
            "model": self.s.asr_model,
            "audioEncoding": "WAV",
        }
        if language_bias:
            request_part["languageBias"] = list(language_bias)
        files = {
            "request": (None, json.dumps(request_part), "application/json"),
            "audio": ("audio.wav", wav, "audio/wav"),
        }
        resp = requests.post(
            self.s.meta_base_url.rstrip("/") + "/asr/transcribe",
            headers={"Authorization": f"Bearer {self.s.meta_api_key}"},
            params={"sessionId": session_id},
            files=files,
            timeout=300,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Voice Transcribe {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        self.usage.add_call("asr")
        self.usage.audio_seconds += (data.get("audioDurationMs") or 0) / 1000
        return data.get("transcript", "")


class GlimmerClient:
    """Muse Glimmer (open weights) servi par un fournisseur compatible OpenAI (Chat Completions)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        if not (self.s.glimmer_base_url and self.s.glimmer_model):
            raise RuntimeError("GLIMMER_BASE_URL et GLIMMER_MODEL doivent être définis dans .env")
        from openai import OpenAI

        self.client = OpenAI(base_url=self.s.glimmer_base_url, api_key=self.s.glimmer_api_key or "none")

    def ask(self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 1200) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(img)}})
        resp = self.client.chat.completions.create(
            model=self.s.glimmer_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
