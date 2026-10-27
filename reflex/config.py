"""Configuration lue depuis l'environnement (fichier .env en local)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv est optionnel
    pass


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _ids(name: str) -> FrozenSet[int]:
    """Liste d'identifiants Telegram séparés par des virgules ou des espaces."""
    raw = os.environ.get(name, "")
    try:
        return frozenset(int(x) for x in re.split(r"[\s,;]+", raw) if x)
    except ValueError:
        raise ValueError(f"{name} doit contenir des identifiants numériques séparés par des virgules") from None


@dataclass(frozen=True)
class Settings:
    # Meta Model API
    meta_api_key: str = field(default_factory=lambda: os.environ.get("MODEL_API_KEY", ""))
    meta_base_url: str = field(default_factory=lambda: os.environ.get("META_BASE_URL", "https://api.meta.ai/v1"))
    spark_model: str = field(default_factory=lambda: os.environ.get("SPARK_MODEL", "muse-spark-1.3"))
    # Mode Contributor = moins cher, mais Meta peut entraîner sur les données : uniquement pour des données publiques.
    spark_cheap_model: str = field(default_factory=lambda: os.environ.get("SPARK_CHEAP_MODEL", "muse-spark-1.3-contributor"))
    sam_model: str = field(default_factory=lambda: os.environ.get("SAM_MODEL", "sam-3.1"))
    asr_model: str = field(default_factory=lambda: os.environ.get("ASR_MODEL", "muse-voice-transcribe-1.0"))

    # Muse Glimmer (open weights) servi par un fournisseur compatible OpenAI (Fireworks, Modal, Baseten...)
    glimmer_base_url: str = field(default_factory=lambda: os.environ.get("GLIMMER_BASE_URL", ""))
    glimmer_api_key: str = field(default_factory=lambda: os.environ.get("GLIMMER_API_KEY", ""))
    glimmer_model: str = field(default_factory=lambda: os.environ.get("GLIMMER_MODEL", ""))

    # Laya (étage local)
    laya_model: str = field(default_factory=lambda: os.environ.get("LAYA_MODEL", "multilingual"))
    # p(phishing) >= block_threshold  -> bloqué sans appel au gros modèle
    block_threshold: float = field(default_factory=lambda: _float("LAYA_BLOCK_THRESHOLD", 0.97))
    # p(phishing) <= close_threshold  -> classé légitime sans appel au gros modèle
    close_threshold: float = field(default_factory=lambda: _float("LAYA_CLOSE_THRESHOLD", 0.03))
    # Garde-fou : au-dessus de ce seuil, l'agent ne peut pas conclure "légitime" sans preuve
    risky_threshold: float = field(default_factory=lambda: _float("LAYA_RISKY_THRESHOLD", 0.5))

    # Étage 2 : l'agent Meta examine les cas incertains (escalate). AGENT_ENABLED=0 pour le couper.
    agent_enabled: bool = field(default_factory=lambda: os.environ.get("AGENT_ENABLED", "1") not in ("0", "false", "no"))

    # Telegram
    telegram_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    # Comptes autorisés à utiliser le bot. Vide = personne (le bot répond avec l'identifiant à ajouter).
    telegram_allowed_users: FrozenSet[int] = field(default_factory=lambda: _ids("TELEGRAM_ALLOWED_USERS"))
    # Langue des réponses du bot : "en" (démo devant un jury anglophone) ou "fr"
    bot_lang: str = field(default_factory=lambda: os.environ.get("BOT_LANG", "en"))

    # Pi-hole
    pihole_url: str = field(default_factory=lambda: os.environ.get("PIHOLE_URL", "http://localhost:8081"))
    pihole_password: str = field(default_factory=lambda: os.environ.get("PIHOLE_PASSWORD", ""))

    # OCR
    ocr_langs: str = field(default_factory=lambda: os.environ.get("OCR_LANGS", "fra+eng"))


def get_settings() -> Settings:
    return Settings()
