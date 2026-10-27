"""Chaîne locale d'un signalement : étage 0 (extraction) puis étage 1 (Laya).

Indépendante de Telegram : le bot (reflex/bot.py) reçoit les fichiers et affiche le résultat.
Les cas `escalate` partent ensuite à l'agent Meta (reflex/agent.py), lancé par le bot.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import Settings, get_settings
from .extract import Extraction, extract_email, extract_image, extract_text, find_urls, to_wav
from .triage import LayaTriage

# Rien de lisible dans le signalement : on ne consulte pas Laya.
UNREADABLE = "unreadable"


def text_extraction(text: str) -> Extraction:
    """Texte collé ou transféré dans le bot.

    Un lien seul est mis en forme comme les URLs de l'entraînement (canal url, texte vide).
    """
    text = (text or "").strip()
    urls = find_urls(text)
    if len(urls) == 1 and len(text.split()) == 1:
        return Extraction(channel="url", text="", urls=urls)
    return extract_text(text, channel="sms")


@dataclass
class Analysis:
    route: str                                         # close | block | escalate | unreadable
    p_phishing: Optional[float]                        # None si Laya n'a pas été consulté
    channel: str
    urls: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)   # signaux des URLs, sans doublon
    has_qr: bool = False
    latency_ms: float = 0.0
    extraction: Optional[Extraction] = field(default=None, repr=False)  # pour l'agent de l'étage 2


class Pipeline:
    def __init__(self, settings: Optional[Settings] = None, triage: Optional[LayaTriage] = None):
        self.s = settings or get_settings()
        self.triage = triage or LayaTriage(self.s)
        # Laya n'est pas prévu pour des appels simultanés : un signalement à la fois.
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        """Charge Laya tout de suite, sinon la première réponse du bot prend plusieurs secondes."""
        self.from_text("warm-up message")

    def analyze(self, ex: Extraction) -> Analysis:
        if not ((ex.text or "").strip() or ex.urls or ex.qr):
            return Analysis(route=UNREADABLE, p_phishing=None, channel=ex.channel, extraction=ex)
        with self._lock:
            decision = self.triage.decide(ex)
        signals = list(dict.fromkeys(s for info in ex.url_infos for s in info.signals))
        return Analysis(route=decision.route, p_phishing=decision.p_phishing, channel=ex.channel,
                        urls=list(ex.urls), signals=signals, has_qr=bool(ex.qr),
                        latency_ms=decision.latency_ms, extraction=ex)

    def _timed(self, make: Callable[[], Extraction]) -> Analysis:
        """Analyse complète, extraction comprise : c'est le délai que voit l'agent."""
        t0 = time.perf_counter()
        result = self.analyze(make())
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def from_text(self, text: str) -> Analysis:
        return self._timed(lambda: text_extraction(text))

    def from_image(self, image: bytes) -> Analysis:
        return self._timed(lambda: extract_image(image, self.s.ocr_langs))

    def from_email(self, raw: bytes) -> Analysis:
        return self._timed(lambda: extract_email(raw))

    def from_voice(self, audio: bytes) -> Analysis:
        """Message vocal : conversion locale en WAV, puis Voice Transcribe (Meta, hors mode Contributor)."""
        from .meta import MetaClient

        def make() -> Extraction:
            transcript = MetaClient(self.s).transcribe(to_wav(audio))
            return extract_text(transcript, channel="voice")

        return self._timed(make)
