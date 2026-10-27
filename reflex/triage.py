"""Étage 1 : Laya trie en local. Les cas évidents sont réglés ici, les autres partent à l'agent Meta."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .config import Settings, get_settings
from .extract import Extraction
from .state import PHISHING, QUESTIONS, build_state

CLOSE, BLOCK, ESCALATE = "close", "block", "escalate"


@dataclass
class Decision:
    route: str            # close | block | escalate
    p_phishing: float     # probabilité calibrée donnée par Laya
    confidence: float
    latency_ms: float
    state: Dict[str, Any]

    @property
    def risky(self) -> bool:
        return self.route != CLOSE


def route_for(p_phishing: float, s: Settings) -> str:
    if p_phishing >= s.block_threshold:
        return BLOCK
    if p_phishing <= s.close_threshold:
        return CLOSE
    return ESCALATE


class LayaTriage:
    """Enveloppe autour de Laya. `predict_fn` permet de tester sans charger le modèle."""

    def __init__(self, settings: Optional[Settings] = None,
                 predict_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None):
        self.s = settings or get_settings()
        self._predict_fn = predict_fn
        self._agent = None

    def _predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self._predict_fn is not None:
            return self._predict_fn(state)
        if self._agent is None:
            import laya

            self._agent = laya.load(self.s.laya_model)
        result = self._agent.predict(state, QUESTIONS)
        return result.get("answers", result)

    def score(self, state: Dict[str, Any]) -> tuple[float, float]:
        answer = self._predict(state)["verdict"]
        p = float(answer["probabilities"][PHISHING])
        conf = float(answer.get("answer_confidence", answer.get("confidence", max(p, 1 - p))))
        return p, conf

    def decide(self, ex: Extraction) -> Decision:
        state = build_state(ex)
        t0 = time.perf_counter()
        p, conf = self.score(state)
        latency = (time.perf_counter() - t0) * 1000
        return Decision(route=route_for(p, self.s), p_phishing=p, confidence=conf,
                        latency_ms=latency, state=state)
