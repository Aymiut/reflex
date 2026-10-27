"""Ce que Laya lit : un état compact construit à partir d'une Extraction.

La MÊME fonction sert à fabriquer les données d'entraînement (scripts/build_laya_dataset.py)
et à l'inférence (reflex/triage.py). Si on change le format ici, il faut réentraîner Laya.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .extract import Extraction, url_info

MAX_TEXT_CHARS = 1500
MAX_URLS = 5

# La question posée à Laya. Deux options au nom neutre plutôt qu'un oui/non :
# le model card signale un biais des questions "noul" vers leurs libellés true/false.
QUESTIONS: Dict[str, Dict[str, Any]] = {
    "verdict": {
        "type": "choice",
        "instructions": "Is the report in `text` (with its `urls`) a phishing or scam attempt, or legitimate?",
        "criteria": {
            "phishing": "phishing, scam or fraud: tries to steal credentials, money or personal data, or leads to a malicious page",
            "legitimate": "a normal, harmless message or page",
        },
    }
}
PHISHING, LEGITIMATE = "phishing", "legitimate"


def _url_line(url: str) -> str:
    info = url_info(url)
    line = info.url[:200]
    if info.signals:
        line += "  [" + ", ".join(info.signals) + "]"
    return line


def build_state(ex: Extraction) -> Dict[str, Any]:
    state: Dict[str, Any] = {"channel": ex.channel}
    if ex.email is not None:
        if ex.email.sender:
            state["from"] = ex.email.sender[:150]
        if ex.email.reply_to and ex.email.reply_to != ex.email.sender:
            state["reply_to"] = ex.email.reply_to[:150]
        if ex.email.auth:
            state["auth"] = " ".join(f"{k}={v}" for k, v in sorted(ex.email.auth.items()))
        if ex.email.subject:
            state["subject"] = ex.email.subject[:200]
        if ex.email.attachments:
            state["attachments"] = ex.email.attachments[:5]
    urls: List[str] = list(dict.fromkeys(ex.urls))[:MAX_URLS]
    if urls:
        state["urls"] = [_url_line(u) for u in urls]
    if ex.qr:
        state["qr"] = [q[:200] for q in ex.qr[:3]]
    text = " ".join((ex.text or "").split())
    state["text"] = text[:MAX_TEXT_CHARS]
    return state
