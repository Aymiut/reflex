"""Textes des réponses du bot : anglais par défaut (démo), français avec BOT_LANG=fr."""
from __future__ import annotations

import re
from typing import Dict

from .pipeline import UNREADABLE, Analysis
from .triage import BLOCK, CLOSE, ESCALATE

# Verdicts de l'agent (reflex/agent.py), recopiés ici pour ne pas importer le navigateur dans les réponses.
PHISHING, LEGITIMATE, UNSURE = "phishing", "legitimate", "unsure"
CAPTION_MAX = 1024  # limite Telegram pour la légende d'une photo

MAX_LINKS = 3
MAX_SIGNS = 5

TEXTS: Dict[str, Dict[str, str]] = {
    "en": {
        BLOCK: "🚫 Phishing. Don't click, don't reply, delete the message.",
        ESCALATE: "⚠️ Suspicious. Don't click or reply until IT confirms it's safe.",
        CLOSE: "✅ Looks legitimate.",
        UNREADABLE: "🤔 I couldn't read anything here. Send a sharper screenshot, or paste the text.",
        "score": "Phishing risk: {p:.0%}",
        "signs": "Warning signs:",
        "links": "Links found (made unclickable):",
        "qr": "Contains a QR code.",
        "footer": "Analyzed in {s:.1f} s.",
        "start": ("Send me anything that looks suspicious: a screenshot, a photo of a screen, an email (.eml file), "
                  "a voice message, or just paste the text or the link. I'll tell you within seconds if it's a scam."),
        "denied": "Access denied. Your Telegram ID is {uid}: send it to your IT team so they can add you.",
        "too_big": "This file is too large (max {mb} MB). Send a screenshot instead.",
        "unsupported": "I can read screenshots, photos, .eml emails, voice messages and text. Please send one of those.",
        "voice_failed": "I couldn't transcribe this voice message right now. Paste the text, or try again later.",
        "error": "Something went wrong during the analysis. Please try again.",
        "investigating": "🔎 Taking a closer look (about 20 s)…",
        "agent_" + PHISHING: "🚫 Confirmed: phishing{brand}. Don't click, don't reply, delete the message.",
        "agent_" + LEGITIMATE: "✅ Closer look: this looks legitimate.",
        "agent_" + UNSURE: "⚠️ Still unclear after a closer look. Don't click: IT will check it.",
        "agent_brand": " (pretends to be {b})",
        "agent_guardrail": "The first check found it risky, so I won't call it safe without clear proof.",
        "agent_why": "Why:",
        "agent_proof": "Proof: {x}",
        "agent_page": "The link leads to: {u}",
        "agent_unreachable": "I couldn't open the link ({e}).",
        "agent_footer": "Closer look in {s:.0f} s.",
        "agent_failed": "I couldn't finish the closer look. Treat this message as suspicious until IT confirms.",
    },
    "fr": {
        BLOCK: "🚫 Phishing. Ne clique pas, ne réponds pas, supprime le message.",
        ESCALATE: "⚠️ Suspect. Ne clique pas et ne réponds pas tant que l'informatique n'a pas confirmé.",
        CLOSE: "✅ Ça semble légitime.",
        UNREADABLE: "🤔 Je n'arrive à rien lire. Envoie une capture plus nette, ou colle le texte.",
        "score": "Risque de phishing : {p:.0%}",
        "signs": "Signaux d'alerte :",
        "links": "Liens trouvés (rendus non cliquables) :",
        "qr": "Contient un QR code.",
        "footer": "Analysé en {s:.1f} s.",
        "start": ("Envoie-moi tout ce qui te paraît louche : capture d'écran, photo d'écran, e-mail (fichier .eml), "
                  "message vocal, ou colle simplement le texte ou le lien. Je te dis en quelques secondes si c'est une arnaque."),
        "denied": "Accès refusé. Ton identifiant Telegram est {uid} : envoie-le au service informatique pour qu'il t'ajoute.",
        "too_big": "Ce fichier est trop gros ({mb} Mo maximum). Envoie plutôt une capture d'écran.",
        "unsupported": "Je lis les captures, photos, e-mails .eml, messages vocaux et textes. Envoie l'un de ceux-là.",
        "voice_failed": "Je n'arrive pas à transcrire ce vocal pour l'instant. Colle le texte, ou réessaie plus tard.",
        "error": "Un problème est survenu pendant l'analyse. Réessaie.",
        "investigating": "🔎 J'examine de plus près (environ 20 s)…",
        "agent_" + PHISHING: "🚫 Confirmé : phishing{brand}. Ne clique pas, ne réponds pas, supprime le message.",
        "agent_" + LEGITIMATE: "✅ Après examen : ça semble légitime.",
        "agent_" + UNSURE: "⚠️ Toujours incertain après examen. Ne clique pas : l'informatique va vérifier.",
        "agent_brand": " (se fait passer pour {b})",
        "agent_guardrail": "Le premier tri le jugeait risqué : je ne le déclare pas sûr sans preuve claire.",
        "agent_why": "Pourquoi :",
        "agent_proof": "Preuve : {x}",
        "agent_page": "Le lien mène à : {u}",
        "agent_unreachable": "Je n'ai pas pu ouvrir le lien ({e}).",
        "agent_footer": "Examen en {s:.0f} s.",
        "agent_failed": "Je n'ai pas pu terminer l'examen. Considère ce message comme suspect jusqu'à confirmation de l'informatique.",
    },
}

# Signaux produits par reflex.extract.url_info, en mots simples.
SIGNS: Dict[str, Dict[str, str]] = {
    "en": {
        "brand-lookalike": "pretends to be {x}",
        "tld": "unusual domain ending .{x}",
        "shortener": "shortened link hides the real address",
        "ip-address": "raw IP address instead of a website name",
        "punycode": "look-alike characters in the address",
        "at-sign": "an @ in the address hides the real site",
        "many-subdomains": "unusually long chain of subdomains",
        "long-url": "very long link",
        "no-https": "not encrypted (http)",
    },
    "fr": {
        "brand-lookalike": "se fait passer pour {x}",
        "tld": "extension de domaine inhabituelle .{x}",
        "shortener": "lien raccourci qui cache la vraie adresse",
        "ip-address": "adresse IP brute au lieu d'un nom de site",
        "punycode": "caractères trompeurs dans l'adresse",
        "at-sign": "un @ dans l'adresse cache le vrai site",
        "many-subdomains": "chaîne de sous-domaines anormalement longue",
        "long-url": "lien très long",
        "no-https": "connexion non chiffrée (http)",
    },
}


def texts(lang: str) -> Dict[str, str]:
    return TEXTS.get(lang, TEXTS["en"])


def defang(url: str, max_len: int = 90) -> str:
    """hxxps://exemple[.]com/... : lisible, mais ni cliquable ni prévisualisé par Telegram."""
    out = re.sub(r"^http", "hxxp", url, flags=re.IGNORECASE).replace(".", "[.]")
    return out if len(out) <= max_len else out[: max_len - 1] + "…"


_DOMAIN_LIKE = re.compile(r"\b(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?", re.IGNORECASE)


def defang_text(text: str) -> str:
    """Désarme les adresses glissées dans une phrase (Telegram rend cliquable un simple « exemple.com »)."""
    return _DOMAIN_LIKE.sub(lambda m: defang(m.group(0)), text)


def describe(signal: str, lang: str) -> str:
    key, _, value = signal.partition(":")
    template = SIGNS.get(lang, SIGNS["en"]).get(key)
    if template is None:
        return signal
    return template.format(x=value.capitalize() if key == "brand-lookalike" else value)


def format_reply(a: Analysis, lang: str = "en", investigating: bool = False) -> str:
    t = texts(lang)
    if a.route == UNREADABLE:
        return t[UNREADABLE]
    lines = [t[a.route]] + ([t["investigating"]] if investigating else []) + [t["score"].format(p=a.p_phishing)]
    if a.signals:
        lines += ["", t["signs"]] + [f"• {describe(s, lang)}" for s in a.signals[:MAX_SIGNS]]
    if a.urls:
        lines += ["", t["links"]] + [f"• {defang(u)}" for u in a.urls[:MAX_LINKS]]
    if a.has_qr:
        lines += ["", t["qr"]]
    lines += ["", t["footer"].format(s=a.latency_ms / 1000)]
    return "\n".join(lines)


def format_finding(f, lang: str = "en") -> str:
    """Réponse de l'agent (reflex.agent.Finding), assez courte pour tenir en légende de photo."""
    t = texts(lang)
    brand = t["agent_brand"].format(b=f.brand) if f.brand else ""
    lines = [t["agent_" + f.verdict].format(brand=brand)]
    if f.guardrail:
        lines.append(t["agent_guardrail"])
    if f.reasons:
        lines += ["", t["agent_why"]] + [f"• {defang_text(r)}" for r in f.reasons]
    if f.proof:
        lines.append(t["agent_proof"].format(x=defang_text(f.proof)))
    if f.visited is not None:
        lines.append("")
        if f.visited.ok:
            lines.append(t["agent_page"].format(u=defang(f.visited.final_url or f.visited.url)))
        else:
            lines.append(t["agent_unreachable"].format(e=defang_text(f.visited.error)[:120]))
    lines += ["", t["agent_footer"].format(s=f.latency_ms / 1000)]
    text = "\n".join(lines)
    return text if len(text) <= CAPTION_MAX else text[: CAPTION_MAX - 1] + "…"
