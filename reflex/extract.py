"""Étage 0 : extraction locale et gratuite, avant tout modèle.

Transforme un signalement brut (mail .eml, photo/capture, texte, audio) en éléments que Laya peut lire :
texte, URLs, QR codes, en-têtes d'authentification du mail, signaux sur les domaines.
Aucun appel réseau ici.
"""
from __future__ import annotations

import email
import email.policy
import io
import ipaddress
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlsplit

import tldextract

# Liste publique des suffixes embarquée dans tldextract : pas de téléchargement au démarrage.
_TLD = tldextract.TLDExtract(suffix_list_urls=())

# Marques souvent imitées en France (services publics, santé, banques, colis, tech).
BRANDS = (
    "microsoft", "office365", "office", "outlook", "onedrive", "sharepoint", "teams", "azure",
    "google", "gmail", "apple", "icloud", "paypal", "amazon", "netflix", "docusign", "dropbox",
    "ameli", "impots", "caf", "franceconnect", "ants", "antai", "urssaf", "pole-emploi", "francetravail",
    "doctolib", "mutuelle", "laposte", "colissimo", "chronopost", "dhl", "ups", "fedex", "mondialrelay",
    "creditagricole", "credit-agricole", "bnp", "bnpparibas", "societegenerale", "lcl", "boursorama",
    "caisse-epargne", "banquepopulaire", "labanquepostale", "cic", "orange", "sfr", "free", "bouygues",
)
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rebrand.ly", "shorturl.at", "t.ly")
# TLD fréquents dans les campagnes (signal faible, jamais suffisant seul).
SUSPICIOUS_TLDS = ("xyz", "top", "zip", "mov", "click", "shop", "live", "icu", "buzz", "rest", "cfd", "sbs", "online", "site", "info")

_URL_RE = re.compile(
    r"(?:(?:https?|hxxps?)://|www\.)[^\s<>\"'()\[\]{}]+"
    r"|\b(?:[a-z0-9-]+(?:\.|\[\.\]))+[a-z]{2,24}/[^\s<>\"']*",
    re.IGNORECASE,
)


def refang(text: str) -> str:
    """hxxp://exemple[.]com -> http://exemple.com (formes "désarmées" courantes dans les signalements)."""
    return (text.replace("[.]", ".").replace("(.)", ".").replace("[:]", ":")
            .replace("hxxps://", "https://").replace("hxxp://", "http://")
            .replace("HXXPS://", "https://").replace("HXXP://", "http://"))


def find_urls(text: str) -> List[str]:
    """URLs trouvées dans un texte, dédoublonnées, dans l'ordre d'apparition."""
    seen, out = set(), []
    for raw in _URL_RE.findall(refang(text or "")):
        url = raw.rstrip(".,;:!?)»\"'")
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "http://" + url
        host = urlsplit(url).hostname or ""
        if not host or not (_TLD(host).suffix or _is_ip(host)):
            continue
        if url.lower() not in seen:
            seen.add(url.lower())
            out.append(url)
    return out


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


# Vrais domaines des marques ci-dessus : une URL chez eux n'est pas une imitation.
LEGIT_DOMAINS = {
    "microsoft.com", "microsoftonline.com", "office.com", "office365.com", "live.com", "outlook.com",
    "sharepoint.com", "onedrive.com", "azure.com", "google.com", "google.fr", "gmail.com", "apple.com",
    "icloud.com", "paypal.com", "paypal.fr", "amazon.com", "amazon.fr", "netflix.com", "docusign.com",
    "docusign.net", "dropbox.com", "ameli.fr", "impots.gouv.fr", "caf.fr", "franceconnect.gouv.fr",
    "ants.gouv.fr", "antai.gouv.fr", "urssaf.fr", "pole-emploi.fr", "francetravail.fr", "doctolib.fr",
    "laposte.fr", "laposte.net", "colissimo.fr", "chronopost.fr", "dhl.com", "dhl.fr", "ups.com",
    "fedex.com", "mondialrelay.fr", "credit-agricole.fr", "bnpparibas", "bnpparibas.com", "mabanque.bnpparibas",
    "societegenerale.fr", "lcl.fr", "boursorama.com", "boursobank.com", "caisse-epargne.fr",
    "banquepopulaire.fr", "labanquepostale.fr", "cic.fr", "orange.fr", "sfr.fr", "free.fr", "bouyguestelecom.fr",
}


def brand_lookalike(subdomain: str, domain_label: str, registered: str) -> Optional[str]:
    """Marque citée dans un domaine qui n'appartient pas à cette marque (ex. microsoft-verif.xyz)."""
    if registered in LEGIT_DOMAINS:
        return None
    tokens = set(t for t in re.split(r"[-.\d_]+", subdomain + "." + domain_label) if t)
    joined = subdomain.replace("-", "") + domain_label.replace("-", "")
    for brand in BRANDS:
        # Les noms courts ("ups", "caf", "free") seulement comme mot entier, sinon trop de faux positifs.
        if brand in tokens or (len(brand) >= 6 and brand.replace("-", "") in joined):
            return brand
    return None


@dataclass
class UrlInfo:
    url: str
    host: str
    registered_domain: str
    signals: List[str] = field(default_factory=list)


def url_info(url: str) -> UrlInfo:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    ext = _TLD(host)
    registered = ext.top_domain_under_public_suffix if hasattr(ext, "top_domain_under_public_suffix") else ext.registered_domain
    registered = registered or host
    signals: List[str] = []
    if _is_ip(host):
        signals.append("ip-address")
    if host.startswith("xn--") or ".xn--" in host:
        signals.append("punycode")
    if "@" in (parts.netloc or ""):
        signals.append("at-sign")
    if host in SHORTENERS or registered in SHORTENERS:
        signals.append("shortener")
    if ext.suffix and ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS:
        signals.append("tld:" + ext.suffix)
    sub = (ext.subdomain or "").lower()
    if sub.count(".") >= 3:
        signals.append("many-subdomains")
    brand = brand_lookalike(sub, (ext.domain or "").lower(), registered)
    if brand:
        signals.append("brand-lookalike:" + brand)
    if len(url) > 120:
        signals.append("long-url")
    if parts.scheme == "http":
        signals.append("no-https")
    return UrlInfo(url=url, host=host, registered_domain=registered, signals=signals)


# --------------------------------------------------------------------------- e-mail
@dataclass
class EmailInfo:
    subject: str = ""
    sender: str = ""
    reply_to: str = ""
    return_path: str = ""
    auth: Dict[str, str] = field(default_factory=dict)  # {"spf": "fail", "dkim": "pass", "dmarc": "fail"}
    body: str = ""
    urls: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    message_id: str = ""


_AUTH_RE = re.compile(r"\b(spf|dkim|dmarc)=(\w+)", re.IGNORECASE)


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()


def html_links(html: str) -> List[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True) if a["href"].lower().startswith(("http", "hxxp", "www."))]


def parse_eml(raw: bytes) -> EmailInfo:
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    info = EmailInfo(
        subject=str(msg.get("Subject", "") or ""),
        sender=str(msg.get("From", "") or ""),
        reply_to=str(msg.get("Reply-To", "") or ""),
        return_path=str(msg.get("Return-Path", "") or ""),
        message_id=str(msg.get("Message-ID", "") or ""),
    )
    for header in msg.get_all("Authentication-Results", []) or []:
        for key, value in _AUTH_RE.findall(str(header)):
            info.auth.setdefault(key.lower(), value.lower())

    plain, html = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if filename:
            info.attachments.append(filename)
            continue
        ctype = part.get_content_type()
        try:
            payload = part.get_content()
        except (LookupError, ValueError):
            payload = (part.get_payload(decode=True) or b"").decode("utf-8", "replace")
        if ctype == "text/plain":
            plain.append(str(payload))
        elif ctype == "text/html":
            html.append(str(payload))

    info.body = "\n".join(plain).strip() or html_to_text("\n".join(html))
    links = [l for h in html for l in html_links(h)]
    info.urls = find_urls("\n".join(links + [info.body]))
    return info


# --------------------------------------------------------------------------- images
def decode_qr(image: bytes) -> List[str]:
    """Contenu des QR codes présents dans l'image (OpenCV, aucune dépendance système)."""
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return []
    detector = cv2.QRCodeDetector()
    found: List[str] = []
    try:
        ok, decoded, _points, _ = detector.detectAndDecodeMulti(arr)
        if ok:
            found.extend(d for d in decoded if d)
    except cv2.error:
        pass
    if not found:
        data, _points, _ = detector.detectAndDecode(arr)
        if data:
            found.append(data)
    return list(dict.fromkeys(found))


def ocr_image(image: bytes, langs: str = "fra+eng") -> str:
    """Texte lu dans l'image avec Tesseract. Renvoie "" si Tesseract n'est pas installé."""
    if not shutil.which("tesseract"):
        return ""
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(image)).convert("RGB")
    # Les captures d'écran ont souvent un petit texte : on agrandit les petites images.
    if img.width < 1200:
        factor = 1200 / img.width
        img = img.resize((int(img.width * factor), int(img.height * factor)))
    available = set(pytesseract.get_languages(config=""))
    wanted = [l for l in langs.split("+") if l in available] or ["eng"]
    return pytesseract.image_to_string(img, lang="+".join(wanted)).strip()


# --------------------------------------------------------------------------- audio
def to_wav(audio: bytes, sample_rate: int = 16000) -> bytes:
    """Convertit n'importe quel audio (OGG/Opus de Telegram, m4a...) en WAV mono PCM 16 bits."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg introuvable (brew install ffmpeg)")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-map_metadata", "-1", "-f", "wav", "pipe:1"],
        input=audio, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg : " + proc.stderr.decode("utf-8", "replace")[:300])
    return proc.stdout


# --------------------------------------------------------------------------- signalement
@dataclass
class Extraction:
    """Tout ce que l'étage local a pu tirer d'un signalement."""
    channel: str                      # email | sms | url | page | image | voice
    text: str = ""
    urls: List[str] = field(default_factory=list)
    qr: List[str] = field(default_factory=list)
    email: Optional[EmailInfo] = None

    @property
    def url_infos(self) -> List[UrlInfo]:
        return [url_info(u) for u in self.urls]


def extract_email(raw: bytes) -> Extraction:
    info = parse_eml(raw)
    return Extraction(channel="email", text=info.body, urls=info.urls, email=info)


def extract_image(image: bytes, ocr_langs: str = "fra+eng") -> Extraction:
    text = ocr_image(image, ocr_langs)
    qr = decode_qr(image)
    urls = find_urls("\n".join([text] + qr))
    return Extraction(channel="image", text=text, urls=urls, qr=qr)


def extract_text(text: str, channel: str = "sms") -> Extraction:
    return Extraction(channel=channel, text=text, urls=find_urls(text))
