import shutil

import pytest

from reflex.extract import (
    brand_lookalike, decode_qr, extract_image, find_urls, parse_eml, refang, to_wav, url_info,
)
from reflex.fixtures import DEMO_URL, fake_login_png

EML = b"""From: "Service Informatique" <support@hopital-rh-securite.top>
Reply-To: helpdesk.verif@gmail.com
To: agent@ch-exemple.fr
Subject: Action requise : votre mot de passe expire aujourd'hui
Authentication-Results: mx.ch-exemple.fr; spf=fail smtp.mailfrom=hopital-rh-securite.top; dkim=none; dmarc=fail
Message-ID: <abc@hopital-rh-securite.top>
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="b1"

--b1
Content-Type: text/plain; charset=utf-8

Bonjour, votre mot de passe expire. Validez ici : hxxps://login-microsoft365[.]verif-session[.]xyz/auth
--b1
Content-Type: text/html; charset=utf-8

<html><body><p>Bonjour</p><a href="https://login-microsoft365.verif-session.xyz/auth">Valider</a></body></html>
--b1--
"""


def test_refang_and_find_urls():
    text = "Cliquez sur hxxps://evil[.]example[.]com/login ou www.impots-remboursement.top/x, pas sur fichier.txt"
    assert refang("a[.]b") == "a.b"
    urls = find_urls(text)
    assert "https://evil.example.com/login" in urls
    assert "http://www.impots-remboursement.top/x" in urls
    assert not any("fichier.txt" in u for u in urls)


def test_bare_domain_with_path_from_sms():
    assert find_urls("Colis en attente : colissimo-suivi.info/fr/12345 frais 1,99€") == [
        "http://colissimo-suivi.info/fr/12345"
    ]


def test_url_signals():
    info = url_info("http://login.microsoftonline.com.secure-check.xyz/a")
    assert info.registered_domain == "secure-check.xyz"
    assert "brand-lookalike:microsoft" in info.signals
    assert "tld:xyz" in info.signals
    assert "no-https" in info.signals
    assert url_info("http://192.168.1.10/login").signals[0] == "ip-address"


def test_brand_lookalike_ignores_real_domains_and_short_words():
    assert brand_lookalike("login", "microsoftonline", "microsoftonline.com") is None
    assert brand_lookalike("", "groups", "groups.io") is None          # "ups" n'est pas un mot entier
    assert brand_lookalike("", "cafe-du-port", "cafe-du-port.fr") is None
    assert brand_lookalike("", "ameli-remboursement", "ameli-remboursement.top") == "ameli"


def test_parse_eml():
    info = parse_eml(EML)
    assert info.subject.startswith("Action requise")
    assert info.auth == {"spf": "fail", "dkim": "none", "dmarc": "fail"}
    assert "gmail.com" in info.reply_to
    assert info.urls == ["https://login-microsoft365.verif-session.xyz/auth"]


def test_decode_qr_on_fixture():
    assert decode_qr(fake_login_png()) == [DEMO_URL]


def test_extract_image_finds_qr_url_even_without_ocr():
    ex = extract_image(fake_login_png())
    assert ex.channel == "image"
    assert DEMO_URL in ex.urls


@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract non installé")
def test_ocr_reads_the_address_bar():
    ex = extract_image(fake_login_png(with_qr=False), ocr_langs="eng")
    assert "verif-session" in ex.text.lower().replace(" ", "")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg non installé")
def test_to_wav_converts_to_mono_16k():
    import io
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"".join(struct.pack("<hh", int(8000 * math.sin(i / 20)), 0) for i in range(44100)))
    out = to_wav(buf.getvalue())
    with wave.open(io.BytesIO(out)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 16000, 2)
