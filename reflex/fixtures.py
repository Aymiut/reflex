"""Images de test générées localement (aucune vraie page de phishing)."""
from __future__ import annotations

import io

DEMO_URL = "https://login-microsoft365.verif-session.xyz/auth"


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def fake_login_png(url: str = DEMO_URL, with_qr: bool = True) -> bytes:
    """Fausse page « session expirée » avec URL, champs et QR code — pour les tests et la démo."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1280, 800), "white")
    d = ImageDraw.Draw(img)
    # barre d'adresse
    d.rectangle([0, 0, 1280, 56], fill=(235, 235, 235))
    d.rounded_rectangle([20, 10, 1000, 46], radius=18, fill="white", outline=(190, 190, 190))
    d.text((40, 16), url, fill=(40, 40, 40), font=_font(22))
    # carte de connexion
    d.rectangle([380, 140, 900, 680], outline=(200, 200, 200), width=2)
    d.rectangle([420, 180, 470, 230], fill=(242, 80, 34))
    d.rectangle([475, 180, 525, 230], fill=(127, 186, 0))
    d.rectangle([420, 235, 470, 285], fill=(0, 164, 239))
    d.rectangle([475, 235, 525, 285], fill=(255, 185, 0))
    d.text((420, 310), "Votre session Microsoft 365 a expire", fill="black", font=_font(28))
    d.text((420, 355), "Reconnectez-vous sous 24h pour garder", fill=(60, 60, 60), font=_font(20))
    d.text((420, 382), "l'acces a votre messagerie.", fill=(60, 60, 60), font=_font(20))
    d.text((420, 430), "Adresse e-mail", fill="black", font=_font(20))
    d.rectangle([420, 458, 860, 498], outline=(120, 120, 120), width=2)
    d.text((420, 515), "Mot de passe", fill="black", font=_font(20))
    d.rectangle([420, 543, 860, 583], outline=(120, 120, 120), width=2)
    d.rectangle([420, 605, 600, 650], fill=(0, 103, 184))
    d.text((455, 615), "Se connecter", fill="white", font=_font(20))
    if with_qr:
        import qrcode

        qr = qrcode.make(url).get_image().convert("RGB").resize((220, 220))
        img.paste(qr, (980, 300))
        d.text((960, 530), "Ou scannez avec votre mobile", fill=(60, 60, 60), font=_font(18))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def draw_boxes(image: bytes, boxes, color=(220, 20, 60), label: str = "") -> bytes:
    """Entoure les boîtes SAM sur l'image (overlay pour l'utilisateur)."""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(image)).convert("RGB")
    d = ImageDraw.Draw(img)
    for b in boxes:
        b = b.scaled(img.width, img.height)
        pad = 8
        d.ellipse([b.x1 - pad * 3, b.y1 - pad * 2, b.x2 + pad * 3, b.y2 + pad * 2], outline=color, width=5)
        if label:
            d.text((b.x1, max(0, b.y1 - 34)), label, fill=color, font=_font(24))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
