"""Vérifie en 1 minute que tout ce dont Reflex a besoin côté Meta fonctionne avec TA clé.

    python scripts/smoke_meta.py            # tout
    python scripts/smoke_meta.py --only sam # une seule étape (models, spark, sam, voice, glimmer)

Les sorties (image annotée, réponses) sont écrites dans data/smoke/.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reflex.config import get_settings  # noqa: E402
from reflex.extract import to_wav  # noqa: E402
from reflex.fixtures import DEMO_URL, draw_boxes, fake_login_png  # noqa: E402

OUT = ROOT / "data" / "smoke"
REQUIRED = ("muse-spark-1.3", "sam-3.1", "muse-voice-transcribe-1.0")
SAM_PHRASES = ("QR code", "text field", "button", "logo", "address bar")
VISHING_FR = ("Bonjour, ici le service informatique de l'hôpital. On a détecté un problème sur votre "
              "compte, pouvez-vous me donner le code que vous venez de recevoir par SMS pour que je le débloque ?")

results: dict = {}


def step(name):
    def wrap(fn):
        def run(*a, **kw):
            print(f"\n=== {name} ===")
            t0 = time.perf_counter()
            try:
                fn(*a, **kw)
                results[name] = "OK"
                print(f"--> OK ({time.perf_counter() - t0:.1f} s)")
            except Exception as e:  # on veut le diagnostic de chaque étape, pas un arrêt au premier échec
                results[name] = f"ÉCHEC : {type(e).__name__}: {e}"
                print(f"--> ÉCHEC : {type(e).__name__}: {e}")
        return run
    return wrap


@step("models")
def check_models(client):
    ids = client.list_models()
    print("Modèles activés :", ", ".join(ids) or "(aucun)")
    missing = [m for m in REQUIRED if m not in ids]
    if missing:
        raise RuntimeError("manquants : " + ", ".join(missing))


@step("spark")
def check_spark(client, image):
    t0 = time.perf_counter()
    answer = client.spark_json(
        "Tu analyses un signalement de phishing envoyé par un agent hospitalier. "
        "Voici une capture de la page qu'il a reçue. Donne : is_phishing (bool), imitated_brand (str|null), "
        "urls_seen (liste), password_requested (bool), reasons (liste de phrases courtes en français).",
        [image],
    )
    print(f"Latence : {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(json.dumps(answer, ensure_ascii=False, indent=2))
    (OUT / "spark.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2))


@step("sam")
def check_sam(client, image):
    annotated = image
    found_any = False
    for phrase in SAM_PHRASES:
        t0 = time.perf_counter()
        boxes = client.segment(image, phrase)
        ms = (time.perf_counter() - t0) * 1000
        desc = ", ".join(f"[{b.x1},{b.y1}→{b.x2},{b.y2} c={b.confidence}]" for b in boxes) or "rien"
        print(f"« {phrase} » ({ms:.0f} ms) : {desc}")
        if boxes:
            found_any = True
            annotated = draw_boxes(annotated, boxes, label=phrase)
    (OUT / "sam_overlay.png").write_bytes(annotated)
    print("Overlay :", OUT / "sam_overlay.png")
    if not found_any:
        raise RuntimeError("SAM n'a rien trouvé : essaie d'autres expressions (plan B : boîtes données par Muse Spark)")


def _french_audio() -> bytes:
    if platform.system() != "Darwin" or not shutil.which("say"):
        raise RuntimeError("génération audio auto seulement sur macOS (commande `say`) ; utilise --audio fichier")
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "vishing.aiff"
        subprocess.run(["say", "-v", "Thomas", "-o", str(aiff), VISHING_FR], check=True)
        return aiff.read_bytes()


@step("voice")
def check_voice(client, audio_path):
    raw = Path(audio_path).read_bytes() if audio_path else _french_audio()
    wav = to_wav(raw)
    t0 = time.perf_counter()
    text = client.transcribe(wav, language_bias=("French",))
    print(f"Latence : {(time.perf_counter() - t0) * 1000:.0f} ms")
    print("Transcription :", text)
    if not text.strip():
        raise RuntimeError("transcription vide")


@step("glimmer")
def check_glimmer(image):
    from reflex.meta import GlimmerClient

    g = GlimmerClient()
    print(g.ask("Is this page a phishing attempt? Answer in one sentence and name the imitated brand.", [image]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["models", "spark", "sam", "voice", "glimmer"])
    ap.add_argument("--audio", help="fichier audio à transcrire (sinon généré avec `say` sur macOS)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    image = fake_login_png(DEMO_URL)
    (OUT / "fake_login.png").write_bytes(image)
    print("Image de test :", OUT / "fake_login.png")

    s = get_settings()
    want = lambda name: args.only in (None, name)  # noqa: E731

    if any(want(n) for n in ("models", "spark", "sam", "voice")):
        from reflex.meta import MetaClient

        try:
            client = MetaClient(s)
        except RuntimeError as e:
            print("\n" + str(e))
            return 1
        if want("models"):
            check_models(client)
        if want("spark"):
            check_spark(client, image)
        if want("sam"):
            check_sam(client, image)
        if want("voice"):
            check_voice(client, args.audio)
        u = client.usage
        cost = u.cost_usd()
        print(f"\nConsommation : {u.input_tokens} tokens en entrée, {u.output_tokens} en sortie, "
              f"{u.sam_images} images SAM, {u.audio_seconds:.0f} s d'audio ≈ {cost:.4f} $")

    if want("glimmer"):
        if s.glimmer_base_url and s.glimmer_model:
            check_glimmer(image)
        elif args.only == "glimmer":
            print("Glimmer non configuré (GLIMMER_BASE_URL / GLIMMER_MODEL dans .env)")

    print("\n=== Résumé ===")
    for name, status in results.items():
        print(f"{name:8s} {status}")
    return 0 if all(v == "OK" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
