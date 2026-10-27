"""Construit les données d'entraînement de Laya au format JSONL attendu par `laya-train`.

Chaque ligne : {"state": {...}, "questions": {...}, "expected": {"verdict": "phishing" | "legitimate"}}
L'état est produit par reflex.state.build_state, exactement comme à l'inférence.

Sources :
- ealvaradob/phishing-dataset (Apache-2.0) : e-mails, SMS, URLs, pages web — en anglais ;
- optionnel : PhreshPhish (CC BY 4.0, recherche anti-phishing uniquement) : URL + HTML récents ;
- optionnel : traductions françaises et appels (vishing) synthétiques générés par Muse Spark.

    python scripts/build_laya_dataset.py --per-class 3000
    python scripts/build_laya_dataset.py --per-class 3000 --phreshphish 1500 --translate-fr 300 --synth-voice 200

Sorties dans data/laya/ : train.jsonl, val.jsonl, test.jsonl (+ test_fr.jsonl, test_voice.jsonl si générés).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reflex.extract import Extraction, extract_text, find_urls, html_links, html_to_text  # noqa: E402
from reflex.state import LEGITIMATE, PHISHING, QUESTIONS, build_state  # noqa: E402

EALVARADOB = "ealvaradob/phishing-dataset"
PHRESHPHISH = "phreshphish/phreshphish"


# --------------------------------------------------------------------------- conversion
def to_extraction(channel: str, text: str) -> Optional[Extraction]:
    text = (text or "").strip()
    if not text:
        return None
    if channel == "url":
        urls = find_urls(text) or find_urls("http://" + text)
        return Extraction(channel="url", text="", urls=urls[:1]) if urls else None
    if channel == "page":
        if "<" in text and ">" in text:
            body = html_to_text(text)
            links = html_links(text)
        else:
            body, links = text, []
        return Extraction(channel="page", text=body, urls=find_urls("\n".join(links[:20]))[:5])
    return extract_text(text, channel=channel)


def row(ex: Extraction, label: str, origin: str) -> Dict:
    return {"state": build_state(ex), "questions": QUESTIONS, "expected": {"verdict": label}, "origin": origin}


def label_of(value) -> Optional[str]:
    if value in (1, "1", True, "phishing", "phish", "malicious", "spam", "smishing"):
        return PHISHING
    if value in (0, "0", False, "benign", "legitimate", "ham", "safe"):
        return LEGITIMATE
    return None


# --------------------------------------------------------------------------- sources
def _channel_from_filename(name: str) -> Optional[str]:
    n = name.lower()
    if "combined" in n:
        return None  # on prend les sous-ensembles pour connaître le canal de chaque ligne
    # texts.json mélange e-mails (~83 %) et SMS sans colonne pour les distinguer : on le range en "email".
    for key, channel in (("email", "email"), ("sms", "sms"), ("url", "url"), ("web", "page"), ("site", "page"),
                         ("text", "email")):
        if key in n:
            return channel
    return None


def _read_table(path: str):
    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)


def load_ealvaradob() -> Dict[str, List[tuple]]:
    """{canal: [(texte, label), ...]} depuis les fichiers bruts du dépôt (sans script de chargement)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(EALVARADOB, repo_type="dataset")
    data_files = [f for f in files if f.endswith((".json", ".jsonl", ".csv", ".parquet"))]
    print("Fichiers du dépôt :", ", ".join(data_files))
    by_channel: Dict[str, List[tuple]] = defaultdict(list)
    for f in data_files:
        channel = _channel_from_filename(f)
        if channel is None:
            continue
        df = _read_table(hf_hub_download(EALVARADOB, f, repo_type="dataset"))
        text_col = "text" if "text" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else df.columns[-1]
        for text, lab in zip(df[text_col], df[label_col]):
            label = label_of(lab)
            if label and isinstance(text, str):
                by_channel[channel].append((text, label))
        print(f"  {f} -> {channel} : {len(df)} lignes")
    if not by_channel:
        raise SystemExit("Aucun sous-ensemble reconnu : regarde la liste des fichiers ci-dessus et adapte "
                         "_channel_from_filename().")
    return by_channel


def load_phreshphish(n_per_class: int, seed: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset(PHRESHPHISH, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    counts: Counter = Counter()
    out: List[tuple] = []
    for r in ds:
        label = label_of(str(r.get("label", "")).lower())
        if not label or counts[label] >= n_per_class:
            if all(counts[l] >= n_per_class for l in (PHISHING, LEGITIMATE)):
                break
            continue
        html = (r.get("html") or "")[:200_000]
        ex = Extraction(channel="page", text=html_to_text(html) if html else "",
                        urls=find_urls(r.get("url", ""))[:1] + find_urls("\n".join(html_links(html)[:20]))[:4])
        if ex.urls or ex.text:
            out.append((ex, label))
            counts[label] += 1
    print(f"PhreshPhish : {dict(counts)}")
    return out


# --------------------------------------------------------------------------- génération par Muse Spark
TRANSLATE_PROMPT = (
    "Traduis ce message en français naturel, comme s'il avait été reçu en France. "
    "Adapte les noms de services si utile (ex. banque, administration), mais garde EXACTEMENT les URLs, "
    "adresses e-mail et numéros tels quels. Réponds en JSON : {\"text\": \"...\"}.\n\nMessage :\n"
)
VOICE_PROMPT = (
    "Génère {n} transcriptions réalistes d'appels téléphoniques reçus par du personnel d'hôpital ou de mairie en France, "
    "à l'oral, 2 à 5 phrases chacune, uniquement ce que dit l'appelant. Type : {kind}. "
    "Varie les prétextes, les tons et les interlocuteurs. Réponds en JSON : {{\"calls\": [\"...\", ...]}}"
)
VOICE_KINDS = {
    PHISHING: "arnaques (faux support informatique qui demande un code ou un accès à distance, faux conseiller bancaire, "
              "faux fournisseur qui change de RIB, fausse CPAM/ameli, faux directeur qui demande un virement urgent)",
    LEGITIMATE: "appels légitimes (collègue, vrai service informatique qui rappelle un ticket sans demander de secret, "
                "fournisseur qui confirme une livraison, patient qui demande un rendez-vous, mairie voisine)",
}


def translate_rows(rows: List[Dict], n: int, seed: int) -> List[Dict]:
    from reflex.meta import MetaClient

    client = MetaClient()
    rnd = random.Random(seed)
    candidates = [r for r in rows if r["state"]["channel"] in ("email", "sms") and r["state"].get("text")]
    out = []
    for r in rnd.sample(candidates, min(n, len(candidates))):
        try:
            fr = client.spark_json(TRANSLATE_PROMPT + r["state"]["text"], model=client.s.spark_cheap_model)["text"]
        except Exception as e:
            print("  traduction ignorée :", e)
            continue
        ex = extract_text(fr, channel=r["state"]["channel"])
        out.append(row(ex, r["expected"]["verdict"], "spark-fr"))
    print(f"Traductions françaises : {len(out)}")
    return out


def synth_voice(n_per_class: int) -> List[Dict]:
    from reflex.meta import MetaClient

    client = MetaClient()
    out = []
    for label, kind in VOICE_KINDS.items():
        made = 0
        while made < n_per_class:
            batch = min(25, n_per_class - made)
            try:
                calls = client.spark_json(VOICE_PROMPT.format(n=batch, kind=kind),
                                          model=client.s.spark_cheap_model, max_output_tokens=6000)["calls"]
            except Exception as e:
                print("  lot ignoré :", e)
                continue
            for c in calls[:batch]:
                out.append(row(extract_text(c, channel="voice"), label, "spark-voice"))
            made += len(calls[:batch])
    print(f"Appels synthétiques : {len(out)}")
    return out


# --------------------------------------------------------------------------- assemblage
def dedupe(rows: Iterable[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in rows:
        key = json.dumps(r["state"], sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def split(rows: List[Dict], seed: int, val: float = 0.1, test: float = 0.1):
    rnd = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["state"]["channel"], r["expected"]["verdict"])].append(r)
    train, va, te = [], [], []
    for g in groups.values():
        rnd.shuffle(g)
        n_te, n_va = int(len(g) * test), int(len(g) * val)
        te += g[:n_te]
        va += g[n_te:n_te + n_va]
        train += g[n_te + n_va:]
    for part in (train, va, te):
        rnd.shuffle(part)
    return train, va, te


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = Counter((r["state"]["channel"], r["expected"]["verdict"]) for r in rows)
    print(f"{path.name}: {len(rows)} lignes  " + "  ".join(f"{c}/{l}={n}" for (c, l), n in sorted(stats.items())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "laya"))
    ap.add_argument("--per-class", type=int, default=3000, help="max par canal et par classe (ealvaradob)")
    ap.add_argument("--phreshphish", type=int, default=0, help="pages PhreshPhish par classe (0 = aucune)")
    ap.add_argument("--image-copies", type=float, default=0.15,
                    help="part des e-mails/SMS/pages recopiés avec channel=image (l'OCR donne ce genre de texte)")
    ap.add_argument("--translate-fr", type=int, default=0, help="nb de messages traduits en français par Muse Spark")
    ap.add_argument("--synth-voice", type=int, default=0, help="nb d'appels synthétiques par classe (Muse Spark)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for channel, items in load_ealvaradob().items():
        by_label = defaultdict(list)
        for text, label in items:
            by_label[label].append(text)
        for label, texts in by_label.items():
            for text in rnd.sample(texts, min(args.per_class, len(texts))):
                ex = to_extraction(channel, text)
                if ex is not None:
                    rows.append(row(ex, label, "ealvaradob"))
    if args.phreshphish:
        rows += [row(ex, label, "phreshphish") for ex, label in load_phreshphish(args.phreshphish, args.seed)]

    rows = dedupe(rows)
    train, val, test = split(rows, args.seed)

    # Copies "image" : à l'inférence, le texte d'une capture vient de l'OCR (canal image).
    for part in (train, val):
        extra = []
        for r in part:
            if r["state"]["channel"] in ("email", "sms", "page") and rnd.random() < args.image_copies:
                copy = json.loads(json.dumps(r))
                copy["state"]["channel"] = "image"
                copy["origin"] += "+image"
                extra.append(copy)
        part += extra

    if args.translate_fr:
        fr = translate_rows(train, args.translate_fr, args.seed)
        fr_train, fr_val, fr_test = split(fr, args.seed, val=0.1, test=0.2)
        train += fr_train
        val += fr_val
        write_jsonl(out_dir / "test_fr.jsonl", fr_test)
    if args.synth_voice:
        voice = synth_voice(args.synth_voice)
        v_train, v_val, v_test = split(voice, args.seed, val=0.1, test=0.2)
        train += v_train
        val += v_val
        write_jsonl(out_dir / "test_voice.jsonl", v_test)

    rnd.shuffle(train)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    # Le script d'évaluation voyage avec les données (pratique pour Kaggle).
    shutil.copy(ROOT / "scripts" / "eval_laya.py", out_dir / "eval_laya.py")
    print("\nVérifie le format avant d'entraîner :\n"
          f"  laya-train --data {out_dir / 'train.jsonl'} --base multilingual --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
