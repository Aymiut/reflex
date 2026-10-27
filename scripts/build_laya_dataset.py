"""Construit les données d'entraînement de Laya au format JSONL attendu par `laya-train`.

Chaque ligne : {"state": {...}, "questions": {...}, "expected": {"verdict": "phishing" | "legitimate"}}
L'état est produit par reflex.state.build_state, exactement comme à l'inférence.

Sources :
- ealvaradob/phishing-dataset (Apache-2.0) : e-mails, SMS, URLs, pages web — en anglais ;
- optionnel : PhreshPhish (CC BY 4.0, recherche anti-phishing uniquement) : vraies pages de 2024-2025,
  en page entière (URL + HTML) et en lien seul ;
- optionnel : traductions françaises et appels (vishing) synthétiques générés par Muse Spark.

    python scripts/build_laya_dataset.py --per-class 3000
    python scripts/build_laya_dataset.py --per-class 3000 --phreshphish 2500
    python scripts/build_laya_dataset.py --per-class 3000 --phreshphish 2500 --translate-fr 300 --synth-voice 200

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
    # "text" : texts.json mélange e-mails et SMS ; le canal est choisi ligne par ligne (_text_channel).
    for key, channel in (("email", "email"), ("sms", "sms"), ("url", "url"), ("web", "page"), ("site", "page"),
                         ("text", "text")):
        if key in n:
            return channel
    return None


def _text_channel(text: str) -> str:
    """E-mail ou SMS, pour une ligne de texts.json (pas de colonne qui le dit).

    Les e-mails du corpus (Enron, spam) sont en minuscules avec la ponctuation détachée (« hi , there ») ;
    les SMS sont courts et gardent leurs majuscules. Le bot range un texte collé dans le canal sms.
    """
    t = text.strip()
    return "sms" if len(t) <= 400 and t != t.lower() and " , " not in t else "email"


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
                by_channel[_text_channel(text) if channel == "text" else channel].append((text, label))
        print(f"  {f} -> {'email + sms' if channel == 'text' else channel} : {len(df)} lignes")
    print("  par canal : " + ", ".join(f"{c}={len(v)}" for c, v in sorted(by_channel.items())))
    if not by_channel:
        raise SystemExit("Aucun sous-ensemble reconnu : regarde la liste des fichiers ci-dessus et adapte "
                         "_channel_from_filename().")
    return by_channel


def phreshphish_extraction(url: str, html: str) -> Optional[Extraction]:
    """Une page PhreshPhish (adresse + HTML) vue comme le canal page : l'adresse d'abord, puis les liens de la page."""
    html = (html or "")[:200_000]
    ex = Extraction(channel="page", text=html_to_text(html) if html else "",
                    urls=find_urls(url or "")[:1] + find_urls("\n".join(html_links(html)[:20]))[:4])
    return ex if ex.urls or ex.text else None


def load_phreshphish(n_per_class: int, seed: int, shards: List[str]) -> List[tuple]:
    """[(ligne page, ligne lien seul), ...] tirés du split train de PhreshPhish (juil. 2024 → sept. 2025).

    Le split complet pèse environ 145 Go : on télécharge seulement quelques fichiers (shards) d'environ 300 à 550 Mo,
    chacun mêlant toute la période. On tire les pages sur la seule colonne label, puis on ne lit le HTML que par lots.
    Le split test reste réservé à build_phreshphish_test.py.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rnd = random.Random(seed)
    files, pool = {}, defaultdict(list)
    for shard in shards:
        if not shard.startswith("train-"):
            raise SystemExit("--phreshphish-shards : seulement des shards train-* (le test sert à l'évaluation)")
        files[shard] = pq.ParquetFile(hf_hub_download(PHRESHPHISH, f"data/{shard}.parquet", repo_type="dataset"))
        for i, lab in enumerate(files[shard].read(columns=["label"]).column("label").to_pylist()):
            label = label_of(str(lab).lower())
            if label:
                pool[label].append((shard, i))
    picked = {key for label in (PHISHING, LEGITIMATE)
              for key in rnd.sample(pool[label], min(n_per_class, len(pool[label])))}

    out: List[tuple] = []
    for shard, pf in files.items():
        i = 0
        for batch in pf.iter_batches(batch_size=200, columns=["url", "html", "label"]):
            cols = batch.to_pydict()
            for url, html, lab in zip(cols["url"], cols["html"], cols["label"]):
                if (shard, i) in picked:
                    label = label_of(str(lab).lower())
                    page = phreshphish_extraction(url, html)
                    link = to_extraction("url", url or "")
                    if page is not None:
                        out.append((row(page, label, "phreshphish"),
                                    row(link, label, "phreshphish") if link is not None else None))
                i += 1
    print(f"PhreshPhish ({', '.join(shards)}) : " + ", ".join(
        f"{l}={sum(1 for p, _ in out if p['expected']['verdict'] == l)}" for l in (PHISHING, LEGITIMATE)) +
        " pages, chacune aussi en lien seul")
    return out


def split_pairs(pairs: List[tuple], seed: int):
    """Découpe des couples (page, lien seul) : les deux lignes d'un même site vont dans la même partie,
    sinon le modèle verrait à l'entraînement le lien d'une page de test."""
    pages = []
    for i, (page, _) in enumerate(pairs):
        page["_pair"] = i
        pages.append(page)
    parts = split(dedupe(pages), seed)
    seen = set()
    for part in parts:
        for r in list(part):
            link = pairs[r.pop("_pair")][1]
            key = link and json.dumps(link["state"], sort_keys=True, ensure_ascii=False)
            if link is not None and key not in seen:
                seen.add(key)
                part.append(link)
    return parts


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
    ap.add_argument("--phreshphish", type=int, default=0,
                    help="pages PhreshPhish récentes par classe, chacune aussi en lien seul (0 = aucune)")
    ap.add_argument("--phreshphish-shards", nargs="+", default=["train-001"],
                    help="fichiers du split train à télécharger (train-001 : 320 Mo, 5 780 pages)")
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
    rows = dedupe(rows)
    train, val, test = split(rows, args.seed)
    if args.phreshphish:
        for part, extra in zip((train, val, test),
                               split_pairs(load_phreshphish(args.phreshphish, args.seed, args.phreshphish_shards),
                                           args.seed)):
            part += extra

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
