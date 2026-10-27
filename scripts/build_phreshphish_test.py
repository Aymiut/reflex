"""Jeu de test indépendant tiré de PhreshPhish : vraies pages collectées de septembre à décembre 2025.

Laya n'a jamais vu ces pages (split « test » de PhreshPhish, postérieur à ses données d'entraînement).
Deux fichiers, pour les deux cas du bot :
- test_phreshphish_page.jsonl : adresse + contenu de la page (canal page) ;
- test_phreshphish_url.jsonl  : l'adresse seule (canal url), comme un lien collé dans Telegram.

    python scripts/build_phreshphish_test.py [--shard test-000]
    python scripts/eval_laya.py --model ./models/laya-reflex --close-threshold 0.0235 --block-threshold 0.9751 \\
        --data data/laya/test_phreshphish_page.jsonl --data data/laya/test_phreshphish_url.jsonl

Un fichier (shard) suffit : test-000 pèse 55 Mo pour 1 000 pages, le split complet environ 9 Go.
PhreshPhish : CC BY 4.0, réservé à la recherche anti-phishing (Dalton et al., 2025, arXiv:2507.10854).
Les fichiers restent dans data/ (hors git) : les pages peuvent contenir des données personnelles.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_laya_dataset as build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="test-000", help="fichier du split test, ex. test-000")
    ap.add_argument("--out", default=str(ROOT / "data" / "laya"))
    args = ap.parse_args()
    if not args.shard.startswith("test-"):
        raise SystemExit("Seul le split test sert ici : le split train pourrait recouper un futur entraînement.")

    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(build.PHRESHPHISH, f"data/{args.shard}.parquet", repo_type="dataset")
    df = pd.read_parquet(path, columns=["url", "html", "label", "lang", "date"])
    print(f"{args.shard} : {len(df)} pages du {df.date.min()} au {df.date.max()}")

    pages, urls = [], []
    for url, html, lab, lang in zip(df.url, df.html, df.label, df.lang):
        label = build.label_of(str(lab).lower())
        if label is None:
            continue
        origin = f"phreshphish:{lang}"
        page = build.phreshphish_extraction(url, html)
        if page is not None:
            pages.append(build.row(page, label, origin))
        link = build.to_extraction("url", url or "")
        if link is not None:
            urls.append(build.row(link, label, origin))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    build.write_jsonl(out / "test_phreshphish_page.jsonl", build.dedupe(pages))
    build.write_jsonl(out / "test_phreshphish_url.jsonl", build.dedupe(urls))
    langs = Counter(df.lang)
    print("langues : " + ", ".join(f"{k}={v}" for k, v in langs.most_common(8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
