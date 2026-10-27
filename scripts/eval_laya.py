"""Évalue un checkpoint Laya sur un JSONL de test et choisit les seuils de la cascade.

Autonome (dépend seulement de `laya` et `numpy`) pour tourner aussi bien sur Kaggle que sur le Mac :

    python eval_laya.py --model ./laya-reflex --data test.jsonl [--data test_fr.jsonl ...] --out report.json

Ce qui compte pour Reflex n'est pas seulement la précision, mais :
- la COUVERTURE : la part des signalements que Laya règle seul (sans appel au gros modèle) ;
- les ATTAQUES RATÉES : du phishing que Laya classe "légitime" tout seul — l'erreur à ne pas faire ;
- les FAUX BLOCAGES : du légitime que Laya bloque tout seul.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PHISHING = "phishing"


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def predict_probs(agent, rows: List[Dict], batch_size: int = 32) -> np.ndarray:
    questions = rows[0]["questions"]
    states = [r["state"] for r in rows]
    probs = []
    for i in range(0, len(states), batch_size):
        out = agent.predict_batch(states[i:i + batch_size], questions, batch_size=batch_size)
        for r in out:
            ans = r.get("answers", r)["verdict"]
            probs.append(float(ans["probabilities"][PHISHING]))
    return np.array(probs)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def cascade(p: np.ndarray, y: np.ndarray, close_t: float, block_t: float) -> Dict:
    closed, blocked = p <= close_t, p >= block_t
    auto = closed | blocked
    n = len(p)
    return {
        "close_threshold": close_t,
        "block_threshold": block_t,
        "coverage": float(auto.mean()),                                   # réglé par Laya seul
        "escalated": float((~auto).mean()),                              # envoyé à l'agent Meta
        "missed_attacks": int((closed & (y == 1)).sum()),                # phishing fermé à tort
        "missed_attack_rate": float((closed & (y == 1)).sum() / max(1, (y == 1).sum())),
        "false_blocks": int((blocked & (y == 0)).sum()),                 # légitime bloqué à tort
        "false_block_rate": float((blocked & (y == 0)).sum() / max(1, (y == 0).sum())),
        "n": n,
    }


def best_thresholds(p: np.ndarray, y: np.ndarray, max_missed: float, max_false_block: float) -> Dict:
    grid = [0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
    best = None
    for c in grid:
        for b in (0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
            r = cascade(p, y, c, b)
            if r["missed_attack_rate"] <= max_missed and r["false_block_rate"] <= max_false_block:
                if best is None or r["coverage"] > best["coverage"]:
                    best = r
    return best or cascade(p, y, 0.0, 1.0 + 1e-9)


def evaluate(agent, path: str, max_missed: float, max_false_block: float) -> Dict:
    rows = read_jsonl(path)
    y = np.array([1 if r["expected"]["verdict"] == PHISHING else 0 for r in rows])
    t0 = time.perf_counter()
    p = predict_probs(agent, rows)
    elapsed = time.perf_counter() - t0
    pred = (p >= 0.5).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    by_channel = {}
    for ch in sorted({r["state"]["channel"] for r in rows}):
        m = np.array([r["state"]["channel"] == ch for r in rows])
        by_channel[ch] = {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean())}
    return {
        "file": Path(path).name,
        "n": len(rows),
        "accuracy": float((pred == y).mean()),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "ece": ece(p, y),
        "ms_per_item_batched": elapsed * 1000 / max(1, len(rows)),
        "by_channel": by_channel,
        "cascade_default": cascade(p, y, 0.03, 0.97),
        "cascade_best": best_thresholds(p, y, max_missed, max_false_block),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dossier du checkpoint ou alias (multilingual, english)")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-missed", type=float, default=0.005, help="taux max d'attaques fermées à tort")
    ap.add_argument("--max-false-block", type=float, default=0.02, help="taux max de légitimes bloqués à tort")
    ap.add_argument("--out", default="laya_report.json")
    args = ap.parse_args()

    import laya

    agent = laya.load(args.model, device=args.device)
    report = [evaluate(agent, d, args.max_missed, args.max_false_block) for d in args.data]
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for r in report:
        b = r["cascade_best"]
        print(f"\n{r['file']} (n={r['n']}) accuracy={r['accuracy']:.3f} precision={r['precision']:.3f} "
              f"recall={r['recall']:.3f} ECE={r['ece']:.3f} {r['ms_per_item_batched']:.1f} ms/item")
        print(f"  seuils retenus : fermer si p<={b['close_threshold']}, bloquer si p>={b['block_threshold']} "
              f"-> Laya règle {b['coverage']:.1%} seul, {b['escalated']:.1%} vont à l'agent, "
              f"{b['missed_attacks']} attaques ratées, {b['false_blocks']} faux blocages")
    print("\nRapport complet :", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
