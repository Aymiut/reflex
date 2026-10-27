import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load("build_laya_dataset")
ev = _load("eval_laya")


def test_to_extraction_by_channel():
    assert build.to_extraction("url", "paypal-secure-login.top/verify").urls == ["http://paypal-secure-login.top/verify"]
    page = build.to_extraction("page", '<html><a href="https://x.example.com/a">go</a><p>Sign in</p></html>')
    assert page.channel == "page" and "Sign in" in page.text and page.urls == ["https://x.example.com/a"]
    assert build.to_extraction("sms", "   ") is None


def test_texts_file_is_split_into_email_and_sms():
    assert build._channel_from_filename("texts.json") == "text"
    assert build._channel_from_filename("combined_reduced.json") is None
    assert build._text_channel("WINNER!! You have won a £900 prize. Call 09061701461 now") == "sms"
    assert build._text_channel("re : meeting tomorrow , please confirm the room . thanks") == "email"
    assert build._text_channel("Dear user, " + "x" * 500) == "email"


def test_phreshphish_page_puts_its_address_first():
    ex = build.phreshphish_extraction(
        "https://secure-login.example.top/verify",
        '<html><a href="https://cdn.example.com/a">go</a><p>Sign in to continue</p></html>')
    assert ex.channel == "page" and "Sign in" in ex.text
    assert ex.urls == ["https://secure-login.example.top/verify", "https://cdn.example.com/a"]
    assert build.phreshphish_extraction("", "") is None


def test_split_pairs_keeps_page_and_link_together():
    pairs = []
    for i in range(60):
        label = "phishing" if i % 2 else "legitimate"
        url = f"https://site{i}.example.com/login"
        pairs.append((build.row(build.phreshphish_extraction(url, f"<p>Page {i}</p>"), label, "t"),
                      build.row(build.to_extraction("url", url), label, "t")))
    parts = build.split_pairs(pairs, seed=3)
    where = {}
    for name, part in zip(("train", "val", "test"), parts):
        assert all("_pair" not in r for r in part)
        for r in part:
            where.setdefault(r["state"]["urls"][0], set()).add(name)
    assert len(where) == 60 and all(len(v) == 1 for v in where.values())
    assert sum(len(p) for p in parts) == 120


def test_labels_and_rows():
    assert build.label_of(1) == "phishing" and build.label_of("ham") == "legitimate" and build.label_of("?") is None
    r = build.row(build.to_extraction("sms", "Hello"), "legitimate", "test")
    assert set(r) == {"state", "questions", "expected", "origin"} and r["expected"] == {"verdict": "legitimate"}


def test_dedupe_and_split_keep_classes():
    rows = [build.row(build.to_extraction("sms", f"msg {i}"), "phishing" if i % 2 else "legitimate", "t")
            for i in range(100)]
    rows += rows[:10]
    rows = build.dedupe(rows)
    assert len(rows) == 100
    train, val, test = build.split(rows, seed=1)
    assert len(train) + len(val) + len(test) == 100
    assert {r["expected"]["verdict"] for r in test} == {"phishing", "legitimate"}


def test_cascade_metrics():
    p = np.array([0.001, 0.02, 0.5, 0.96, 0.999, 0.01])
    y = np.array([0, 0, 1, 1, 1, 1])
    r = ev.cascade(p, y, close_t=0.03, block_t=0.97)
    assert r["coverage"] == 4 / 6
    assert r["missed_attacks"] == 1          # le 0.01 était du phishing
    assert r["false_blocks"] == 0
    best = ev.best_thresholds(p, y, max_missed=0.0, max_false_block=0.0)
    assert best["missed_attacks"] == 0


def test_thresholds_follow_compressed_scores():
    # Scores d'un modèle calibré, tous entre 0.02 et 0.98 : la grille fixe seule ne trouvait rien.
    p = np.array([0.021, 0.022, 0.024, 0.975, 0.976, 0.977])
    y = np.array([0, 0, 1, 0, 1, 1])
    best = ev.best_thresholds(p, y, max_missed=0.0, max_false_block=0.0)
    assert 0.022 <= best["close_threshold"] < 0.024 and 0.975 < best["block_threshold"] <= 0.976
    assert best["coverage"] == 4 / 6 and best["missed_attacks"] == 0 and best["false_blocks"] == 0
