import pytest

from reflex.config import Settings
from reflex.fixtures import DEMO_URL
from reflex.pipeline import UNREADABLE, Pipeline, text_extraction
from reflex.replies import defang, describe, format_reply
from reflex.triage import BLOCK, ESCALATE, LayaTriage


def fake_laya(p):
    return lambda state: {"verdict": {"probabilities": {"phishing": p, "legitimate": 1 - p}}}


def pipeline(p, **kw):
    s = Settings(block_threshold=0.97, close_threshold=0.03, **kw)
    return Pipeline(s, triage=LayaTriage(s, predict_fn=fake_laya(p)))


def test_link_alone_is_formatted_like_training_urls():
    ex = text_extraction("  paypal-secure-login.top/verify ")
    assert ex.channel == "url" and ex.text == "" and ex.urls == ["http://paypal-secure-login.top/verify"]
    msg = text_extraction(f"Your password expires today, verify now: {DEMO_URL}")
    assert msg.channel == "sms" and msg.urls == [DEMO_URL] and "password" in msg.text


def test_analysis_carries_route_signals_and_latency():
    a = pipeline(0.99).from_text(f"Verify your account: {DEMO_URL}")
    assert a.route == BLOCK and a.p_phishing == 0.99 and a.urls == [DEMO_URL]
    assert "brand-lookalike:microsoft" in a.signals and "tld:xyz" in a.signals
    assert a.latency_ms > 0


def test_nothing_readable_skips_laya():
    def boom(state):
        raise AssertionError("Laya ne doit pas être appelé")

    s = Settings()
    a = Pipeline(s, triage=LayaTriage(s, predict_fn=boom)).from_text("   ")
    assert a.route == UNREADABLE and a.p_phishing is None
    assert "screenshot" in format_reply(a, "en")


def test_reply_never_contains_a_clickable_link():
    a = pipeline(0.5).from_text(f"Hello, see {DEMO_URL} and https://bit.ly/x1")
    assert a.route == ESCALATE
    for lang in ("en", "fr"):
        reply = format_reply(a, lang)
        assert "http" not in reply and "verif-session.xyz" not in reply
        assert "hxxps://login-microsoft365[.]verif-session[.]xyz/auth" in reply
    assert "Pretends to be Microsoft".lower() in format_reply(a, "en").lower()
    assert "se fait passer pour Microsoft" in format_reply(a, "fr")


def test_defang_and_describe():
    assert defang("http://a.b.example.com/x.html") == "hxxp://a[.]b[.]example[.]com/x[.]html"
    assert defang("https://x.com/" + "a" * 200).endswith("…")
    assert describe("tld:xyz", "en") == "unusual domain ending .xyz"
    assert describe("inconnu", "fr") == "inconnu"


def test_allowed_users_parsing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123, 456 789")
    assert Settings().telegram_allowed_users == {123, 456, 789}
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "")
    assert Settings().telegram_allowed_users == frozenset()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123,abc")
    with pytest.raises(ValueError):
        Settings()


def test_bot_only_serves_allowed_users():
    telegram = pytest.importorskip("telegram")
    from reflex.bot import build_app, is_allowed

    s = Settings(telegram_token="123456:TEST", telegram_allowed_users=frozenset({42}))
    user = lambda uid: telegram.Update(1, message=telegram.Message(
        1, None, telegram.Chat(uid, "private"), from_user=telegram.User(uid, "x", False), text="hi"))
    assert is_allowed(s, user(42)) and not is_allowed(s, user(7))
    app = build_app(s, pipeline(0.5))  # aucune requête réseau à la construction
    assert app.bot_data["settings"] is s and len(app.handlers[0]) == 6
