import json

from reflex.config import Settings
from reflex.extract import extract_text
from reflex.state import QUESTIONS, build_state
from reflex.triage import BLOCK, CLOSE, ESCALATE, LayaTriage, route_for


def fake_laya(p):
    return lambda state: {"verdict": {"probabilities": {"phishing": p, "legitimate": 1 - p},
                                      "answer_confidence": max(p, 1 - p)}}


def test_routes():
    s = Settings(block_threshold=0.97, close_threshold=0.03)
    assert route_for(0.99, s) == BLOCK
    assert route_for(0.01, s) == CLOSE
    assert route_for(0.5, s) == ESCALATE


def test_decide_uses_state_and_model():
    s = Settings(block_threshold=0.97, close_threshold=0.03)
    ex = extract_text("Votre colis est bloqué, payez 1,99€ : colissimo-suivi.info/fr/1", channel="sms")
    d = LayaTriage(s, predict_fn=fake_laya(0.6)).decide(ex)
    assert d.route == ESCALATE and d.risky
    assert d.state["channel"] == "sms"
    assert "brand-lookalike:colissimo" in d.state["urls"][0]


def test_state_is_json_and_short():
    ex = extract_text("x " * 5000 + " https://a.example.com/" + " ".join(f"https://u{i}.example.com" for i in range(9)))
    state = build_state(ex)
    assert len(state["text"]) <= 1500
    assert len(state["urls"]) == 5
    json.dumps(state)


def test_questions_have_neutral_two_options():
    q = QUESTIONS["verdict"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"phishing", "legitimate"}
