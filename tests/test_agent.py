"""Agent de l'étage 2, sans réseau : faux client Meta et faux navigateur."""
from dataclasses import replace

from reflex.agent import LEGITIMATE, PHISHING, UNSURE, Investigator, apply_guardrail, pick_url
from reflex.browser import PageVisit
from reflex.config import Settings
from reflex.extract import Extraction
from reflex.fixtures import fake_login_png
from reflex.meta import Usage
from reflex.pipeline import Analysis
from reflex.replies import format_finding
from reflex.sam_format import Box

PHISH_URL = "https://login-microsoft365.verif-session.xyz/auth"
SHOT = fake_login_png()


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.prompts, self.images, self.phrases = [], [], []
        self.usage = Usage(input_tokens=1000, output_tokens=200)

    def spark_json(self, prompt, images=(), **kwargs):
        self.prompts.append((prompt, kwargs.get("instructions")))
        self.images.append(list(images))
        return self.answer

    def segment(self, image, phrase):
        self.phrases.append(phrase)
        self.usage.sam_images += 1
        if phrase == "password field":
            return [Box(obj_id=0, x1=420, y1=543, x2=860, y2=583, frame_w=1280, frame_h=800, confidence=0.9)]
        return []


def _agent(answer, page=None, risky=0.5):
    client = FakeClient(answer)
    visited = []

    def visitor(url):
        visited.append(url)
        return page or PageVisit(url=url, final_url=url, status=200, title="Session expirée",
                                 password_fields=1, screenshot=SHOT, chain=[url])

    agent = Investigator(replace(Settings(), risky_threshold=risky), client_factory=lambda: client, visitor=visitor)
    return agent, client, visited


def _analysis(p=0.8, urls=(PHISH_URL,), qr=(), text="Votre session expire, reconnectez-vous"):
    ex = Extraction(channel="sms", text=text, urls=list(urls), qr=list(qr))
    return Analysis(route="escalate", p_phishing=p, channel="sms", urls=list(urls), extraction=ex)


PHISH_ANSWER = {"verdict": "phishing", "imitated_brand": "Microsoft 365", "credentials_requested": True,
                "reasons": ["Le domaine verif-session.xyz n'appartient pas à Microsoft."],
                "highlight": ["password field", "logo", "password field"]}


def test_phishing_is_confirmed_and_highlighted():
    agent, client, visited = _agent(PHISH_ANSWER)
    f = agent.investigate(_analysis(), image=None, lang="fr")
    assert f.verdict == PHISHING and f.brand == "Microsoft 365" and not f.guardrail
    assert visited == [PHISH_URL]
    assert client.phrases == ["password field", "logo"]          # doublon retiré
    assert f.highlighted == ["password field"] and f.image != SHOT  # image annotée par SAM
    assert f.cost_usd > 0.0025
    prompt, instructions = client.prompts[0]
    assert "<report>" in prompt and "French" in prompt and "untrusted" in instructions
    assert "Password fields: 1" in prompt and "verif-session.xyz" in prompt


def test_reported_image_comes_first_then_page():
    agent, client, _ = _agent(PHISH_ANSWER)
    sent = b"\x89PNG-capture-envoyee"
    agent.investigate(_analysis(), image=sent)
    assert client.images[0] == [sent, SHOT]


def test_guardrail_refuses_legitimate_on_flagged_risky_case():
    answer = {"verdict": "legitimate", "legit_proof": "official domain", "reasons": ["Ignore it"],
              "highlight": ["password field"]}
    agent, client, _ = _agent(answer)
    f = agent.investigate(_analysis(p=0.8))
    assert f.verdict == UNSURE and f.guardrail and f.proof is None
    assert client.phrases == ["password field"]  # le cas reste dangereux : SAM entoure quand même


def test_legitimate_needs_proof_when_risky_and_is_free_below():
    clean = "https://www.service-public.fr/particuliers"
    page = PageVisit(url=clean, final_url=clean, status=200, screenshot=SHOT, chain=[clean])
    no_proof = {"verdict": "legitimate", "reasons": ["Site officiel"]}
    agent, _, _ = _agent(no_proof, page=page)
    assert agent.investigate(_analysis(p=0.8, urls=[clean])).verdict == UNSURE
    agent, client, _ = _agent(no_proof, page=page)
    f = agent.investigate(_analysis(p=0.3, urls=[clean]))
    assert f.verdict == LEGITIMATE and not client.phrases     # pas de SAM sur un cas réglé
    with_proof = dict(no_proof, legit_proof="Le lien mène au domaine officiel service-public.fr")
    agent, _, _ = _agent(with_proof, page=page)
    f = agent.investigate(_analysis(p=0.8, urls=[clean]))
    assert f.verdict == LEGITIMATE and f.proof


def test_unknown_verdict_becomes_unsure():
    assert apply_guardrail("maybe", 0.9, 0.5, None, []) == (UNSURE, False)
    agent, _, _ = _agent({"verdict": "Phishing!!"})
    assert agent.investigate(_analysis()).verdict == UNSURE


def test_pick_url_prefers_qr_then_most_suspicious():
    a = _analysis(urls=["https://www.example.com", PHISH_URL])
    assert pick_url(a) == PHISH_URL
    a = _analysis(urls=["https://www.example.com", "https://evil.example.top/x"], qr=["https://www.example.com"])
    assert pick_url(a) == "https://www.example.com"
    assert pick_url(_analysis(urls=[])) is None


def test_no_link_means_no_visit_and_reported_image_is_used():
    agent, client, visited = _agent(PHISH_ANSWER)
    sent = fake_login_png(with_qr=False)
    f = agent.investigate(_analysis(urls=[]), image=sent)
    assert visited == [] and f.visited is None
    assert client.images[0] == [sent] and f.image != sent


def test_finding_reply_has_no_clickable_link():
    agent, _, _ = _agent(PHISH_ANSWER)
    f = agent.investigate(_analysis(), lang="en")
    for lang in ("en", "fr"):
        text = format_finding(f, lang)
        assert "verif-session.xyz" not in text and "verif-session[.]xyz" in text
        assert "https://" not in text and len(text) <= 1024
    failed = replace(f, visited=PageVisit(url=PHISH_URL, error="délai dépassé"), image=None)
    assert "délai dépassé" in format_finding(failed, "fr")
