"""Teste la forme des requêtes envoyées à la Meta Model API contre un faux serveur local."""
import json
from email.parser import BytesParser
from email.policy import default as email_policy
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from reflex.config import Settings
from reflex.fixtures import fake_login_png
from reflex.meta import MetaClient

SAM_TEXT = "<0f>0<|box;x1=980;y1=300;x2=1199;y2=519;w=1280;h=800;c=0.93|><|mask;x=0;y=0;data=2,2,~ab|>"
seen = {}


def _response(text):
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "model": "muse-spark-1.3", "status": "completed",
        "output": [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"object": "list", "data": [{"id": m, "object": "model", "created": 0, "owned_by": "meta"}
                                                   for m in ("sam-3.1", "muse-spark-1.3", "muse-voice-transcribe-1.0")]})

    def do_POST(self):
        seen["auth"] = self.headers.get("Authorization")
        if self.path.startswith("/v1/responses"):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.setdefault("responses", []).append(body)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [
                    {"type": "response.output_text.delta", "sequence_number": 1, "item_id": "msg_1",
                     "output_index": 0, "content_index": 0, "delta": SAM_TEXT[:30], "logprobs": []},
                    {"type": "response.output_text.delta", "sequence_number": 2, "item_id": "msg_1",
                     "output_index": 0, "content_index": 0, "delta": SAM_TEXT[30:], "logprobs": []},
                    {"type": "response.completed", "sequence_number": 3, "response": _response(SAM_TEXT)},
                ]
                for e in events:
                    self.wfile.write(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode())
                self.wfile.flush()
                return
            self._json(_response('```json\n{"is_phishing": true, "imitated_brand": "Microsoft"}\n```'))
        elif self.path.startswith("/v1/asr/transcribe"):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            msg = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
            parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
            seen["asr_request"] = json.loads(parts["request"].get_payload(decode=True))
            seen["asr_audio_len"] = len(parts["audio"].get_payload(decode=True))
            seen["asr_query"] = self.path
            self._json({"sessionId": "reflex", "transcript": "bonjour ici le service informatique",
                        "audioDurationMs": 4200, "turns": []})


@pytest.fixture(scope="module")
def client():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    s = Settings(meta_api_key="test-key", meta_base_url=f"http://127.0.0.1:{server.server_port}/v1")
    yield MetaClient(s)
    server.shutdown()


def test_models(client):
    assert client.list_models() == ["muse-spark-1.3", "muse-voice-transcribe-1.0", "sam-3.1"]


def test_spark_sends_image_as_data_url_and_parses_json(client):
    answer = client.spark_json("Analyse", [fake_login_png()])
    assert answer == {"is_phishing": True, "imitated_brand": "Microsoft"}
    content = seen["responses"][-1]["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image" and content[1]["image_url"].startswith("data:image/png;base64,")
    assert seen["auth"] == "Bearer test-key"
    assert client.usage.input_tokens >= 1000


def test_segment_streams_and_parses_boxes(client):
    boxes = client.segment(fake_login_png(), "QR code")
    body = seen["responses"][-1]
    assert body["model"] == "sam-3.1" and body["stream"] is True
    assert body["metadata"] == {"include_confidence": "true"}
    assert body["input"][0]["content"][0] == {"type": "input_text", "text": "QR code"}
    assert len(boxes) == 1 and boxes[0].confidence == 0.93 and boxes[0].has_mask


def test_transcribe_multipart(client):
    text = client.transcribe(b"RIFF" + b"\0" * 100, language_bias=("French",))
    assert text == "bonjour ici le service informatique"
    assert seen["asr_request"] == {"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0",
                                   "audioEncoding": "WAV", "languageBias": ["French"]}
    assert seen["asr_audio_len"] == 104
    assert "sessionId=reflex" in seen["asr_query"]
    assert client.usage.audio_seconds == pytest.approx(4.2)
