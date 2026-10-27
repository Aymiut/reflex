"""Navigateur de l'agent : règles réseau, proxy et visite réelle d'une fausse page servie en local."""
import asyncio
import http.server
import threading

import pytest

from reflex.browser import Blocked, ip_blocked, port_allowed, resolve_public, start_proxy, visit
from reflex.fixtures import fake_login_html


def test_only_public_addresses_are_allowed():
    for ip in ("127.0.0.1", "10.0.0.5", "172.16.3.4", "192.168.1.1", "169.254.169.254", "100.64.0.1",
               "0.0.0.0", "224.0.0.1", "::1", "fd00::1", "fe80::1", "::ffff:192.168.1.1"):
        assert ip_blocked(ip), ip
    for ip in ("8.8.8.8", "93.184.215.14", "2606:4700:4700::1111"):
        assert not ip_blocked(ip), ip
    assert port_allowed(443) and port_allowed(8443) and port_allowed(3000)
    assert not port_allowed(22) and not port_allowed(25)


def test_resolve_refuses_private_targets():
    with pytest.raises(Blocked):
        asyncio.run(resolve_public("192.168.1.1", 80))
    with pytest.raises(Blocked):
        asyncio.run(resolve_public("8.8.8.8", 25))
    assert asyncio.run(resolve_public("192.168.1.1", 80, allow_private=True)) == "192.168.1.1"


@pytest.fixture()
def local_site():
    html = fake_login_html().encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()


async def _through_proxy(request: bytes, allow_private: bool) -> bytes:
    server, port, blocked = await start_proxy(allow_private)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(request)
        await writer.drain()
        data = await asyncio.wait_for(reader.read(), 10)
        writer.close()
        return data
    finally:
        server.close()


def test_proxy_blocks_local_network(local_site):
    answer = asyncio.run(_through_proxy(f"CONNECT {local_site} HTTP/1.1\r\nHost: {local_site}\r\n\r\n".encode(),
                                        allow_private=False))
    assert answer.startswith(b"HTTP/1.1 403") and b"X-Reflex-Blocked" in answer
    answer = asyncio.run(_through_proxy(f"GET http://{local_site}/ HTTP/1.1\r\nHost: {local_site}\r\n\r\n".encode(),
                                        allow_private=False))
    assert answer.startswith(b"HTTP/1.1 403")


def test_proxy_forwards_plain_http(local_site):
    answer = asyncio.run(_through_proxy(
        f"GET http://{local_site}/login HTTP/1.1\r\nHost: {local_site}\r\nProxy-Connection: keep-alive\r\n\r\n".encode(),
        allow_private=True))
    assert answer.startswith(b"HTTP/1.0 200") or answer.startswith(b"HTTP/1.1 200")
    assert "Mot de passe".encode() in answer


def test_visit_refuses_other_schemes():
    assert "http" in visit("file:///etc/passwd").error
    assert "http" in visit("javascript:alert(1)").error


def _visit_or_skip(url: str, allow_private: bool):
    pytest.importorskip("playwright")
    page = visit(url, allow_private=allow_private)
    if "Executable doesn't exist" in page.error or "playwright install" in page.error:
        pytest.skip("Chromium de Playwright non installé (python -m playwright install --only-shell chromium)")
    return page


def test_visit_reads_a_login_page(local_site):
    page = _visit_or_skip(f"http://{local_site}/login", allow_private=True)
    assert page.ok, page.error
    assert page.status == 200 and "Session expirée" in page.title
    assert page.password_fields == 1 and page.inputs == 2
    assert page.form_actions == [f"http://{local_site}/collect"]
    assert page.screenshot[:2] == b"\xff\xd8"  # JPEG


def test_visit_blocks_local_addresses(local_site):
    page = _visit_or_skip(f"http://{local_site}/login", allow_private=False)
    assert not page.ok and "non publique" in page.error
    assert page.blocked
