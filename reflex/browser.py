"""Navigateur de l'agent : ouvre un lien suspect dans un Chromium headless jetable et rapporte ce qu'on y voit.

Côté Reflex, on appelle `visit(url)`. La visite tourne dans un processus Python à part (`python -m reflex.browser`),
avec un délai maximal : si la page bloque ou fait planter le navigateur, seul ce processus meurt (avec Chromium).

Précautions :
- le processus de visite ne reçoit aucun secret : environnement réduit, et ce module n'importe pas reflex.config (.env) ;
- sandbox de Chromium activée, contexte neuf : aucun cookie, mot de passe ou profil ;
- téléchargements, service workers, WebSockets, popups et boîtes de dialogue refusés ; WebRTC ne sort que par le proxy ;
- tout le trafic passe par un petit proxy local qui résout lui-même chaque nom de domaine et refuse les adresses non publiques
  (localhost, réseau local, box, Pi-hole, intranet de l'hôpital, plages réservées), y compris après une redirection ;
- on ne tape rien dans les formulaires et on ne clique sur rien.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlsplit

NAV_TIMEOUT_MS = 15_000       # chargement de la page
SETTLE_MS = 4_000             # attente du calme réseau (pages qui se construisent en JavaScript)
VISIT_BUDGET_S = 35           # budget total d'une visite, dans le processus enfant
PROCESS_TIMEOUT_S = 45        # au-delà, le parent tue le processus et Chromium
MAX_TEXT = 3000
VIEWPORT = {"width": 1280, "height": 800}
WEB_PORTS = {80, 443, 8080, 8443}
# Variables transmises au processus de visite (rien d'autre : surtout pas les clés de .env).
SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PLAYWRIGHT_BROWSERS_PATH")


# --------------------------------------------------------------------------- règles réseau
class Blocked(Exception):
    """Destination refusée par le proxy."""


def ip_blocked(ip: str) -> bool:
    """Vrai si l'adresse n'est pas une adresse publique d'Internet."""
    addr = ipaddress.ip_address(ip.split("%", 1)[0])
    if addr.version == 6 and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return not addr.is_global or addr.is_multicast


def port_allowed(port: int) -> bool:
    return port in WEB_PORTS or port >= 1024


async def resolve_public(host: str, port: int, allow_private: bool = False) -> str:
    """Résout `host` et renvoie une adresse à joindre, ou lève Blocked.

    Le proxy se connecte ensuite à cette adresse précise : un nom qui change d'adresse entre deux
    résolutions (DNS rebinding) ne peut pas contourner la vérification.
    """
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        addrs = [host]
    except ValueError:
        try:
            infos = await asyncio.wait_for(
                asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM), 10)
        except (OSError, asyncio.TimeoutError) as e:
            raise Blocked(f"nom introuvable : {host}") from e
        addrs = list(dict.fromkeys(info[4][0] for info in infos))
    if not addrs:
        raise Blocked(f"nom introuvable : {host}")
    if not allow_private:
        if not port_allowed(port):
            raise Blocked(f"port refusé : {port}")
        if any(ip_blocked(a) for a in addrs):
            raise Blocked(f"adresse non publique : {host}")
    return addrs[0]


# --------------------------------------------------------------------------- proxy local
_FORBIDDEN = b"HTTP/1.1 403 Forbidden\r\nX-Reflex-Blocked: 1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_BAD = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_HOP_HEADERS = {"proxy-connection", "connection", "keep-alive", "proxy-authorization", "te", "upgrade"}


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await asyncio.wait_for(reader.read(65536), 30)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _proxy_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *,
                        allow_private: bool, blocked: List[str]) -> None:
    """Une connexion de Chromium : CONNECT (https, wss) ou requête http en forme absolue."""
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15)
        lines = [l for l in head.decode("latin-1").split("\r\n") if l]
        method, target, version = lines[0].split(" ", 2)
        if method == "CONNECT":
            parts = urlsplit("//" + target)
            host, port = parts.hostname, parts.port or 443
        else:
            parts = urlsplit(target)
            if parts.scheme != "http" or not parts.hostname:
                writer.write(_BAD)
                return
            host, port = parts.hostname, parts.port or 80
        try:
            ip = await resolve_public(host, port, allow_private)
        except Blocked as e:
            blocked.append(str(e))
            writer.write(_FORBIDDEN)
            return
        up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(ip, port), 10)
        if method == "CONNECT":
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        else:
            path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            headers = [l for l in lines[1:] if l.split(":", 1)[0].strip().lower() not in _HOP_HEADERS]
            # Une requête par connexion : Chromium ne peut pas réutiliser ce tunnel pour un autre site.
            up_writer.write(f"{method} {path} {version}\r\n".encode("latin-1")
                            + "".join(h + "\r\n" for h in headers).encode("latin-1")
                            + b"Connection: close\r\n\r\n")
        await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))
    except Exception:
        pass
    finally:
        try:
            await writer.drain()
            writer.close()
        except Exception:
            pass


async def start_proxy(allow_private: bool = False):
    """Démarre le proxy sur 127.0.0.1, port libre. Renvoie (serveur, port, liste des refus)."""
    blocked: List[str] = []
    server = await asyncio.start_server(
        functools.partial(_proxy_client, allow_private=allow_private, blocked=blocked), "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1], blocked


# --------------------------------------------------------------------------- visite (processus enfant)
_PAGE_FACTS = """() => {
  const visible = el => !['hidden', 'submit', 'button', 'checkbox', 'radio'].includes((el.type || '').toLowerCase());
  return {
    password_fields: document.querySelectorAll('input[type=password]').length,
    inputs: [...document.querySelectorAll('input')].filter(visible).length,
    form_actions: [...document.forms].map(f => f.action).filter(Boolean).slice(0, 5),
    iframes: document.querySelectorAll('iframe').length,
    text: document.body ? document.body.innerText : '',
  };
}"""


async def _visit(url: str, allow_private: bool) -> dict:
    from playwright.async_api import async_playwright

    out: dict = {"url": url, "chain": []}
    server, port, blocked = await start_proxy(allow_private)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                chromium_sandbox=True,
                proxy={"server": f"http://127.0.0.1:{port}"},
                args=["--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                      "--webrtc-ip-handling-policy=disable_non_proxied_udp"],
            )
            major = browser.version.split(".")[0]
            context = await browser.new_context(
                accept_downloads=False,
                service_workers="block",
                viewport=VIEWPORT,
                locale="en-US",
                # Certains kits de phishing montrent une page anodine aux navigateurs « headless ».
                user_agent=(f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                            f"(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"),
            )

            async def only_web(route):
                if urlsplit(route.request.url).scheme in ("http", "https", "data", "blob"):
                    await route.continue_()
                else:
                    await route.abort("blockedbyclient")

            async def no_websocket(ws):
                await ws.close()

            await context.route("**/*", only_web)
            await context.route_web_socket("**/*", no_websocket)
            page = await context.new_page()
            context.on("page", lambda popup: asyncio.ensure_future(popup.close()))
            page.on("dialog", lambda dialog: asyncio.ensure_future(dialog.dismiss()))
            page.on("framenavigated",
                    lambda frame: out["chain"].append(frame.url) if frame == page.main_frame else None)

            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                if response is not None and response.headers.get("x-reflex-blocked"):
                    raise Blocked("lien vers une adresse non publique : refusé")
                if response is not None:
                    out["status"] = response.status
                    req, hops = response.request, []
                    while req.redirected_from is not None:
                        req = req.redirected_from
                        hops.insert(0, req.url)
                    out["chain"] = hops + out["chain"]
                try:
                    await page.wait_for_load_state("networkidle", timeout=SETTLE_MS)
                except Exception:
                    pass
                facts = await page.evaluate(_PAGE_FACTS)
                out.update(
                    final_url=page.url,
                    title=(await page.title())[:300],
                    text=(facts.pop("text") or "")[:MAX_TEXT],
                    **facts,
                )
                shot = await page.screenshot(type="jpeg", quality=80)
                out["screenshot_b64"] = base64.b64encode(shot).decode()
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            await browser.close()
    finally:
        server.close()
    out["chain"] = list(dict.fromkeys(u for u in out["chain"] if u and u != "about:blank"))
    out["blocked"] = list(dict.fromkeys(blocked))[:10]
    return out


def _child_main() -> int:
    ap = argparse.ArgumentParser(description="Visite isolée d'une URL (usage interne de Reflex).")
    ap.add_argument("--allow-private", action="store_true", help="tests uniquement : autorise localhost")
    ap.add_argument("url")
    args = ap.parse_args()
    try:
        result = asyncio.run(asyncio.wait_for(_visit(args.url, args.allow_private), VISIT_BUDGET_S))
    except Exception as e:
        result = {"url": args.url, "error": f"{type(e).__name__}: {e}"[:300]}
    sys.stdout.write(json.dumps(result))
    return 0


# --------------------------------------------------------------------------- côté Reflex
@dataclass
class PageVisit:
    url: str
    final_url: str = ""
    status: Optional[int] = None
    chain: List[str] = field(default_factory=list)      # adresses traversées (redirections comprises)
    title: str = ""
    text: str = ""
    password_fields: int = 0
    inputs: int = 0
    form_actions: List[str] = field(default_factory=list)
    iframes: int = 0
    screenshot: Optional[bytes] = None                  # JPEG de la partie visible de la page
    blocked: List[str] = field(default_factory=list)    # requêtes refusées par le proxy
    error: str = ""
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.screenshot is not None


def visit(url: str, allow_private: bool = False, timeout_s: float = PROCESS_TIMEOUT_S) -> PageVisit:
    """Ouvre `url` dans un processus isolé. Ne lève pas d'exception : l'erreur est dans `PageVisit.error`."""
    t0 = time.perf_counter()
    if urlsplit(url).scheme not in ("http", "https"):
        return PageVisit(url=url, error="seuls les liens http(s) sont ouverts")
    cmd = [sys.executable, "-I", "-m", "reflex.browser"] + (["--allow-private"] if allow_private else []) + ["--", url]
    env = {k: os.environ[k] for k in SAFE_ENV if k in os.environ}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
    try:
        raw, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # le processus et tout Chromium avec lui
        proc.communicate()
        return PageVisit(url=url, error="délai dépassé", latency_ms=(time.perf_counter() - t0) * 1000)
    try:
        data = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        data = {"error": "réponse illisible du navigateur"}
    shot = data.pop("screenshot_b64", None)
    known = {k: v for k, v in data.items() if k in PageVisit.__dataclass_fields__}
    result = PageVisit(**{"url": url, **known})
    result.screenshot = base64.b64decode(shot) if shot else None
    if not result.ok and not result.error:
        result.error = "page non chargée"
    result.latency_ms = (time.perf_counter() - t0) * 1000
    return result


if __name__ == "__main__":
    sys.exit(_child_main())
