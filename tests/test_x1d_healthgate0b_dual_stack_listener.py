"""The listener serves IPv4 and IPv6, or it does not start (X1D-HEALTHGATE0-B).

Staging answered 502 on its public URL for its entire life. The deployment was
SUCCESS, the instance RUNNING, the application healthy, and `uvicorn --host ::`
had produced a single IPv6-only listener: Railway's private network reached it
over IPv6 so xerbs-core kept working, while Railway's public edge connects over
IPv4 and was refused at the TCP layer. Nothing detected it because nothing
asserted that a deployed instance actually answers.

These tests are that assertion, at the listener level. The one that matters
most is ``test_ipv4_loopback_serves``: it fails against any implementation that
binds a single IPv6 socket, which is exactly the regression to guard.

Nothing here needs Railway, PostgreSQL, OpenAI, or the network beyond loopback.
A stub ASGI application stands in for ``app.main:app`` so importing this module
never initialises a database.
"""

import asyncio
import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from app.server import DEFAULT_PORT, build_sockets, resolve_port
from app import server as server_module


# --------------------------------------------------------------------------
# stub application + harness
# --------------------------------------------------------------------------

def _make_stub():
    """An ASGI app that answers 200 and counts its lifespan startups."""
    startups = []

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    startups.append(time.time())
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        body = json.dumps({"status": "healthy"}).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    return app, startups


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _get(url: str):
    """Loopback GET with proxies disabled, so a developer's proxy cannot skew it."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    response = opener.open(url, timeout=10)
    return response.status, response.read().decode()


@contextlib.contextmanager
def _serving():
    """Run one uvicorn server over both sockets, as app.server does."""
    app, startups = _make_stub()
    port = _free_port()
    sockets = build_sockets(port)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": sockets},
                              daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start over the supplied sockets"
    try:
        yield port, server, sockets, startups
    finally:
        server.should_exit = True
        thread.join(timeout=20)
        for sock in sockets:
            with contextlib.suppress(Exception):
                sock.close()


# --------------------------------------------------------------------------
# 1. both families are bound
# --------------------------------------------------------------------------

def test_build_sockets_binds_one_socket_per_family():
    port = _free_port()
    sockets = build_sockets(port)
    try:
        families = sorted(s.family for s in sockets)
        assert families == sorted([socket.AF_INET, socket.AF_INET6])
        assert len(sockets) == 2

        by_family = {s.family: s for s in sockets}
        assert by_family[socket.AF_INET].getsockname()[:2] == ("0.0.0.0", port)
        assert by_family[socket.AF_INET6].getsockname()[:2] == ("::", port)

        # Explicitly IPv6-only, which is what keeps the two binds disjoint and
        # what leaves the working private path semantically unchanged.
        assert by_family[socket.AF_INET6].getsockopt(
            socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
    finally:
        for sock in sockets:
            sock.close()


# --------------------------------------------------------------------------
# 2-3. both families actually serve -- connectivity, not just bind()
# --------------------------------------------------------------------------

def test_ipv4_loopback_serves():
    """The regression sentinel: fails against a single IPv6-only listener."""
    with _serving() as (port, _server, _sockets, _startups):
        status, body = _get(f"http://127.0.0.1:{port}/health")
        assert status == 200
        assert json.loads(body)["status"] == "healthy"


def test_ipv6_loopback_serves():
    with _serving() as (port, _server, _sockets, _startups):
        status, body = _get(f"http://[::1]:{port}/health")
        assert status == 200
        assert json.loads(body)["status"] == "healthy"


# --------------------------------------------------------------------------
# 4. one server, one lifespan, two listeners
# --------------------------------------------------------------------------

def test_both_listeners_share_one_server_lifecycle():
    with _serving() as (port, server, _sockets, startups):
        assert len(server.servers) == 2, "expected one asyncio server per socket"
        assert _get(f"http://127.0.0.1:{port}/health")[0] == 200
        assert _get(f"http://[::1]:{port}/health")[0] == 200
        # Two listeners must not mean two applications.
        assert len(startups) == 1


# --------------------------------------------------------------------------
# 5. partial bind failure is atomic
# --------------------------------------------------------------------------

def test_partial_bind_failure_closes_what_it_made(monkeypatch):
    """Deterministic: the second bind raises, the first must not survive it."""
    created = []
    real_listen = server_module._listen

    def failing_listen(family, host, port, backlog, v6only=None):
        if family == socket.AF_INET:
            raise OSError("simulated IPv4 bind failure")
        sock = real_listen(family, host, port, backlog, v6only=v6only)
        created.append(sock)
        return sock

    monkeypatch.setattr(server_module, "_listen", failing_listen)

    with pytest.raises(OSError):
        build_sockets(_free_port())

    assert created, "the IPv6 socket should have been created before the failure"
    assert all(s.fileno() == -1 for s in created), \
        "a failed startup must not leave a listening socket behind"


def test_occupied_ipv4_port_prevents_startup():
    """Realistic: something already holding the IPv4 wildcard must fail us closed."""
    port = _free_port()
    occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupier.bind(("0.0.0.0", port))
    occupier.listen(8)
    try:
        with pytest.raises(OSError):
            build_sockets(port)
        # The IPv6 half must have been released, not leaked.
        probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        try:
            probe.bind(("::", port))
        finally:
            probe.close()
    finally:
        occupier.close()


# --------------------------------------------------------------------------
# 6. PORT semantics, unchanged from ${PORT:-8080}
# --------------------------------------------------------------------------

def test_port_defaults_to_8080_when_unset(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert resolve_port() == DEFAULT_PORT == 8080


def test_port_defaults_to_8080_when_empty(monkeypatch):
    monkeypatch.setenv("PORT", "")
    assert resolve_port() == 8080


def test_port_env_is_respected(monkeypatch):
    monkeypatch.setenv("PORT", "9137")
    assert resolve_port() == 9137


def test_invalid_port_fails_rather_than_guessing(monkeypatch):
    monkeypatch.setenv("PORT", "not-a-port")
    with pytest.raises(ValueError):
        resolve_port()


# --------------------------------------------------------------------------
# 7. why the launcher exists, pinned as a fact about asyncio
# --------------------------------------------------------------------------

def test_asyncio_host_bind_is_ipv6_only():
    """`uvicorn --host ::` cannot be dual-stack, because asyncio forbids it.

    Asserted on the socket option rather than on connectivity, so the result is
    the same on every platform the suite runs on.
    """
    async def check():
        srv = await asyncio.get_running_loop().create_server(
            asyncio.Protocol, host="::", port=0)
        try:
            sock = srv.sockets[0]
            assert sock.family == socket.AF_INET6
            return sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
        finally:
            srv.close()
            await srv.wait_closed()

    assert asyncio.run(check()) == 1


# --------------------------------------------------------------------------
# 8. shutdown releases both
# --------------------------------------------------------------------------

def test_shutdown_closes_both_sockets():
    app, _ = _make_stub()
    port = _free_port()
    sockets = build_sockets(port)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": sockets},
                              daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started

    server.should_exit = True
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert all(s.fileno() == -1 for s in sockets)
