"""Application launch: bind IPv4 and IPv6 explicitly, then serve both.

`uvicorn --host ::` does not produce a dual-stack listener here, and staging
spent its whole life answering 502 on the public URL because of it. asyncio
creates the socket itself in that path and unconditionally sets

    setsockopt(IPPROTO_IPV6, IPV6_V6ONLY, True)

overriding the container's permissive net.ipv6.bindv6only=0. The result was a
single IPv6-only listener: Railway's private network reached it (xerbs-core
resolves the AAAA record and connects over IPv6, so the clinical pipeline kept
working), while Railway's public edge connects over IPv4 and was refused at the
TCP layer -- a ~5 ms connection refusal the platform reported as a healthy
deployment, because nothing was checking.

So the sockets are built here instead. asyncio does not touch IPV6_V6ONLY on a
socket handed to it, so what is configured below is what gets served.

Two explicit sockets rather than one dual-stack socket. A dual-stack socket is
fewer lines, but a single bind() cannot attest that IPv4 traffic actually
reaches this process -- it succeeds whether or not the IPv4 half took effect,
which is precisely the "IPv4 failed but IPv6 started anyway" outcome this must
never produce. An explicit AF_INET bind either succeeds or raises. It also
leaves the IPv6 socket configured exactly as asyncio already configures it
today, so the currently-working private path gains a sibling rather than new
semantics, and IPv4 clients keep their native addresses instead of arriving as
::ffff:10.x.x.x.

Both binds are required. A partial bind closes what it made and re-raises, so
the process cannot come up serving one family.
"""

import contextlib
import logging
import os
import socket

import uvicorn

# uvicorn's own logger, so the startup line below renders identically to the
# "Uvicorn running on ..." message it replaces. uvicorn suppresses that message
# when sockets are passed to Server.run(), and it is the line every deployment
# check in this project reads to decide the application came up.
LOG = logging.getLogger("uvicorn.error")

DEFAULT_PORT = 8080


def resolve_port() -> int:
    """PORT when it is set and non-empty, else 8080.

    Matches the shell's `${PORT:-8080}` that this launcher replaces. A PORT that
    is not an integer raises rather than falling back: listening on a different
    port than the platform routes to is the failure we are here to fix, and it
    is better to not start than to start somewhere nobody is looking.
    """
    raw = os.environ.get("PORT") or ""
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        raise ValueError("PORT is not an integer; refusing to guess a port") from None


def _listen(family: int, host: str, port: int, backlog: int,
            v6only: int | None = None) -> socket.socket:
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if v6only is not None:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, v6only)
    sock.bind((host, port))
    sock.listen(backlog)
    # Deliberately not set_inheritable(True): these sockets are passed as Python
    # objects to loop.create_server(sock=...) inside this one process. uvicorn
    # only shares descriptors across processes in its multi-worker path, which
    # this launcher does not use.
    return sock


def build_sockets(port: int, backlog: int = 2048) -> list[socket.socket]:
    """One AF_INET6 and one AF_INET listener on the same port, or nothing.

    IPV6_V6ONLY=1 keeps the families disjoint, so the two binds do not collide.
    """
    made: list[socket.socket] = []
    try:
        made.append(_listen(socket.AF_INET6, "::", port, backlog, v6only=1))
        made.append(_listen(socket.AF_INET, "0.0.0.0", port, backlog))
        return made
    except BaseException:
        for sock in made:
            with contextlib.suppress(Exception):
                sock.close()
        raise


def main() -> None:
    port = resolve_port()
    # Built first: constructing Config installs uvicorn's log handlers, which the
    # startup line below needs, and carries the backlog the sockets should use.
    config = uvicorn.Config("app.main:app", log_level="info")
    sockets = build_sockets(port, backlog=config.backlog)
    # Only after both families are bound.
    LOG.info("Uvicorn running on http://0.0.0.0:%d and http://[::]:%d", port, port)
    uvicorn.Server(config).run(sockets=sockets)


if __name__ == "__main__":
    main()
