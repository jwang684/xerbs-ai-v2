FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
COPY app ./app
# Release sequence: migrate, establish the governed Golden Corpus, then serve.
#
# The corpus bootstrap runs here because a target environment has no other way
# to reach it: governance mutation endpoints are deliberately closed, and the
# database is private. It is idempotent, so a restart re-verifies rather than
# duplicates, and it walks the real review lifecycle rather than writing SQL.
#
# It runs before uvicorn on purpose. If the corpus cannot be established the
# service does not start, because a running service with no governed formula
# would answer consumers with an empty catalogue instead of an error.
#
# app.server binds IPv4 and IPv6 explicitly instead of `uvicorn --host ::`.
#
# `--host ::` was chosen here because xerbs-core reaches this service over
# Railway's private network, which does resolve an AAAA record. But that
# network is not IPv6-only -- *.railway.internal resolves A as well -- and
# `--host ::` is not dual-stack: asyncio sets IPV6_V6ONLY on the socket it
# creates itself, so the container listened on IPv6 alone. The private call
# kept working and Railway's public edge, which connects over IPv4, was
# refused at the TCP layer. app.server binds both families and hands the
# sockets to one uvicorn server; its module docstring explains why that is two
# sockets rather than one dual-stack socket. PORT handling is unchanged.
CMD ["sh", "-c", "alembic upgrade head && python -c \"from app.bootstrap.golden_corpus import bootstrap_golden_corpus as b; import json; print('golden_corpus:', json.dumps(b(), ensure_ascii=False))\" && python -m app.server"]
