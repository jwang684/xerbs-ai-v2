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
# uvicorn binds :: rather than 0.0.0.0 because Railway's private network
# (*.railway.internal) is IPv6-only: on 0.0.0.0 the public URL answers but
# xerbs-core's private call is refused. A dual-stack container serves both
# families from ::.
CMD ["sh", "-c", "alembic upgrade head && python -c \"from app.bootstrap.golden_corpus import bootstrap_golden_corpus as b; import json; print('golden_corpus:', json.dumps(b(), ensure_ascii=False))\" && uvicorn app.main:app --host :: --port ${PORT:-8080}"]
