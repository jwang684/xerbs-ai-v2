"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P13: run the coverage harness against Staging, read-only.

    python scripts/governed_coverage_harness.py --source-text flu2025.txt --out report.json

Reads, over HTTP GET only:
  * ai-v2 corpus read APIs (sources, their entities, entity detail,
    relationships, governance binding) with AI_V2_SERVICE_TOKEN;
  * Core via Supabase REST with SUPABASE_SERVICE_KEY: source-verification
    attestations (to prove a verification is live), ``herbal_formulas`` names,
    APPROVED ``product_formula_mappings``, and trace rows -- which are
    sanitized in memory the moment they arrive (see coverage.measure).

Environment: AI_V2_BASE_URL, AI_V2_SERVICE_TOKEN, SUPABASE_URL,
SUPABASE_SERVICE_KEY. None of them is ever printed. The script refuses any
host that is not a Staging host, refuses any HTTP method but GET, opens no
database session and writes nothing but the local report file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.services.coverage import harness as H  # noqa: E402
from app.services.coverage.measure import build_report, sanitize_trace_rows  # noqa: E402

AI_V2_PREFIX = "/api/v1"


class ReadOnlyViolation(RuntimeError):
    pass


class GetOnlyClient:
    """An HTTP client that can only GET. Anything else is a bug, and raises."""

    def __init__(self, base_url, headers, *, host_must_contain, transport=None):
        if host_must_contain not in (base_url or "").lower():
            raise ReadOnlyViolation("refusing a host without %r in it" % host_must_contain)
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=30.0,
                                    transport=transport)
        self.requests = []

    def get(self, path, **params):
        self.requests.append(("GET", path))
        r = self._client.get(path, params=params or None)
        r.raise_for_status()
        return r.json()

    def __getattr__(self, name):
        if name in ("post", "put", "patch", "delete", "request", "stream", "send"):
            raise ReadOnlyViolation("the coverage harness never calls %s" % name.upper())
        raise AttributeError(name)


def _live_verifications(core, environment):
    rows = core.get("/rest/v1/source_verification_attestations",
                    select="governance_object_type,semantic_object_id,object_version,decision,"
                           "target_system,target_environment,revoked_at")
    return {(r["governance_object_type"], r["semantic_object_id"], int(r["object_version"]))
            for r in rows if r.get("decision") == "SOURCE_VERIFIED" and not r.get("revoked_at")
            and r.get("target_system") == "xerbs-ai-v2" and r.get("target_environment") == environment}


def _is_live(verified, object_type, semantic_id, current_version):
    """ai-v2 records the verification at verified_version + 1 (source_verification._apply_source_verification)."""
    return bool(semantic_id) and (object_type, semantic_id, int(current_version) - 1) in verified


def load_snapshot(ai, core, environment="staging"):
    verified = _live_verifications(core, environment)
    sources, entity_refs = [], {}
    offset = 0
    while True:
        page = ai.get(AI_V2_PREFIX + "/knowledge/clinical/sources", limit=100, offset=offset)
        for s in page.get("results", []):
            sources.append(H.SourceSnap(s["source_id"], s["review_status"], int(s["version"]),
                                        live_verification=s["review_status"] == H.SOURCE_VERIFIED
                                        and _is_live(verified, "SOURCE", s["source_id"], s["version"]),
                                        title=s.get("title")))
            for e in ai.get(AI_V2_PREFIX + "/knowledge/clinical/sources/%s/entities" % s["source_id"]).get(
                    "results", []):
                entity_refs[e["entity_id"]] = e["entity_type"]
        offset += 100
        if len(page.get("results", [])) < 100:
            break
    for item in ai.get(AI_V2_PREFIX + "/governance/pending-review", limit=1000).get("items", []):
        if item.get("object_type") == "CLINICAL_ENTITY":
            entity_refs.setdefault(item["object_id"], None)

    entities = []
    for eid, etype in sorted(entity_refs.items()):
        detail = None
        for t in ([etype] if etype else ["pattern", "formula", "herb"]):
            try:
                detail = ai.get(AI_V2_PREFIX + "/knowledge/clinical/entities/%s/%s" % (t, eid))
                break
            except httpx.HTTPStatusError:
                continue
        if detail is None:
            continue
        binding = ai.get(AI_V2_PREFIX + "/governance/pending-review/CLINICAL_ENTITY/%s" % eid)
        content = detail.get("content") or {}
        status = detail["review_status"]
        entities.append(H.EntitySnap(
            eid, detail["entity_type"], detail.get("name", ""), status, int(detail["version"]),
            aliases=tuple(content.get("aliases") or ()),
            source_ids=tuple(s["source_id"] for s in detail.get("sources") or ()),
            live_verification=status == H.SOURCE_VERIFIED and _is_live(
                verified, "CLINICAL_ENTITY", binding.get("semantic_object_id"), detail["version"]),
            clinical_ranking_eligible=bool(detail.get("clinical_ranking_eligible")),
            retired=detail.get("retired_at") is not None,
            external_id=binding.get("semantic_object_id"), source_scope=content.get("source_scope"),
            applicability=content.get("applicability", H.NO_APPLICABILITY)))

    rels = {}
    for e in entities:
        if e.entity_type != "pattern":
            continue
        for r in ai.get(AI_V2_PREFIX + "/safety/relationships", entity_id=e.entity_id).get("results", []):
            rels[r["id"]] = r
    relationships = []
    for rid, r in sorted(rels.items()):
        b = ai.get(AI_V2_PREFIX + "/governance/pending-review/CLINICAL_RELATIONSHIP/%s" % rid)
        version = int(b.get("object_version") or 0)
        relationships.append(H.RelationshipSnap(
            rid, r["source_entity_id"], r["target_entity_id"], r["review_status"], version,
            evidence_source_ids=tuple(x["source_id"] for x in b.get("evidence") or ()) or (r.get("source_id"),),
            live_verification=r["review_status"] == H.SOURCE_VERIFIED and _is_live(
                verified, "CLINICAL_RELATIONSHIP", b.get("semantic_object_id"), version),
            relationship_type=r["relationship_type"]))

    formulas = core.get("/rest/v1/herbal_formulas", select="id,name")
    by_id = {f["id"]: (f.get("name") or "").strip() for f in formulas}
    mapped = [m["formula_id"] for m in core.get("/rest/v1/product_formula_mappings", select="formula_id",
                                                status="eq.APPROVED")]
    return H.CorpusSnapshot(environment=environment, sources=tuple(sources), entities=tuple(entities),
                            relationships=tuple(relationships),
                            core_formula_names=tuple(by_id.values()),
                            core_mapped_formula_names=tuple(sorted(by_id[i] for i in set(mapped)
                                                                   if mapped.count(i) == 1 and i in by_id)))


def load_trace_turns(core, page_size=500):
    raw, offset = [], 0
    while True:
        page = core.get("/rest/v1/integration_recommendation_trace",
                        select="turn_count,recommendation_state,pattern_hypotheses:ai_response_summary->"
                               "pattern_hypotheses,uncertainty_flags:ai_response_summary->uncertainty_flags,"
                               "formula_candidates:ai_response_summary->formula_candidates",
                        order="id", limit=page_size, offset=offset)
        raw.extend(page)
        offset += page_size
        if len(page) < page_size:
            break
    turns, redacted = sanitize_trace_rows(raw)
    raw.clear()
    return turns, redacted


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--environment", default="staging")
    ap.add_argument("--source-text", help="plain text of 流行性感冒诊疗方案（2025年版）, to verify pair locators")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    env = os.environ
    ai = GetOnlyClient(env["AI_V2_BASE_URL"], {"Authorization": "Bearer " + env["AI_V2_SERVICE_TOKEN"]},
                       host_must_contain=args.environment)
    key = env["SUPABASE_SERVICE_KEY"]
    core = GetOnlyClient(env["SUPABASE_URL"], {"apikey": key, "Authorization": "Bearer " + key},
                         host_must_contain="supabase.co")
    snap = load_snapshot(ai, core, args.environment)
    turns, redacted = load_trace_turns(core)
    text = Path(args.source_text).read_text(encoding="utf-8") if args.source_text else None
    report = build_report(snap, turns, source_text=text, redacted_names=redacted)
    report["http"] = {"ai_v2_requests": len(ai.requests), "core_requests": len(core.requests),
                      "methods": sorted({m for m, _ in ai.requests + core.requests})}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("%s -> %s" % (report["label"], args.out))


if __name__ == "__main__":
    main()
