from app.schemas.intake import RecommendationRequest
from app.schemas.reasoning import StructuredSymptom, MissingInformation, PatternAssessment, EvidenceItem, ReasoningResponse
from app.services.knowledge.persistent_clinical import PersistentClinicalStore
from app.services.knowledge.resolver import KnowledgeResolver
from app.services.reasoning.contradictions import ContradictionEngine

CORE_FIELDS = [
    ("duration", "病程/起病时间尚不明确", "症状持续多久？何时开始？"),
    ("temperature", "寒热信息尚不明确", "目前怕冷、怕热或有发热吗？"),
    ("appetite", "饮食信息尚不明确", "近期食欲和进食情况如何？"),
    ("stool", "二便信息尚不完整", "大便情况如何？"),
    ("sleep", "睡眠信息尚不明确", "近期睡眠情况如何？"),
]

class DiagnosticReasoningEngine:
    def __init__(self, corpus=None, resolver=None):
        self.corpus = corpus or PersistentClinicalStore()
        # X1D-KNOWLEDGE1B: the pattern lookup goes through the resolver seam.
        # It delegates to the same corpus search and returns the same value;
        # the only addition is that a miss is recorded instead of vanishing.
        self.resolver = resolver or KnowledgeResolver(self.corpus)
        self.contradictions = ContradictionEngine()

    def analyze(self, request: RecommendationRequest, model_patterns: list[dict] | None = None) -> ReasoningResponse:
        symptoms=[]; seen=set()
        for raw in request.symptoms:
            name=raw.strip()
            if name and name not in seen:
                symptoms.append(StructuredSymptom(name=name)); seen.add(name)
        text=request.text_input or ""
        missing=[]; questions=[]
        markers={
            "duration":["天","周","月","年","小时","开始","持续"],
            "temperature":["发热","怕冷","恶寒","怕热","寒","热"],
            "appetite":["食欲","胃口","进食"],
            "stool":["大便","便秘","腹泻","泄泻"],
            "sleep":["睡眠","失眠","入睡","早醒"],
        }
        joined=" ".join([text,*request.symptoms])
        for field,reason,q in CORE_FIELDS:
            if not any(x in joined for x in markers[field]):
                missing.append(MissingInformation(field=field,reason=reason,question=q,priority="HIGH" if field in {"duration","temperature"} else "MEDIUM")); questions.append(q)

        assessments=[]; source_verified_match=False
        for p in model_patterns or []:
            name=str(p.get("name","")).strip()
            if not name: continue
            # X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P6: bounded name match, so "风寒束表，肺气失宣"
            # finds the reviewed "风寒束表" and a fragment such as "风寒" finds nothing.
            matches=self.resolver.match_pattern(name,limit=3)
            if matches and matches[0].get("governance_basis")=="SOURCE_VERIFIED": source_verified_match=True
            support=[]
            reasoning=str(p.get("reasoning","")).strip()
            if reasoning: support.append(EvidenceItem(text=reasoning,source="model_reasoning"))
            assessments.append(PatternAssessment(pattern_id=(matches[0].get("pattern_id") if matches else None),name=name,model_confidence=float(p.get("confidence",0)),supporting_evidence=support,contradictions=[],corpus_match=bool(matches)))
        flags=[]
        if missing: flags.append("MISSING_CLINICAL_INFORMATION")
        if assessments and not any(x.corpus_match for x in assessments): flags.append("PATTERN_NOT_VERIFIED_IN_REVIEWED_CORPUS")
        if not assessments: flags.append("NO_PATTERN_HYPOTHESIS")
        # X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P7: said out loud, never implied.
        if source_verified_match: flags.append("PATTERN_SOURCE_VERIFIED_NOT_CLINICALLY_REVIEWED")
        ready=bool(assessments) and any(x.corpus_match for x in assessments) and not any(x.priority=="HIGH" for x in missing)
        response=ReasoningResponse(request_id=request.request_id,structured_symptoms=symptoms,missing_information=missing,followup_questions=questions,pattern_assessments=assessments,uncertainty_flags=flags,ready_for_formula_retrieval=ready)
        return self.contradictions.annotate(response, joined)
