"""公开合成资料的真实多轮验证；独立临时数据库，不读取日常文档或旧40题。"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid
from fastapi.testclient import TestClient

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--partition",choices=["development","holdout","regression"],required=True)
    parser.add_argument("--case-id",help="仅复查指定已见场景，不作为完整集通过率")
    args=parser.parse_args()
    from app import main as api
    from app.config import settings
    from app.store import Store
    from app.services import KnowledgeFlow
    from app.rag.memory import answer_units
    config=replace(settings,data_dir=Path(tempfile.mkdtemp(prefix="rag-public-enhanced-")),
                   vector_enabled=False,reranker_enabled=False,ocr_enabled=False)
    if config.llm_mode=="mock":raise RuntimeError("real LLM configuration required")
    api.settings=config;api.store=Store(config.data_dir);api.flow=KnowledgeFlow(api.store,config)
    suite_raw=(args.data/"suite.json").read_bytes();suite=json.loads(suite_raw)
    # 绑定实际运行的应用字节及非敏感配置，保证语义复核能追溯到提示词版本。
    import app as application
    digest=hashlib.sha256()
    app_root=Path(application.__file__).parent
    for source in sorted(app_root.rglob("*.py")):
        digest.update(source.relative_to(app_root).as_posix().encode());digest.update(source.read_bytes())
    report=dict(case_filter=args.case_id,implementation_sha256=digest.hexdigest(),
                runtime=dict(enable_thinking=config.llm_enable_thinking,
                    resolution_timeout_seconds=config.resolution_timeout_seconds,
                    enhanced_timeout_seconds=config.enhanced_timeout_seconds),date="2026-10-15",partition=args.partition,mode="real_llm_isolated_api",
                model=config.llm_model,data_origin="public_synthetic_only",
                suite_sha256=hashlib.sha256(suite_raw).hexdigest(),cases=[],semantic_review="pending",
                browser_acceptance="separate_required",daily_documents_mounted=False)
    with TestClient(api.app) as client:
        def upload(session,fixture):
            name={"report.md":"报告要求","attendance.md":"考勤说明"}.get(fixture,Path(fixture).stem)
            key=str(uuid.uuid4());data=(args.data/"fixtures"/fixture).read_bytes()
            response=client.post("/ingestions",data=dict(session_id=session,client_request_id=key,batch_id=key,batch_size="1"),
                                 files={"file":(name+"-"+key+".md",data,"text/markdown")})
            response.raise_for_status();job=response.json();deadline=time.monotonic()+30
            while job["state"] not in {"ready","failed","interrupted","cancelled"} and time.monotonic()<deadline:
                time.sleep(.05);job=client.get("/ingestions/"+job["id"]).json()
            if job["state"]!="ready":raise RuntimeError("synthetic upload failed")
            return job["document_id"]
        def seed(session,doc,repeated=False,refusal=False,choice="方法用途"):
            row=api.store.chunks((doc,))[0]
            citation=dict(citation_number=1,chunk_id=row["id"],document_id=doc,filename=row["filename"],
                          page=None,position=row["position"],snippet=row["text"],score=None)
            text="1. 方法名称：待解释。[1]\n2. "+choice+"：待解释。[1]"
            if repeated:text+="\n1. 补充主题：待解释。[1]\n2. 常见局限：待解释。[1]"
            turn,_=api.flow.conversations.begin_turn(session,str(uuid.uuid4()),"公开合成旧回答夹具")
            units=answer_units(text,[citation])
            api.flow.conversations.finish_turn(turn,dict(response=dict(answer=text,answer_status="answered",
                 answer_basis="document_only",citations=[citation]),answer_units=units,
                 resolution={"resolved_query":"解释方法","document_ids":[doc]}))
            if refusal:
                other,_=api.flow.conversations.begin_turn(session,str(uuid.uuid4()),"公开合成旧拒答")
                api.flow.conversations.finish_turn(other,dict(response=dict(answer="资料不足。",
                    answer_status="insufficient_evidence",citations=[]),answer_units=[]))
            return turn
        def ask(case,session,question,label,expected=None,basis=None):
            started=time.perf_counter()
            response=client.post("/chat",json=dict(session_id=session,message=question,request_id=str(uuid.uuid4())))
            entry=dict(label=label,question=question,http_status=response.status_code,elapsed_seconds=round(time.perf_counter()-started,3))
            case["steps"].append(entry)
            if response.status_code!=200:
                entry.update(error=response.json(),automatic_checks_passed=False)
                failed_run=entry["error"].get("detail",{}).get("run_id") if isinstance(entry["error"].get("detail"),dict) else None
                if failed_run:
                    entry["run"]=client.get("/runs/"+failed_run).json()
                return None
            value=response.json();entry["response"]=value
            run=client.get("/runs/"+value["run_id"]);run.raise_for_status();entry["run"]=run.json()
            valid=True
            for c in value["citations"]:
                result=client.get("/documents/"+c["document_id"]+"/chunks/"+c["chunk_id"])
                current=result.json()
                valid=valid and result.status_code==200 and current.get("text")==c["snippet"] and current.get("page")==c["page"]
                valid=valid and c["document_id"] in value["scope_snapshot"]["document_ids"]
            section_valid=all(not s["citation_numbers"] for s in value["answer_sections"] if s["source_type"]=="model_knowledge")
            entry.update(source_locations_valid=valid,model_sections_have_no_document_citations=section_valid,
                         automatic_checks_passed=valid and section_valid and
                         (not expected or value["answer_status"] in expected) and
                         (not basis or value["answer_basis"] in basis))
            return value
        for definition in suite["cases"]:
            if args.partition!="regression" and definition["partition"]!=args.partition:continue
            if args.case_id and definition["id"]!=args.case_id:continue
            case=dict(id=definition["id"],expected=definition,steps=[],semantic_review="pending")
            report["cases"].append(case)
            try:
                session=client.post("/conversations",json={"answer_policy":"document_enhanced"}).json()["id"]
                case["session_id"]=session;kind=definition["kind"];doc=None
                if definition.get("fixture"):doc=upload(session,definition["fixture"])
                if kind in {"pending","refusal_focus"}:
                    case["seeded_history"]=True
                    case["seed_turn_id"]=seed(session,doc,repeated=kind=="pending",
                        refusal=kind=="refusal_focus",choice=definition.get("choice","方法用途"))
                first=ask(case,session,definition["question"],"initial",definition["statuses"],definition.get("basis"))
                if kind=="pending":
                    ask(case,session,definition["choice"],"clarification_choice",["answered"],["document_and_model","model_only","document_only"])
                elif kind=="missing_fact":
                    ask(case,session,definition["followup"],"general_explanation",["answered"],["model_only","document_and_model"])
                elif kind=="attachments":
                    b=upload(session,definition["second_fixture"])
                    second=ask(case,session,definition["question"],"new_document",["answered"],["document_only"])
                    if second:
                        case["new_document_selected"]=bool(second["citations"]) and all(c["document_id"]==b for c in second["citations"])
                    third=ask(case,session,definition["followup"],"return_original",["answered"])
                    if third:
                        case["original_document_selected"]=bool(third["citations"]) and all(c["document_id"]==doc for c in third["citations"])
                elif kind=="strict_switch":
                    client.patch("/conversations/"+session+"/policy",json={"answer_policy":"document_strict"}).raise_for_status()
                    ask(case,session,"只根据文档，指出刚才说法的原文位置。","strict_no_model_evidence",["clarification","insufficient_evidence","no_evidence"],["none"])
                if definition.get("model_section_required") and first:
                    case["model_section_present"]=any(s["source_type"]=="model_knowledge" for s in first["answer_sections"])
                if definition.get("anchor") and first:
                    case["answer_anchor_present"]=definition["anchor"] in first["answer"]
                case["automatic_checks_passed"]=bool(case["steps"]) and all(s["automatic_checks_passed"] for s in case["steps"])
                for key in ["new_document_selected","original_document_selected","model_section_present","answer_anchor_present"]:
                    if key in case:case["automatic_checks_passed"] &= case[key]
            except Exception as exc:
                case.update(automatic_checks_passed=False,error_type=type(exc).__name__)
            print("CASE_JSON="+json.dumps(case,ensure_ascii=True),flush=True)
            print("CASE "+case["id"]+" "+str(case["automatic_checks_passed"]),flush=True)
    report["automatic_checks_passed"]=all(c["automatic_checks_passed"] for c in report["cases"])
    print("REPORT_JSON="+json.dumps(report,ensure_ascii=True),flush=True)

if __name__=="__main__":main()
