"""公开英文合成资料的真实追问验收；不读取或外发用户PDF。"""
import json
from pathlib import Path
import time
import uuid
import httpx


def main():
    """先概述英文原文，再以中文指代追问，独立核对当前规范引用。"""
    key=str(uuid.uuid4())
    data=("# Public synthetic tutorial\n"
          "## Background\nThis public tutorial discusses redundant robot arms, and compares online kinematic control methods.\n"
          "## Goal and scope\nOur goal is to compare four online control methods: simple Jacobian control, null-space optimization, task-priority augmentation, and inverse kinematic functions.\n"
          "## Retention rules\nThe public archive retains experiment records for seven years, and checks the archive every month.\n"
          "<!-- "+key+" -->").encode()
    output=Path("evaluation/results/conversation-v2/followup-live.json")
    report={"date":"2026-10-15","data_origin":"public_synthetic_english_only","cases":[],
            "semantic_review":"pending","browser_acceptance":"not_run"}
    doc=None;created=False
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with httpx.Client(base_url="http://127.0.0.1:18000",timeout=180) as client:
        try:
            r=client.post("/conversations");r.raise_for_status();session=r.json()["id"]
            r=client.post("/ingestions",data={"session_id":session,"client_request_id":key,"batch_id":key,"batch_size":"1"},
                          files={"file":("public-followup-"+key+".md",data,"text/markdown")})
            r.raise_for_status();job=r.json();deadline=time.monotonic()+30
            while job["state"] not in {"ready","failed","cancelled","interrupted"} and time.monotonic()<deadline:
                time.sleep(.1);r=client.get("/ingestions/"+job["id"]);r.raise_for_status();job=r.json()
            assert job["state"]=="ready",job.get("error")
            doc=job["document_id"];created=not job.get("duplicate",False)
            def ask(case_id,question):
                r=client.post("/chat",json={"session_id":session,"message":question,"request_id":str(uuid.uuid4())})
                r.raise_for_status();response=r.json()
                r=client.get("/runs/"+response["run_id"]);r.raise_for_status();run=r.json()
                valid=True
                for c in response["citations"]:
                    r=client.get("/documents/"+c["document_id"]+"/chunks/"+c["chunk_id"]);r.raise_for_status()
                    current=r.json()
                    valid=valid and current["text"]==c["snippet"] and current["page"]==c["page"] and c["document_id"]==doc
                entry={"id":case_id,"question":question,"response":response,"run":run,"source_locations_valid":valid}
                report["cases"].append(entry);save()
                print(case_id,response["answer_status"],flush=True)
                return entry
            first=ask("F01","请用中文概述全文，只列三个主条目：1.背景；2.目标与范围；3.保留规则。")
            second=ask("F02","刚才第二点具体是什么意思")
            anchor=next(e["output"] for e in second["run"]["events"] if e["node"]=="reference_evidence")
            report["passed"]=all(c["response"]["answer_status"]=="answered" and c["source_locations_valid"] for c in report["cases"])
            report["passed"]=report["passed"] and anchor.get("anchor_count",0)>0 and any(
                "Our goal" in c["snippet"] for c in second["response"]["citations"])
            report["external_input_restricted_to_public_synthetic"]=True
        finally:
            if doc and created:
                r=client.delete("/documents/"+doc);report["synthetic_document_cleanup_ok"]=r.is_success
            save()
    print(json.dumps({"passed":report.get("passed"),"cases":len(report["cases"])}))


if __name__=="__main__":main()
