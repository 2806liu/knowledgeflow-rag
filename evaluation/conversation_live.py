"""真实容器会话验收：合成资料、真实OCR/LLM、独立规范chunk校验；语义仍需复核。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import uuid
import httpx

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",default="http://127.0.0.1:18000")
    parser.add_argument("--output",type=Path,default=Path("evaluation/results/conversation-v2/live-model.json"))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent/"conversation-v2/fixtures"
    run=str(uuid.uuid4())
    report={"date":"2026-10-15","run_id":run,"mode":"real_api","cases":[],"uploads":[],"cleanup":[],
            "semantic_review":"pending","browser_acceptance":"separate_required","passed":False}
    created=set()
    with httpx.Client(base_url=args.base_url,timeout=300) as client:
        def save():
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        def conversation():
            response=client.post("/conversations");response.raise_for_status();return response.json()["id"]
        def upload(session,name,data,mime="text/markdown",batch=None,size=1):
            started=time.perf_counter()
            key=str(uuid.uuid4())
            response=client.post("/ingestions",data={"session_id":session,"client_request_id":key,
                "batch_id":batch or str(uuid.uuid4()),"batch_size":str(size)},files={"file":(name,data,mime)})
            response.raise_for_status();job=response.json()
            received=time.perf_counter()-started
            while job["state"] not in {"ready","failed","cancelled","interrupted"}:
                time.sleep(.15);response=client.get("/ingestions/"+job["id"]);response.raise_for_status();job=response.json()
            report["uploads"].append({"job":job,"accepted_seconds":received,"elapsed_seconds":time.perf_counter()-started})
            save()
            if job["state"]!="ready":raise RuntimeError("upload failed: "+json.dumps(job.get("error")))
            if not job.get("duplicate"):created.add(job["document_id"])
            return job["document_id"]
        def check_sources(data):
            checks=[]
            for citation in data.get("citations",[]):
                response=client.get("/documents/"+citation["document_id"]+"/chunks/"+citation["chunk_id"])
                response.raise_for_status();canonical=response.json()
                checks.append(canonical["text"]==citation["snippet"] and canonical["page"]==citation["page"] and
                              canonical["filename"]==citation["filename"] and citation["document_id"] in data["scope_snapshot"]["document_ids"])
            return all(checks)
        def ask(case_id,session,question,expected=("answered",),anchors=(),request_id=None):
            started=time.perf_counter()
            response=client.post("/chat",json={"session_id":session,"message":question,"request_id":request_id or str(uuid.uuid4())})
            entry={"id":case_id,"question":question,"http_status":response.status_code,
                   "elapsed_seconds":round(time.perf_counter()-started,3),"checks_passed":False}
            report["cases"].append(entry)
            if response.is_success:
                data=response.json();entry["response"]=data
                entry["source_locations_valid"]=check_sources(data)
                run_response=client.get("/runs/"+data["run_id"]);run_response.raise_for_status()
                entry["run"]=run_response.json()
                entry["checks_passed"]=data["answer_status"] in expected and entry["source_locations_valid"] and all(x in data["answer"] for x in anchors)
                entry["anchor_check_only"]=True
            else:entry["error"]=response.json()
            save();print(case_id+": "+str(entry["checks_passed"]),flush=True)
            return entry.get("response",{})
        try:
            session=conversation()
            report["ui_config"]=client.get("/ui/config").json()
            assert report["ui_config"]["answer_mode"]=="llm" and report["ui_config"]["ocr_enabled"]
            suffix=("\n<!-- synthetic run "+run+" -->").encode()
            report_data=(root/"report.md").read_bytes()+suffix
            a=upload(session,"report-"+run+".md",report_data)
            first=ask("L01",session,"这篇文章讲什么",anchors=("学号","报告"))
            repeat=ask("L02",session,"这篇文章讲什么")
            ask("L03",session,"刚才第二点再详细解释一下")
            # 显式重新概述后展开，验证泛化概述追问不会再被BM25零命中截断。
            ask("L04a",session,"概括全文主要内容")
            ask("L04",session,"再详细一点")
            b=upload(session,"attendance-"+run+".md",(root/"attendance.md").read_bytes()+suffix)
            result=ask("L05",session,"这篇文章讲什么",anchors=("考勤",))
            if report["cases"][-1]["checks_passed"]:
                report["cases"][-1]["checks_passed"]=all(c["document_id"]==b for c in result["citations"])
            batch=str(uuid.uuid4())
            upload(session,"report-"+run+".md",report_data,batch=batch,size=2)
            upload(session,"attendance-"+run+".md",(root/"attendance.md").read_bytes()+suffix,batch=batch,size=2)
            ask("L06",session,"这篇文章讲什么",expected=("clarification",))
            ask("L07",session,"请分别概述这些资料")
            scope=client.patch("/conversations/"+session+"/scope",json={"mode":"selected","document_ids":[a],"active_document_id":a})
            scope.raise_for_status()
            ask("L08",session,"这篇文章讲什么")
            ask("L09",session,"资料明确指定了哪种电脑品牌和型号？",expected=("insufficient_evidence","no_evidence"))
            long_doc=upload(session,"long-"+run+".md",(root/"long.md").read_bytes()+suffix)
            # selected范围的最新批次也应追加新资料并使其成为当前焦点。
            long_answer=ask("L10",session,"概括这篇文章的全部章节")
            report["cases"].append({"id":"L11","derived_from_run":long_answer.get("run_id"),
                "checks_passed":"7年" in long_answer.get("answer","") and any("7年" in c["snippet"] for c in long_answer.get("citations",[])),
                "purpose":"tail_keypoint_and_supporting_source","semantic_review":"pending"})
            client.patch("/conversations/"+session+"/scope",json={"mode":"selected","document_ids":[a],"active_document_id":a}).raise_for_status()
            client.post("/documents/"+a+"/reindex").raise_for_status()
            ask("L12",session,"这篇文章讲什么")
            from PIL import Image,ImageDraw,ImageFont,PngImagePlugin
            image=Image.new("RGB",(1400,370),"white");draw=ImageDraw.Draw(image)
            font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",42)
            for index,text in enumerate(["课程报告提交说明","命名格式为：学号-姓名-报告。","提交PDF，截止时间6月30日18:00。"]):
                draw.text((45,40+95*index),text,font=font,fill="black")
            stream=io.BytesIO();metadata=PngImagePlugin.PngInfo();metadata.add_text("run",run)
            image.save(stream,format="PNG",pnginfo=metadata)
            upload(session,"scan-"+run+".png",stream.getvalue(),"image/png")
            ask("L13",session,"这篇资料主要讲什么",anchors=("学号",))
            # 同页混排：原生段落加包含命名规则的真实栅格区域。
            from pypdf import PdfWriter
            from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject,NumberObject
            writer=PdfWriter();page=writer.add_blank_page(width=595,height=842)
            f=DictionaryObject({NameObject("/Type"):NameObject("/Font"),NameObject("/Subtype"):NameObject("/Type1"),NameObject("/BaseFont"):NameObject("/Helvetica")})
            raster=DecodedStreamObject();raster.set_data(image.tobytes())
            raster.update({NameObject("/Type"):NameObject("/XObject"),NameObject("/Subtype"):NameObject("/Image"),
                NameObject("/Width"):NumberObject(image.width),NameObject("/Height"):NumberObject(image.height),
                NameObject("/ColorSpace"):NameObject("/DeviceRGB"),NameObject("/BitsPerComponent"):NumberObject(8)})
            page[NameObject("/Resources")]=DictionaryObject({NameObject("/Font"):DictionaryObject({NameObject("/F1"):f}),
                NameObject("/XObject"):DictionaryObject({NameObject("/Im"):writer._add_object(raster)})})
            content=DecodedStreamObject();content.set_data(b"BT /F1 11 Tf 40 780 Td (This document explains the report submission process and naming rules.) Tj ET\nq 520 0 0 137 35 520 cm /Im Do Q")
            page[NameObject("/Contents")]=writer._add_object(content);writer.add_metadata({"/Title":run})
            mixed=io.BytesIO();writer.write(mixed)
            upload(session,"mixed-"+run+".pdf",mixed.getvalue(),"application/pdf")
            ask("L14",session,"这篇文章讲什么",anchors=("学号",))
            one=ask("L15a",session,"报告的命名格式是什么",request_id="same-question-"+run)
            two=ask("L15",session,"报告的命名格式是什么",request_id="same-question-"+run)
            report["idempotent_turn_same_run"]=one.get("run_id")==two.get("run_id")
            empty=conversation();ask("L16",empty,"这篇文章讲什么",expected=("clarification",))
            latencies=[]
            for _ in range(20):
                start=time.perf_counter();client.get("/conversations/"+session).raise_for_status();latencies.append(time.perf_counter()-start)
            report["status_query_latency"]={"samples":20,"p50":statistics.median(latencies),"p95":sorted(latencies)[18]}
            report["passed"]=all(c["checks_passed"] for c in report["cases"]) and report["idempotent_turn_same_run"]
        except Exception as exc:
            report["failure_type"]=type(exc).__name__
            print("live check stopped: "+type(exc).__name__,flush=True)
        finally:
            for document in created:
                response=client.delete("/documents/"+document)
                report["cleanup"].append({"document_id":document,"ok":response.is_success})
            report["passed"]=report["passed"] and all(c["ok"] for c in report["cleanup"])
            save()
    print(json.dumps({"passed":report["passed"],"cases":len(report["cases"]),
          "failed":[c["id"] for c in report["cases"] if not c["checks_passed"]],"semantic_review":"pending"}))
    if not report["passed"]:raise SystemExit(1)

if __name__=="__main__":main()
