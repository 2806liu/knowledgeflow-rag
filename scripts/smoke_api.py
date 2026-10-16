"""验证项目 A 的上传、问答、引用及删除闭环；只使用随仓库发布的合成资料。"""

import argparse
import json
import uuid
from pathlib import Path

import httpx


def main():
    """真实模型验证需要显式 --real-llm；失败也输出实际结果，不把机械检查称为语义验收。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--real-llm", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "examples" / "public-demo.md"
    report = {"mode": "real_llm" if args.real_llm else "mock", "cases": [], "passed": False,
              "semantic_review": "not_performed"}
    document_id = None
    created_document = False
    failure = None
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=300) as client:
        try:
            client.get("/health").raise_for_status()
            # 同时验证工作台资源，防止镜像只打包 API 而漏掉页面。
            workspace = client.get("/")
            workspace.raise_for_status()
            assert "text/html" in workspace.headers.get("content-type", ""), "workspace missing"
            for asset in ("/static/app.js", "/static/styles.css"):
                client.get(asset).raise_for_status()
            capabilities = client.get("/ui/config")
            capabilities.raise_for_status()
            assert set(capabilities.json()) == {"answer_mode", "ocr_enabled", "max_upload_bytes"}
            report["workspace_assets_available"] = True
            # 每次演示使用独立内容指纹，避免去重命中用户已经上传的示例。
            fixture = source.read_bytes() + ("\n\n演示运行编号：" + str(uuid.uuid4())).encode("utf-8")
            response = client.post("/documents/upload", files={
                "file": ("public-demo.md", fixture, "text/markdown")})
            response.raise_for_status()
            document_id = response.json()["document_id"]
            created_document = not response.json().get("duplicate", False)
            session = "public-smoke-" + str(uuid.uuid4())
            queries = [
                ("format", "课程报告的命名格式是什么？", "answer"),
                ("brand", "课程报告资料推荐购买哪个品牌的电脑？", "refuse"),
                ("missing", "zqxjkvnonexistenttoken987654321", "no_evidence")]
            # Mock 仅验证链路；不能让它承担“理解证据不足”的语义任务。
            if not args.real_llm:
                queries = [queries[0], queries[2]]
            for case_id, question, expected in queries:
                response = client.post("/chat", json={
                    "session_id": session + "-" + case_id, "message": question,
                    "document_ids": [document_id]})
                response.raise_for_status()
                data = response.json()
                run = client.get("/runs/" + data["run_id"])
                run.raise_for_status()
                event = next((x for x in run.json()["events"] if x["node"] == "llm"), {})
                entry = {"id": case_id, "answer_status": data["answer_status"], "answer": data["answer"],
                         "citations": data["citations"], "llm": event.get("output"),
                         "checks_passed": False}
                report["cases"].append(entry)
                citations = data["citations"]
                if expected == "answer":
                    allowed = {"answered"} if args.real_llm else {"mock"}
                    assert data["answer_status"] in allowed, "answer status mismatch"
                    assert citations and all(x["document_id"] == document_id for x in citations), "invalid source"
                    assert all(x["chunk_id"] and x["snippet"] and x["filename"] == "public-demo.md"
                               for x in citations), "missing provenance"
                    assert "学号-姓名-报告" in data["answer"], "required answer point missing"
                else:
                    assert data["answer_status"] == (
                        "insufficient_evidence" if expected == "refuse" else "no_evidence"), "refusal status mismatch"
                    assert not citations, "refusal must not cite evidence"
                entry["checks_passed"] = True
            report["passed"] = True
        except Exception as exc:
            # 不输出异常正文，避免 SDK/HTTP 异常混入密钥或请求体。
            failure = type(exc).__name__
            report["failure_type"] = failure
        finally:
            if document_id is not None and created_document:
                deleted = client.delete("/documents/" + document_id)
                report["cleanup_ok"] = deleted.is_success
                if not deleted.is_success:
                    report["passed"] = False
            elif document_id is not None:
                report["cleanup_skipped_existing_document"] = True
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "failure_type": failure, "semantic_review": "not_performed"}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
