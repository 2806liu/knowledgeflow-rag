"""使用项目 A 的本地配置验证真实 LLM；临时知识库与日常服务数据隔离，最多四次逻辑模型调用。"""

import json
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.services import KnowledgeFlow
from app.store import Store


def main():
    """分别验证两个真实适配器的 JSON 回答、拒答和无证据分支，逐条保存实际输出。"""
    if not (settings.llm_api_key and settings.llm_base_url and settings.llm_model):
        print(json.dumps({"status": "skipped_missing_llm_configuration"}))
        raise SystemExit(2)
    from app import main as api
    root = Path(__file__).resolve().parents[1]
    output = root / "evaluation" / "results" / "llm-v2-live-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = root / "tests" / ".tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    source = (root / "examples" / "public-demo.md").read_bytes()
    report = {"cases": [], "passed": False, "scope": "synthetic_fixture_only",
              "semantic_review": "pending", "adapters": ["openai_compatible", "langchain_openai"]}
    old_store, old_flow = api.store, api.flow
    try:
        for mode in report["adapters"]:
            with tempfile.TemporaryDirectory(dir=temporary) as directory:
                store = Store(Path(directory))
                config = replace(settings, llm_mode=mode, llm_answer_format="json",
                                 vector_enabled=False, reranker_enabled=False, retrieval_strategy="bm25_v1",
                                 llm_max_retries=1, llm_max_output_tokens=2048, llm_timeout_seconds=60)
                api.store, api.flow = store, KnowledgeFlow(store, config)
                try:
                    # 共用服务生命周期，避免每次请求创建新事件循环。
                    with TestClient(api.app) as client:
                        upload = client.post("/documents/upload", files={
                            "file": ("public-demo.md", source, "text/markdown")})
                        upload.raise_for_status()
                        document_id = upload.json()["document_id"]
                        for case_id, question, expected in (
                                ("format", "课程报告的命名格式是什么？", "answered"),
                                ("brand", "课程报告资料推荐购买哪个品牌的电脑？", "insufficient_evidence"),
                                ("missing", "zqxjkvnonexistenttoken987654321", "no_evidence")):
                            response = client.post("/chat", json={
                                "session_id": "llm-v2-" + str(uuid.uuid4()), "message": question,
                                "document_ids": [document_id]})
                            data = response.json()
                            entry = {"mode": mode, "id": case_id, "expected": expected,
                                     "http_status": response.status_code, "response": data, "passed": False}
                            run_id = data.get("run_id") or data.get("detail", {}).get("run_id")
                            if run_id:
                                entry["run"] = store.get_run(run_id)
                            if response.is_success:
                                correct = data["answer_status"] == expected
                                if expected == "answered":
                                    correct = correct and "学号-姓名-报告" in data["answer"] and bool(data["citations"])
                                    correct = correct and all(c["document_id"] == document_id and c["chunk_id"]
                                                              for c in data["citations"])
                                else:
                                    correct = correct and not data["citations"]
                                entry["passed"] = bool(correct)
                            report["cases"].append(entry)
                            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        client.delete("/documents/" + document_id).raise_for_status()
                finally:
                    api.store, api.flow = old_store, old_flow
                    store.conn.close()
    finally:
        api.store, api.flow = old_store, old_flow
        report["passed"] = len(report["cases"]) == 6 and all(x["passed"] for x in report["cases"])
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "semantic_review": report["semantic_review"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
