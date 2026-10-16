# KnowledgeFlow RAG

可评测的多模态文档知识库与问答服务。面向讲义、题目 PDF 和学习资料，提供上传、解析、索引、检索、带来源回答、删除与重新解析闭环。

当前架构、长文概述、多轮记忆、双篇比较、历史会话及验收边界汇总见 [项目 A 总览](docs/project-a-overview.md)。

## 架构与技术栈

Python / FastAPI / Pydantic / LangChain / SQLite / Qdrant / BM25 / RRF / Cross-Encoder / RapidOCR / ONNX Runtime。

PDF、Markdown、TXT、图片 → 原生提取与 OCR → 清洗和结构化切块 → BM25 / Dense → RRF 混合召回 → 可选重排 → Top-K 与相邻证据补齐 → LLM 回答或拒答 → 引用与评测。

- SQLite 是文档、页、chunk 和来源的主存储；Qdrant 提供可选向量索引。
- 扫描 PDF 和混排 PDF 的图片区域可以 OCR。
- 列表切块保留父标题和章节标识，相邻证据按数量和上下文预算补齐。
- 跨文档相同文本保留各自来源，引用包含文件名、物理页码、chunk ID 和原文。
- 这是固定 RAG 工作流，项目 A 独立运行，不依赖其他项目。

## 本地启动

要求 Python 3.10+；推荐 3.12。先复制 .env.example 为 .env，再填写自己的配置。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ocr,vector,reranker]"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000/ 使用文档工作台；http://127.0.0.1:8000/docs 为开发者接口文档。默认配置是 BM25 与 Mock，无需模型密钥；Mock 返回证据原文，不代表生成式问答质量。

## 配置真实 LLM

在 .env 中设置以下字段；聊天模型与 Embedding 模型分别配置。

```dotenv
STUDYFLOW_LLM_MODE=langchain_openai
STUDYFLOW_LLM_BASE_URL=你的兼容接口地址，通常以/v1结尾
STUDYFLOW_LLM_MODEL=你的聊天模型名
STUDYFLOW_LLM_API_KEY=仅在本地填写
STUDYFLOW_LLM_TIMEOUT_SECONDS=60
STUDYFLOW_LLM_MAX_RETRIES=2
STUDYFLOW_LLM_MAX_OUTPUT_TOKENS=1024
STUDYFLOW_LLM_ANSWER_FORMAT=text
```

- langchain_openai 使用 ChatOpenAI；openai_compatible 使用直接 HTTP。两者共用调用限制、重试与答案校验。
- 仅网络异常、超时、408、429、5xx 重试；鉴权和响应格式错误立即失败。最多尝试 max_retries + 1 次，限时按每次请求计算。
- temperature 默认为空，不发送该参数；需要时在配置中显式填写。
- text 保持既有自由文本契约。上游支持 JSON object 时可切换 json，返回明确的 answered / insufficient_evidence 状态，拒答由服务归一为固定文案。
- JSON 格式不合法、引用声明与正文不一致时明确标记 invalid_response；不会把失败伪装为成功。
- 调用错误返回 502 / 503 / 504，并提供 run_id。运行记录包含尝试次数、耗时、实际 token 用量和脱敏错误类别。
- 当前 /chat 返回完整 JSON。

## 检索、OCR 与重排

检索策略：lexical_v1、lexical_v2、bm25_v1、bm25_idf_v2、dense_v1、hybrid_rrf_v1。默认 bm25_idf_v2；向量模式需要独立 Qdrant 和 Embedding 配置。

```powershell
python -m app.rag.ocr --prepare
python -m app.rag.reranker --prepare
```

模型必须显式准备，请求期间不自动下载。重排默认关闭，ONNX INT8 AVX2 模型需要兼容 CPU。详细配置见 .env.example。

## 文档与问答 API

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | /documents/upload | 上传并解析、切块、索引 |
| GET | /documents | 列出文档 |
| GET | /documents/{id}/pages | 查看页级原文和 OCR 元数据 |
| GET | /documents/{id}/source | 下载原文件 |
| POST | /documents/{id}/reindex | 对保存文本重新切块并索引，不重新 OCR |
| POST | /documents/{id}/reparse | 从原文件重新解析/OCR 并更新索引 |
| DELETE | /documents/{id} | 删除文档及索引 |
| POST | /chat | 指定文档范围或查询全库 |
| GET | /runs/{id} | 查询成功或失败运行的诊断 |
| GET | /health | 进程健康检查，不调用付费模型 |

reparse 保留 document ID，但 chunk ID 可能改变；历史问答保留原快照。SQLite 更新使用事务，向量同步失败时标记 failed，可通过 reindex 重试，不宣称跨存储原子事务。

## 验证与评测

```powershell
python -m unittest discover -s tests -v
python scripts/smoke_api.py
# 启动真实 LLM 服务后，显式执行真实调用并保存结果。
python scripts/smoke_api.py --real-llm --output evaluation/results/public-llm-smoke.json
```

smoke 使用公开合成资料，验证上传、来源、回答/拒答、运行记录和删除；其机械检查不替代人工语义验收。模块测试使用模拟模型，不访问外网。

历史开发集包含 35 题检索查询与 40 题答案案例。列表切块修正后，35 题开发集 BM25 IDF² Recall@4 为 0.9714 → 1.0000，MRR@4 为 0.9238 → 0.9310。


## Docker 与公开发布

Docker 启动、持久卷、Qdrant 配置和模型准备见 [部署说明](docs/deployment.md)。
GitHub 使用白名单导出目录，步骤见 [发布说明](docs/release-checklist.md)。

```powershell
docker compose up --build -d
python scripts/smoke_api.py
```

默认容器演示关闭 OCR、向量和真实 LLM；按部署说明显式配置后启用。运行数据保存在独立卷，不覆盖本地 .studyflow。

## 文档工作台

根路径 / 提供本地中文工作台，页面资源随 Python 安装包和 Docker 镜像分发，不依赖 CDN 或前端构建服务。

- 知识问答：选择全部或指定文档范围，查看带编号引用的回答与明确拒答状态。
- 资料库：拖放或选择多份资料上传，搜索文件名，查看解析原文，重新索引、重新解析与删除。
- 引用来源：独立展示文件、页码、chunk ID 和原文快照，可打开页级文本并下载原文件。
- 运行记录：查看本浏览器见过的 run_id，或手动查找既有记录，展示调用耗时、实际 token 和失败诊断。

页面根据服务能力提示是否支持图片识别。回答和文件内容以文本节点呈现，不执行文档中的 HTML 或脚本。工作台的引用编号校验不等于自动语义核验。


## 会话式文档问答（v2）

工作台支持文档概述、连续附件、任务取消/恢复和多轮追问。
后台摄取阶段分别显示排队、解析/OCR、切块、索引与就绪；客户端停止传输不代表服务端已确认取消。

新客户端先POST /conversations，再将返回id作为 /chat 的session_id。POST /ingestions受理后返回202和任务id，使用GET /ingestions/{id}查询；取消与重试使用对应子接口。
会话范围、消息与附件持久化到本项目SQLite；旧 /documents/upload 与旧问答客户端保留兼容行为。
概述绕过关键词命中门槛，按原始chunk取证；长文分批汇总，缺失范围明确标记部分回答。引用可通过 /documents/{document_id}/chunks/{chunk_id}核验。

近期完整轮次、滚动摘要和结构化主题用于理解追问；历史摘要不单独作为事实证据。所有模型阶段统一检查输入预算。
可配置 STUDYFLOW_CONTEXT_WINDOW、STUDYFLOW_HISTORY_BUDGET、STUDYFLOW_MEMORY_SUMMARY_BUDGET、STUDYFLOW_OVERVIEW_MAX_OUTPUT_TOKENS（默认4096）、STUDYFLOW_OVERVIEW_MAX_CALLS 与 STUDYFLOW_INGESTION_TIMEOUT_SECONDS。
未知模型采用明确标记的保守计数，不能把估算值当服务商实际token。

运行 python -m evaluation.conversation_validation 记录自动回归；node tests/frontend_upload_tests.js仅为客户端故障单测，不代替真实浏览器验收。
真实模型合成资料验证：python -m evaluation.conversation_live。语义复核与浏览器检查单独记录，自动通过不表示完整验收。
完整模块边界与门槛见 [最终设计](docs/conversation-rag-v2-final-design.md)。


## 文档增强回答与问题理解（v1）

工作台的新会话默认 `document_enhanced`，可切换为 `document_strict`。旧 API 未指定策略仍按严格文档问答运行。

- **文档概述、论文实验数值、报告要求**始终要求原文。不能用模型知识补造论文结果。
- **概念、原理、用途、比较和常见局限**可由 LLM 补充；无附件时也可问明确的通用问题。
- 回答分为文档依据、模型通用知识、范围与缺失说明。模型知识没有文档引用，也没有联网核实。
- 结构化问题解析保留最近实质回答。拒答和澄清不覆盖主题；多个“第二点”先展示候选，回复标题即可接续。
- 每轮冻结文档范围和策略。追问读取当前原文 chunk，检查来源版本；模型知识不会升级成文档事实。
- 通常两次前台逻辑调用，最多三次，只允许一次补检索。第3次也可用于首次结构不合规后的重新生成，不能与补检索循环叠加。增强总期限默认120秒，解析20秒；长文概述使用原有独立期限。

新接口：`POST /conversations` 可指定 `{"answer_policy":"document_enhanced"}`；
`PATCH /conversations/{id}/policy` 切换后续轮次模式。`POST /chat` 也可显式覆盖本轮策略。
返回字段包括 `answer_policy`、`effective_policy`、`answer_basis`、`answer_sections`、`unresolved_aspects`。
`answer_basis` 为 `document_only`、`document_and_model`、`model_only` 或 `none`，由服务端校验后计算。
概述即使选择增强模式，其 `effective_policy` 仍为严格文档模式。

上游支持 JSON object 时建议 `STUDYFLOW_LLM_ANSWER_FORMAT=json`。可选 `STUDYFLOW_LLM_ENABLE_THINKING` 只用于支持该扩展的服务商，留空不发送；第三方代理是否透传必须实测。
Qwen 的该参数语义见 [模型服务文档](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)。
本模块不自动开启 Dense、Qdrant、联网搜索，也不解决既有长概述超时。

自动回归：`python -m evaluation.enhanced_validation --all`；
公开合成真实模型开发集：`python -m evaluation.enhanced_live --data evaluation/document-enhanced-v1 --partition development`。
后者会调用当前配置的真实模型，建议使用隔离配置和临时数据目录。开发集、冻结验收集、逐事实语义复核和浏览器点击验收分别记录。
模块设计及验收门槛见 [文档增强问答设计](docs/document-enhanced-rag-v1-design.md)。
