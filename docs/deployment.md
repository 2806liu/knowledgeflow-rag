# Docker 部署

要求 Docker Engine/Desktop 与 Compose 2.24+。所有命令从项目 A 或其独立发布目录执行。

## 默认演示

```powershell
docker compose up --build -d
docker compose ps
python scripts/smoke_api.py
```

访问 http://127.0.0.1:8000/ 打开文档工作台；/docs 为接口调试文档。宿主机端口占用时设置 RAG_PORT=8001。容器以非 root 用户运行，单 worker，SQLite 与原文件位于 rag_data 卷。health 仅检查 API 进程，不验证真实模型可用性。

## 真实模型

复制 .env.docker.example 为 .env.docker，填写聊天接口、模型与密钥，设置 STUDYFLOW_LLM_MODE=langchain_openai。上游支持 JSON object 时可设置 STUDYFLOW_LLM_ANSWER_FORMAT=json，否则使用 text。

```powershell
docker compose --env-file .env.docker up -d --force-recreate api
python scripts/smoke_api.py --real-llm --output evaluation/results/docker-llm-smoke.json
```

.env.docker 只在运行时传入，不复制进镜像。容器内访问宿主机模型接口时，Docker Desktop 可使用 host.docker.internal，不能使用容器自身的 127.0.0.1。

## 向量检索

在 .env.docker 中填入 Embedding 接口、模型与密钥，启用 STUDYFLOW_VECTOR_ENABLED=true，设置检索策略为 dense_v1 或 hybrid_rrf_v1。

```powershell
docker compose --profile vector up --build -d
```

容器网络内 Qdrant 地址固定为 http://qdrant:6333，不映射 Qdrant 端口到宿主机。向量服务启动后再上传或 reindex；如同步失败，检查 vector_status 后重试 reindex。切换向量维度时使用新 collection，不能混用旧向量。

## OCR 与重排模型

```powershell
docker compose run --rm api python -m app.rag.ocr --prepare
docker compose run --rm api python -m app.rag.reranker --prepare
```

模型写入同一个 rag_data 卷。准备成功后在 .env.docker 启用对应开关并重建 api 容器。运行请求不下载模型。重排 AVX2 模型要求兼容 CPU，准备命令会校验模型；首次准备需要网络。

停止服务使用 docker compose down。删除持久卷会清除资料和索引，只有明确要清空数据时才使用 down -v。

## 环境限制与验证状态

Dockerfile 与 Compose 文件存在不等于容器已验证。发布前应完成 build、up、健康检查和 smoke；CI 会构建并启动容器、执行 Mock smoke 和离线测试。真实 LLM smoke 需要自己的密钥，不放在公共 CI。

## Windows 终端找不到 docker

用户级 Docker Desktop 可能安装在 %LOCALAPPDATA%\Programs\DockerDesktop。安装后重新打开 PowerShell；当前终端可以临时补充 PATH：

```powershell
$env:Path = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:Path"
docker version
docker compose version
```

应同时看到 Client 和 Server。只有 Client 时需先启动 Docker Desktop。镜像拉取超时可先单独执行 docker pull python:3.12-slim-bookworm，区分镜像仓库网络与应用构建问题。Docker Desktop 代理设置与容器应用自身的 HTTP 代理应分别核对。

Dockerfile 使用 Debian 官方 HTTPS 源和有限下载重试，处理首次构建期间的临时网关错误。

## 本轮实机验收（2026-10-14）

Windows Docker Desktop、Linux/amd64、Python 3.12：镜像构建、健康检查、99 项离线回归（0 跳过）、Mock 2 例和真实 LLM 3 例通过。真实 LLM 使用 LangChain 与 JSON 回答契约，验证命名答案、固定拒答和无命中不调用模型；助手核对实际引用，未替代独立人工复审或全量 40 题验收。

本地 8001 已被占用，本轮使用 18000。启动时显式设置端口：

```powershell
$env:RAG_PORT = "18000"
docker compose --env-file .env.docker up -d --wait
```

打开 http://127.0.0.1:18000/ 使用文档工作台；接口调试文档位于 /docs。本地 .env.docker 是含密钥的运行配置，已被 Git 忽略，不进入发布目录。容器使用独立持久卷，不自动迁入旧 .studyflow 数据；测试上传已删除，可以重新上传演示资料。

本轮实际服务为 BM25 + 真实 LLM；OCR、Qdrant 与重排按上述说明另行准备和启用，不把模块离线测试称为所有模型在容器内均已实测。


## 文档工作台验证（2026-10-15）

首页提供知识问答、资料库、来源面板与运行记录；静态资源随 Python 包及 Docker 镜像发布，无独立前端服务。
上传、重建索引与重新解析使用现有接口，删除需要页面确认。OCR 未启用时会明确提示；运行模式来自只读能力接口，不在页面暴露密钥和模型地址。

本轮宿主机 103 项回归、Linux 安装包 103 项回归（0 跳过）、3 例真实 LLM 接口冒烟通过；另在隔离 Mock 服务验证 2 例。浏览器连接超时，布局和点击交互仍待人工验收；接口测试不能替代浏览器验证。
