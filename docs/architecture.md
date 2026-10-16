# 项目 A 架构

项目 A 是独立 RAG 服务；配置、代码、数据与启动流程均在本项目内，不调用项目 B。

## 数据流

1. FastAPI/Pydantic 校验输入，原文件保存在配置的数据目录。
2. parsers、pdf_layout、ocr 产生页级文本与 OCR 元数据。
3. quality、chunking 清洗和切块，SQLite 保存文档、页、chunk 及独立来源。
4. retrieval 提供词法/BM25，embedding/vector_index 提供 Qdrant Dense，hybrid 使用 RRF 融合。
5. reranker 可选重排；evidence_expansion 补齐同文档、同章节邻近片段。
6. solver 构造证据上下文，llm_client 统一适配器、超时、重试、用量和脱敏错误。
7. services 保存引用、会话和运行事件；main 将上游调用错误映射为明确 HTTP 状态。
8. evaluation 与 tests 分别承担效果评测、人工证据复核和离线模块回归。

## 关键约束

- 不合并不同文档或页的独立来源。补充片段有自己的引用编号。
- 模型调用失败与证据不足是不同状态，不能用 Mock 或空答案掩盖外部调用失败。
- JSON 回答只校验结构、状态和引用编号；不自动证明事实成立。
- 单次调用有超时；重试有上限，SDK 内部重试关闭。总耗时还包括多次尝试及退避。
- 记录用量只采用上游实际返回值；缺失为 null。
- 固定工作流不是 LangGraph 或 Multi-Agent 实现。

## 会话式文档问答改进（实现与验收）

四项改进的模块边界、拟定接口、状态机和验收门槛见 [最终设计与验收方案](conversation-rag-v2-final-design.md)。六模块已接入；自动、真实模型、人工语义与浏览器验收分别记录。设计门槛不能替代实际验收结果。
