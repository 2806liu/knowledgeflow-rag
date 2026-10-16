/* 页面只调用本项目同源 API；所有文件名、问题、回答与原文都用文本节点渲染，禁止解释文档 HTML。 */
"use strict";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = {docs: [], selected: new Set(), config: {}, page: "chat", asking: false, uploading: false, updating: new Set(),
  session: crypto.randomUUID(), activeCitations: [], deleteId: null, runId: null, detailRequest: 0, toastTimer: null};
const pageNames = {chat: "知识问答", documents: "资料库", runs: "运行记录"};
const answerNames = {answered: "知识库回答", mock: "流程演示", insufficient_evidence: "依据不足",
  no_evidence: "未找到依据", invalid_citations: "引用校验未通过", invalid_response: "回答格式异常", clarification: "请明确对象", partial_answer: "部分回答", error: "请求失败"};

/* 元素工厂只接受代码中指定的属性；来自接口的内容通过 textContent 写入。 */
function node(tag, attrs = {}, children = []) {
  const element = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child !== null && child !== undefined) element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#i-" + name);
  svg.setAttribute("aria-hidden", "true"); svg.append(use); return svg;
}
function action(label, handler, className = "") {
  const button = node("button", {type: "button", class: className}, label);
  button.addEventListener("click", handler); return button;
}
function toast(message, error = false) {
  clearTimeout(state.toastTimer); const box = $("#toast");
  box.textContent = message; box.className = "toast" + (error ? " error" : ""); box.hidden = false;
  state.toastTimer = setTimeout(() => { box.hidden = true; }, error ? 9000 : 4500);
}
function readableError(status, detail) {
  const messages = {response_length: "模型输出达到长度上限，请缩小问题范围或检查概述输出预算。", authentication_error: "模型服务鉴权失败，请检查服务端配置。", rate_limited: "模型服务繁忙，请稍后再试。",
    timeout: "模型响应超时，请稍后重试。", connection_error: "暂时无法连接模型服务，请稍后重试。",
    upstream_http_error: "模型服务返回异常，请查看本次运行记录。", provider_error: "模型服务暂不可用，请查看运行记录。",
    invalid_response: "答案未通过结构与来源校验，请查看运行记录后重试。"};
  if (detail && typeof detail === "object" && !Array.isArray(detail)) return messages[detail.code] || detail.message || "服务处理失败，请查看运行记录。";
  if (Array.isArray(detail)) return "请求参数不符合要求，请检查输入后重试。";
  if (status === 413) return "文件过大，请选择较小的资料。";
  if (status === 415) return "文件格式不支持，请选择页面列出的格式。";
  if (typeof detail === "string" && /ocr.*disabled|disabled.*ocr/i.test(detail)) return "这份资料需要图片识别，当前服务尚未启用 OCR。";
  if (status === 404) return "未找到对应记录，文档可能已删除或记录 ID 有误。";
  if (status === 503) return "所需服务或识别模型尚未就绪，请检查服务配置。";
  return typeof detail === "string" ? detail.slice(0, 220) : "请求失败，请稍后重试。";
}
/* 请求出错时保留 run_id，便于直接打开失败诊断；不把异常当作正常拒答。 */
async function request(path, options = {}) {
  // 请求有明确期限；超时只停止前端等待，后台状态通过任务标识恢复。
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), path === "/chat" ? 300000 : 10000);
  try {
    const response = await fetch(path, {...options, signal: options.signal || controller.signal});
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(readableError(response.status, data.detail));
      error.runId = data.detail?.run_id; error.code = data.detail?.code; error.status = response.status;
      throw error;
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("请求等待超时，后台状态可通过任务记录恢复。");
    if (error instanceof TypeError) throw new Error("无法连接本地服务，请恢复连接后查询任务状态。");
    throw error;
  } finally { clearTimeout(timer); }
}
function showPage(page) {
  if (!pageNames[page]) return;
  // 页面切换时收起资料浮层，避免返回问答页后遮住消息。
  if (page !== "chat") $("#materials-menu").open = false;
  if (page !== "chat") HistoryUI.drawer(false);
  state.page = page;
  $$(".page").forEach(element => { const active = element.id === "page-" + page; element.hidden = !active; element.classList.toggle("active", active); });
  $$(".nav-item").forEach(button => { const active = button.dataset.page === page; button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current"); });
  $("#breadcrumb-current").textContent = pageNames[page];
  if (page === "runs") renderRuns();
}
function updateScope() {
  const count = state.selected.size;
  let label = "全部资料";
  if (count === 1) label = state.docs.find(item => state.selected.has(item.document_id))?.filename || "已选 1 份资料";
  else if (count > 1) label = "已选 " + count + " 份资料";
  const conversation = state.conversation;
  if (conversation) {
    if (conversation.scope_mode === "all") label = "全部知识库";
    else if (!conversation.document_ids.length) label = "本会话尚无资料";
    const focus = state.docs.find(doc => doc.document_id === conversation.active_document_id);
    if (conversation.pending_focus) label += " · 新附件处理中";
    else if (focus) label += " · 当前：" + focus.filename;
    else if (count > 1) label += " · 需明确对象";
  }
  $("#scope-label").textContent = label; $("#scope-button").title = label;
}
function dateText(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"});
}
async function refreshDocuments() {
  const docs = await request("/documents");
  state.docs = docs.filter(item => item.status !== "deleted");
  const active = new Set(state.docs.map(item => item.document_id));
  state.selected = new Set([...state.selected].filter(id => active.has(id)));
  $("#stat-docs").replaceChildren(document.createTextNode(String(state.docs.length)), node("small", {}, "份文档"));
  $("#stat-chunks").replaceChildren(document.createTextNode(String(state.docs.reduce((sum, item) => sum + item.chunk_count, 0))), node("small", {}, "个片段"));
  $("#nav-doc-count").textContent = state.docs.length; $("#library-count").textContent = state.docs.length;
  $("#welcome-description").textContent = state.docs.length ? "选择问答范围，或者直接向资料库提问。" : "可以直接询问通用知识，或上传资料后获得文档依据。";
  $("#welcome-upload").hidden = state.docs.length > 0;
  updateScope(); renderDocuments();
}
function ensureIdle() {
  if (state.asking) { toast("请等待当前回答完成后再操作。"); return false; }
  return true;
}
function documentPath(id, suffix = "") { return "/documents/" + encodeURIComponent(id) + suffix; }
function renderDocuments() {
  const rows = $("#document-rows"); rows.replaceChildren();
  const search = $("#document-search").value.trim().toLocaleLowerCase();
  const docs = state.docs.filter(item => item.filename.toLocaleLowerCase().includes(search));
  $("#documents-empty").hidden = docs.length > 0;
  $("#documents-empty h3").textContent = search ? "没有找到匹配的文档" : "还没有资料";
  $("#documents-empty p").textContent = search ? "试试其他名称，或清除搜索条件。" : "上传学习资料、讲义或题目，建立你的知识库。";
  for (const doc of docs) {
    const type = doc.filename.includes(".") ? doc.filename.split(".").pop().toUpperCase().slice(0, 5) : "FILE";
    const name = node("div", {class: "doc-name-cell"}, [node("span", {class: "doc-type"}, type),
      node("div", {}, [node("div", {class: "doc-name"}, doc.filename), node("div", {class: "doc-date"}, dateText(doc.created_at))])]);
    const facts = (doc.page_count ? doc.page_count + " 页 · " : "") + doc.chunk_count + " 个片段";
    const ready = doc.chunk_count > 0;
    const menu = node("details", {class: "doc-menu"});
    menu.append(node("summary", {"aria-label": doc.filename + " 更多操作"}, icon("more")));
    const items = node("div", {class: "menu-items"});
    const closeAnd = handler => () => { menu.open = false; handler(); };
    items.append(action("查看解析文本", closeAnd(() => openPages(doc.document_id, doc.filename))));
    if (doc.source_available) items.append(node("a", {href: documentPath(doc.document_id, "/source"), target: "_blank", rel: "noopener"}, "下载原文件"));
    items.append(action("重新建立索引", closeAnd(() => mutateDocument(doc, "reindex", "索引已更新"))));
    items.append(action("重新解析文档", closeAnd(() => mutateDocument(doc, "reparse", "文档已重新解析"))));
    items.append(action("删除文档", closeAnd(() => askDelete(doc)), "danger-text"));
    menu.append(items);
    const buttons = node("div", {class: "row-actions"}, [
      action("提问", async () => {
        if (!ensureIdle()) return;
        try { await ConversationUI.setScope("selected",[doc.document_id],doc.document_id);showPage("chat");$("#question").focus(); }
        catch (error) { toast(error.message,true); }
      }), menu]);
    rows.append(node("tr", {}, [node("td", {}, name), node("td", {class: "doc-facts"}, facts),
      node("td", {}, node("span", {class: "badge" + (ready ? "" : " warning")}, ready ? "可检索" : "待检查")), node("td", {}, buttons)]));
  }
}
/* 多文件按顺序提交，进度使用真实文件计数；不虚构解析或上传百分比。 */
function uploadFiles(files) {
  // 每个文件独立管理，不再通过全局uploading锁禁用全部入口。
  return UploadManager.add(files);
}
function openScope() {
  if (!ensureIdle()) return;
  const options = $("#scope-options"); options.replaceChildren();
  if (!state.docs.length) options.append(node("p", {class: "subtitle"}, "请先上传资料。"));
  state.docs.forEach(doc => {
    const checkbox = node("input", {type: "checkbox", value: doc.document_id}); checkbox.checked = state.selected.has(doc.document_id);
    options.append(node("label", {class: "scope-option"}, [checkbox, node("span", {}, [doc.filename, node("small", {}, doc.chunk_count + " 个片段")])]));
  });
  $("#scope-dialog").showModal();
}
async function newChat() {
  if (!ensureIdle()) return;
  try {
    await ConversationUI.create();
    state.activeCitations=[];state.runId=null;
    $$(".message,.loading-message",$("#conversation")).forEach(item=>item.remove());
    $("#welcome").hidden=false;$("#question").value="";$("#send-button").disabled=true;
    renderSources([]);showPage("chat");UploadManager.render();HistoryUI.drawer(false);
    // 新会话属于最近列表；清除旧搜索和归档筛选，确保它立即可见。
    HistoryUI.archived=false;$("#history-search").value="";
    $$("[data-history-filter]").forEach(item=>item.classList.toggle("selected",item.dataset.historyFilter==="active"));
    await HistoryUI.refresh();$("#question").focus();
  } catch (error) { toast(error.message,true); }
}
/* 最小 Markdown 呈现仅支持粗体与引用；未知 HTML、链接和脚本全部保留为文本。 */
function renderAnswer(text, citations) {
  const body = node("div", {class: "answer-content"});
  const valid = new Set(citations.map(item => item.citation_number));
  for (const line of String(text).split("\n")) {
    const paragraph = node("p");
    const clean = line.replace(/^#{1,4}\s+/, "");
    const parts = clean.split(/(\*\*[^*\n]+\*\*|\[\d+\])/g);
    parts.forEach(part => {
      const citation = /^\[(\d+)\]$/.exec(part);
      if (citation && valid.has(Number(citation[1]))) {
        paragraph.append(action(citation[1], () => renderSources(citations, Number(citation[1])), "citation-button"));
      } else if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
        paragraph.append(node("strong", {}, part.slice(2, -2)));
      } else paragraph.append(document.createTextNode(part));
    });
    if (clean) body.append(paragraph);
  }
  return body;
}
function showAnswer(data,target=$("#conversation"),showSources=true) {
  const citations = data.citations || []; const status = data.answer_status || "error";
  const name = AnswerPolicyUI.basisLabel(data, answerNames[status] || status); const warning = ["insufficient_evidence", "no_evidence", "mock"].includes(status);
  const bad = ["invalid_citations", "invalid_response", "error"].includes(status);
  const logo = node("span", {class: "assistant-logo"}, icon("spark"));
  const heading = node("div", {class: "assistant-heading"}, [logo, "KnowledgeFlow",
    node("span", {class: "status-label" + (bad ? " error" : warning ? " warning" : "")}, name)]);
  const footer = node("div", {class: "message-footer"});
  if (citations.length) footer.append(action(citations.length + " 处引用", () => renderSources(citations)));
  if (data.run_id) footer.append(action("查看运行详情", () => openRun(data.run_id)));
  if (data.scope_snapshot) {
    const names = data.scope_snapshot.document_ids.map(id => state.docs.find(doc=>doc.document_id===id)?.filename || id);
    footer.append(node("span",{class:"turn-scope"},"本轮范围：" + names.join("、")));
  }
  const message = node("article", {class: "message assistant"}, [heading, data.answer_sections?.length ? AnswerPolicyUI.render(data,citations) :
    renderAnswer(data.answer || "未返回回答内容。", citations), footer]);
  target.append(message);
  if (showSources) renderSources(citations);
}
async function askQuestion(event) {
  event.preventDefault(); const input = $("#question"); const question = input.value.trim();
  if (state.asking || !question || state.conversation?.history_meta?.archived_at) return;
  if (state.updating.size) { toast("资料正在更新，请完成后再提问。"); return; }
  state.asking = true; $("#answer-policy").disabled = true; $("#send-button").disabled = true; $("#new-chat").disabled = true; $("#scope-button").disabled = true;
  $("#welcome").hidden = true;
  $("#conversation").append(node("article", {class: "message user"}, node("div", {class: "user-bubble"}, question)));
  const pending = node("div", {class: "loading-message"}, [node("span", {class: "spinner"}), "正在检索资料并生成回答…"]);
  $("#conversation").append(pending); input.value = ""; input.style.height = ""; scrollConversation();
  const started = Date.now();
  const timer = setInterval(() => { $("#composer-status").textContent = "已等待 " + Math.floor((Date.now() - started) / 1000) + " 秒"; }, 1000);
  const payload = {session_id: state.session, message: question, stream: false};
  // 工作台显式选择策略；旧API省略参数仍保持严格默认。
  payload.answer_policy = $("#answer-policy").value;
  payload.request_id = crypto.randomUUID();
  payload.scope_revision = state.conversation?.revision;
  // 服务端按会话范围构建本轮快照，空会话不能回落到全库。
  try {
    const data = await request("/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    pending.remove(); showAnswer(data); rememberRun(data.run_id, question, data.answer_status);
  } catch (error) {
    pending.remove(); showAnswer({answer: error.message, answer_status: "error", run_id: error.runId, citations: []});
    if (error.runId) rememberRun(error.runId, question, "error");
    toast(error.message, true);
  } finally {
    clearInterval(timer); state.asking = false; $("#new-chat").disabled = false;
    $("#composer-status").textContent = "Enter 发送 · Shift + Enter 换行";
    ConversationUI.setArchived(Boolean(state.conversation?.history_meta?.archived_at));
    ConversationUI.render(); scrollConversation(); HistoryUI.refresh();
  }
}
function scrollConversation() { const box = $("#conversation"); box.scrollTop = box.scrollHeight; }
function renderSources(citations, selectedNumber = null, autoOpen = true) {
  state.activeCitations = citations; $("#source-count").textContent = citations.length;
  const content = $("#sources-content");
  if (!citations.length) {
    content.replaceChildren(node("div", {class: "sources-empty"}, [
      node("div", {class: "empty-icon"}, icon("link")), node("h3", {}, "暂未引用来源"),
      node("p", {}, "有依据的回答会在这里展示原文；模型通用知识不附文档引用。")]));
    return;
  }
  content.replaceChildren();
  citations.forEach(citation => {
    const card = node("article", {class: "source-card" + (citation.citation_number === selectedNumber ? " selected" : ""), "data-citation": citation.citation_number});
    const origin = citation.page ? "第 " + citation.page + " 页" : "文本片段";
    const detail = node("div", {}, [node("div", {class: "source-file"}, citation.filename || "未命名文档"), node("div", {class: "source-page"}, origin)]);
    card.append(node("div", {class: "source-card-header"}, [node("span", {class: "source-number"}, citation.citation_number), detail]),
      node("div", {class: "source-snippet"}, citation.snippet || ""));
    const doc = state.docs.find(item => item.document_id === citation.document_id);
    const buttons = node("div", {class: "source-actions"});
    if (doc) buttons.append(action("查看解析原文", () => openPages(citation.document_id, citation.filename || "原文", citation.page)));
    if (doc?.source_available) buttons.append(node("a", {href: documentPath(citation.document_id, "/source"), target: "_blank", rel: "noopener"}, "下载原文件"));
    card.append(buttons);
    const metadata = node("details", {class: "source-meta"}, node("summary", {}, "来源详情"));
    metadata.append(node("p", {}, "Chunk ID：" + citation.chunk_id));
    if (typeof citation.retrieval_score === "number") metadata.append(node("p", {}, "检索分数：" + citation.retrieval_score.toFixed(4)));
    if (typeof citation.rerank_score === "number") metadata.append(node("p", {}, "重排分数：" + citation.rerank_score.toFixed(4)));
    if (typeof citation.ocr_confidence === "number") metadata.append(node("p", {}, "OCR 识别置信度：" + (citation.ocr_confidence * 100).toFixed(1) + "%（不代表答案正确率）"));
    if (citation.evidence_role === "neighbor") metadata.append(node("p", {}, "来源类型：同章节相邻证据"));
    card.append(metadata); content.append(card);
  });
  content.append(node("p", {class: "source-snapshot-note"}, "以上片段为回答时的引用快照。重新解析或删除文档后，请结合原文件核对。"));
  // 恢复历史轮次时更新来源内容，但窄屏上不自动用来源抽屉盖住对话。
  if (autoOpen) $("#sources-panel").classList.add("is-open");
  if (selectedNumber !== null) $('[data-citation="' + selectedNumber + '"]', content)?.scrollIntoView({block: "nearest"});
}
async function openPages(id, title, page = null) {
  // 快速切换来源时丢弃过期响应，避免标题与其他文档原文混配。
  const requestId = ++state.detailRequest;
  $("#detail-title").textContent = title; $("#detail-subtitle").textContent = "解析后的原文 · 可对照原文件核验";
  const content = $("#detail-content"); content.replaceChildren(node("p", {class: "subtitle"}, "正在读取原文…"));
  const dialog = $("#detail-dialog"); if (!dialog.open) dialog.showModal();
  try {
    const pages = await request(documentPath(id, "/pages"));
    if (requestId !== state.detailRequest || !dialog.open) return;
    content.replaceChildren();
    if (!pages.length) { content.append(node("p", {}, "没有保存可查看的页级文本。")); return; }
    const select = node("select", {class: "page-select", "aria-label": "选择原文页码"});
    pages.forEach(item => select.append(node("option", {value: item.page_number}, "第 " + item.page_number + " 页")));
    if (pages.some(item => item.page_number === page)) select.value = String(page);
    const text = node("div", {class: "parsed-text"});
    const show = () => { const item = pages.find(item => String(item.page_number) === select.value); text.textContent = item?.text || "此页未提取到文本。"; };
    select.addEventListener("change", show); show(); content.append(select, text);
  } catch (error) { if (requestId === state.detailRequest && dialog.open) content.replaceChildren(node("p", {}, error.message)); }
}
async function mutateDocument(doc, operation, message) {
  if (!ensureIdle()) return;
  if (operation === "reparse" && !doc.source_available) { toast("原文件不可用，无法重新解析。", true); return; }
  // 同一文档只允许一次重建操作，避免连续点击导致并发重解析。
  if (state.updating.has(doc.document_id)) { toast("这份文档正在处理中，请稍候。"); return; }
  state.updating.add(doc.document_id);
  toast("正在处理文档，请稍候…");
  try { await request(documentPath(doc.document_id, "/" + operation), {method: "POST"}); await refreshDocuments(); toast(message + "。"); }
  catch (error) { toast(error.message, true); }
  finally { state.updating.delete(doc.document_id); }
}
function askDelete(doc) {
  if (!ensureIdle()) return;
  if (state.updating.has(doc.document_id)) { toast("文档正在处理中，完成后才能删除。"); return; }
  state.deleteId = doc.document_id;
  $("#delete-description").textContent = "将删除「" + doc.filename + "」及其原文件和检索索引。历史回答保留引用快照。";
  $("#delete-dialog").showModal();
}
async function deleteDocument() {
  const id = state.deleteId; if (!id) return;
  const button = $("#confirm-delete"); button.disabled = true;
  try {
    await request(documentPath(id), {method: "DELETE"}); $("#delete-dialog").close(); await refreshDocuments();
    state.activeCitations = state.activeCitations.filter(item => item.document_id !== id);
    renderSources(state.activeCitations); toast("文档已删除。");
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; state.deleteId = null; }
}
/* 运行列表只保存本浏览器见过的 ID 与问题摘要；旧记录也可以粘贴 ID 查询，不伪造全库历史。 */
function readRuns() {
  try { const data = JSON.parse(sessionStorage.getItem("knowledgeflow-runs") || "[]"); return Array.isArray(data) ? data.slice(0, 30) : []; }
  catch (_) { return []; }
}
function rememberRun(id, question, status) {
  if (!id) return;
  const rows = [{id, question: question.slice(0, 90), status, date: new Date().toISOString()}, ...readRuns().filter(item => item.id !== id)].slice(0, 30);
  try { sessionStorage.setItem("knowledgeflow-runs", JSON.stringify(rows)); } catch (_) { /* 隐私模式禁用存储时仍可使用当前问答与手动查找。 */ }
}
function renderRuns() {
  const list = $("#run-list"); list.replaceChildren(); const rows = readRuns(); $("#runs-empty").hidden = rows.length > 0;
  rows.forEach(item => {
    const button = node("button", {class: "run-item panel", type: "button"}, [
      node("span", {class: "run-icon"}, icon("clock")),
      node("div", {class: "run-copy"}, [node("strong", {}, item.question || "运行记录"), node("small", {}, item.id + " · " + dateText(item.date))]),
      node("span", {}, answerNames[item.status] || item.status || "查看")]);
    button.addEventListener("click", () => openRun(item.id)); list.append(button);
  });
}
async function openRun(id) {
  const requestId = ++state.detailRequest;
  if (!id?.trim()) { toast("请先输入运行 ID。"); return; }
  $("#detail-title").textContent = "运行详情"; $("#detail-subtitle").textContent = id;
  const content = $("#detail-content"); content.replaceChildren(node("p", {class: "subtitle"}, "正在读取运行记录…"));
  if (!$("#detail-dialog").open) $("#detail-dialog").showModal();
  try {
    const run = await request("/runs/" + encodeURIComponent(id));
    if (requestId !== state.detailRequest || !$("#detail-dialog").open) return;
    content.replaceChildren();
    // 使用每次真实模型调用累计 Token；末尾 llm 事件只是摘要，不能当作整轮费用。
    const calls = run.events.filter(item => item.node === "model_call" && typeof item.output?.usage?.total_tokens === "number");
    const legacyUsage = run.events.find(item => item.node === "llm")?.output?.usage?.total_tokens;
    const totalTokens = calls.length ? calls.reduce((sum, item) => sum + item.output.usage.total_tokens, 0) :
      typeof legacyUsage === "number" ? legacyUsage : null;
    const workflow = run.events.find(item => item.node === "workflow");
    const stats = node("div", {class: "run-stats"});
    [["运行状态", run.status === "completed" ? "已完成" : "失败"], ["总耗时", typeof workflow?.latency_ms === "number" ? (workflow.latency_ms / 1000).toFixed(2) + " 秒" : "未记录"],
      ["模型 Token 合计", totalTokens === null ? "未提供" : String(totalTokens)]].forEach(([label, value]) =>
      stats.append(node("div", {class: "run-stat"}, [node("small", {}, label), node("strong", {}, value)])));
    content.append(stats);
    const names = {planner: "任务识别", task_lookup: "历史比较任务定位", scope_check: "文档范围核对", resolution_recovery: "历史引用恢复", reranker: "候选重排", evidence_expansion: "证据补齐", aspect_retrieval: "按方面检索", comparison_evidence: "双文档证据覆盖", comparison_supplemental_retrieval: "缺失文档补检索", comparison_focus: "比较任务焦点", supplemental_retrieval: "补充检索", answer_validation: "答案结构校验", llm: "模型调用", solver: "答案整理", critic: "引用编号检查", workflow: "流程结束"};
    const comparisonTask = run.events.find(item => item.node === "query_resolution")?.output?.comparison_task;
    const taskLookup = run.events.find(item => item.node === "task_lookup")?.output;
    const documentName = id => comparisonTask?.filenames?.[id] || taskLookup?.filenames?.[id] || id;
    const timeline = node("div", {class: "timeline"});
    run.events.forEach(item => {
      let title = names[item.node] || item.node;
      if (item.node === "tool") title = item.tool === "search_history" ? "读取会话上下文" : "检索资料";
      const skip = item.status?.startsWith("skipped") || item.status === "disabled";
      const failed = item.status === "error"; const latency = item.latency_ms ?? item.output?.latency_ms ?? item.output?.total_ms;
      const details = [];
      if (typeof item.result_count === "number") details.push("返回 " + item.result_count + " 条记录");
      if (item.node === "llm" && item.output) {
        details.push("尝试 " + (item.output.attempts || 0) + " 次");
        if (item.output.model) details.push(item.output.model);
      }
      if (item.node === "critic") details.push("已校验引用编号；未自动判定语义支持");
      if (["task_lookup", "scope_check", "resolution_recovery"].includes(item.node)) {
        // 定位与恢复节点只展示文件名；内部任务 ID 仍保留在完整运行信息中供排查。
        const ids = item.output?.document_ids || [];
        if (ids.length) details.push("资料：" + ids.map(documentName).join("、"));
      }
      if (item.node === "aspect_retrieval" && item.output?.document_ids?.some(Boolean)) {
        // 比较题逐篇展示查询与候选数，用户无需从完整 JSON 猜测哪一侧漏检。
        item.output.document_ids.forEach((id, index) => details.push(documentName(id) + "：" +
          (item.output.candidate_counts?.[index] ?? 0) + " 条候选；检索词 " + (item.output.queries?.[index] || "")));
      }
      if (item.node === "comparison_evidence") {
        item.output?.document_ids?.forEach(id => details.push(documentName(id) + "：" +
          (item.output.selected_counts?.[id] ?? 0) + " 条入选；chunk " +
          (item.output.selected_chunk_ids?.[id]?.join("、") || "无")));
      }
      if (item.node === "comparison_supplemental_retrieval") {
        details.push("仅补检索 " + (item.output?.document_ids || []).map(documentName).join("、") +
          "；新增 " + (item.output?.new_chunk_count ?? 0) + " 条");
      }
      const stateText = item.status === "no_new_evidence" ? "无新增证据" : item.status === "retrying" ? "修复中" :
        item.status === "matched" ? "已定位" : item.status === "out_of_scope" ? "当前范围未选中" :
        item.status === "replanned_current_version" ? "按当前版本重检索" :
        item.status === "version_changed" ? "来源版本变化" : item.status === "not_found" ? "未找到可信任务" :
        item.status === "ambiguous_names" ? "文件名有歧义" : item.status === "unavailable" ? "资料未就绪" :
        item.status === "not_comparison" ? "转为对象澄清" :
        item.status === "blocked" ? "需调整文档范围" :
        item.status === "missing_document" ? "缺少一侧证据" : item.status === "invalidated_source_change" ? "来源变化，已失效" :
        failed ? "失败" : skip ? "跳过" : item.status === "fallback" ? "已回退" : "完成";
      timeline.append(node("div", {class: "timeline-item"}, [
        node("span", {class: "timeline-dot" + (failed ? " error" : skip ? " skip" : "")}),
        node("div", {class: "timeline-text"}, [node("strong", {}, title), node("span", {}, stateText + (typeof latency === "number" ? " · " + (latency / 1000).toFixed(2) + " 秒" : "")),
          node("p", {}, details.join(" · "))])]));
    });
    content.append(timeline);
    const raw = node("details", {}, [node("summary", {}, "查看完整运行信息"), node("pre", {}, JSON.stringify(run, null, 2))]); content.append(raw);
  } catch (error) { if (requestId === state.detailRequest && $("#detail-dialog").open) content.replaceChildren(node("p", {}, error.message)); }
}
async function initialize() {
  HistoryUI.initialize();
  // 历史按钮负责打开抽屉，只有带 data-page 的按钮才切换工作页面。
  $$(".nav-item[data-page]").forEach(item => item.addEventListener("click", () => showPage(item.dataset.page)));
  ["upload-button", "welcome-upload", "composer-upload", "dropzone"].forEach(id => $("#" + id).addEventListener("click", () => $("#file-input").click()));
  $("#file-input").addEventListener("change", event => uploadFiles([...event.target.files]));
  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add("drag-over"); }));
  ["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove("drag-over"); }));
  dropzone.addEventListener("drop", event => uploadFiles([...event.dataTransfer.files]));
  $("#new-chat").addEventListener("click", newChat);
  $("#scope-button").addEventListener("click", openScope);
  // 原生 details 提供键盘展开；点到浮层外或按 Escape 时恢复对话阅读空间。
  document.addEventListener("pointerdown", event => {
    const menu=$("#materials-menu");
    if (menu.open && !menu.contains(event.target)) menu.open=false;
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") $("#materials-menu").open=false;
  });
  $("#scope-all").addEventListener("click", async () => {
    try { await ConversationUI.setScope("all",state.conversation.document_ids,state.conversation.active_document_id);$("#scope-dialog").close(); }
    catch (error) { toast(error.message,true); }
  });
  $("#scope-apply").addEventListener("click", async () => {
    const ids=$$("#scope-options input:checked").map(item=>item.value);
    try { await ConversationUI.setScope("selected",ids,ids.length===1 ? ids[0] : null);$("#scope-dialog").close(); }
    catch (error) { toast(error.message,true); }
  });
  $("#detail-dialog").addEventListener("close", () => { state.detailRequest++; });
  $$("[data-close]").forEach(item => item.addEventListener("click", () => $("#" + item.dataset.close).close()));
  $$("dialog").forEach(dialog => dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }));
  $("#confirm-delete").addEventListener("click", deleteDocument);
  $("#close-sources").addEventListener("click", () => $("#sources-panel").classList.remove("is-open"));
  $("#document-search").addEventListener("input", renderDocuments);
  $("#refresh-docs").addEventListener("click", () => refreshDocuments().catch(error => toast(error.message, true)));
  $("#chat-form").addEventListener("submit", askQuestion);
  $("#turns-more").addEventListener("click", () => ConversationUI.loadOlder());
  // 模式持久化到本会话，只影响后续请求；失败时恢复原选择。
  $("#answer-policy").addEventListener("change",async event=>{
    const previous=state.conversation?.answer_policy || "document_enhanced";
    event.target.disabled=true;
    try {
      const value=await request("/conversations/"+state.session+"/policy",{
        method:"PATCH",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({answer_policy:event.target.value})});
      ConversationUI.apply(value);
    } catch(error) { event.target.value=previous;toast(error.message,true); }
    finally { event.target.disabled=state.asking || Boolean(state.conversation?.history_meta?.archived_at); }
  });
  $("#question").addEventListener("input", event => { $("#send-button").disabled = state.asking || Boolean(state.conversation?.history_meta?.archived_at) || !event.target.value.trim();
    event.target.style.height = "auto"; event.target.style.height = Math.min(event.target.scrollHeight, 140) + "px"; });
  $("#question").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("#chat-form").requestSubmit(); } });
  $$("[data-question]").forEach(item => item.addEventListener("click", () => { $("#question").value = item.dataset.question; $("#question").dispatchEvent(new Event("input")); $("#question").focus(); }));
  $("#run-form").addEventListener("submit", event => { event.preventDefault(); openRun($("#run-id").value.trim()); });
  try {
    state.config = await request("/ui/config");
    $("#stat-mode").textContent = state.config.answer_mode === "mock" ? "流程演示" : "真实模型";
    const formats = state.config.ocr_enabled ? "PDF、Markdown、TXT、PNG、JPEG、WebP" : "PDF、Markdown、TXT（当前未启用图片识别）";
    const limit = Math.round(state.config.max_upload_bytes / 1024 / 1024);
    $("#upload-formats").textContent = "支持 " + formats + " · 单文件最大 " + limit + " MB";
    $("#file-input").accept = state.config.ocr_enabled ? ".pdf,.md,.txt,.png,.jpg,.jpeg,.webp" : ".pdf,.md,.txt";
    await refreshDocuments();
    await ConversationUI.initialize(); await UploadManager.restore(); await HistoryUI.refresh(); await request("/health");
    $("#health-dot").className = "online"; $("#health-text").textContent = "服务连接正常";
  } catch (error) { $("#health-dot").className = "offline"; $("#health-text").textContent = "服务连接异常"; toast(error.message, true); }
}
initialize();
