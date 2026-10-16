"""独立于滚动摘要的会话焦点：拒答和澄清不覆盖实质讨论，旧会话可推导恢复。"""
import re

SUBSTANTIVE = {"answered", "partial_answer", "mock"}

def substantive(turn):
    response = turn.get("response", {})
    return response.get("answer_status") in SUBSTANTIVE and bool(response.get("answer", "").strip())

def context(repo, session, memory):
    """以已完成轮次为权威恢复焦点；不读取其他会话或把模型知识升级成文档事实。"""
    saved = (repo.get(session) or {}).get("dialogue_state", {})
    turns = memory["turns"]
    valid = [t for t in turns if substantive(t)]
    latest = valid[-1] if valid else None
    result = dict(saved)
    if latest:
        # 历史手动种入、旧客户端及上次未写焦点时也能恢复。
        if result.get("last_substantive_turn_id") != latest["id"]:
            result.update(last_substantive_turn_id=latest["id"],
                          current_topic=latest.get("resolution", {}).get("resolved_query", latest["question"]),
                          topic_source_type=latest["response"].get("answer_basis", "document_only"),pending_reference=None)
    return result

def candidates(memory, focus, question, limit=36):
    """保留稳定的turn:unit键；过去引用编号[2]不作为回答条目编号。"""
    turns = memory["turns"]
    selected = [t for t in turns if substantive(t)][-6:]
    explicit = re.search(r"第(\d+)轮", question)
    if explicit and 0 < int(explicit[1]) <= len(turns):
        selected = [turns[int(explicit[1])-1]]
    else:
        pending = focus.get("pending_reference") or {}
        wanted = {focus.get("last_substantive_turn_id"), pending.get("turn_id")}
        selected = [t for t in turns if t in selected or t["id"] in wanted]
    result = []
    for turn in reversed(selected):
        for unit in turn.get("answer_units", []):
            match = re.match(r"^\s*(?:\*\*)?(\d+)[.、）)]", unit.get("text", ""))
            result.append(dict(key=turn["id"]+":"+unit["unit_id"], turn_id=turn["id"], unit_id=unit["unit_id"],
                display_number=unit.get("display_number", int(match[1]) if match else unit["ordinal"]),
                title=unit["title"], excerpt=unit.get("text", "")[:400],
                document_ids=unit.get("document_ids", []),
                source_type=unit.get("source_type", turn["response"].get("answer_basis", "document_only"))))
            if len(result) >= limit:
                return result
    return result

def update_in_transaction(repo, db, session, turn, resolution, focus):
    """与回答同一事务发布焦点；附件或模式并发变更使用最新会话行，不覆盖其状态。"""
    import json
    row = db.execute("SELECT payload FROM conversations WHERE id=?", (session,)).fetchone()
    value = json.loads(row[0])
    state = dict(focus)
    # 纠正目标单独保留，后续“它/该方法”不会依赖可压缩的历史叙述。
    if resolution.get("user_correction"):
        state["user_correction"] = resolution["user_correction"]
    if resolution.get("clear_pending"):
        state["pending_reference"]=None
    # 比较题失败后保留所选文档、版本与维度；澄清或无关问答不冲掉待继续任务。
    comparison = resolution.get("comparison_task")
    if comparison:
        if turn["response"].get("answer_status") in {"insufficient_evidence", "partial_answer"}:
            state["pending_comparison_task"] = comparison
        elif turn["response"].get("answer_status") == "answered":
            state["pending_comparison_task"] = None
    if resolution["intent"] == "clarification":
        choices = resolution.get("candidate_keys", [])
        if choices:
            state["pending_reference"] = dict(turn_id=resolution.get("candidate_turn_id"),
                                              candidate_keys=choices, document_ids=resolution["document_ids"])
    elif substantive(turn):
        state.update(last_substantive_turn_id=turn["id"], current_topic=resolution["resolved_query"],
                     topic_source_type=turn["response"].get("answer_basis", "document_only"),
                     pending_reference=None)
    value["dialogue_state"] = state
    repo.save(db, value)
