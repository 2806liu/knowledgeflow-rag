"""会话记忆：完整轮次、可审计的滚动摘要与结构化主题。历史不作为原文证据。"""
import asyncio
import hashlib
import json
import re
from app.rag.context_builder import ContextBuilder

def answer_units(answer, citations):
    """从实际回答的显式条目建立稳定顺序；引用[2]绝不被当作第二点。"""
    # 数字主条目的子项目保持在父条目中；不能把每个短横线算成下一“点”。
    # 分批概述可能重置数字，保存显示编号，让解析器发现多个“第二点”并澄清。
    numbered = list(re.finditer(r"(?m)^\s*(?:\*\*)?(\d+)[.、）)]\s*", answer))
    if numbered:
        items = [answer[m.start():numbered[i+1].start() if i+1<len(numbered) else len(answer)].strip()
                 for i,m in enumerate(numbered)]
        display_numbers = [int(m.group(1)) for m in numbered]
    else:
        groups = re.split(r"(?m)^\s*(?=[-*]\s+\S)", answer)
        items = [g.strip() for g in groups if re.match(r"^\s*[-*]\s+",g)]
        display_numbers = list(range(1,len(items)+1))
    by_number = {c["citation_number"]:c for c in citations}
    result = []
    for index, text in enumerate(items,1):
        clean = re.sub(r"^\s*(?:\d+[.、）)]\s*|[-*]\s+)","",text).replace("**","")
        title = re.split(r"[:：\n。]",clean,maxsplit=1)[0]
        title = re.sub(r"\[\d+\]","",title).strip()
        numbers = {int(n) for n in re.findall(r"\[(\d+)\]",text)}
        result.append(dict(unit_id=str(index),ordinal=index,display_number=display_numbers[index-1],title=title,text=text,
                           document_ids=list(dict.fromkeys(by_number[n]["document_id"] for n in numbers if n in by_number)),
                           cited_chunk_ids=[by_number[n]["chunk_id"] for n in numbers if n in by_number]))
    return result

class MemoryManager:
    def __init__(self, conversations, config, client):
        self.repo, self.config, self.client = conversations, config, client
        self.builder = ContextBuilder(config)
        self.pending = {}

    def state(self, session):
        with self.repo.connect() as db:
            row = db.execute("SELECT payload,revision FROM conversation_memory WHERE session_id=?", (session,)).fetchone()
        return {**json.loads(row[0]),"revision":row[1]} if row else {"summary":"","covered_until":0,"revision":0,"topic":""}

    def snapshot(self, session):
        """保留最近完整轮次；摘要覆盖区间不重复塞回历史。"""
        turns = self.repo.turns(session,limit=2000)
        state = self.state(session)
        uncovered = [t for t in turns if t["created"]>state.get("covered_until",0)]
        # 概述条目已有独立索引；长概述正文不再作为最近轮次反复送给回答模型。
        recent = [t for t in uncovered if t["response"].get("intent") != "overview"][-self.config.memory_recent_rounds:]
        rounds = [[{"role":"user","content":t["question"]},
                   {"role":"assistant","content":t["response"]["answer"]}] for t in recent]
        # 摘要是待核验的对话定位信息，不以system身份赋予其事实权威。
        # 摘要单独进入生成payload，不伪装成一轮用户发言。
        return dict(turns=turns,rounds=rounds,state=state,latest_attempt=(self.repo.turns(session,limit=1,include_failed=True) or [None])[-1],
                    trace={"recent_complete_turns":len(recent),"summary_revision":state["revision"],
                           "summary_present":bool(state.get("summary"))})

    def schedule(self, session):
        """后台增量压缩；失败只记录状态，不能使已成功的回答丢失。"""
        task = self.pending.get(session)
        if task and not task.done():
            return
        task = asyncio.create_task(self.compact(session))
        self.pending[session] = task
        def record(done):
            if done.cancelled():return
            try:
                outcome=done.result()
                if outcome.get("status")=="not_needed":return
                with self.repo.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    row=db.execute("SELECT payload,revision FROM conversation_memory WHERE session_id=?",(session,)).fetchone()
                    value=json.loads(row[0]) if row else {"summary":"","covered_until":0}
                    value["last_compaction"]=outcome
                    db.execute("INSERT OR REPLACE INTO conversation_memory VALUES(?,?,?)",
                               (session,json.dumps(value,ensure_ascii=False),row[1] if row else 0))
            except Exception:
                # 数据库关闭/退出期间不把后台诊断失败变成已完成回答的异常。
                return
        task.add_done_callback(record)

    async def compact(self, session):
        state = self.state(session)
        turns = [t for t in self.repo.turns(session,limit=2000) if t["created"]>state.get("covered_until",0)]
        rounds = [[{"role":"user","content":t["question"]},{"role":"assistant","content":t["response"]["answer"]}] for t in turns]
        if len(turns)<=self.config.memory_recent_rounds or self.builder.count([m for p in rounds for m in p])<=self.config.history_budget:
            return {"status":"not_needed","attempts":0}
        older = turns[:-self.config.memory_recent_rounds]
        # 分批摘要最早的未覆盖区间，不让压缩请求本身超过预算。
        accepted=[]
        system=("压缩对话用于理解追问。只概括给定历史，不新增事实；保留主题、用户纠正、撤回、待解决问题和说法归属。"
                "用户猜测和助手说法不是已核实文档事实。model_only及model_knowledge是模型通用知识，"
                "摘要必须保留该归属，不能改成论文结论或已核验文档事实。输出JSON且只有summary字段，"
                "summary为简短中文字符串，尽量不超过150个汉字。")
        built = None
        for turn in older:
            payload={"previous_summary":state.get("summary",""),"turns":[{"id":t["id"],"question":t["question"],
                     "answer":t["response"]["answer"],
                     "answer_basis":t["response"].get("answer_basis","document_only"),
                     "answer_sections":t["response"].get("answer_sections",[])} for t in accepted+[turn]]}
            try:
                candidate=self.builder.build(system,payload)
            except ValueError:
                break
            accepted.append(turn); built=candidate
        if not accepted:
            return {"status":"input_too_large"}
        if self.config.llm_mode=="mock":
            return {"status":"skipped_mock","attempts":0}
        try:
            generation,trace=await self.client.generate(built.messages)
            data=json.loads(generation.content)
            summary=data.get("summary")
            if set(data)!={"summary"} or not isinstance(summary,str) or not summary.strip():
                raise ValueError("invalid summary")
            if self.builder.counter(summary)>self.config.memory_summary_budget:
                raise ValueError("summary exceeds budget")
            ids=state.get("source_turn_ids",[])+[t["id"] for t in accepted]
            value={"summary":summary,"covered_until":accepted[-1]["created"],"source_turn_ids":ids,
                   "history_digest":hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   "topic":accepted[-1].get("resolution",{}).get("resolved_query",""),
                   "trace":trace,"context_budget":built.trace,"status":"ready"}
            with self.repo.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row=db.execute("SELECT revision FROM conversation_memory WHERE session_id=?",(session,)).fetchone()
                actual=row[0] if row else 0
                if actual!=state["revision"]:
                    return {"status":"stale_discarded"}
                db.execute("INSERT OR REPLACE INTO conversation_memory VALUES(?,?,?)",
                           (session,json.dumps(value,ensure_ascii=False),actual+1))
            return {"status":"updated","revision":actual+1}
        except asyncio.CancelledError:
            return {"status":"interrupted"}
        except Exception as exc:
            # 保留旧摘要和原始消息；错误诊断只记类别，避免泄露上游异常正文。
            return {"status":"failed","error_type":type(exc).__name__}

    async def close(self):
        tasks=list(self.pending.values())
        for task in tasks:
            if not task.done(): task.cancel()
        if tasks: await asyncio.gather(*tasks,return_exceptions=True)
