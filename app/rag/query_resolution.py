"""检索前理解：确定规则优先，受限LLM处理复杂指代，歧义明确澄清。"""
import json
import re
from pathlib import Path
from app.rag.context_builder import ContextBuilder

_NUMBERS={"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}

class QueryResolver:
    def __init__(self, config, client):
        self.config,self.client=config,client
        self.builder=ContextBuilder(config)

    async def resolve(self, question, snapshot, memory, documents):
        ids=list(snapshot["document_ids"])
        result=dict(intent="qa",resolved_query=question,document_ids=ids,method="explicit_question",
                    referenced_turn_id=None,referenced_unit_id=None,clarification_question=None)
        def clarify(text, reason):
            return {**result,"intent":"clarification","method":reason,"clarification_question":text}
        if not ids:
            return clarify("当前会话还没有就绪资料，请先添加或选择资料。","empty_scope")
        # 明确文档名优于活动焦点；名称重名且无法区分时不能猜。
        named=[]
        for doc in documents:
            stem=Path(doc["filename"]).stem.split("_")[0]
            if stem and stem in question: named.append(doc["id"])
        plural=bool(re.search(r"这几[篇份]|这些[文章资料文档]+|分别.*(?:概述|总结|概括)|所有.*(?:文档|资料)",question))
        overview=bool(re.search(r"讲(?:的|了)?什么|关于什么|主要内容|核心要点|全文|整篇|整体概述|概述|(?:总结|概括)",question))
        generic_reference=overview or bool(re.search(r"这[篇份]|该(?:文章|文档|资料)",question))
        if named:
            if len(named)>1 and not plural: return clarify("有多份同名资料，请明确选择要讨论的文档。","ambiguous_filename")
            result["document_ids"]=named
        elif generic_reference and not plural:
            if snapshot.get("pending_focus"):
                return clarify("新添加的资料还在处理中，待就绪后可继续；也可以明确选择已有资料。","attachment_not_ready")
            if snapshot.get("unavailable_focus"):
                return clarify("当前讨论的资料已删除或不在范围内，请重新选择。","unavailable_document")
            if snapshot.get("ambiguous_batch"):
                return clarify("本次添加了多份资料，请选择要讨论的一份，或明确要求分别概述。","ambiguous_batch")
            focus=snapshot.get("active_document_id")
            if focus in ids: result["document_ids"]=[focus]
            elif len(ids)==1: result["document_ids"]=ids
            else: return clarify("你想讨论哪份资料？请选择一份，或明确要求分别概述这些资料。","ambiguous_document")
        if overview:
            return {**result,"intent":"overview","method":"overview_rule"}
        ordinal=re.search(r"第([一二三四五六七八九十\d]+)[点项条个]",question)
        followup=bool(re.search(r"再.*(?:详细|解释|展开)|详细一点|展开(?:说|讲)|继续|刚才|上面|上述|它|这个原则|这些原则|之前提到",question))
        turns=memory["turns"]
        turn_reference=re.search(r"第(\d+)轮",question)
        previous=turns[-1] if turns else None
        if turn_reference:
            index=int(turn_reference.group(1))-1
            if not 0<=index<len(turns):
                return clarify("没有找到你指定的对话轮次，请明确说明。","missing_turn")
            previous=turns[index]
        units=previous.get("answer_units",[]) if previous else []
        # 指定主题可消除分批摘要的重复编号歧义，也兼容升级前保存的旧条目。
        titled=[u for u in units if len(u.get("title",""))>1 and u["title"] in question]
        wants_unit=bool(ordinal and (followup or len(question)<22) or
                        titled and (followup or turn_reference or re.search(r"解释|展开|说明",question)))
        if wants_unit and previous:
            if not turn_reference and memory.get("latest_attempt") and memory["latest_attempt"]["status"]!="completed":
                return clarify("上一请求失败或中断，请明确主题或重试。","previous_request_failed")
            matches=titled
            if not matches and ordinal:
                raw=ordinal.group(1);number=int(raw) if raw.isdigit() else _NUMBERS.get(raw)
                def visible_number(unit):
                    match=re.match(r"^\s*(?:\*\*)?(\d+)[.、）)]",unit.get("text",""))
                    return unit.get("display_number",int(match.group(1)) if match else unit["ordinal"])
                matches=[u for u in units if visible_number(u)==number]
            if len(matches)>1:
                choices="、".join(dict.fromkeys(u["title"] for u in matches))
                return clarify("上一回答有多组相同编号或主题（"+choices+"），请明确要展开哪个主题。","ambiguous_answer_unit")
            if not matches:
                return clarify("上一回答没有可定位的这个条目，请说明你想展开的主题。","missing_unit")
            unit=matches[0]
            unit_ids=[i for i in unit.get("document_ids",[]) if i in ids]
            if unit.get("document_ids") and not unit_ids:
                return clarify("该条目的来源已不在当前范围，请重新选择资料。","historical_source_unavailable")
            # 去掉展示序号，避免BM25把“2”当关键词召回英文论文中的无关数字。
            body=re.sub(r"^\s*(?:\*\*)?(?:\d+[.、）)]|[-*])\s*","",unit.get("text",unit["title"]))
            body=re.sub(r"\[\d+\]","",body)
            if body.startswith(unit["title"]+"："):body=body[len(unit["title"])+1:]
            # 保留实际追问，不能把“是什么意思”一律扩大成文档未提供的完整技术讲解。
            query=unit["title"]+"："+body[:300]+"；用户追问："+question
            return {**result,"resolved_query":query,
                    "document_ids":unit_ids or result["document_ids"],"method":"answer_unit",
                    "referenced_turn_id":previous["id"],"referenced_unit_id":unit["unit_id"]}
        if followup:
            if memory.get("latest_attempt") and memory["latest_attempt"]["status"]!="completed":
                return clarify("上一请求失败或中断，请明确主题或重试。","previous_request_failed")
            if not turns or turns[-1]["response"]["answer_status"] not in {"answered","partial_answer","mock"}:
                return clarify("请明确要展开的主题；上一轮没有可供展开的有效回答。","no_substantive_turn")
            previous=turns[-1]
            prior=previous.get("resolution",{}).get("resolved_query",previous["question"])
            if re.search(r"再.*(?:详细|解释|展开)|详细一点|展开(?:说|讲)|继续",question):
                # 概述后的“展开”仍需全篇取证，不能把泛化旧问句直接送入BM25导致零命中。
                prior_ids=previous.get("resolution",{}).get("document_ids",ids)
                available=[i for i in prior_ids if i in ids]
                if not available:
                    return clarify("上一主题的资料已不在当前范围，请重新选择。","historical_source_unavailable")
                is_overview=previous["response"].get("intent")=="overview"
                return {**result,"resolved_query":prior+"；"+question,
                        "intent":"overview" if is_overview else "qa",
                        "document_ids":result["document_ids"] if named else available,
                        "method":"previous_topic","referenced_turn_id":previous["id"]}
            # 复杂指代仅一次逻辑生成，返回ID必须属于当前范围，历史只用于理解。
            if self.config.llm_mode!="mock":
                system=("解析文档问答的追问，不回答事实。给定范围外的文档ID不可使用。"
                        "有歧义返回clarification。输出JSON且仅含intent、resolved_query、document_ids、clarification_question。"
                        "intent为qa/overview/clarification，resolved_query是独立完整问题。")
                built=self.builder.build(system,{"question":question,"allowed_document_ids":ids,
                    "topic_summary":memory["state"].get("summary",""),"last_question":prior,
                    "answer_units":previous.get("answer_units",[])})
                generation,trace=await self.client.generate(built.messages)
                try:
                    data=json.loads(generation.content)
                    valid=set(data)=={"intent","resolved_query","document_ids","clarification_question"}
                    valid=valid and data["intent"] in {"qa","overview","clarification"} and isinstance(data["resolved_query"],str)
                    valid=valid and isinstance(data["document_ids"],list) and all(isinstance(i,str) and i in ids for i in data["document_ids"])
                    if not valid or (data["intent"]!="clarification" and not data["document_ids"]):
                        raise ValueError("resolver scope invalid")
                    return {**result,**data,"method":"llm","trace":trace,"context_budget":built.trace}
                except (ValueError,TypeError):
                    return {**clarify("无法确定你指代的主题，请补充说明。","invalid_resolution"),"trace":trace,"context_budget":built.trace}
            return clarify("请明确你指代的主题或条目。","unresolved_reference")
        return result
