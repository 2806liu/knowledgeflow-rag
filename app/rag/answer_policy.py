"""回答策略与严格数据契约；来源权限由服务端校验，不放宽旧严格校验器。"""
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt

PolicyName = Literal["document_strict", "document_enhanced"]
Intent = Literal["overview", "document_fact", "concept_explanation", "comparison", "general_question", "clarification"]
CITATION = re.compile(r"\[(\d+)\]")

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class Aspect(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    question: str = Field(min_length=1, max_length=500)
    requires_document: bool = False

class ResolvedQuestion(StrictModel):
    intent: Intent
    resolved_query: str = Field(min_length=1, max_length=2500)
    document_ids: list[str] = Field(max_length=1000)
    requested_aspects: list[Aspect] = Field(min_length=1, max_length=8)
    referenced_turn_id: str | None = None
    referenced_unit_id: str | None = None
    referenced_unit_ids: list[str] = Field(default_factory=list, max_length=8)
    candidate_keys: list[str] = Field(default_factory=list, max_length=12)
    clarification_question: str | None = Field(default=None, max_length=1000)

class AnswerSection(StrictModel):
    source_type: Literal["document", "model_knowledge", "limitation"]
    content: str = Field(min_length=1, max_length=16000)
    citation_numbers: list[StrictInt] = Field(default_factory=list, max_length=64)
    aspect_ids: list[str] = Field(default_factory=list, max_length=8)

class EnhancedDecision(StrictModel):
    action: Literal["answer", "retrieve_more"]
    sections: list[AnswerSection] = Field(default_factory=list, max_length=16)
    unresolved_aspect_ids: list[str] = Field(default_factory=list, max_length=8)
    supplemental_query: str | None = Field(default=None, max_length=500)

def document_specific(question):
    """对论文数据、文件要求等强约束作额外保护；一般概念解释仍允许知识补充。"""
    # “实验参数的一般作用”是概念；“论文实验参数具体是多少”仍是文档事实。
    general=bool(re.search(r"一般作用|通用(?:含义|解释|知识)|什么是|概念|定义",question))
    concrete=bool(re.search(r"具体|数值|取值|是多少|采用了|使用了|论文.{0,10}(?:指出|实验|参数)|文中.{0,10}(?:给出|指出)",question))
    if general and not concrete:
        return False
    return bool(re.search(
        r"实验参数|实验结果|样本量|数据集|准确率|召回率|电脑品牌|电脑型号|截止时间|截止日期|命名格式|"
        r"(?:论文|文中|作者|资料|报告).{0,30}(?:参数值|数值|多少|百分比|采用了|使用了|得出了)|"
        r"(?:experiment|paper).{0,25}(?:parameter|result|sample size)", question, re.I))

def apply_policy(resolution):
    """混合问题逐方面约束；具体文档事实不得被模型标成可自由补充。"""
    value = dict(resolution)
    aspects = []
    for raw in value["requested_aspects"]:
        item = dict(raw)
        text=item["question"]
        explicit_attribution=bool(re.search(r"仅(?:依|根|限)|只(?:依|根)|(?:论文|文中|作者|原文).{0,12}(?:指出|认为|证明|发现|给出|明确|实际|实验)",text))
        generic=bool(re.search(r"原理|含义|概念|用途|局限|一般|通用|解释|比较|区别|适用条件|使用条件",text))
        if value["intent"]=="overview" or document_specific(text) or explicit_attribution:
            item["requires_document"]=True
        elif generic:
            # 通用方面仍优先使用文档，但缺少证据时允许单独标记的模型知识。
            item["requires_document"]=False
        elif value["intent"]=="document_fact":
            item["requires_document"]=True
        aspects.append(item)
    value["requested_aspects"] = aspects
    return value

def validate_decision(raw, resolution, evidence, final=False):
    """验证结构、逐区块引用与方面权限；不冒称能自动证明语义正确。"""
    data = EnhancedDecision.model_validate_json(raw)
    available = {e.citation_number for e in evidence}
    aspects = {a["id"]: a for a in resolution["requested_aspects"]}
    if len(aspects) != len(resolution["requested_aspects"]):
        raise ValueError("duplicate aspect IDs")
    if data.action == "retrieve_more":
        if not data.supplemental_query or data.sections or data.unresolved_aspect_ids:
            raise ValueError("invalid supplemental retrieval")
        if final:
            # 补检索额度耗尽时明确未解决项，不能继续调用或伪装已完成回答。
            stopped=dict(action="answer",supplemental_query=None,
                unresolved_aspect_ids=list(aspects),
                sections=[dict(source_type="limitation",content="本轮补检索次数已用完，尚不能完整回答这些方面。",
                               citation_numbers=[],aspect_ids=list(aspects))])
            return validate_decision(__import__("json").dumps(stopped,ensure_ascii=False),resolution,evidence,True)
        return data, None
    if data.supplemental_query:
        raise ValueError("final answer contains unused retrieval query")
    if len(set(data.unresolved_aspect_ids)) != len(data.unresolved_aspect_ids) or not set(data.unresolved_aspect_ids) <= set(aspects):
        raise ValueError("unknown unresolved aspects")
    answered, cited = set(), []
    kinds = set()
    for section in data.sections:
        if not section.aspect_ids or not set(section.aspect_ids) <= set(aspects):
            raise ValueError("section must reference valid aspects")
        markers = set(int(n) for n in CITATION.findall(section.content))
        numbers = section.citation_numbers
        if len(numbers) != len(set(numbers)) or any(n < 1 for n in numbers):
            raise ValueError("duplicate or invalid citation numbers")
        if markers != set(numbers) or not markers <= available:
            raise ValueError("citation declarations do not match current evidence")
        if resolution.get("comparison_task") and re.search(r"\bd[1-9][0-9]*\b", section.content, re.I):
            raise ValueError("comparison answer exposes internal document alias")
        if section.source_type == "document":
            if not markers:
                raise ValueError("document section requires canonical citation")
            if resolution.get("comparison_task"):
                # 双篇比较按方面核对引用所属文档；编号存在并不代表引用了正确的论文。
                source_docs = {e.document_id for e in evidence if e.citation_number in markers}
                for aspect_id in section.aspect_ids:
                    aspect = aspects[aspect_id]
                    required = aspect.get("document_id")
                    pair = aspect.get("comparison_document_ids")
                    if required and source_docs != {required}:
                        raise ValueError("comparison aspect cites the wrong document")
                    if pair and not set(pair).issubset(source_docs):
                        raise ValueError("comparison difference lacks both document sources")
            answered.update(section.aspect_ids)
            cited.extend(n for n in numbers if n not in cited)
            kinds.add("document")
        elif section.source_type == "model_knowledge":
            if markers or any(aspects[a]["requires_document"] for a in section.aspect_ids):
                raise ValueError("model knowledge cannot replace document facts or cite documents")
            answered.update(section.aspect_ids)
            kinds.add("model_knowledge")
        elif markers:
            # 缺失说明可描述本次证据状态，不将说明文本伪装成新的文档事实。
            raise ValueError("limitation sections must not attach document citations")
    unresolved = set(data.unresolved_aspect_ids) | (set(aspects) - answered)
    # 已解答与尚未解决不能同时指向同一方面，防止回答状态自相矛盾。
    if answered & set(data.unresolved_aspect_ids):
        raise ValueError("answered aspect is also unresolved")
    basis = ("document_and_model" if len(kinds) == 2 else
             "document_only" if kinds == {"document"} else
             "model_only" if kinds == {"model_knowledge"} else "none")
    status = "partial_answer" if answered and unresolved else "answered" if answered else "insufficient_evidence"
    result = dict(answer_status=status, answer_basis=basis, cited_numbers=cited,
                  answer_sections=[s.model_dump() for s in data.sections],
                  unresolved_aspects=[aspects[a]["question"] for a in aspects if a in unresolved])
    # 缺失方面由服务端补齐，不能因为模型省略区块就把结果显示为完整。
    if unresolved:
        result["answer_sections"].append(dict(source_type="limitation",
            content="当前文档证据不足以确认：" + "；".join(result["unresolved_aspects"]),
            citation_numbers=[], aspect_ids=[a for a in aspects if a in unresolved]))
    if not result["answer_sections"]:
        result["answer_sections"] = [dict(source_type="limitation", content="资料不足，无法根据当前文档回答。",
                                          citation_numbers=[], aspect_ids=list(aspects))]
    labels = {"document": "文档依据", "model_knowledge": "通用知识补充（模型知识，未由文档直接证明）", "limitation": "范围与缺失说明"}
    result["answer"] = "\n\n".join("### " + labels[s["source_type"]] + "\n" + s["content"]
                                   for s in result["answer_sections"])
    return data, result
