"""真实LLM的公开合成记忆测试；只发送本文件构造的虚拟对话，不读取用户文档。"""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import uuid
from app.config import settings
from app.store import Store
from app.conversation import ConversationService
from app.rag.memory import MemoryManager
from app.rag.llm_client import LLMClient

async def main():
    config=replace(settings,data_dir=Path(tempfile.mkdtemp(prefix="rag-public-memory-")),
                   memory_recent_rounds=2,history_budget=1000)
    store=Store(config.data_dir);repo=ConversationService(store);session=repo.create()["id"]
    memory=MemoryManager(repo,config,LLMClient(config))
    transcript=[
      ("先讨论课程报告命名。","文档命名格式是学号-姓名-报告。"),
      ("课程报告提交什么格式？","文档要求PDF格式。"),
      ("文档截止时间是什么？","原文是6月30日18:00。"),
      ("之后也要讨论实验室考勤。","原文考勤规则是每天4次。"),
      ("我猜电脑品牌是Quantum。","这只是用户猜测，文档未核实品牌，不能当成已知事实。"),
      ("纠正之前的讨论目标：我希望自己的计划目标改成7月1日，原文仍是6月30日。","已记录用户个人计划目标7月1日，未修改原文规定。"),
      ("后续请只依据原文回答，不要把猜测写成事实。","确认只按原文取证，用户偏好和文档事实分别说明。"),
      ("我需要记住命名规则和考勤两类主题。","已记录报告命名与实验室考勤主题。"),
      ("最近继续聊报告。","继续讨论课程报告。"),
      ("先保留这轮对话。","保留完整轮次。")]
    def seed(pairs):
        for question,answer in pairs:
            turn,_=repo.begin_turn(session,str(uuid.uuid4()),question)
            repo.finish_turn(turn,{"response":{"answer":answer,"answer_status":"answered","citations":[]},
                                  "resolution":{"resolved_query":question},"answer_units":[]})
    seed(transcript)
    first=await memory.compact(session);state1=memory.state(session)
    second_pairs=[
      ("继续查看历史主题。","之前讨论过报告、PDF和考勤。"),
      ("命名规则仍按原文。","继续使用原文命名规则。"),
      ("撤回我之前7月1日的个人计划目标；后续只讨论报告命名。","已撤回个人7月1日目标，当前主题为报告命名，原文截止要求未被修改。"),
      ("确认用户猜测品牌没有成为事实。","品牌仍未核实。"),
      ("请保留我的撤回与主题更改。","记录撤回及当前报告命名主题。"),
      ("命名规则用于后续追问。","已保留命名主题。"),
      ("这是最近一轮。","保留最近完整轮次。"),
      ("还有什么需要我明确？","等待明确的新问题。")]
    seed(second_pairs);second=await memory.compact(session);state2=memory.state(session)
    report={"date":"2026-10-15","data_origin":"public_synthetic_transcript_only","mode":config.llm_mode,
            "model":config.llm_model,"first_result":first,"first_state":state1,"second_result":second,"second_state":state2,
            "transcript":transcript+second_pairs,"semantic_review":"pending",
            "structural_checks":{"original_turns_retained":len(repo.turns(session))==18,
              "source_ids_unique":len(state2.get("source_turn_ids",[]))==len(set(state2.get("source_turn_ids",[]))),
              "watermark_advanced":state2.get("covered_until",0)>state1.get("covered_until",0)},
            "passed":first.get("status")=="updated" and second.get("status")=="updated"}
    store.conn.close()
    print(json.dumps(report,ensure_ascii=True))
    return report

if __name__=="__main__":asyncio.run(main())
