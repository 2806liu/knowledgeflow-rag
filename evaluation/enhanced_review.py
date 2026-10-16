"""将真实增强回答与来源并排保存，助手初审与人工确认始终分别记录。"""
import argparse
import hashlib
import html
import json
from pathlib import Path


def build(directory):
    """报告哈希绑定原始响应；未知或待审项不自动转为通过。"""
    records=[];hashes={}
    annotation_path=directory/"assistant-annotations.json"
    annotations=json.loads(annotation_path.read_text(encoding="utf-8")) if annotation_path.exists() else {}
    partitions=["regression"] if (directory/"live-regression.json").exists() else ["development","holdout"]
    for partition in partitions:
        source=directory/("live-"+partition+".json")
        if not source.exists():continue
        raw=source.read_bytes();hashes[source.name]=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        expected=annotations.get("report_hashes",{}).get(source.name)
        if expected and expected!=hashes[source.name]:raise ValueError("review annotation report hash mismatch")
        for case in data["cases"]:
            for step in case["steps"]:
                key=case["id"]+":"+step["label"]
                records.append(dict(key=key,partition=partition,question=step["question"],
                    response=step.get("response",{}),automatic_checks_passed=step["automatic_checks_passed"],
                    assistant_review=annotations.get("steps",{}).get(key,{"status":"pending"}),
                    human_confirmation="pending"))
    review=dict(reviewer="Codex (assistant initial review)",review_type="assistant_initial_review",
        human_confirmation="pending",report_hashes=hashes,records=records,
        limitations=["结构和引用定位通过不等于事实正确", "模型知识独立复核，不能因没有文档引用直接判错", "未逐原子事实标注前不统计引用支持率"])
    (directory/"assistant-review.json").write_text(json.dumps(review,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    escape=lambda value:html.escape(str(value))
    cards=[]
    for row in records:
        response=row["response"];note=row["assistant_review"]
        citations="".join("<li><b>[%s] %s</b><pre>%s</pre></li>"%(c["citation_number"],escape(c["filename"]),escape(c["snippet"])) for c in response.get("citations",[]))
        cards.append('<article><h2>'+escape(row["key"])+" · "+escape(note.get("status","pending"))+"</h2><p>"+escape(row["question"])+
            '</p><div class="grid"><pre>'+escape(response.get("answer","请求失败"))+"</pre><div><h3>支持原文</h3><ul>"+citations+
            "</ul><h3>助手初审</h3><pre>"+escape(json.dumps(note,ensure_ascii=False,indent=2))+"</pre></div></div>"+
            '<label><input type="checkbox" data-key="'+escape(row["key"])+'"> 我已核对本条回答、来源与初审意见</label></article>')
    # 只导出人工确认记录，不覆盖源报告；下载时保留绑定哈希及实际复核身份。
    payload=json.dumps(dict(report_hashes=hashes,assistant_review_file="assistant-review.json"),ensure_ascii=False).replace("<","\\u003c")
    script="""<script>
const bound=BOUND;
document.querySelector('#download').onclick=()=>{
 const reviewer=document.querySelector('#reviewer').value.trim();
 const checks=[...document.querySelectorAll('[data-key]')];
 if(!reviewer||checks.some(x=>!x.checked)){alert('请填写复核身份并逐项核对；未检查的条目保持待审。');return;}
 const data={...bound,review_type:'human_confirmation',reviewer,confirmed_at:new Date().toISOString(),
 decisions:checks.map(x=>({key:x.dataset.key,confirmed:true}))};
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
 a.download='document-enhanced-human-confirmation.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};</script>""".replace("BOUND",payload)
    page="""<!doctype html><meta charset="utf-8"><title>文档增强问答初审</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:16px;background:#f5f8f6;color:#20362f}article,header{background:white;padding:24px;margin:20px 0;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui}.grid{display:grid;grid-template-columns:1fr 1fr;gap:28px}input,button{padding:10px}@media(max-width:800px){.grid{display:block}}</style>
<header><h1>文档增强问答 · 助手初审</h1><p>人工确认仍待完成。请逐项核对回答、实际引用、模型补充及初审问题；勾选仅表示你完成核对，不会修改原始报告。</p><p>发现错误时应保留初审缺陷记录。此页的机械通过不等于语义全通过。</p><input id="reviewer" placeholder="你的复核身份"><button id="download">下载人工复核确认</button></header>"""+"".join(cards)+script
    target=directory/"assistant-review.html";target.write_text(page,encoding="utf-8")
    return dict(records=len(records),human_confirmation="pending",path=str(target))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=Path("evaluation/results/document-enhanced-v1"))
    args=parser.parse_args();print(json.dumps(build(args.directory),ensure_ascii=True))

if __name__=="__main__":main()
