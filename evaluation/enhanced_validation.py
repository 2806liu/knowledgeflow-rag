"""记录48项新增场景与全量回归，区分结构通过与语义/浏览器待验收。"""
import argparse
import json
from pathlib import Path
import re
import sys
import unittest
from evaluation.conversation_validation import RecordingResult

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("evaluation/results/document-enhanced-v1/module-tests.json"))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"tests"))
    suite=unittest.defaultTestLoader.discover(str(root/"tests"),pattern="test*.py" if args.all else "test_enhanced*.py")
    result=unittest.TextTestRunner(verbosity=1,resultclass=RecordingResult).run(suite)
    for item in result.records:
        match=re.search(r"\.test_(E[A-Z]\d{2})_",item["test"])
        if match:item["enhanced_case_id"]=match[1]
    enhanced=[r for r in result.records if r["test"].startswith("test_enhanced")]
    identified=[r for r in enhanced if r.get("enhanced_case_id")]
    report=dict(date="2026-10-15",test_count=result.testsRun,passed=result.wasSuccessful(),
        passed_count=sum(r["status"]=="passed" for r in result.records),skipped=len(result.skipped),
        enhanced_case_count=len(enhanced),identified_case_count=len(identified),enhanced_passed=sum(r["status"]=="passed" for r in enhanced),
        semantic_acceptance="requires_real_model_review",browser_acceptance="separate_required",records=result.records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="records"}))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=="__main__":main()
