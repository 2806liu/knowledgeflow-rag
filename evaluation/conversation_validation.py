"""记录真实单元/集成测试结果，区分自动场景与待人工/浏览器验收。"""
import argparse
import json
import re
import sys
import time
import unittest
from pathlib import Path

class RecordingResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.records=[];self.failed_ids=set();self.skipped_ids=set()
    def startTest(self,test):
        self.started=time.perf_counter();super().startTest(test)
    def addFailure(self,test,err):
        self.failed_ids.add(test.id());super().addFailure(test,err)
    def addError(self,test,err):
        self.failed_ids.add(test.id());super().addError(test,err)
    def addSubTest(self,test,subtest,err):
        if err:self.failed_ids.add(test.id())
        super().addSubTest(test,subtest,err)
    def addSkip(self,test,reason):
        self.skipped_ids.add(test.id());super().addSkip(test,reason)
    def stopTest(self,test):
        match=re.search(r"\.test_([OAUCMB]\d{2})_",test.id())
        self.records.append({"test":test.id(),"case_id":match.group(1) if match else None,
            "status":"skipped" if test.id() in self.skipped_ids else "failed" if test.id() in self.failed_ids else "passed",
            "elapsed_ms":round((time.perf_counter()-self.started)*1000,3)})
        super().stopTest(test)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("evaluation/results/conversation-v2/automated-tests.json"))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"tests"))
    suite=unittest.defaultTestLoader.discover(str(root/"tests"))
    result=unittest.TextTestRunner(verbosity=1,resultclass=RecordingResult).run(suite)
    new=[r for r in result.records if r["case_id"]]
    report={"date":"2026-10-15","test_count":result.testsRun,"passed":result.wasSuccessful(),"skipped":len(result.skipped),
        "passed_count":sum(r["status"]=="passed" for r in result.records),"new_scenario_count":len(new),"new_scenarios_passed":sum(r["status"]=="passed" for r in new),
        "scenarios_have_unique_ids":len({r["case_id"] for r in new})==len(new),
        "semantic_acceptance":"pending_real_model_and_review","browser_acceptance":"separate_required",
        "records":result.records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="records"}))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=="__main__":main()
