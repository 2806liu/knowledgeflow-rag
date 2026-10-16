"""通过Node执行上传客户端的故障单测；此结果不代替浏览器真实操作。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import unittest

class FrontendUploadTests(unittest.TestCase):
    def test_upload_faults_with_simulated_xhr(self):
        node=shutil.which("node")
        if not node:self.skipTest("Node runtime unavailable; browser validation still required")
        root=Path(__file__).resolve().parents[1]
        import app
        env={**os.environ,"KF_STATIC_DIR":str(Path(app.__file__).resolve().parent/"static")}
        result=subprocess.run([node,str(root/"tests/frontend_upload_tests.js")],cwd=root,
                              text=True,capture_output=True,timeout=15,env=env)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout.strip())
        self.assertEqual(report["checks"],8)
        self.assertFalse(report["browser_acceptance"])
        # 保存实际Node结果，Linux应用镜像不需要额外安装开发时Node运行时。
        output=root/"evaluation/results/conversation-v2/frontend-unit.json"
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
