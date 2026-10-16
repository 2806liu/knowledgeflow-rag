"""用公开临时夹具验证发布边界，避免把本地文档或运行配置传到 GitHub。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_release import ROOT_FILES, export, validate_public_name


class ReleasePolicyTests(unittest.TestCase):
    """发布只接受白名单文件，并在写入目标目录前完成检查。"""

    def setUp(self):
        """在项目测试临时目录构造独立源码树，不读取用户数据。"""
        self.parent = Path("tests/.tmp").resolve()
        self.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="release-policy-", dir=self.parent)
        self.root = Path(self.temporary.name) / "project"
        names = list(ROOT_FILES) + [
            "app/__init__.py", "app/rag/__init__.py", "app/static/index.html",
            "tests/test_public.py", "examples/public-demo.md", "scripts/smoke_api.py",
            "scripts/prepare_release.py", ".github/workflows/ci.yml",
            "tests/frontend_answer_tests.js", "tests/frontend_upload_tests.js",
            "evaluation/conversation-v2/acceptance-plan.json",
            "evaluation/conversation-v2/fixtures/public.md",
            "evaluation/document-enhanced-v1/fixtures/public.md", "docs/private.md",
            ".env", ".env.docker", ".studyflow/private.db", "evaluation/results/private.json",
        ]
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("public fixture\n", encoding="utf-8")

    def tearDown(self):
        """清理前验证绝对路径，确保只删除本次生成的测试目录。"""
        if not Path(self.temporary.name).resolve().is_relative_to(self.parent):
            raise AssertionError("unexpected temporary directory")
        self.temporary.cleanup()

    def test_export_excludes_docs_secrets_and_runtime(self):
        """源码与配置模板可发布，docs、真实环境和数据库不能进入清单。"""
        destination = self.root / "dist/public"
        # 模拟 Windows 原文件，导出后统一换行以保证 Git 发布哈希可核对。
        (self.root / "README.md").write_bytes(b"public fixture\r\n")
        result = export(self.root, destination)
        manifest = json.loads((destination / "release-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result["files"], len(manifest["files"]))
        self.assertIn("README.md", manifest["files"])
        self.assertIn("app/static/index.html", manifest["files"])
        self.assertIn(".env.example", manifest["files"])
        self.assertEqual((destination / "README.md").read_bytes(), b"public fixture\n")
        for name in ["docs/private.md", ".env", ".env.docker", ".studyflow/private.db",
                     "evaluation/results/private.json"]:
            self.assertNotIn(name, manifest["files"])
            self.assertFalse((destination / name).exists())

    def test_expanding_allowlist_cannot_bypass_exclusions(self):
        """后续扩展白名单时仍阻止敏感目录与越界路径，失败不留发布包。"""
        for name in ["docs/private.md", "../outside.txt", ".env.docker", "data/private.db",
                     "evaluation/results/private.json"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_public_name(name)
        destination = self.root / "dist/blocked"
        with patch("scripts.prepare_release.release_names", return_value=["docs/private.md"]):
            with self.assertRaisesRegex(ValueError, "excluded release source"):
                export(self.root, destination)
        self.assertFalse(destination.exists())

    def test_possible_secret_is_rejected_without_echo(self):
        """密钥模式只用合成字符串验证，异常不能回显疑似密钥。"""
        token = "sk-" + "A" * 32
        (self.root / "README.md").write_text(token, encoding="utf-8")
        destination = self.root / "dist/blocked"
        with self.assertRaisesRegex(ValueError, "possible secret") as raised:
            export(self.root, destination)
        self.assertNotIn(token, str(raised.exception))
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
