"""工作台 HTTP 边界测试：静态资源完整、能力信息脱敏、既有 API 继续可用。"""

import unittest
from dataclasses import replace

from fastapi.testclient import TestClient


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        """只替换页面读取的配置；不上传文件、不调用模型或修改日常知识库。"""
        from app import main
        self.main = main
        self.original = main.settings
        main.settings = replace(main.settings, llm_mode="langchain_openai",
                                llm_api_key="workspace-private-key", llm_base_url="https://private.invalid/v1",
                                ocr_enabled=False, max_upload_bytes=123456)
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        self.main.settings = self.original

    def test_workspace_and_assets_are_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("文档知识工作台", response.text)
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        for path, content_type in (("/static/app.js", "javascript"), ("/static/styles.css", "text/css")):
            asset = self.client.get(path)
            self.assertEqual(asset.status_code, 200)
            self.assertIn(content_type, asset.headers["content-type"])

    def test_capabilities_never_expose_credentials_or_endpoint(self):
        response = self.client.get("/ui/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer_mode": "llm", "ocr_enabled": False, "max_upload_bytes": 123456})
        self.assertNotIn("workspace-private-key", response.text)
        self.assertNotIn("private.invalid", response.text)

    def test_static_mount_cannot_read_project_configuration(self):
        for path in ("/static/.env", "/static/%2e%2e/%2e%2e/.env", "/static/config.py"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_api_documentation_and_health_remain_available(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/chat", schema["paths"])
        self.assertIn("/documents/upload", schema["paths"])


if __name__ == "__main__":
    unittest.main()
