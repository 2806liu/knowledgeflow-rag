"""项目 A 的本地工作台：提供静态页面和最小能力信息，不对外暴露配置或密钥。"""

from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings


STATIC_DIR = Path(__file__).resolve().parent / "static"


def attach_workspace(app: FastAPI, settings_provider: Callable[[], Settings]) -> None:
    """将工作台挂到根路径；同源调用既有 API，原有 /docs 与业务接口继续使用。"""
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="workspace-static")

    @app.get("/", include_in_schema=False)
    def workspace() -> FileResponse:
        """仅发送固定 HTML，不接受用户指定文件路径；页面脚本和字体不依赖外部站点。"""
        return FileResponse(
            STATIC_DIR / "index.html", media_type="text/html",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
                    "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
                ),
            },
        )

    @app.get("/ui/config", include_in_schema=False)
    def workspace_config() -> dict:
        """只返回影响用户操作的能力标志；不得序列化 Settings 或返回模型密钥、路径、地址。"""
        config = settings_provider()
        return {
            "answer_mode": "mock" if config.llm_mode == "mock" else "llm",
            "ocr_enabled": config.ocr_enabled,
            "max_upload_bytes": config.max_upload_bytes,
        }
