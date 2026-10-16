"""将项目 A 的公开文件白名单导出为独立发布目录，隔离用户资料、密钥和项目 B。"""

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT_FILES = ("README.md", "pyproject.toml", "Dockerfile", "compose.yaml", ".dockerignore",
              ".gitignore", ".gitattributes", ".env.example", ".env.docker.example")


def release_names(root: Path):
    """仅枚举公开源码和合成夹具；设计说明集中在 README，docs 留在本地。"""
    names = list(ROOT_FILES)
    for directory in ("app", "tests", "evaluation"):
        names.extend(p.relative_to(root).as_posix() for p in (root / directory).glob("*.py"))
        # app/rag 是当前唯一需要发布的应用子包；不递归复制运行数据和临时目录。
    names.extend(p.relative_to(root).as_posix() for p in (root / "app" / "rag").glob("*.py"))
    names += ["examples/public-demo.md", "scripts/smoke_api.py", "scripts/prepare_release.py",
              ".github/workflows/ci.yml"]
    # 静态页面使用明确扩展名白名单，防止漏发资源或复制运行数据。
    names.extend(p.relative_to(root).as_posix() for p in (root / "app" / "static").iterdir()
                 if p.is_file() and p.suffix in {".html", ".css", ".js"})
    # 只发布本轮公开合成夹具、固定验收计划与客户端单测，不导出真实模型报告。
    names += ["tests/frontend_answer_tests.js", "tests/frontend_upload_tests.js", "evaluation/conversation-v2/acceptance-plan.json"]
    names.extend(p.relative_to(root).as_posix() for p in (root / "evaluation" / "conversation-v2" / "fixtures").iterdir()
                 if p.is_file() and p.suffix in {".md", ".json"})
    # 增强问答只分发冻结的公开合成集，不导出真实响应、语义评审和日常数据库。
    names.extend(p.relative_to(root).as_posix() for p in (root / "evaluation" / "document-enhanced-v1").rglob("*")
                 if p.is_file() and p.suffix in {".md", ".json"})
    return sorted(set(names))


def validate_public_name(name: str):
    """白名单扩展时仍拒绝文档归档、真实配置、运行数据和目录越界。"""
    relative = Path(name)
    forbidden = {"docs", ".git", ".venv", ".studyflow", ".skillflow", "references",
                 "dist", "results", ".tmp", "data"}
    if (relative.is_absolute() or ".." in relative.parts
            or any(part in forbidden for part in relative.parts)
            or relative.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".pdf", ".tar"}
            or (relative.name.startswith(".env") and relative.name not in ROOT_FILES)):
        raise ValueError("excluded release source: " + name)


def export(root: Path, destination: Path):
    """先检查全部来源再创建发布包；不覆盖旧包，不修改父工作区的 Git。"""
    root = root.resolve()
    destination = destination.resolve()
    if not destination.is_relative_to(root / "dist") or destination.exists():
        raise ValueError("destination must be a new directory under project A/dist")
    files = []
    for name in release_names(root):
        validate_public_name(name)
        source = root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError("missing or linked release source: " + name)
        text = source.read_text(encoding="utf-8")
        # 限定常见真实密钥前缀；此检查不能代替发布前人工检查，但不会打印疑似密钥。
        if re.search(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})", text):
            raise ValueError("possible secret in release source: " + name)
        files.append((name, source))
    destination.mkdir(parents=True)
    manifest = {}
    for name, source in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # 公开文件均为 UTF-8 文本；统一 LF，保证跨系统 Git 与发布清单哈希一致。
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        manifest[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "release-manifest.json").write_text(
        json.dumps({"files": manifest, "count": len(manifest)}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return {"directory": str(destination), "files": len(manifest)}


def main():
    """导出不触碰原仓库，GitHub 上传只针对输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="knowledgeflow-rag")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.name):
        parser.error("invalid release directory name")
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(export(root, root / "dist" / args.name)))


if __name__ == "__main__":
    main()
