# 项目 A 独立发布到 GitHub

父工作区包含多个独立项目。只导出项目 A 的公开文件，发布包不包含项目 B、用户上传资料、运行数据库、模型、历史问答报告、求职材料或参考仓库。

## 导出

```powershell
python scripts/prepare_release.py
```

输出 dist/knowledgeflow-rag。脚本使用白名单、校验常见密钥前缀并生成 SHA256 清单；目标已存在时拒绝覆盖。需要新快照可使用 --name knowledgeflow-rag-v2。发布前查看输出内容和 .env 示例，真实密钥仅在本地配置。

## 验证导出目录

在导出目录创建独立虚拟环境，安装 .[ocr,vector,reranker]，运行 unittest；再执行 docker compose config、docker build、启动和 smoke。历史私有数据集未分发，公开演示使用 examples/public-demo.md。

## 建立独立仓库并上传

先在 GitHub 创建空仓库，选择自己的公开/私有设置，再在导出目录执行：

```powershell
git init -b main
git add .
git diff --cached --stat
git commit -m "Initial KnowledgeFlow RAG release"
git remote add origin <你的GitHub仓库地址>
git push -u origin main
```

不从父工作区执行 git add .，不把项目 B 一起上传。GitHub 登录和仓库权限需要本人账号。仓库 LICENSE 应由本人确认代码与资料授权后选择，不预先声称所有资料均可再分发。
