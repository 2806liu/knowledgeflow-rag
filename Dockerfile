# 项目 A 独立镜像；不复制密钥、用户资料和运行数据库。
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STUDYFLOW_OCR_ENABLED=false \
    STUDYFLOW_DATA_DIR=/data STUDYFLOW_OCR_MODEL_DIR=/data/ocr_models \
    STUDYFLOW_RERANKER_MODEL_DIR=/data/reranker_models
WORKDIR /app
# 使用官方 HTTPS 软件源与有限重试，避免首次构建遇到临时网关错误。
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# 先安装依赖，再复制应用；仅调整页面时复用依赖层，避免重新下载 OCR 等大型包。
COPY pyproject.toml ./
RUN python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; d=p['dependencies']+sum([p['optional-dependencies'][x] for x in ('ocr','vector','reranker')],[]); open('/tmp/rag-requirements.txt','w').write('\n'.join(d))" \
    && pip install "setuptools>=68" wheel -r /tmp/rag-requirements.txt \
    && groupadd --gid 10001 rag && useradd --uid 10001 --gid rag --no-create-home rag \
    && mkdir -p /data && chown rag:rag /data
COPY README.md ./
COPY app ./app
# 安装当前代码及静态资源；依赖已经就绪，不触发重复解析或网络请求。
RUN pip install --no-deps --no-build-isolation .
USER rag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
