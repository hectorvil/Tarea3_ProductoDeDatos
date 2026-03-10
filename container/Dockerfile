FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/program/.venv/bin:/opt/program:${PATH}"

WORKDIR /opt/program

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /opt/program/
RUN uv sync --frozen --no-dev --no-install-project

COPY src /opt/program/src
COPY container/train /opt/program/train
COPY container/serve /opt/program/serve
COPY container/predictor.py /opt/program/predictor.py
COPY container/wsgi.py /opt/program/wsgi.py

RUN chmod +x /opt/program/train /opt/program/serve