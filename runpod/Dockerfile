FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LTX_APP_DATA_DIR=/workspace/ltx-data
ENV LTX_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen

COPY backend ./
COPY runpod/start.sh /usr/local/bin/ltx-runpod-start
RUN chmod +x /usr/local/bin/ltx-runpod-start

EXPOSE 8000
CMD ["ltx-runpod-start"]
