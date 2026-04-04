FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --upgrade pip && \
    pip install ".[all]"

COPY . .

RUN python -m spacy download en_core_web_sm

EXPOSE 8080

ENTRYPOINT ["provenance"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
