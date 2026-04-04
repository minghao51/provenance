FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

COPY pyproject.toml uv.lock ./

RUN uv pip install --system ".[all]"

COPY . .

RUN python -m spacy download en_core_web_sm

EXPOSE 8080

ENTRYPOINT ["provenance"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
