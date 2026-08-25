# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

RUN groupadd -r helmet && useradd --no-log-init -r -g helmet helmet

WORKDIR /app

COPY --from=builder /install /usr/local
COPY api/ api/
COPY cost_report.py .
COPY .env.example .env

RUN chown -R helmet:helmet /app

USER helmet

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
