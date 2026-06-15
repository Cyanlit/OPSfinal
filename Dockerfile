# Stage 1: Build wheels
FROM python:3.10-slim AS builder
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY requirements.txt .
# Only backend deps are needed inside the container
RUN grep -v "^#" requirements.txt | grep -v "Pillow\|requests\|tkinterdnd2" > requirements.backend.txt
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.backend.txt

# Stage 2: Minimal runtime image
FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .
COPY .env.example .env

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
