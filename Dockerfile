# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile native extensions (faiss, pymupdf, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY rag/ ./rag/

# Install core + optional extras; skip GPU-heavy OCR and paddle
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
        "python-docx>=1.1.0" \
        "openpyxl>=3.1.0" \
        "ragas>=0.1.0" \
        "langchain" \
        "langchain-openai" \
 && pip install --no-cache-dir -e .


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libraries (libGL for pymupdf rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY rag/ ./rag/
COPY configs/ ./configs/
COPY .env.example ./.env.example

# Data directory (bind-mounted at runtime; pre-create so permissions are right)
RUN mkdir -p /app/data

# Streamlit port
EXPOSE 8501

# MCP server port
EXPOSE 8000

# Default: run the Streamlit Studio UI
CMD ["streamlit", "run", "rag/app/studio/studio.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
