# Create the requirements.txt to download dependencies from poetry
FROM python:3.11 as builder

RUN pip install poetry

WORKDIR /tmp

COPY pyproject.toml poetry.lock ./

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --only main


# 1. Base Docker Image (contains linux and a small python version)
FROM python:3.11-slim

# 2. Env variables optimize for python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# 3. Install system dependencies
#RUN apt-get update && apt-get install -y \
#    curl \
#    libgl1 \
#    libglib2.0-0 \
#    && rm -rf /var/lib/apt/lists/*

# 4. Define a new user (to avoid the use of root)
RUN useradd -m -u 1000 appuser

# 5. Define a subfolder
WORKDIR /app
RUN chown appuser:appuser /app

# 6. Go to the created user
USER appuser

COPY --from=builder /tmp/requirements.txt .

# 7. Install Python dependencies 
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 8. Copy the other part of the files
COPY --chown=appuser:appuser .src.app .

# 9. Add permission
RUN chmod +x start.sh 

# 10. Open ports (8000=API, 7860=Gradio)
EXPOSE 8000 7860

# 11. Health check (Ping l'API toutes les 30s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

# 12. Start
CMD ["./start.sh"]