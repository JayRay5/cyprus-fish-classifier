# 1. Ultra-light base image compatible with Python 3.11
FROM python:3.11-slim

# 2. Optimize Python, Poetry (v2.x) and Hugging Face environments
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.0.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    HF_HOME=/tmp/hf_cache

# 3. Install minimal system dependencies required for image processing (Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Install the exact version of Poetry
RUN pip install "poetry==$POETRY_VERSION"

# 6. Optimization Layer: Copy ONLY dependency configurations first
# This ensures Docker caches the heavy library installations (optimum, fastapi).
# If you edit your API code later, Docker will skip this slow step entirely.
COPY pyproject.toml poetry.lock* ./

# 7. Install ONLY the production backend dependencies (ignores dev, training, frontend)
RUN poetry install --only main --no-interaction --no-ansi --no-root

# 8. Production Isolation: Copy ONLY the backend API folder.
# The 'src/' folder (training logic) is deliberately ignored to keep the image pure.
COPY apps/backend/ ./apps/backend/

# 9. Expose the default FastAPI port
EXPOSE 8000

# 10. Official startup command for Azure App Service
# Uvicorn looks for the "app" instance inside apps/backend/api.py
CMD ["uvicorn", "apps.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]