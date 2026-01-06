# Create the requirements.txt to download dependencies from poetry
FROM python:3.11 as builder

RUN pip install poetry poetry-plugin-export

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

# 7. Install Python dependencies 
COPY --from=builder /tmp/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 8. Copy the other part of the files
COPY --chown=appuser:appuser . .

#RUN echo "Je suis ici :" && pwd && echo "Voici les fichiers :" && ls -laR

# 9. Add permission
RUN chmod +x ./src/app/start.sh 

# 10. 
EXPOSE 7860

# 11. Health check (Ping l'API toutes les 30s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/docs || exit 1

# 12. Start
CMD ["./src/app/start.sh"]