FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# OpenEnv standard ports
EXPOSE 8000 7860

# Environment configuration (OpenEnv standard)
ENV PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    MAX_CONCURRENT_ENVS=100 \
    ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Start server
CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
