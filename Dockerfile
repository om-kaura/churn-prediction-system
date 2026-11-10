FROM python:3.12-slim

WORKDIR /app

# libgomp1 is required by XGBoost's OpenMP-based tree builder on slim Debian images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Only the API's own dependencies are installed here (no matplotlib/seaborn/pytest) —
# see requirements-api.txt vs requirements.txt for the full explanation.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY outputs/ ./outputs/
COPY api/ ./api/

# Run as a non-root user rather than the container default of root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
