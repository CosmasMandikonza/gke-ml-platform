FROM python:3.11-slim as base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
FROM base as dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
FROM base as production
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin
COPY app/ ./app/
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]