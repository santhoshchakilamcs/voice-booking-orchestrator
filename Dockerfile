FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY frontend ./frontend
# Data directory is created at runtime. In production, replace SQLite
# with the user's own RDS Postgres by setting DATABASE_URL in their
# AWS account. No keys or account IDs are baked into this image.
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
