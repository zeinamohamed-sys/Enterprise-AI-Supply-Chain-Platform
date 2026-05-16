# Use standard minimal stable environment core
FROM python:3.10-slim

# Enforce clean terminal buffer executions
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Establish application layer target boundaries
WORKDIR /app

# Ingest underlying operating systems packages for visualization engine bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache requirements layers separately to boost build efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ingest remaining repository operational assets
COPY . .

# Force baseline transactional table synthesis during deployment execution
RUN python data_manager.py

# Expose individual port listeners for FastAPI [8000] and Streamlit UI [8501]
EXPOSE 8000
EXPOSE 8501

# Run multiple servers asynchronously for testing deployment setup
CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port 8000 & streamlit run app_ui.py --server.port 8501 --server.address 0.0.0.0"]
