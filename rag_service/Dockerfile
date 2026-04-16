FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy agent_integration code
COPY agent_integration /app/agent_integration

# Copy rag_service code
COPY rag_service /app/rag_service

# Install Python dependencies
COPY agent_integration/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory to rag_service
WORKDIR /app/rag_service

# Expose port
EXPOSE 8001

# Run the service
CMD ["python", "main.py"]
