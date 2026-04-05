FROM python:3.10-slim

# Prevent Python from buffering logs

ENV PYTHONUNBUFFERED=1

# Set working directory

WORKDIR /app

# Install system dependencies (important for torch + stability)

RUN apt-get update && apt-get install -y 
build-essential 
&& rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)

COPY requirements.txt .

# Install Python dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project

COPY . .

# Expose port

EXPOSE 10000

# Run FastAPI app (dynamic port support)

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
