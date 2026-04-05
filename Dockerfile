FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (FIXED syntax)

RUN apt-get update && apt-get install -y 
build-essential 
&& rm -rf /var/lib/apt/lists/*

# Copy requirements

COPY requirements.txt .

# Install Python dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy project

COPY . .

# Expose port

EXPOSE 10000

# Run app

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
