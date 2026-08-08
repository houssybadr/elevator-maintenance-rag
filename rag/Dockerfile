FROM python:3.11-slim

# Set working directory
WORKDIR /app_dir

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY ./app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY ./app ./app

# Expose the port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
