FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install torch CPU version first (updated to fix CVE-2025-32434)
RUN pip install --no-cache-dir torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set a non-user-specific path for Playwright browsers
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install browsers and their dependencies as root
RUN crawl4ai-setup

# Copy application code
COPY . .

# Create a non-root user and set permissions
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app \
    && chown -R app:app /ms-playwright  # Give the app user ownership of browsers

# Switch to the non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
