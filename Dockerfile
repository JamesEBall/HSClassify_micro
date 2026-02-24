# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTENCE_MODEL_NAME=Mead0w1ark/multilingual-e5-small-hs-codes

# System deps for OCR endpoints:
# - tesseract for image OCR
# - poppler-utils for pdf2image PDF conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only runtime files (keeps build context and cache churn smaller)
COPY app.py field_extractor.py hs_dataset.py ./
COPY templates ./templates
COPY static ./static
COPY data ./data
COPY models ./models
RUN mkdir -p uploads

# Expose the port FastAPI will run on
EXPOSE 7860

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
