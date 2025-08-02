FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

COPY *.joblib ./

ENV PYTHONPATH=/app/src

# Run prediction script to verify model
CMD ["python", "src/predict.py"]