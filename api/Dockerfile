FROM python:3.12-slim
WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY api/main.py .

ENV MODEL_PATH=models/xgb_robust.pkl

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
