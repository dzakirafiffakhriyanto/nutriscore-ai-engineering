
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py nutriscore_utils.py ./
COPY outputs_nutriscore/ ./outputs_nutriscore/

EXPOSE 10000

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
