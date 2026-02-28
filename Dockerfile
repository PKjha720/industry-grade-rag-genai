FROM python:3.11.8-slim

WORKDIR /app

# OS security updates + minimal deps
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8050"]