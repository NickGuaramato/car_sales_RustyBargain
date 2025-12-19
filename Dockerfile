FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copiar en orden correcto
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/
COPY main.py ./
COPY config/ ./config/
COPY data/raw/ ./data/raw/

# Instalar después de copiar el código
RUN pip install --no-cache-dir -e .

RUN mkdir -p artifacts/logs
CMD ["python", "main.py"]
