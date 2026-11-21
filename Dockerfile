FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install ReinforceSMEFT and remaining dependencies
RUN pip install --no-cache-dir .

ENTRYPOINT ["python"]
