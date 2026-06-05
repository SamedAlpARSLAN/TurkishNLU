# Reproducible CPU image. For GPU, start FROM a CUDA-enabled PyTorch base
# (e.g. pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime) and drop the torch pin.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the offline core tests so `docker run` proves the image works.
CMD ["python", "tests/test_core.py"]
