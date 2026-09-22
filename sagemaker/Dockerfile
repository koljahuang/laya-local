FROM --platform=linux/amd64 python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    LAYA_MODEL_PATH=/opt/ml/model \
    LAYA_DEVICE=cpu \
    SAGEMAKER_BIND_HOST=0.0.0.0

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.0 \
 && pip install --no-cache-dir \
      laya==0.3.5 \
      fastapi==0.141.1 \
      uvicorn==0.53.0

COPY sagemaker/model/ /opt/ml/model/
COPY sagemaker/serve.py /opt/program/serve.py

EXPOSE 8080

ENTRYPOINT ["python", "/opt/program/serve.py"]
