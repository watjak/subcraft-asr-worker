FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so cold starts only load into VRAM
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('large-v3', device='cpu', compute_type='int8'); \
print('whisper-v3 cached')"

RUN python -c "\
from transformers import pipeline; \
pipeline('automatic-speech-recognition', model='biodatlab/whisper-th-large-v3-combined'); \
print('whisper-th cached')"

COPY src/ src/

CMD ["python", "src/handler.py"]
