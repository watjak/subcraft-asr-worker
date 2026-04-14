FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Models download at first cold start (or mount RunPod Network Volume at /models)
# This keeps image small (~3GB vs ~15GB) and build fast (~3min vs ~20min)

COPY src/ src/

CMD ["python", "src/handler.py"]
