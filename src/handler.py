"""
RunPod Serverless handler — Whisper Large v3 + biodatlab Thai model.

Input:
  {
    "audio_base64": "<base64>",
    "language": "th",
    "model": "whisper-v3" | "whisper-th",
    "format": ".mp3"
  }

Output:
  {
    "language": "th",
    "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": "...", "avg_logprob": -0.5}],
    "words": [{"word": "...", "start": 0.0, "end": 1.0, "confidence": 0.95}]
  }
"""

import base64
import logging
import tempfile
import uuid
from pathlib import Path

import runpod
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Device: %s", DEVICE)

# Lazy-loaded model cache
_models = {}


def get_whisper_v3():
    if "whisper-v3" not in _models:
        from faster_whisper import WhisperModel

        compute = "float16" if DEVICE == "cuda" else "int8"
        logger.info("Loading whisper-v3 (compute=%s)", compute)
        _models["whisper-v3"] = WhisperModel("large-v3", device=DEVICE, compute_type=compute)
        logger.info("whisper-v3 ready")
    return _models["whisper-v3"]


def get_whisper_th():
    if "whisper-th" not in _models:
        from faster_whisper import WhisperModel

        compute = "float16" if DEVICE == "cuda" else "int8"
        logger.info("Loading whisper-th via faster-whisper (compute=%s)", compute)
        _models["whisper-th"] = WhisperModel(
            "Vinxscribe/biodatlab-whisper-th-large-v3-faster",
            device=DEVICE,
            compute_type=compute,
        )
        logger.info("whisper-th ready")
    return _models["whisper-th"]


def transcribe_v3(audio_path: str, language: str):
    model = get_whisper_v3()
    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
    )

    segments = []
    words = []
    for i, seg in enumerate(segments_gen):
        segments.append({
            "id": i,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
        })
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 4) if hasattr(w, "probability") else 1.0,
                })

    return {"language": info.language, "segments": segments, "words": words}


def transcribe_th(audio_path: str, language: str):
    model = get_whisper_th()
    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
    )

    segments = []
    words = []
    for i, seg in enumerate(segments_gen):
        segments.append({
            "id": i,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
        })
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 4) if hasattr(w, "probability") else 1.0,
                })

    return {"language": info.language, "segments": segments, "words": words}


MODELS = {"whisper-v3": transcribe_v3, "whisper-th": transcribe_th}


def handler(event):
    inp = event["input"]

    audio_b64 = inp.get("audio_base64")
    if not audio_b64:
        return {"error": "audio_base64 is required"}

    language = inp.get("language", "th")
    model_key = inp.get("model", "whisper-v3")
    ext = inp.get("format", ".mp3")

    if model_key not in MODELS:
        return {"error": f"Unknown model: {model_key}. Available: {', '.join(MODELS)}"}

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {"error": f"Invalid base64: {e}"}

    tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{ext}"
    try:
        tmp.write_bytes(audio_bytes)
        return MODELS[model_key](str(tmp), language)
    except Exception as e:
        logger.exception("Transcription failed")
        return {"error": f"Transcription failed: {e}"}
    finally:
        tmp.unlink(missing_ok=True)


runpod.serverless.start({"handler": handler})
