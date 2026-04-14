# subcraft-asr-worker

RunPod Serverless worker for Thai speech-to-text.

## Models

| Key | Model | Description |
|-----|-------|-------------|
| `whisper-v3` | `openai/whisper-large-v3` via faster-whisper | Stock Whisper, good general accuracy |
| `whisper-th` | `biodatlab/whisper-th-large-v3-combined` | Thai-finetuned, WER 6.59% on CV13 |

## Deploy

```bash
docker build --platform linux/amd64 -t watjakc/subcraft-asr:latest .
docker push watjakc/subcraft-asr:latest
```

Then create a RunPod Serverless endpoint with this image.

## API

POST to RunPod endpoint:

```json
{
  "input": {
    "audio_base64": "<base64-encoded audio>",
    "language": "th",
    "model": "whisper-th",
    "format": ".mp3"
  }
}
```

Response:

```json
{
  "language": "th",
  "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": "...", "avg_logprob": -0.5}],
  "words": [{"word": "...", "start": 0.0, "end": 1.0, "confidence": 0.95}]
}
```
