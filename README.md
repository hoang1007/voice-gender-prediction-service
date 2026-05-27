# Voice Gender Prediction Service

REST service that classifies speaker gender (female / male) from a voice audio clip. Served via ONNX Runtime with dynamic batching.

---

## Use Cases

- **Voice assistant personalization** — adapt responses based on detected speaker gender.
- **Call-center analytics** — tag recordings by speaker gender for routing or reporting.
- **Content moderation** — demographic metadata on user-generated voice content.

---

## Setup

### Requirements

- Python 3.13+
- NVIDIA GPU (Recommended)
- [`uv`](https://github.com/astral-sh/uv)

### Install

```bash
git clone <repo-url>
cd voice-gender-prediction
uv sync
```

### Export model to ONNX (run once)

Downloads `prithivMLmods/Common-Voice-Gender-Detection` from Hugging Face, exports to ONNX, and converts weights to FP16.

```bash
uv run gender-prediction-export
```

Artifacts are saved to `./onnx_model/` by default.

### Run the service

```bash
uv run gender-prediction
```

Service listens on `http://0.0.0.0:8000` by default.

### Configuration

All settings are read from environment variables (prefix `GENDER_`) or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GENDER_MODEL_ID` | `prithivMLmods/Common-Voice-Gender-Detection` | HuggingFace model ID used during export |
| `GENDER_ONNX_DIR` | `onnx_model` | Path to exported ONNX artifacts |
| `GENDER_PRECISION` | `fp16` | Export precision: `fp32` or `fp16` |
| `GENDER_MAX_BATCH_SIZE` | `32` | Maximum dynamic batch size |
| `GENDER_BATCH_TIMEOUT` | `0.005` | Seconds to wait before dispatching an incomplete batch |
| `GENDER_WORKERS` | `1` | Parallel ORT sessions per GPU (2 recommended) |
| `GENDER_HOST` | `0.0.0.0` | Bind host |
| `GENDER_PORT` | `8000` | Bind port |
| `GENDER_LOG_LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `GENDER_LOG_DIR` | _(unset)_ | Directory for log files (`gender-<pid>.log`, rotated 100 MB × 5). Unset = stdout only. |

---

## API

### `POST /v1/gender/predict`

Classify gender from a base64-encoded audio clip.

**Request headers**

| Header | Description |
|---|---|
| `X-Request-ID` | Optional. Trace ID echoed back in the response header and logged. Auto-generated (UUID4) if omitted. |

**Request body**

```json
{
  "audio": "<base64-encoded audio bytes>"
}
```

Accepted formats: any format supported by `libsndfile` (WAV, FLAC, OGG, …). Audio is resampled to 16 kHz mono automatically.

**Response headers**

| Header | Description |
|---|---|
| `X-Request-ID` | Trace ID from the request (or the auto-generated one). |

**Response body**

```json
{
  "label": "male",
  "scores": {
    "female": 0.0312,
    "male": 0.9688
  }
}
```

| Field | Type | Description |
|---|---|---|
| `label` | `string` | Top predicted class (`"female"` or `"male"`) |
| `scores` | `object` | Softmax probability for each class |

**Example**

```bash
curl -X POST http://localhost:8000/v1/gender/predict \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-trace-id-123" \
  -d "{\"audio\": \"$(base64 -w0 audio.wav)\"}"
```

**Error responses**

| Status | Cause |
|---|---|
| `400` | Missing or malformed `audio` field |
| `500` | Internal inference error |

### `GET /health`

Returns `200 OK` when the service is ready.

---

## Benchmark

```bash
python scripts/benchmark.py --audio audio.wav --concurrency 16 --n 200
```

Reference numbers on a single GPU with `GENDER_WORKERS=2`:

```
p50=38.9ms  p95=86.4ms  p99=88.1ms  throughput=372.8 req/s
```

---

## Docker

### Requirements

- Docker 24+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Build and run

```bash
# 1. Export ONNX model locally first (requires GPU + Python env)
uv run gender-prediction-export

# 2. Build image
docker build -t voice-gender-prediction .

# 3. Run (mount pre-exported model as read-only volume)
docker run --gpus all -p 8000:8000 \
  -v ./onnx_model:/app/onnx_model:ro \
  voice-gender-prediction
```

### docker compose

```bash
cp .env.example .env   # adjust settings if needed
docker compose up -d
```

---

## Model

This service uses **[prithivMLmods/Common-Voice-Gender-Detection](https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection)**.

The model is released under the **Apache License 2.0** by [prithivMLmods](https://huggingface.co/prithivMLmods).

> This project does not redistribute model weights. Weights are downloaded at export time directly from Hugging Face Hub.
