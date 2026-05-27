import json
import logging
import uuid

import litserve as ls
import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from .audio import decode_audio
from .config import settings
from .logging_setup import request_id_var, setup_logging
from .model import GenderModel

logger = logging.getLogger(__name__)


class _RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(req_id)

        # Inject into body so worker processes (separate PIDs) can read it
        if request.method == "POST":
            body = await request.body()
            try:
                data = json.loads(body)
                data["_request_id"] = req_id
                request._body = json.dumps(data).encode()
            except (json.JSONDecodeError, ValueError):
                pass

        logger.info("%s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = req_id
        return response


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class GenderLitAPI(ls.LitAPI):
    def setup(self, device: str):
        # LitServe worker processes do not inherit logging config from main process
        setup_logging(settings.log_level, settings.log_dir)
        self._model = GenderModel()
        logger.info("model loaded on device=%s", device)

    def decode_request(self, request: dict, context: dict) -> np.ndarray:
        req_id = request.get("_request_id", "-")
        context["request_id"] = req_id
        request_id_var.set(req_id)
        b64 = request.get("audio")
        if not b64:
            raise ValueError("missing 'audio' field")
        waveform = decode_audio(b64)
        logger.info("audio decoded samples=%d", len(waveform))
        return waveform

    def batch(self, waveforms: list[np.ndarray]) -> list[np.ndarray]:
        return waveforms

    def predict(self, waveforms: list[np.ndarray]) -> np.ndarray:
        return self._model.infer(waveforms)

    def unbatch(self, logits: np.ndarray) -> list[np.ndarray]:
        return list(logits)

    def encode_response(self, logit: np.ndarray, context: dict) -> dict:
        request_id_var.set(context.get("request_id", "-"))
        probs = _softmax(logit.astype(np.float64)).tolist()
        id2label = self._model.id2label
        label_idx = int(np.argmax(logit))
        label = id2label[label_idx]
        scores_per_label = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}

        logger.info("prediction label=%s scores=%s", label, scores_per_label)
        return {
            "label": label,
            "scores": scores_per_label,
        }


def main():
    setup_logging(settings.log_level, settings.log_dir)
    api = GenderLitAPI(
        api_path="/v1/gender/predict",
        max_batch_size=settings.max_batch_size,
        batch_timeout=settings.batch_timeout,
    )
    server = ls.LitServer(
        api,
        workers_per_device=settings.workers,
    )
    server.app.add_middleware(_RequestIDMiddleware)
    server.run(host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
