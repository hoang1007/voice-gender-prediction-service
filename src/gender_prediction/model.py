from pathlib import Path

import numpy as np
from onnxruntime import GraphOptimizationLevel, SessionOptions
from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import Wav2Vec2FeatureExtractor

from .config import settings


class GenderModel:
    def __init__(self):
        onnx_dir = Path(settings.onnx_dir)

        sess_opts = SessionOptions()
        sess_opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        # single-stream GPU inference: minimize CPU thread overhead
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        self.model = ORTModelForAudioClassification.from_pretrained(
            str(onnx_dir),
            provider="CUDAExecutionProvider",
            session_options=sess_opts,
        )
        # IO binding requires torch CUDA tensors; we pass numpy → disable it
        self.model.use_io_binding = False
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(str(onnx_dir))
        self.id2label: dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }

    def infer(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Run inference on a batch. Returns logits shape (N, num_labels)."""
        inputs = self.processor(
            waveforms,
            sampling_rate=settings.target_sample_rate,
            return_tensors="np",
            padding=True,
        )
        outputs = self.model(**inputs)
        return outputs.logits  # numpy array
