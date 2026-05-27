"""
Export Wav2Vec2 model to ONNX via optimum.

Usage:
    python -m gender_prediction.export [--fp16] [--model-id ID] [--output-dir DIR]

Default: exports FP16 ONNX to settings.onnx_dir.
"""

import argparse
from pathlib import Path

import onnx
from onnxconverter_common.float16 import convert_float_to_float16
from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import Wav2Vec2FeatureExtractor

from .config import settings


def export(model_id: str, output_dir: str, fp16: bool):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_id} → {out} (FP32) ...")
    model = ORTModelForAudioClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(str(out))

    if fp16:
        onnx_path = out / "model.onnx"
        print(f"Converting {onnx_path} to FP16 ...")
        fp32_model = onnx.load(str(onnx_path))
        # keep_io_types=True: inputs/outputs stay float32 so the feature extractor works unchanged
        fp16_model = convert_float_to_float16(fp32_model, keep_io_types=True)
        onnx.save(fp16_model, str(onnx_path))
        print("FP16 conversion done.")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    processor.save_pretrained(str(out))

    print(f"id2label: {model.config.id2label}")
    print(f"Done. Artifacts in {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Export gender detection model to ONNX"
    )
    parser.add_argument("--model-id", default=settings.model_id)
    parser.add_argument("--output-dir", default=settings.onnx_dir)
    parser.add_argument(
        "--fp16", action="store_true", default=(settings.precision == "fp16")
    )
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = parser.parse_args()
    export(args.model_id, args.output_dir, args.fp16)


if __name__ == "__main__":
    main()
