"""Local OCR contract and a lazy RapidOCR adapter (no remote inference)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.metadata
import hashlib
import json
from pathlib import Path
import threading
from typing import Protocol


class OCRUnavailable(RuntimeError):
    """The optional OCR runtime or prepared models are unavailable."""


class OCRExecutionError(RuntimeError):
    """OCR inference failed; never index a partially parsed document."""


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    box: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class OCRResult:
    lines: tuple[OCRLine, ...]
    engine: str = "rapidocr"
    version: str = ""


class OCREngine(Protocol):
    def recognize(self, image) -> OCRResult: ...


def _runtime():
    try:
        from rapidocr import RapidOCR
        import onnxruntime  # fail before attempting model initialization
    except ImportError as exc:
        raise OCRUnavailable('OCR runtime missing; run pip install -e ".[ocr]"') from exc
    return RapidOCR


class RapidOCREngine:
    def __init__(self, model_dir: Path):
        manifest = model_dir / "models.json"
        if not manifest.is_file():
            raise OCRUnavailable("OCR models not prepared; run python -m app.rag.ocr --prepare")
        try:
            models = json.loads(manifest.read_text(encoding="utf-8"))
            params = {"Global.text_score": 0.0, "Global.log_level": "warning",
                      "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                      "EngineConfig.onnxruntime.inter_op_num_threads": 1}
            for stage in ("Det", "Cls", "Rec"):
                path = (model_dir / models[stage]).resolve()
                if not path.is_relative_to(model_dir.resolve()) or not path.is_file():
                    raise ValueError("missing or invalid local model path")
                if hashlib.sha256(path.read_bytes()).hexdigest() != models["sha256"][stage]:
                    raise ValueError("model fingerprint does not match the prepared manifest")
                params[f"{stage}.model_path"] = str(path)
            self._engine = _runtime()(params=params)
        except OCRUnavailable:
            raise
        except Exception as exc:
            raise OCRUnavailable("OCR model initialization failed; prepare valid local models") from exc
        self._lock = threading.Lock()
        self.version = importlib.metadata.version("rapidocr")

    def recognize(self, image) -> OCRResult:
        try:
            with self._lock:
                # PIL inputs let RapidOCR apply its documented RGB -> BGR conversion.
                output = self._engine(image)
            if output.txts is None:
                return OCRResult((), version=self.version)
            lines = tuple(OCRLine(str(text).strip(), float(score),
                                  tuple(tuple(float(value) for value in point) for point in box))
                          for text, score, box in zip(output.txts, output.scores, output.boxes)
                          if str(text).strip())
            return OCRResult(lines, version=self.version)
        except Exception as exc:
            raise OCRExecutionError("local OCR inference failed") from exc


@lru_cache(maxsize=4)
def get_ocr_engine(model_dir: str) -> RapidOCREngine:
    return RapidOCREngine(Path(model_dir))


def prepare_models(model_dir: Path) -> None:
    """Provision models explicitly before serving uploads; network is allowed here."""
    import shutil

    runtime = _runtime()
    import rapidocr
    model_dir.mkdir(parents=True, exist_ok=True)
    names = {"Det": "PP-OCRv6_det_small.onnx", "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
             "Rec": "PP-OCRv6_rec_small.onnx"}
    bundled = Path(rapidocr.__file__).parent / "models"
    for name in names.values():
        source = bundled / name
        if source.is_file() and source.resolve() != (model_dir / name).resolve():
            shutil.copyfile(source, model_dir / name)
    # Current wheels bundle these models; only provisioning may download missing ones.
    if not all((model_dir / name).is_file() for name in names.values()):
        runtime(params={"Global.model_root_dir": str(model_dir.resolve()),
                        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                        "EngineConfig.onnxruntime.inter_op_num_threads": 1})
    models = {**names, "sha256": {}, "rapidocr_version": importlib.metadata.version("rapidocr")}
    for stage, name in names.items():
        source = model_dir / name
        target = model_dir / source.name
        models["sha256"][stage] = hashlib.sha256(target.read_bytes()).hexdigest()
    (model_dir / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    # Verify the exact same offline adapter that the service will use.
    RapidOCREngine(model_dir)


if __name__ == "__main__":
    import argparse
    from app.config import settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--model-dir", type=Path, default=settings.ocr_model_dir)
    args = parser.parse_args()
    prepare_models(args.model_dir)
    print("Local OCR models prepared: " + str(args.model_dir))
