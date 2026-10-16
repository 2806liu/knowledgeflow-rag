"""Page-aware text extraction with a separate, optional local OCR adapter."""

from __future__ import annotations

import io
import json
import logging
import math
import threading
from pathlib import Path

from app.config import Settings, settings
from app.rag.contracts import DocumentPage, ParsedDocument
from app.rag.ocr import OCREngine, OCRUnavailable, get_ocr_engine
from app.rag.pdf_layout import mixed_page_regions, native_lines, normalized

logger = logging.getLogger(__name__)
IMAGE_TYPES = {".png": ("PNG", "image/png"), ".jpg": ("JPEG", "image/jpeg"),
               ".jpeg": ("JPEG", "image/jpeg"), ".webp": ("WEBP", "image/webp")}
_render_lock = threading.Lock()  # PDFium is not thread safe.


class DocumentLimitError(ValueError):
    """A document exceeds the configured OCR resource budget."""


def document_suffix(filename: str, data: bytes) -> str:
    """Detect supported binary formats only when the filename has no suffix.

    Keep the supplied filename for citations. Explicit extensions still go
    through signature and decoder checks, so a mislabeled file is not accepted.
    """
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    if data.startswith(b"%PDF-"):
        return ".pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ""


def _has_images(page: object) -> bool:
    """Find raster scans, including image XObjects nested inside forms."""
    from pypdf.generic import ContentStream
    visited: set[int] = set()

    def has_inline(stream):
        if stream is None:
            return False
        content = stream if isinstance(stream, ContentStream) else ContentStream(stream, None)
        return any(operator == b"INLINE IMAGE" for _, operator in content.operations)

    def walk(resources: object, depth: int = 0) -> bool:
        if not resources or depth > 16:
            return False
        resources = resources.get_object()
        if id(resources) in visited:
            return False
        visited.add(id(resources))
        objects = resources.get("/XObject", {})
        objects = objects.get_object() if hasattr(objects, "get_object") else objects
        for ref in objects.values():
            obj = ref.get_object()
            if obj.get("/Subtype") == "/Image":
                return True
            if obj.get("/Subtype") == "/Form":
                if has_inline(obj) or walk(obj.get("/Resources"), depth + 1):
                    return True
        return False

    return walk(page.get("/Resources")) or has_inline(page.get_contents())


def render_pdf_page(data: bytes, index: int, config: Settings):
    """Return a detached RGB image and close all PDFium native handles."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise OCRUnavailable('PDF OCR requires pip install -e ".[ocr]"') from exc
    with _render_lock:
        with pdfium.PdfDocument(data) as document:
            page = document[index]
            try:
                width, height = page.get_size()
                if width <= 0 or height <= 0:
                    raise ValueError("PDF page has invalid dimensions")
                scale = min(config.ocr_dpi / 72, math.sqrt(config.ocr_max_render_pixels / (width * height)))
                while math.ceil(width * scale) * math.ceil(height * scale) > config.ocr_max_render_pixels:
                    scale *= 0.99
                bitmap = page.render(scale=scale)
                try:
                    view = bitmap.to_pil()
                    try:
                        return view.convert("RGB")
                    finally:
                        view.close()
                finally:
                    bitmap.close()
            finally:
                page.close()


def _ocr_page(image, page_number: int, config: Settings, engine: OCREngine | None) -> DocumentPage:
    if not config.ocr_enabled:
        raise OCRUnavailable("OCR is disabled; enable STUDYFLOW_OCR_ENABLED and prepare local models")
    result = (engine or get_ocr_engine(str(config.ocr_model_dir.resolve()))).recognize(image)
    accepted = [line for line in result.lines if line.confidence >= config.ocr_min_confidence]
    warnings = []
    if len(accepted) != len(result.lines):
        warnings.append("low_confidence_lines_excluded")
    if not accepted:
        warnings.append("no_usable_ocr_text")
    metadata = {
        "extraction_method": "ocr", "ocr_engine": result.engine,
        "ocr_engine_version": result.version, "image_width": image.width,
        "image_height": image.height, "ocr_min_confidence": config.ocr_min_confidence,
        "ocr_mean_confidence": round(sum(line.confidence for line in accepted) / len(accepted), 6) if accepted else None,
        "ocr_dropped_lines": len(result.lines) - len(accepted), "warnings": warnings,
        "ocr_lines": [{"text": line.text, "confidence": line.confidence,
                       "box": line.box, "accepted": line.confidence >= config.ocr_min_confidence}
                      for line in result.lines],
    }
    return DocumentPage(page_number=page_number, text="\n".join(line.text for line in accepted), metadata=metadata)


def _parse_image(filename: str, data: bytes, suffix: str, config: Settings, engine: OCREngine | None) -> ParsedDocument:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise OCRUnavailable('Image OCR requires pip install -e ".[ocr]"') from exc
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format != IMAGE_TYPES[suffix][0]:
                raise ValueError("image format does not match its filename")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("animated or multi-frame images are not supported")
            if source.width * source.height > config.ocr_max_image_pixels:
                raise DocumentLimitError("image exceeds STUDYFLOW_OCR_MAX_IMAGE_PIXELS")
            source.load()
            with ImageOps.exif_transpose(source) as oriented:
                with oriented.convert("RGBA") as rgba:
                    with Image.new("RGBA", rgba.size, "white") as canvas:
                        canvas.alpha_composite(rgba)
                        with canvas.convert("RGB") as image:
                            page = _ocr_page(image, 1, config, engine)
    except Image.DecompressionBombError as exc:
        raise DocumentLimitError("image decompression limit exceeded") from exc
    return _assemble(filename, IMAGE_TYPES[suffix][1], suffix, (page,))


def _mixed_pdf_page(data, index, text, config, engine):
    native, crops, dimensions = mixed_page_regions(data, index, config, _render_lock)
    blocks, fallback = native_lines(native, text)
    warnings = ["native_layout_fallback"] if fallback else []
    regions, lines = [], []
    duplicate_count = 0
    try:
        for region_id, (region, image) in enumerate(crops):
            result = _ocr_page(image, index + 1, config, engine)
            warnings.extend(result.metadata["warnings"])
            regions.append({"region_id": region_id, "box": region["box"], "pdf_box": region["pdf_box"],
                            "ocr_engine": result.metadata["ocr_engine"],
                            "ocr_engine_version": result.metadata["ocr_engine_version"]})
            for line in result.metadata["ocr_lines"]:
                entry = dict(line)
                entry["region_id"] = region_id
                entry["box"] = [(x + region["box"][0], y + region["box"][1]) for x, y in line["box"]]
                candidate = normalized(line["text"])
                duplicate = bool(candidate) and (candidate in normalized(text) if len(candidate) >= 4 else
                                                  candidate in {normalized(s) for s in text.splitlines()})
                entry["duplicate_native"] = duplicate
                entry["indexed"] = line["accepted"] and not duplicate and bool(candidate)
                lines.append(entry)
                duplicate_count += int(duplicate and line["accepted"])
            added = [line for line in lines if line["region_id"] == region_id and line["indexed"]]
            if added:
                # Keep an image's related lines together, rather than interleave
                # a table/diagram's columns with nearby native paragraphs.
                blocks.append({"text": "\n".join(line["text"] for line in added), "box": region["box"],
                               "method": "ocr", "region_id": region_id})
    finally:
        for _, image in crops:
            image.close()
    indexed = [line for line in lines if line["indexed"]]
    metadata = {"extraction_method": "mixed" if indexed else "native_text",
                "ocr_attempted": bool(crops), "native_characters": len(normalized(text)),
                "image_width": dimensions[0], "image_height": dimensions[1],
                "coordinate_system": "rendered_page_pixels_top_left",
                "pdf_coordinate_system": "page_points_bottom_left",
                "ocr_regions": regions, "ocr_lines": lines,
                "ocr_mean_confidence": round(sum(line["confidence"] for line in indexed) / len(indexed), 6) if indexed else None,
                "ocr_dropped_lines": sum(not line["accepted"] for line in lines),
                "ocr_duplicate_native_lines": duplicate_count,
                "warnings": sorted(set(warnings)),
                "text_blocks": sorted(blocks, key=lambda b: (b["box"][1], b["box"][0]))}
    if regions:
        metadata.update({key: regions[0][key] for key in ("ocr_engine", "ocr_engine_version")})
    return DocumentPage(page_number=index + 1,
                        text="\n".join(block["text"] for block in metadata["text_blocks"]), metadata=metadata)


def _assemble(filename: str, content_type: str, suffix: str, pages: tuple[DocumentPage, ...]) -> ParsedDocument:
    ocr_pages = sum(page.metadata.get("extraction_method") in {"ocr", "mixed"} for page in pages)
    metadata = {"suffix": suffix, "page_count": len(pages), "parser_version": "3.1-mixed-ocr",
                "ocr_page_count": ocr_pages, "native_page_count": len(pages) - ocr_pages,
                "mixed_page_count": sum(page.metadata.get("extraction_method") == "mixed" for page in pages),
                "ocr_attempted_page_count": sum(page.metadata.get("ocr_attempted", page.metadata.get("extraction_method") == "ocr") for page in pages),
                "empty_page_count": sum(not page.text.strip() for page in pages),
                "warnings": [{"page": page.page_number, "codes": page.metadata["warnings"]}
                             for page in pages if page.metadata.get("warnings")]}
    logger.info(json.dumps({"event": "document_parsed", "filename": filename, **metadata}, ensure_ascii=False))
    return ParsedDocument(filename=filename, text="\n".join(page.text for page in pages),
                          content_type=content_type, metadata=metadata, pages=pages)


def parse_document(filename: str, data: bytes, *, config: Settings | None = None,
                   ocr_engine: OCREngine | None = None, checkpoint=None) -> ParsedDocument:
    """OCR scan pages or raster regions in mixed pages, retaining native text."""
    config = config or settings
    # 摄取任务在页/阶段边界检查取消与期限；旧同步调用可不传回调。
    checkpoint = checkpoint or (lambda done=None, total=None: None)
    checkpoint()
    if not (0 <= config.ocr_min_confidence <= 1 and config.ocr_dpi > 0
            and config.ocr_max_render_pixels >= 4 and config.ocr_max_image_pixels > 0
            and config.ocr_max_pages > 0 and config.ocr_max_regions_per_page > 0
            and config.ocr_pdf_min_text_characters >= 0):
        raise ValueError("invalid OCR configuration")
    filename = filename or "document.txt"
    suffix = document_suffix(filename, data)
    if suffix in IMAGE_TYPES:
        result = _parse_image(filename, data, suffix, config, ocr_engine)
        checkpoint(1, 1)
        return result
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("encrypted PDFs are not supported; upload an unlocked copy")
        native = []
        for index, page in enumerate(reader.pages):
            checkpoint()  # 原生提取阶段只检查取消，页计数留给下方实际完成的解析/OCR阶段。
            native.append(page.extract_text() or "")
        candidates = {index for index in range(len(native)) if _has_images(reader.pages[index])}
        if len(candidates) > config.ocr_max_pages:
            raise DocumentLimitError("PDF exceeds STUDYFLOW_OCR_MAX_PAGES; split it into smaller files")
        if candidates and not config.ocr_enabled:
            raise OCRUnavailable("PDF contains image regions but OCR is disabled")
        pages = []
        for index, text in enumerate(native):
            checkpoint(index, len(native))
            if index in candidates and len(normalized(text)) < config.ocr_pdf_min_text_characters:
                with render_pdf_page(data, index, config) as image:
                    page = _ocr_page(image, index + 1, config, ocr_engine)
                page.metadata["native_characters"] = len("".join(text.split()))
                pages.append(page)
            elif index in candidates:
                pages.append(_mixed_pdf_page(data, index, text, config, ocr_engine))
            else:
                pages.append(DocumentPage(page_number=index + 1, text=text,
                                          metadata={"extraction_method": "native_text"}))
        checkpoint(len(pages), len(pages))
        return _assemble(filename, "application/pdf", suffix, tuple(pages))
    if suffix not in {".txt", ".md", ""}:
        raise ValueError("unsupported document type")
    return ParsedDocument(filename=filename, text=data.decode("utf-8", errors="ignore"),
                          content_type="text/markdown" if suffix == ".md" else "text/plain",
                          metadata={"suffix": suffix or ".txt", "extraction_method": "native_text"})


def extract_text(filename: str, data: bytes) -> str:
    """Compatibility helper for callers that only need text."""
    return parse_document(filename, data).text
