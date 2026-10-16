"""OCR routing, provenance, failures and the complete document lifecycle.

Fast tests inject recognition, but use real image codecs, PDF extraction and
rasterization. Actual local model inference is covered by evaluation.ocr_smoke.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import shutil
import unittest
from dataclasses import replace
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from app.config import settings
from app.rag.ocr import OCRExecutionError, OCRLine, OCRResult, OCRUnavailable, RapidOCREngine
from app.rag.parsers import DocumentLimitError, parse_document, render_pdf_page
from app.services import KnowledgeFlow
from app.store import Store

HAS_CODECS = all(importlib.util.find_spec(name) for name in ("PIL", "pypdfium2"))
TEXT = "电子版报告和程序按照学号姓名报告格式命名。扫描文档与图片均应保留各自的文件名、实际页码及原文片段，便于核查引用。"
BOX = ((0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0))


class FakeOCR:
    def __init__(self, lines=None):
        self.lines = (OCRLine(TEXT, 0.98, BOX),) if lines is None else lines
        self.images = []

    def recognize(self, image):
        self.images.append((image.size, image.getpixel((0, 0))))
        return OCRResult(tuple(self.lines), engine="fixture", version="1")


def image_bytes(format="PNG", size=(160, 90), *, transparent=False, orientation=None):
    from PIL import Image
    image = Image.new("RGBA" if transparent else "RGB", size, (0, 0, 0, 0) if transparent else "white")
    stream = io.BytesIO()
    options = {}
    if orientation is not None:
        exif = image.getexif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(stream, format=format, **options)
    image.close()
    return stream.getvalue()


def mixed_pdf(scans=1):
    """Native page, blank page, then raster pages with a short native footer."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=400)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 350 Td (Native page preserves existing document text and references.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_blank_page(width=300, height=400)
    if scans:
        from PIL import Image
        image = Image.new("RGB", (160, 90), "white")
        scan_stream = io.BytesIO()
        image.save(scan_stream, format="PDF")
        image.close()
        for _ in range(scans):
            scanned = writer.add_page(PdfReader(io.BytesIO(scan_stream.getvalue())).pages[0])
            scanned["/Resources"][NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font})
            footer = DecodedStreamObject()
            footer.set_data(b"BT /F1 10 Tf 10 10 Td (3) Tj ET")
            scanned[NameObject("/Contents")] = ArrayObject([scanned.raw_get("/Contents"), writer._add_object(footer)])
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def embedded_pdf(*, nested=False, rotation=0, inline=False):
    """More than 40 native characters plus a raster region between paragraphs."""
    from PIL import Image
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=400)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    image = DecodedStreamObject()
    with Image.new("RGB", (160, 90), "white") as raster:
        image.set_data(raster.tobytes())
    image.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"),
                  NameObject("/Width"): NumberObject(160), NameObject("/Height"): NumberObject(90),
                  NameObject("/ColorSpace"): NameObject("/DeviceRGB"), NameObject("/BitsPerComponent"): NumberObject(8)})
    resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
                                  NameObject("/XObject"): DictionaryObject({NameObject("/Im"): writer._add_object(image)})})
    placement = b"q 200 0 0 90 50 190 cm /Im Do Q"
    if inline:
        resources[NameObject("/XObject")] = DictionaryObject()
        placement = b"q 200 0 0 90 50 190 cm BI /W 2 /H 2 /CS /RGB /BPC 8 /F /AHx ID " + b"FF" * 12 + b"> EI Q"
    if nested:
        form = DecodedStreamObject()
        form.set_data(placement)
        form.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Form"),
                     NameObject("/BBox"): ArrayObject([NumberObject(n) for n in (0, 0, 300, 400)]),
                     NameObject("/Resources"): resources})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): resources["/Font"],
            NameObject("/XObject"): DictionaryObject({NameObject("/Fm"): writer._add_object(form)})})
        placement = b"q 1 0 0 1 0 20 cm /Fm Do Q"
    else:
        page[NameObject("/Resources")] = resources
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 10 Tf 20 350 Td (Native introduction before the embedded image.) Tj ET\n" + placement +
                     b"\nBT /F1 10 Tf 20 160 Td (Native conclusion after the embedded image.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    if rotation:
        page.rotate(rotation)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@unittest.skipUnless(HAS_CODECS, 'install the optional .[ocr] extra for codec integration tests')
class OCRIngestionTests(unittest.TestCase):
    def setUp(self):
        from app import main
        self.main = main
        root = Path.cwd() / "tests" / ".tmp"
        root.mkdir(parents=True, exist_ok=True)
        self.temp_path = root / str(uuid.uuid4())
        self.store = Store(self.temp_path)
        self.old = main.store, main.flow, main.settings
        main.settings = replace(settings, vector_enabled=False, llm_mode="mock", retrieval_strategy="bm25_v1", ocr_enabled=True)
        main.store = self.store
        main.flow = KnowledgeFlow(self.store, main.settings)
        self.client = TestClient(main.app)
        self.engine = FakeOCR()
        self.patcher = patch("app.rag.parsers.get_ocr_engine", return_value=self.engine)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.client.close()
        self.main.store, self.main.flow, self.main.settings = self.old
        self.store.conn.close()
        if not self.temp_path.resolve().is_relative_to((Path.cwd() / "tests" / ".tmp").resolve()):
            raise RuntimeError("test cleanup path escaped the workspace")
        shutil.rmtree(self.temp_path)

    def upload(self, data, filename="材料.png", mime="image/png"):
        return self.client.post("/documents/upload", files={"file": (filename, data, mime)})

    def test_mixed_pdf_keeps_native_blank_and_scanned_page_numbers(self):
        parsed = parse_document("mixed.pdf", mixed_pdf(2), config=self.main.settings, ocr_engine=self.engine)
        self.assertEqual([p.page_number for p in parsed.pages], [1, 2, 3, 4])
        self.assertIn("Native page", parsed.pages[0].text)
        self.assertEqual(parsed.pages[1].text, "")
        self.assertEqual(parsed.pages[2].text, TEXT)
        self.assertEqual(parsed.metadata["ocr_page_count"], 2)
        self.assertEqual(len(self.engine.images), 2)

    def test_native_pdf_needs_no_ocr_even_with_ocr_disabled(self):
        parsed = parse_document("native.pdf", mixed_pdf(0),
                                config=replace(self.main.settings, ocr_enabled=False), ocr_engine=self.engine)
        self.assertIn("Native page", parsed.text)
        self.assertEqual(self.engine.images, [])

    def test_embedded_images_are_ocrd_between_native_paragraphs(self):
        parsed = parse_document("mixed.pdf", embedded_pdf(), config=self.main.settings, ocr_engine=self.engine)
        page = parsed.pages[0]
        self.assertEqual(page.metadata["extraction_method"], "mixed")
        self.assertLess(page.text.index("Native introduction"), page.text.index(TEXT))
        self.assertLess(page.text.index(TEXT), page.text.index("Native conclusion"))
        self.assertEqual(parsed.metadata["mixed_page_count"], 1)
        self.assertEqual(self.engine.images[0][0], (500, 225))
        self.assertEqual(page.metadata["ocr_regions"][0]["box"], (125, 300, 625, 525))
        self.assertEqual(page.metadata["ocr_lines"][0]["box"][0], (125, 300))

    def test_nested_and_rotated_image_coordinates(self):
        parsed = parse_document("nested.pdf", embedded_pdf(nested=True), config=self.main.settings, ocr_engine=self.engine)
        self.assertEqual(parsed.pages[0].metadata["ocr_regions"][0]["box"], (125, 250, 625, 475))
        parsed = parse_document("rotated.pdf", embedded_pdf(rotation=90), config=self.main.settings, ocr_engine=self.engine)
        self.assertEqual(self.engine.images[-1][0], (225, 500))
        box = parsed.pages[0].metadata["ocr_regions"][0]["box"]
        self.assertGreaterEqual(min(box), 0)
        self.assertLessEqual(box[2], parsed.pages[0].metadata["image_width"])

    def test_inline_images_in_page_and_nested_forms_are_detected(self):
        for nested in (False, True):
            parsed = parse_document("inline.pdf", embedded_pdf(inline=True, nested=nested),
                                    config=self.main.settings, ocr_engine=self.engine)
            self.assertIn(TEXT, parsed.text)
            self.assertEqual(parsed.metadata["mixed_page_count"], 1)
        self.assertEqual(len(self.engine.images), 2)

    def test_mixed_native_duplicates_and_low_confidence_are_not_indexed(self):
        native = "Native introduction before the embedded image."
        engine = FakeOCR((OCRLine(native, .99, BOX), OCRLine(TEXT, .97, BOX), OCRLine("bad", .1, BOX)))
        parsed = parse_document("mixed.pdf", embedded_pdf(), config=self.main.settings, ocr_engine=engine)
        self.assertEqual(parsed.text.count(native), 1)
        self.assertIn(TEXT, parsed.text)
        self.assertNotIn("bad", parsed.text)
        self.assertEqual(parsed.pages[0].metadata["ocr_duplicate_native_lines"], 1)
        self.assertEqual(parsed.pages[0].metadata["ocr_mean_confidence"], .97)
        with self.assertRaises(OCRUnavailable):
            parse_document("mixed.pdf", embedded_pdf(), config=replace(self.main.settings, ocr_enabled=False))

    def test_native_glyph_positions_preserve_logical_order(self):
        from app.rag.pdf_layout import native_lines
        # Different font/glyph heights must not reorder a source line.
        blocks = [{"text": "A", "box": (10, 12, 20, 30)}, {"text": ",", "box": (21, 25, 23, 30)},
                  {"text": "B", "box": (25, 9, 35, 30)}]
        lines, fallback = native_lines(blocks, "A,B")
        self.assertFalse(fallback)
        self.assertEqual(lines[0]["text"], "A,B")
        lines, fallback = native_lines(blocks, "A,B missing glyph")
        self.assertTrue(fallback)
        self.assertEqual(lines[0]["text"], "A,B missing glyph")

    def test_reparse_preserves_document_identity_and_updates_vectors(self):
        from qdrant_client import QdrantClient
        from app.rag.vector_index import QdrantVectorIndex
        class FixtureEmbedding:
            model = "reparse-fixture"
            def embed(self, texts):
                return [[1., float("程序" in text)] for text in texts]
        self.main.flow.vector_index = QdrantVectorIndex(self.store, FixtureEmbedding(), "reparse", client=QdrantClient(":memory:"))
        data = embedded_pdf()
        # Seed an older parser result to reproduce the actual migration.
        from app.rag.contracts import ParsedDocument, DocumentPage
        old_text = "Original native text without embedded image content and with enough characters."
        old = ParsedDocument("mixed.pdf", old_text, "application/pdf", {}, (DocumentPage(1, old_text),))
        with patch("app.main.parse_document", return_value=old):
            doc_id = self.upload(data, "mixed.pdf", "application/pdf").json()["document_id"]
        other = self.upload(b"Another document must retain its independent source and chunks.", "other.txt", "text/plain").json()["document_id"]
        before = dict(self.store.document(doc_id))
        other_chunks = [dict(row) for row in self.store.chunks((other,))]
        old_ids = tuple(row["id"] for row in self.store.chunks((doc_id,)))
        response = self.client.post(f"/documents/{doc_id}/reparse")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["document_id"], doc_id)
        self.assertEqual(response.json()["created_at"], before["created_at"])
        self.assertEqual(response.json()["content_hash"], before["content_hash"])
        self.assertEqual(response.json()["vector_status"], "indexed")
        self.assertEqual(self.client.get(f"/documents/{doc_id}/source").content, data)
        self.assertEqual([dict(row) for row in self.store.chunks((other,))], other_chunks)
        self.assertEqual(self.store.chunks_by_ids(old_ids), [])
        hits = self.main.flow.vector_index.search("程序报告", document_ids=(doc_id,))
        self.assertTrue(hits)
        self.assertTrue(all(hit.metadata["extraction_method"] == "mixed" for hit in hits))
        self.assertTrue(any(TEXT in hit.text for hit in hits))
        self.assertTrue(all(hit.metadata["ocr_regions"] for hit in hits))

    def test_reparse_failures_retain_previous_document_pages_and_chunks(self):
        doc_id = self.upload(image_bytes()).json()["document_id"]
        before = dict(self.store.document(doc_id)), [dict(r) for r in self.store.document_pages(doc_id)], [dict(r) for r in self.store.chunks((doc_id,))]
        for lines, error, expected in (((), None, 400), (None, OCRExecutionError("fixture failure"), 502),
                                       (None, OCRUnavailable("fixture unavailable"), 503)):
            self.engine.lines = lines
            with patch.object(self.engine, "recognize", side_effect=error) if error else patch.object(self.engine, "lines", lines):
                result = self.client.post(f"/documents/{doc_id}/reparse")
            self.assertEqual(result.status_code, expected, result.text)
            after = dict(self.store.document(doc_id)), [dict(r) for r in self.store.document_pages(doc_id)], [dict(r) for r in self.store.chunks((doc_id,))]
            self.assertEqual(after, before)

    def test_reparse_missing_deleted_source_and_fingerprint_guards(self):
        self.assertEqual(self.client.post("/documents/unknown/reparse").status_code, 404)
        doc_id = self.upload(image_bytes()).json()["document_id"]
        self.store.save_source(doc_id, image_bytes(size=(161, 90)))
        self.assertEqual(self.client.post(f"/documents/{doc_id}/reparse").status_code, 409)
        self.store.delete_source(doc_id)
        self.assertEqual(self.client.post(f"/documents/{doc_id}/reparse").status_code, 404)
        self.client.delete(f"/documents/{doc_id}")
        self.assertEqual(self.client.post(f"/documents/{doc_id}/reparse").status_code, 409)

    def test_reparse_vector_failure_is_visible_and_reindex_can_retry(self):
        from unittest.mock import Mock
        doc_id = self.upload(image_bytes()).json()["document_id"]
        self.main.flow.vector_index = Mock()
        self.main.flow.vector_index.replace_document.side_effect = RuntimeError("fixture offline")
        self.engine.lines = (OCRLine("重新识别后的报告程序命名和原文引用规则。" * 4, .98, BOX),)
        response = self.client.post(f"/documents/{doc_id}/reparse")
        self.assertEqual(response.status_code, 502)
        self.assertIn("reindex", response.json()["detail"])
        self.assertEqual(self.store.document(doc_id)["vector_status"], "failed")
        self.assertIn("重新识别", self.store.document(doc_id)["content"])
        self.main.flow.vector_index.replace_document.side_effect = None
        self.main.flow.vector_index.replace_document.return_value = 1
        self.assertEqual(self.client.post(f"/documents/{doc_id}/reindex").status_code, 200)
        self.assertEqual(self.store.document(doc_id)["vector_status"], "indexed")

    def test_reparse_sql_transaction_rolls_back_on_chunk_write_failure(self):
        import sqlite3
        from app.rag.contracts import ParsedDocument, DocumentPage
        doc_id = self.upload(image_bytes()).json()["document_id"]
        before = dict(self.store.document(doc_id)), [dict(r) for r in self.store.document_pages(doc_id)], [dict(r) for r in self.store.chunks((doc_id,))]
        parsed = ParsedDocument("replacement.txt", "replacement text", "text/plain", {}, (DocumentPage(1, "replacement text"),))
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.replace_parsed_document(doc_id, parsed,
                [("same-id", doc_id, "one", 0, {}), ("same-id", doc_id, "two", 1, {})],
                fingerprint="fixture", size_bytes=10, chunk_strategy="fixture", chunk_version="1")
        after = dict(self.store.document(doc_id)), [dict(r) for r in self.store.document_pages(doc_id)], [dict(r) for r in self.store.chunks((doc_id,))]
        self.assertEqual(after, before)

    def test_mixed_region_limit_and_overlapping_tiles(self):
        writer = PdfWriter()
        writer.add_page(PdfReader(io.BytesIO(embedded_pdf())).pages[0])
        page = writer.pages[0]
        stream = DecodedStreamObject()
        stream.set_data(page.get_contents().get_data() + b"\nq 200 0 0 90 50 190 cm /Im Do Q")
        page[NameObject("/Contents")] = writer._add_object(stream)
        output = io.BytesIO()
        writer.write(output)
        data = output.getvalue()
        with self.assertRaises(DocumentLimitError):
            parse_document("tiles.pdf", data, config=replace(self.main.settings, ocr_max_regions_per_page=1), ocr_engine=self.engine)
        self.assertFalse(self.engine.images)
        parsed = parse_document("tiles.pdf", data, config=self.main.settings, ocr_engine=self.engine)
        self.assertEqual(len(self.engine.images), 1)
        self.assertEqual(len(parsed.pages[0].metadata["ocr_regions"]), 1)

    def test_image_metadata_low_confidence_and_transparency(self):
        engine = FakeOCR((OCRLine(TEXT, 0.98, BOX), OCRLine("不可作为引用的低置信度字", 0.2, BOX)))
        parsed = parse_document("透明.png", image_bytes(transparent=True), config=self.main.settings, ocr_engine=engine)
        self.assertEqual(engine.images[0][1], (255, 255, 255))
        self.assertEqual(parsed.pages[0].text, TEXT)
        metadata = parsed.pages[0].metadata
        self.assertEqual(metadata["ocr_dropped_lines"], 1)
        self.assertFalse(metadata["ocr_lines"][1]["accepted"])
        self.assertIn("low_confidence_lines_excluded", metadata["warnings"])

    def test_jpeg_exif_orientation_and_webp(self):
        parse_document("photo.jpg", image_bytes("JPEG", (80, 40), orientation=6),
                       config=self.main.settings, ocr_engine=self.engine)
        self.assertEqual(self.engine.images[0][0], (40, 80))
        parsed = parse_document("photo.webp", image_bytes("WEBP"), config=self.main.settings, ocr_engine=self.engine)
        self.assertEqual(parsed.content_type, "image/webp")

    def test_upload_query_duplicate_reindex_source_and_delete(self):
        from qdrant_client import QdrantClient
        from app.rag.vector_index import QdrantVectorIndex
        class FixtureEmbedding:
            model = "ocr-lifecycle-fixture"

            def embed(self, texts):
                return [[1.0, float("报告" in text)] for text in texts]

        self.main.flow.vector_index = QdrantVectorIndex(self.store, FixtureEmbedding(), "ocr_chunks", client=QdrantClient(":memory:"))
        data = image_bytes()
        uploaded = self.upload(data)
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        doc_id = uploaded.json()["document_id"]
        self.assertTrue(uploaded.json()["indexed"])
        self.assertEqual(uploaded.json()["metadata"]["ocr_page_count"], 1)
        self.assertEqual(self.client.get(f"/documents/{doc_id}/source").content, data)
        self.assertTrue(self.client.get(f"/documents/{doc_id}").json()["source_available"])
        pages = self.client.get(f"/documents/{doc_id}/pages").json()
        self.assertEqual(pages[0]["metadata"]["ocr_lines"][0]["text"], TEXT)
        duplicate = self.upload(data)
        self.assertTrue(duplicate.json()["duplicate"])
        self.assertEqual(len(self.engine.images), 1)
        for _ in range(2):
            chat = self.client.post("/chat", json={"session_id": "ocr", "message": "电子版报告程序命名格式", "document_ids": [doc_id]})
            self.assertEqual(chat.status_code, 200, chat.text)
            cite = chat.json()["citations"][0]
            self.assertEqual(cite["filename"], "材料.png")
            self.assertEqual(cite["page"], 1)
            self.assertEqual(cite["extraction_method"], "ocr")
            self.assertEqual(cite["ocr_confidence"], 0.98)
            self.assertTrue(cite["chunk_id"])
            self.assertIn("学号姓名", cite["snippet"])
            self.assertEqual(self.client.post(f"/documents/{doc_id}/reindex").status_code, 200)
        self.assertTrue(self.main.flow.vector_index.search("推免", document_ids=(doc_id,)))
        self.assertEqual(self.client.delete(f"/documents/{doc_id}").status_code, 200)
        self.assertFalse(self.store.source_path(doc_id).exists())
        self.assertEqual(self.main.flow.vector_index.search("推免", document_ids=(doc_id,)), [])
        self.assertEqual(self.client.get(f"/documents/{doc_id}/pages").status_code, 404)
        self.assertEqual(self.client.get(f"/documents/{doc_id}/source").status_code, 404)

    def test_identical_ocr_text_retains_pdf_and_image_sources(self):
        image = self.upload(image_bytes()).json()["document_id"]
        pdf_response = self.upload(mixed_pdf(), "扫描.pdf", "application/pdf")
        self.assertEqual(pdf_response.status_code, 200, pdf_response.text)
        pdf = pdf_response.json()["document_id"]
        hits = self.main.flow.retriever.search("电子版报告程序命名格式", document_ids=(image, pdf))
        self.assertEqual({hit.document_id for hit in hits}, {image, pdf})
        self.assertEqual({hit.page for hit in hits}, {1, 3})

    def test_ocr_upload_does_not_merge_into_legacy_equal_text_source(self):
        old_id = str(uuid.uuid4())
        self.store.add_document(old_id, "旧资料.txt", TEXT)
        response = self.upload(image_bytes())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotEqual(response.json()["document_id"], old_id)
        self.assertFalse(response.json()["duplicate"])
        self.assertIsNone(self.store.document(old_id)["content_hash"])

    def test_extensionless_images_keep_filename_and_ocr_provenance(self):
        for format in ("PNG", "JPEG", "WEBP"):
            with self.subTest(format=format):
                filename = "无扩展名" + format
                response = self.upload(image_bytes(format), filename=filename, mime="application/octet-stream")
                self.assertEqual(response.status_code, 200, response.text)
                doc_id = response.json()["document_id"]
                self.assertEqual(response.json()["filename"], filename)
                self.assertTrue(response.json()["indexed"])
                pages = self.client.get(f"/documents/{doc_id}/pages").json()
                self.assertEqual(pages[0]["metadata"]["extraction_method"], "ocr")
                hits = self.main.flow.retriever.search("报告程序命名格式", document_ids=(doc_id,))
                self.assertEqual(hits[0].filename, filename)
                self.assertEqual(hits[0].page, 1)

    def test_extensionless_pdf_uses_pdf_parser_and_preserves_physical_pages(self):
        response = self.upload(mixed_pdf(), filename="无扩展名PDF", mime="application/octet-stream")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["page_count"], 3)
        self.assertEqual(response.json()["metadata"]["ocr_page_count"], 1)
        self.assertEqual(response.json()["filename"], "无扩展名PDF")

    def test_unknown_extensionless_binary_is_rejected_before_ocr(self):
        response = self.upload(b"PK\x03\x04binary-office-data", filename="无扩展名", mime="application/octet-stream")
        self.assertEqual(response.status_code, 415)
        self.assertIn("recognizable PDF or image signature", response.json()["detail"])
        self.assertEqual(self.engine.images, [])

    def test_disabled_and_missing_model_return_503_without_persistence(self):
        self.main.settings = replace(self.main.settings, ocr_enabled=False)
        self.assertEqual(self.upload(image_bytes()).status_code, 503)
        self.main.settings = replace(self.main.settings, ocr_enabled=True)
        with patch("app.rag.parsers.get_ocr_engine", side_effect=OCRUnavailable("models not prepared")):
            self.assertEqual(self.upload(mixed_pdf(), "scan.pdf", "application/pdf").status_code, 503)
        self.assertEqual(self.store.documents(), [])

    def test_inference_failure_returns_502_without_partial_document(self):
        with patch.object(self.engine, "recognize", side_effect=OCRExecutionError("OCR inference failed")):
            response = self.upload(mixed_pdf(), "scan.pdf", "application/pdf")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.store.documents(), [])

    def test_no_usable_ocr_text_returns_400_without_index(self):
        self.engine.lines = (OCRLine("疑似文字", 0.1, BOX),)
        self.assertEqual(self.upload(image_bytes()).status_code, 400)
        self.assertEqual(self.store.documents(), [])

    def test_resource_limits_before_inference_and_render_budget(self):
        self.main.settings = replace(self.main.settings, ocr_max_image_pixels=10)
        self.assertEqual(self.upload(image_bytes()).status_code, 413)
        config = replace(self.main.settings, ocr_max_pages=1)
        with self.assertRaises(DocumentLimitError):
            parse_document("scan.pdf", mixed_pdf(2), config=config, ocr_engine=self.engine)
        self.assertEqual(self.engine.images, [])
        with render_pdf_page(mixed_pdf(), 2, replace(config, ocr_max_render_pixels=10000)) as image:
            self.assertLessEqual(image.width * image.height, 10000)

    def test_signature_mime_corrupt_image_and_encrypted_pdf_rejected(self):
        self.assertEqual(self.upload(b"not an image").status_code, 400)
        self.assertEqual(self.upload(image_bytes(), mime="image/jpeg").status_code, 415)
        self.assertEqual(self.upload(image_bytes("JPEG"), filename="bad.png").status_code, 400)
        self.assertEqual(self.upload(b"\x89PNG\r\n\x1a\ncorrupt").status_code, 400)
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("secret")
        output = io.BytesIO()
        writer.write(output)
        self.assertEqual(self.upload(output.getvalue(), "locked.pdf", "application/pdf").status_code, 400)
        self.assertEqual(self.store.documents(), [])


class OCRAdapterTests(unittest.TestCase):
    def test_missing_prepared_manifest_does_not_initialize_or_download(self):
        with patch("app.rag.ocr._runtime") as runtime:
            with self.assertRaises(OCRUnavailable):
                RapidOCREngine(Path("tests/.tmp/nonexistent-model-directory"))
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
