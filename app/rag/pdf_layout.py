"""Bounded raster-region rendering and native text positions for mixed PDFs."""

import ctypes
import math

from app.rag.ocr import OCRUnavailable


def normalized(text):
    return "".join(text.split())


def mixed_page_regions(data, index, config, lock):
    """Return detached crops; all PDFium calls run under the shared render lock.

    Coordinates of crops/native blocks are pixels from the rendered page's top
    left. PDF bounds use points from the bottom left. PageToDevice handles page
    rotation and crop boxes; nested Form matrices are applied explicitly.
    """
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
    except ImportError as exc:
        raise OCRUnavailable('Mixed PDF OCR requires pip install -e ".[ocr]"') from exc
    crops = []
    try:
        with lock, pdfium.PdfDocument(data) as document:
            page = document[index]
            try:
                width, height = page.get_size()
                if width <= 0 or height <= 0:
                    raise ValueError("PDF page has invalid dimensions")
                scale = min(config.ocr_dpi / 72, math.sqrt(config.ocr_max_render_pixels / (width * height)))
                while math.ceil(width * scale) * math.ceil(height * scale) > config.ocr_max_render_pixels:
                    scale *= 0.99
                pixel_width, pixel_height = math.ceil(width * scale), math.ceil(height * scale)

                def bounds(obj):
                    rect = obj.get_bounds()
                    parent = obj.container
                    while parent is not None:
                        rect = parent.get_matrix().on_rect(*rect)
                        parent = parent.container
                    points = []
                    for x, y in ((rect[0], rect[1]), (rect[0], rect[3]),
                                 (rect[2], rect[1]), (rect[2], rect[3])):
                        px, py = ctypes.c_int(), ctypes.c_int()
                        if not raw.FPDF_PageToDevice(page, 0, 0, pixel_width, pixel_height, 0,
                                                     x, y, ctypes.byref(px), ctypes.byref(py)):
                            raise ValueError("cannot map PDF object to rendered page")
                        points.append((px.value, py.value))
                    box = (max(0, min(p[0] for p in points)), max(0, min(p[1] for p in points)),
                           min(pixel_width, max(p[0] for p in points)), min(pixel_height, max(p[1] for p in points)))
                    return box, rect

                native = []
                regions = []
                textpage = page.get_textpage()
                try:
                    for obj in page.get_objects(textpage=textpage):
                        if obj.type not in (raw.FPDF_PAGEOBJ_TEXT, raw.FPDF_PAGEOBJ_IMAGE):
                            continue
                        box, rect = bounds(obj)
                        if box[2] <= box[0] or box[3] <= box[1]:
                            continue
                        if obj.type == raw.FPDF_PAGEOBJ_TEXT:
                            text = obj.extract().strip()
                            if text:
                                native.append({"text": text, "box": box, "method": "native_text"})
                        else:
                            regions.append({"box": box, "pdf_box": rect})
                            if len(regions) > config.ocr_max_regions_per_page:
                                # Imported lazily to avoid a parser/layout cycle.
                                from app.rag.parsers import DocumentLimitError
                                raise DocumentLimitError("PDF page exceeds STUDYFLOW_OCR_MAX_REGIONS_PER_PAGE")
                finally:
                    textpage.close()
                # Overlapping image tiles/masks share one crop and one OCR pass.
                merged = []
                for region in regions:
                    pending = region
                    changed = True
                    while changed:
                        changed = False
                        for other in list(merged):
                            a, b = pending["box"], other["box"]
                            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                                pending = {"box": (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                                           "pdf_box": tuple(min(pending["pdf_box"][i], other["pdf_box"][i]) if i < 2 else
                                                            max(pending["pdf_box"][i], other["pdf_box"][i]) for i in range(4))}
                                merged.remove(other)
                                changed = True
                                break
                    merged.append(pending)
                if merged:
                    bitmap = page.render(scale=scale)
                    try:
                        view = bitmap.to_pil()
                        try:
                            for region in sorted(merged, key=lambda r: (r["box"][1], r["box"][0])):
                                with view.crop(region["box"]) as crop:
                                    crops.append((region, crop.convert("RGB")))
                        finally:
                            view.close()
                    finally:
                        bitmap.close()
                return native, crops, (pixel_width, pixel_height)
            finally:
                page.close()
    except BaseException:
        for _, image in crops:
            image.close()
        raise


def native_lines(blocks, original):
    """Keep pypdf's logical text verbatim, locating its lines using PDFium.

    Sorting individual glyph boxes by their top edge scrambles mixed fonts and
    punctuation. Match in content-stream order first, then order whole lines.
    """
    stream = ""
    positions = []
    for block in blocks:
        value = normalized(block["text"])
        stream += value
        positions.extend([block["box"]] * len(value))
    lines = []
    cursor = 0
    for text in original.splitlines():
        value = normalized(text)
        if not value:
            continue
        start = stream.find(value, cursor)
        if start < 0:
            # Preserve native text if the engines disagree on glyph decoding.
            return [{"text": original, "box": (0, 0, 0, 0), "method": "native_text"}], True
        cursor = start + len(value)
        boxes = positions[start:cursor]
        box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes))
        lines.append({"text": text, "box": box, "method": "native_text"})
    return lines, False
