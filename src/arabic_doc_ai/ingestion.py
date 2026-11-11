from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGES | {".pdf"}


def iter_document_pages(path: Path, *, dpi: int, max_side: int, max_pages: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield one page at a time as an RGB uint8 array to keep RAM bounded."""
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        yield from _iter_pdf_pages(path, dpi=dpi, max_side=max_side, max_pages=max_pages)
    else:
        yield 1, _load_image(path, max_side=max_side)


def _resize_if_needed(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    try:
        import cv2

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((new_width, new_height), Image.Resampling.LANCZOS))


def _load_image(path: Path, *, max_side: int) -> np.ndarray:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required. Install it with: python -m pip install Pillow") from exc

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        image = np.asarray(im, dtype=np.uint8)
    return _resize_if_needed(image, max_side)


def _iter_pdf_pages(path: Path, *, dpi: int, max_side: int, max_pages: int | None) -> Iterator[tuple[int, np.ndarray]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF files. Install it with: python -m pip install PyMuPDF") from exc

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    with fitz.open(path) as document:
        page_count = len(document)
        if page_count == 0:
            raise ValueError("The PDF contains no pages.")
        limit = min(page_count, max_pages) if max_pages else page_count
        for page_index in range(limit):
            page = document.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            yield page_index + 1, _resize_if_needed(image, max_side)
            del image, pix, page
