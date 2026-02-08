from __future__ import annotations

"""Utility helpers for preprocessing brokerage screenshots before OCR/LLM parsing."""

import logging
from io import BytesIO
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:  # Pillow is optional but strongly recommended
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageFilter = None  # type: ignore
    ImageOps = None  # type: ignore

try:  # pytesseract is optional; we fall back gracefully if missing
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore


def preprocess_image_for_ocr(
    image_bytes: bytes,
    target_format: str = "PNG",
) -> Tuple[bytes, dict]:
    """Apply lightweight denoising/contrast boosts to help OCR accuracy."""

    metadata: dict = {"steps": []}

    if not Image:
        metadata["skipped"] = "Pillow not installed"
        return image_bytes, metadata

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            metadata["original_mode"] = img.mode
            image = img.convert("L")  # grayscale simplifies downstream models
            metadata["steps"].append("grayscale")

            if ImageOps:
                image = ImageOps.autocontrast(image)
                metadata["steps"].append("autocontrast")

            if ImageEnhance:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.4)
                metadata["steps"].append("contrast_x1.4")

            if ImageFilter:
                image = image.filter(ImageFilter.MedianFilter(size=3))
                metadata["steps"].append("median_filter")

            buffer = BytesIO()
            image.save(buffer, format=target_format)
            processed_bytes = buffer.getvalue()
            metadata["output_format"] = target_format
            metadata["output_mime"] = f"image/{target_format.lower()}"
            return processed_bytes, metadata
    except Exception as exc:  # pragma: no cover - defensive fallback
        metadata["error"] = str(exc)
        logger.warning("OCR preprocessing failed: %s", exc)
        return image_bytes, metadata


def extract_text_hint(image_bytes: bytes, lang: str = "eng") -> Optional[str]:
    """Run pytesseract (if available) to capture raw text for Gemini prompt hints."""

    if not pytesseract or not Image:
        return None

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            text = pytesseract.image_to_string(img, lang=lang)
    except Exception as exc:  # pragma: no cover - depends on runtime env
        logger.warning("Tesseract OCR hint failed: %s", exc)
        return None

    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return cleaned or None


def run_pipeline(
    image_bytes: bytes,
    mime_type: Optional[str] = None,
) -> Tuple[bytes, str, dict, Optional[str]]:
    """Full OCR pipeline hook used by the screenshot ingestion flow."""

    processed_bytes, metadata = preprocess_image_for_ocr(image_bytes)
    text_hint = extract_text_hint(processed_bytes)

    # If enhanced image produced no hint, fall back to the raw image.
    if text_hint is None and processed_bytes != image_bytes:
        text_hint = extract_text_hint(image_bytes)

    output_mime = metadata.get("output_mime") or mime_type or "image/png"
    metadata.setdefault("output_mime", output_mime)

    return processed_bytes, output_mime, metadata, text_hint
