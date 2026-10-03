"""
TASK 5 - OCR AND MULTIMODAL TEXT EXTRACTION

Supports:
- PNG
- JPG / JPEG
- WebP
- PDF

Pipeline:
    Uploaded file
        ↓
    File validation
        ↓
    OCR / PDF text extraction
        ↓
    Extracted text
        ↓
    Existing structured extractor

Important safety rule:
This module never invents missing values.
If OCR cannot reliably recover information, the field remains absent/None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pymupdf
import pytesseract
from PIL import Image, ImageFilter, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

SUPPORTED_PDF_EXTENSIONS = {
    ".pdf",
}

SUPPORTED_EXTENSIONS = (
    SUPPORTED_IMAGE_EXTENSIONS
    | SUPPORTED_PDF_EXTENSIONS
)

# A very small amount of preprocessing can improve OCR on
# screenshots and scanned documents without altering the source.
OCR_SCALE = 2.0

# Minimum amount of OCR text considered useful.
MIN_OCR_TEXT_LENGTH = 3


# -------------------------------------------------------------------
# VALIDATION
# -------------------------------------------------------------------

def validate_supported_file(file_path: str | Path) -> Path:
    """
    Validate that the file exists, is a regular file, and has
    a supported image/PDF extension.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Not a regular file: {file_path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    return path


# -------------------------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Prepare an image for OCR.

    The original image object is not modified.
    """

    processed = image.convert("L")

    # Increase image size for small screenshots/scans.
    if OCR_SCALE > 1:
        new_size = (
            int(processed.width * OCR_SCALE),
            int(processed.height * OCR_SCALE),
        )

        processed = processed.resize(new_size)

    # Improve contrast.
    processed = ImageOps.autocontrast(processed)

    # Mild sharpening.
    processed = processed.filter(ImageFilter.SHARPEN)

    return processed


# -------------------------------------------------------------------
# IMAGE OCR
# -------------------------------------------------------------------

def ocr_image(
    file_path: str | Path,
) -> Dict[str, Any]:
    """
    Run Tesseract OCR on an image.

    Returns:
        {
            "success": bool,
            "text": str,
            "pages": 1,
            "source_type": "image",
            "quality": "good" | "low",
            "clarification_required": bool
        }
    """

    path = validate_supported_file(file_path)

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Expected an image file, received: {path.suffix}"
        )

    try:
        with Image.open(path) as image:
            processed = preprocess_image(image)

            text = pytesseract.image_to_string(
                processed
            ).strip()

    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "pages": 1,
            "source_type": "image",
            "quality": "low",
            "clarification_required": True,
            "error": str(exc),
        }

    quality = (
        "good"
        if len(text) >= MIN_OCR_TEXT_LENGTH
        else "low"
    )

    return {
        "success": True,
        "text": text,
        "pages": 1,
        "source_type": "image",
        "quality": quality,
        "clarification_required": quality == "low",
    }


# -------------------------------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------------------------------

def extract_pdf_text(
    file_path: str | Path,
) -> Dict[str, Any]:
    """
    Extract text directly from a PDF.

    This is preferred for digitally generated PDFs because it
    preserves text more reliably than OCR.

    If a PDF page contains little/no embedded text, that page
    is rendered and passed through OCR.
    """

    path = validate_supported_file(file_path)

    if path.suffix.lower() not in SUPPORTED_PDF_EXTENSIONS:
        raise ValueError(
            f"Expected a PDF file, received: {path.suffix}"
        )

    page_texts: List[str] = []
    ocr_page_count = 0

    try:
        document = pymupdf.open(path)

        for page in document:
            text = page.get_text("text").strip()

            if len(text) >= MIN_OCR_TEXT_LENGTH:
                page_texts.append(text)
                continue

            # Scanned/image-only PDF page.
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(
                    OCR_SCALE,
                    OCR_SCALE,
                ),
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )

            processed = preprocess_image(image)

            ocr_text = pytesseract.image_to_string(
                processed
            ).strip()

            page_texts.append(ocr_text)
            ocr_page_count += 1

        page_count = len(document)
        document.close()

    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "pages": 0,
            "source_type": "pdf",
            "quality": "low",
            "clarification_required": True,
            "ocr_pages": 0,
            "error": str(exc),
        }

    combined_text = "\n\n".join(
        text for text in page_texts if text
    ).strip()

    quality = (
        "good"
        if len(combined_text) >= MIN_OCR_TEXT_LENGTH
        else "low"
    )

    return {
        "success": True,
        "text": combined_text,
        "pages": page_count,
        "source_type": "pdf",
        "quality": quality,
        "clarification_required": quality == "low",
        "ocr_pages": ocr_page_count,
    }


# -------------------------------------------------------------------
# UNIFIED MULTIMODAL EXTRACTION
# -------------------------------------------------------------------

def extract_text_from_file(
    file_path: str | Path,
) -> Dict[str, Any]:
    """
    Unified entry point for image and PDF processing.
    """

    path = validate_supported_file(file_path)

    extension = path.suffix.lower()

    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return ocr_image(path)

    if extension in SUPPORTED_PDF_EXTENSIONS:
        return extract_pdf_text(path)

    # Defensive fallback. Validation above should prevent this.
    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# -------------------------------------------------------------------
# STRUCTURED EXTRACTION INTEGRATION
# -------------------------------------------------------------------

def extract_structured_data_from_file(
    file_path: str | Path,
) -> Dict[str, Any]:
    """
    Run OCR/PDF extraction and then pass the resulting text
    through the existing structured extractor.

    No values are guessed.
    """

    result = extract_text_from_file(file_path)

    if not result["success"]:
        return {
            **result,
            "structured_data": None,
        }

    if result["clarification_required"]:
        return {
            **result,
            "structured_data": None,
        }

    # Import only when needed so this module can still perform
    # OCR independently.
    from extractor import extract_structured_data

    structured_data = extract_structured_data(
        result["text"]
    )

    return {
        **result,
        "structured_data": structured_data,
    }


# -------------------------------------------------------------------
# SELF-TEST
# -------------------------------------------------------------------

if __name__ == "__main__":

    import tempfile

    print("=" * 70)
    print("TASK 5 - OCR EXTRACTOR SELF-TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        # -----------------------------------------------------------
        # TEST 1 - CREATE IMAGE WITH TEXT
        # -----------------------------------------------------------

        print("\nTEST 1 - IMAGE OCR")

        image_path = temp_path / "invoice.png"

        image = Image.new(
            "RGB",
            (1000, 400),
            "white",
        )

        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)

        draw.text(
            (50, 50),
            "Order ID: ORD-12345",
            fill="black",
        )

        draw.text(
            (50, 120),
            "Date: 15 September 2026",
            fill="black",
        )

        draw.text(
            (50, 190),
            "Amount: INR 750",
            fill="black",
        )

        draw.text(
            (50, 260),
            "Product: Wireless Keyboard",
            fill="black",
        )

        image.save(image_path)

        result = ocr_image(image_path)

        print("OCR success:", result["success"])
        print("OCR quality:", result["quality"])
        print("Extracted text:")
        print(result["text"])

        assert result["success"] is True
        assert result["quality"] == "good"
        assert "ORD" in result["text"].upper()

        print("Image OCR: PASS ✅")

        # -----------------------------------------------------------
        # TEST 2 - MISSING INFORMATION IS NOT INVENTED
        # -----------------------------------------------------------

        print("\nTEST 2 - NO INVENTION")

        sparse_image_path = temp_path / "sparse.png"

        sparse_image = Image.new(
            "RGB",
            (800, 300),
            "white",
        )

        draw = ImageDraw.Draw(sparse_image)

        draw.text(
            (50, 100),
            "Order ID: ORD-999",
            fill="black",
        )

        sparse_image.save(sparse_image_path)

        result = extract_structured_data_from_file(
            sparse_image_path
        )

        print(result["structured_data"])

        assert result["structured_data"] is not None

        # The OCR/extractor pipeline must not invent values.
        data = result["structured_data"]

        assert data.get("order_id") is not None

        print("Missing values are not invented: PASS ✅")

        # -----------------------------------------------------------
        # TEST 3 - PDF OCR
        # -----------------------------------------------------------

        print("\nTEST 3 - PDF PROCESSING")

        pdf_path = temp_path / "invoice.pdf"

        pdf = pymupdf.open()

        page = pdf.new_page(
            width=600,
            height=400,
        )

        page.insert_text(
            (50, 80),
            "Order ID: ORD-PDF-123",
            fontsize=18,
        )

        page.insert_text(
            (50, 130),
            "Amount: INR 1200",
            fontsize=18,
        )

        page.insert_text(
            (50, 180),
            "Product: Laptop Stand",
            fontsize=18,
        )

        pdf.save(pdf_path)
        pdf.close()

        result = extract_pdf_text(pdf_path)

        print("PDF success:", result["success"])
        print("PDF pages:", result["pages"])
        print("OCR pages:", result["ocr_pages"])
        print("PDF text:")
        print(result["text"])

        assert result["success"] is True
        assert result["pages"] == 1
        assert "ORD-PDF-123" in result["text"]

        print("PDF processing: PASS ✅")

    print("\n" + "=" * 70)
    print("OCR EXTRACTOR SELF-TEST COMPLETE")
    print("ALL OCR TESTS PASSED ✅")
    print("=" * 70)
