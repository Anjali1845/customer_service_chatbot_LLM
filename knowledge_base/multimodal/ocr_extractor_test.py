"""
TASK 5 - OCR EXTRACTOR REGRESSION TEST

Tests:
1. Image OCR
2. Structured extraction from OCR text
3. Missing values are not invented
4. Digitally generated PDF processing
5. Scanned/image-only PDF OCR fallback
6. Unsupported file rejection
7. Empty/low-quality image handling
"""

from pathlib import Path
import tempfile

import pymupdf
from PIL import Image, ImageDraw

from ocr_extractor import (
    ocr_image,
    extract_pdf_text,
    extract_text_from_file,
    extract_structured_data_from_file,
)


def create_text_image(path: Path, text_lines):
    """Create a clear test image containing the supplied text."""

    image = Image.new(
        "RGB",
        (1200, 500),
        "white",
    )

    draw = ImageDraw.Draw(image)

    y = 50

    for line in text_lines:
        draw.text(
            (50, y),
            line,
            fill="black",
        )
        y += 80

    image.save(path)


def main():

    print("=" * 70)
    print("TASK 5 - OCR EXTRACTOR REGRESSION TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:

        test_dir = Path(temp_dir)

        # =========================================================
        # TEST 1 - IMAGE OCR
        # =========================================================

        print("\nTEST 1 - IMAGE OCR")

        image_path = test_dir / "invoice.png"

        create_text_image(
            image_path,
            [
                "Order ID: ORD-TEST-1001",
                "Date: 15 September 2026",
                "Amount: INR 750",
                "Product: Wireless Keyboard",
            ],
        )

        result = ocr_image(image_path)

        print("Success:", result["success"])
        print("Quality:", result["quality"])
        print("Text:")
        print(result["text"])

        assert result["success"] is True
        assert result["quality"] == "good"
        assert "ORD-TEST-1001" in result["text"]

        print("PASS ✅")

        # =========================================================
        # TEST 2 - STRUCTURED EXTRACTION
        # =========================================================

        print("\nTEST 2 - STRUCTURED EXTRACTION FROM IMAGE")

        result = extract_structured_data_from_file(
            image_path
        )

        print(result["structured_data"])

        assert result["structured_data"] is not None

        data = result["structured_data"]

        assert data["order_id"] is not None
        assert data["date"] is not None
        assert data["amount"] is not None
        assert data["product_name"] is not None

        print("Structured fields extracted: PASS ✅")

        # =========================================================
        # TEST 3 - NO INVENTION
        # =========================================================

        print("\nTEST 3 - MISSING VALUES ARE NOT INVENTED")

        sparse_image = test_dir / "sparse.png"

        create_text_image(
            sparse_image,
            [
                "Order ID: ORD-ONLY-123",
            ],
        )

        result = extract_structured_data_from_file(
            sparse_image
        )

        data = result["structured_data"]

        print(data)

        assert data["order_id"] is not None
        assert data["date"] is None
        assert data["amount"] is None
        assert data["product_name"] is None

        print("No missing values invented: PASS ✅")

        # =========================================================
        # TEST 4 - DIGITAL PDF
        # =========================================================

        print("\nTEST 4 - DIGITAL PDF PROCESSING")

        pdf_path = test_dir / "invoice.pdf"

        pdf = pymupdf.open()

        page = pdf.new_page(
            width=600,
            height=400,
        )

        page.insert_text(
            (50, 80),
            "Order ID: ORD-PDF-1001",
            fontsize=18,
        )

        page.insert_text(
            (50, 130),
            "Date: 20 September 2026",
            fontsize=18,
        )

        page.insert_text(
            (50, 180),
            "Amount: INR 1200",
            fontsize=18,
        )

        page.insert_text(
            (50, 230),
            "Product: Laptop Stand",
            fontsize=18,
        )

        pdf.save(pdf_path)
        pdf.close()

        result = extract_pdf_text(pdf_path)

        print("Success:", result["success"])
        print("Pages:", result["pages"])
        print("OCR pages:", result["ocr_pages"])
        print("Text:")
        print(result["text"])

        assert result["success"] is True
        assert result["pages"] == 1
        assert "ORD-PDF-1001" in result["text"]
        assert result["ocr_pages"] == 0

        print("Digital PDF processing: PASS ✅")

        # =========================================================
        # TEST 5 - IMAGE-ONLY PDF / OCR FALLBACK
        # =========================================================

        print("\nTEST 5 - IMAGE-ONLY PDF OCR FALLBACK")

        image_pdf_path = test_dir / "scanned_invoice.pdf"

        source_image = Image.new(
            "RGB",
            (1200, 500),
            "white",
        )

        draw = ImageDraw.Draw(source_image)

        draw.text(
            (50, 80),
            "Order ID: ORD-SCAN-2001",
            fill="black",
        )

        draw.text(
            (50, 160),
            "Amount: INR 950",
            fill="black",
        )

        draw.text(
            (50, 240),
            "Product: USB Hub",
            fill="black",
        )

        # Save temporary image.
        temporary_image = test_dir / "scan_source.png"
        source_image.save(temporary_image)

        # Convert image to PDF using PyMuPDF.
        image_document = pymupdf.open()

        page = image_document.new_page(
            width=600,
            height=250,
        )

        page.insert_image(
            page.rect,
            filename=str(temporary_image),
        )

        image_document.save(image_pdf_path)
        image_document.close()

        result = extract_pdf_text(image_pdf_path)

        print("Success:", result["success"])
        print("Pages:", result["pages"])
        print("OCR pages:", result["ocr_pages"])
        print("Text:")
        print(result["text"])

        assert result["success"] is True
        assert result["pages"] == 1
        assert result["ocr_pages"] == 1
        assert "ORD-SCAN-2001" in result["text"]

        print("Image-only PDF OCR fallback: PASS ✅")

        # =========================================================
        # TEST 6 - UNSUPPORTED FILE
        # =========================================================

        print("\nTEST 6 - UNSUPPORTED FILE")

        unsupported = test_dir / "document.txt"

        unsupported.write_text(
            "This should not be processed as an image or PDF.",
            encoding="utf-8",
        )

        try:
            extract_text_from_file(unsupported)
            raise AssertionError(
                "Unsupported file was incorrectly accepted"
            )

        except ValueError:
            pass

        print("Unsupported file rejected: PASS ✅")

        # =========================================================
        # TEST 7 - EMPTY / LOW QUALITY IMAGE
        # =========================================================

        print("\nTEST 7 - LOW QUALITY / EMPTY IMAGE")

        empty_image = test_dir / "empty.png"

        Image.new(
            "RGB",
            (500, 300),
            "white",
        ).save(empty_image)

        result = ocr_image(empty_image)

        print("Success:", result["success"])
        print("Quality:", result["quality"])
        print(
            "Clarification required:",
            result["clarification_required"],
        )

        assert result["success"] is True
        assert result["quality"] == "low"
        assert result["clarification_required"] is True

        print("Low-quality evidence triggers clarification: PASS ✅")

    print("\n" + "=" * 70)
    print("TASK 5 - OCR EXTRACTOR REGRESSION TEST COMPLETE")
    print("ALL OCR REGRESSION TESTS PASSED ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
