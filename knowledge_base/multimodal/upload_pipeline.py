"""
END-TO-END MULTIMODAL UPLOAD PIPELINE
=====================================

Single documented entry point, ``process_upload``, that walks a
REAL customer upload through every multimodal stage in order:

    1. file_validator.validate_file
       Rejects missing / empty / oversized / unsupported /
       unsafe files.

    2. malicious_detector.scan_uploaded_file
       Filename-level security scan (dangerous extensions).

    3. ocr_extractor.extract_text_from_file
       Real OCR / PDF text extraction.

    4. malicious_detector.prepare_untrusted_content +
       malicious_detector.scan_uploaded_file(filename, text)
       Content-level security gate over the EXTRACTED text.
       Malicious uploads are REJECTED, never silently
       continued, and detected instructions are never
       executed.

    5. extractor.extract_structured_data
       Conservative structured-field extraction. Missing
       information stays None - nothing is invented.

    6. comparator.compare_message_with_document
       Customer message <-> document evidence comparison.
       Conflicts are surfaced, never auto-resolved.

    7. pii_masker.mask_sensitive_fields +
       pii_masker.mask_sensitive_text
       Masking happens BEFORE anything is returned, so the
       masked outputs are safe to log and hand off.

Import behaviour
----------------
This module (and ``process_upload``) import with the standard
library only. ``ocr_extractor`` pulls in pymupdf / PIL /
pytesseract at module scope, so it is imported lazily inside
the OCR step. This keeps ``process_upload`` importable in
environments without the OCR stack without changing the OCR
module's own behaviour.

process_upload contract
-----------------------
``process_upload(file_path, message="", on_notify=None,
threshold_seconds=None) -> dict``

Returns a structured dict with these keys:

    ok / allowed        True only when the file passed EVERY
                        stage. A file that fails validation or
                        scans as malicious is rejected
                        (False/False), never silently
                        continued.
    reason              None on success, otherwise a
                        human-readable explanation of the
                        rejection.
    text                Raw extracted text (None when
                        rejected). NOT log-safe - use
                        masked_text for logs.
    masked_text         PII-masked copy of the full text -
                        log-safe.
    structured_data     Fields extracted from the text (may
                        contain None values; never invented).
    masked              PII-masked copy of structured_data -
                        log-safe for handoff.
    masked_message      PII-masked copy of the customer
                        message - log-safe.
    comparison          compare_message_with_document result
                        (None when rejected).
    clarification_required
                        True when the evidence was too weak
                        to read reliably.
    notifications       Notification events collected during
                        processing (rejections, unreadable
                        evidence, long-running processing).
                        Each is also forwarded to the
                        optional on_notify callback.
    validation          Raw file_validator result (audit).
    security            Final malicious_detector scan result
                        (audit).
    ocr                 OCR summary (success / pages /
                        quality / error).
    elapsed_seconds     Wall-clock duration of the pipeline.

Optional keyword arguments:

    on_notify           Callable[[dict], None] invoked with
                        each notification event. Exceptions it
                        raises are swallowed so a broken
                        sink can never break processing.
                        None by default.
    threshold_seconds   When the pipeline's real running time
                        crosses this value, a
                        "processing_slow" notification is
                        emitted so the customer can be told the
                        upload is still working. Defaults to
                        background_processor
                        .BACKGROUND_THRESHOLD_SECONDS (30s).
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_THIS_DIR = Path(__file__).resolve().parent


def _sibling(module_name: str):
    """
    Import a sibling module of this pipeline.

    Works when this directory is already importable (normal
    case), and when the pipeline itself was loaded from a file
    path by the host application (e.g. support_deps-style
    loading) - in that case the directory is added to sys.path
    once so sibling imports, including the lazy
    ``from extractor import ...`` inside ocr_extractor,
    resolve.
    """

    try:
        return importlib.import_module(module_name)
    except ImportError:
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        return importlib.import_module(module_name)


# Stdlib-only dependencies: safe to import eagerly.
file_validator = _sibling("file_validator")
malicious_detector = _sibling("malicious_detector")
extractor = _sibling("extractor")
comparator = _sibling("comparator")
pii_masker = _sibling("pii_masker")
background_processor = _sibling("background_processor")

# pymupdf / PIL / pytesseract live behind this lazy import.
_ocr_extractor = None


def _get_ocr_extractor():
    """
    Lazily import the OCR module on first use.

    ocr_extractor imports pymupdf / PIL / pytesseract at module
    scope; deferring that import keeps process_upload itself
    importable without the OCR stack while leaving the OCR
    module's behaviour untouched.
    """

    global _ocr_extractor

    if _ocr_extractor is None:
        _ocr_extractor = _sibling("ocr_extractor")

    return _ocr_extractor


def process_upload(
    file_path: str,
    message: str = "",
    on_notify: Optional[
        Callable[[Dict[str, Any]], None]
    ] = None,
    threshold_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Process one real customer upload through the full
    multimodal pipeline: validation -> security scan ->
    OCR/text extraction -> untrusted-content gate ->
    structured extraction -> message/document comparison ->
    PII masking.

    See the module docstring for the full return contract.
    Files that fail validation or scan as malicious are
    rejected (ok=False, allowed=False) - never silently
    continued.
    """

    if threshold_seconds is None:
        threshold_seconds = (
            background_processor
            .BACKGROUND_THRESHOLD_SECONDS
        )

    started = time.time()

    notifications: List[Dict[str, Any]] = []
    slow_notified = False

    def notify(event: Dict[str, Any]) -> None:
        notifications.append(event)

        if on_notify is not None:
            try:
                on_notify(event)
            except Exception:
                # A broken notification sink must never
                # break the pipeline.
                pass

    def maybe_notify_slow(stage: str) -> None:
        """Real-time slow-processing notification."""

        nonlocal slow_notified

        if slow_notified:
            return

        elapsed = time.time() - started

        if elapsed >= threshold_seconds:
            slow_notified = True

            notify(
                {
                    "type": "processing_slow",
                    "stage": stage,
                    "elapsed_seconds": round(elapsed, 3),
                    "threshold_seconds": threshold_seconds,
                    "message": (
                        "Processing your upload is taking "
                        "longer than usual. It is still "
                        "working - you can keep chatting and "
                        "will receive the result shortly."
                    ),
                    "timestamp": time.time(),
                }
            )

    path = Path(file_path)

    result: Dict[str, Any] = {
        "ok": False,
        "allowed": False,
        "reason": None,
        "file_path": str(path),
        "filename": path.name,
        "message": message,
        "text": None,
        "masked_text": None,
        "structured_data": None,
        "masked": None,
        "masked_message": None,
        "comparison": None,
        "clarification_required": False,
        "notifications": notifications,
        "validation": None,
        "security": None,
        "ocr": None,
        "elapsed_seconds": None,
    }

    def reject(
        reason: str,
        stage: str
    ) -> Dict[str, Any]:
        """Stop the pipeline and return a rejection result."""

        result["ok"] = False
        result["allowed"] = False
        result["reason"] = reason
        result["elapsed_seconds"] = round(
            time.time() - started,
            3,
        )

        notify(
            {
                "type": "upload_rejected",
                "stage": stage,
                "reason": reason,
                "message": (
                    "Your file could not be processed: "
                    + reason
                ),
                "timestamp": time.time(),
            }
        )

        return result

    # ---------------------------------------------------------
    # 1. FILE VALIDATION
    # ---------------------------------------------------------

    validation = file_validator.validate_file(
        str(path)
    )
    result["validation"] = validation

    if not validation.get("valid"):
        return reject(
            validation.get(
                "reason",
                "File validation failed.",
            ),
            "validation",
        )

    filename = validation.get("filename", path.name)

    # ---------------------------------------------------------
    # 2. FILENAME-LEVEL SECURITY SCAN
    # ---------------------------------------------------------

    security = malicious_detector.scan_uploaded_file(
        filename,
        ""
    )
    result["security"] = security

    if not security.get("safe"):
        return reject(
            "; ".join(
                security.get("reasons") or []
            )
            or "File failed the security scan.",
            "security",
        )

    # ---------------------------------------------------------
    # 3. OCR / TEXT EXTRACTION (lazy, heavy dependencies)
    # ---------------------------------------------------------

    ocr_module = _get_ocr_extractor()

    ocr = ocr_module.extract_text_from_file(str(path))

    result["ocr"] = {
        key: ocr.get(key)
        for key in (
            "success",
            "pages",
            "source_type",
            "quality",
            "clarification_required",
            "ocr_pages",
            "error",
        )
    }

    if not ocr.get("success"):
        result["clarification_required"] = True
        return reject(
            ocr.get("error")
            or "Text could not be extracted "
            "from the file.",
            "ocr",
        )

    text = ocr.get("text") or ""
    result["text"] = text
    result["clarification_required"] = bool(
        ocr.get("clarification_required")
    )

    if result["clarification_required"]:
        notify(
            {
                "type": "clarification_required",
                "message": (
                    "The uploaded file was hard to read. "
                    "Please confirm the order details in "
                    "your message."
                ),
                "timestamp": time.time(),
            }
        )

    maybe_notify_slow("ocr")

    # ---------------------------------------------------------
    # 4. CONTENT-LEVEL UNTRUSTED-CONTENT GATE
    # ---------------------------------------------------------

    prepared = (
        malicious_detector
        .prepare_untrusted_content(text)
    )

    security = malicious_detector.scan_uploaded_file(
        filename,
        text
    )
    result["security"] = security

    if (
        not prepared.get("allowed")
        or not security.get("safe")
    ):
        return reject(
            prepared.get("reason")
            or "; ".join(
                security.get("reasons") or []
            )
            or "Malicious content detected.",
            "security",
        )

    # ---------------------------------------------------------
    # 5. STRUCTURED EXTRACTION (no invented values)
    # ---------------------------------------------------------

    structured_data = (
        extractor.extract_structured_data(text)
    )
    result["structured_data"] = structured_data

    # ---------------------------------------------------------
    # 6. MESSAGE <-> DOCUMENT COMPARISON
    # ---------------------------------------------------------

    comparison = (
        comparator.compare_message_with_document(
            message or "",
            structured_data,
        )
    )
    result["comparison"] = comparison

    # ---------------------------------------------------------
    # 7. PII MASKING (before anything is returned)
    # ---------------------------------------------------------

    result["masked"] = (
        pii_masker.mask_sensitive_fields(
            structured_data
        )
    )
    result["masked_text"] = (
        pii_masker.mask_sensitive_text(text)
    )
    result["masked_message"] = (
        pii_masker.mask_sensitive_text(message or "")
    )

    # ---------------------------------------------------------
    # SUCCESS
    # ---------------------------------------------------------

    result["ok"] = True
    result["allowed"] = True
    result["reason"] = None
    result["elapsed_seconds"] = round(
        time.time() - started,
        3,
    )

    maybe_notify_slow("pipeline")

    return result


# --------------------------------------------------
# Self-test
# --------------------------------------------------

if __name__ == "__main__":

    import tempfile

    print("=" * 70)
    print("UPLOAD PIPELINE SELF-TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        # ocr_extractor has already imported pymupdf; reuse it
        # here to build a real digital PDF.
        ocr_module = _get_ocr_extractor()
        pymupdf = ocr_module.pymupdf

        pdf_path = temp_path / "invoice.pdf"

        document = pymupdf.open()
        page = document.new_page(
            width=600,
            height=400,
        )

        page.insert_text(
            (50, 80),
            "Order ID: ORD-E2E-1001",
            fontsize=18,
        )
        page.insert_text(
            (50, 130),
            "Amount: INR 1499.00",
            fontsize=18,
        )
        page.insert_text(
            (50, 180),
            "Product Name: Wireless Keyboard",
            fontsize=18,
        )

        document.save(pdf_path)
        document.close()

        outcome = process_upload(
            str(pdf_path),
            "My order ORD-E2E-1001 was charged INR 1499.",
        )

        print(outcome)

        assert outcome["ok"] is True
        assert outcome["allowed"] is True
        assert outcome["structured_data"][
            "order_id"
        ] == "ORD-E2E-1001"
        assert outcome["masked"] is not None

        print("End-to-end pipeline: PASS ✅")

        missing = process_upload(
            str(temp_path / "missing.pdf")
        )

        assert missing["ok"] is False
        assert missing["allowed"] is False
        assert "exist" in missing["reason"]

        print("Missing file rejected: PASS ✅")

    print("\n" + "=" * 70)
    print("UPLOAD PIPELINE SELF-TEST COMPLETE: PASS ✅")
    print("=" * 70)
