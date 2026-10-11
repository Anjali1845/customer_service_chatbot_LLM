"""
DOCUMENT INTAKE BRIDGE

The seam between an upload in the Streamlit UI and the multimodal
pipeline in ``knowledge_base/multimodal/``.

The pipeline modules existed and were tested, but nothing in the
application could reach them: there was no upload widget, no storage
directory, and no caller, so validation, security scanning, OCR,
structured extraction, comparison and PII masking never ran on a real
turn. This module is that caller.

It does three things:

1. Stores an uploaded file under ``uploads/`` with a generated name,
   so a customer-supplied filename can never escape the directory.
2. Runs the real pipeline over it via
   ``upload_pipeline.process_upload`` and hands back its result.
3. Runs the retention sweep, which is what actually deletes expired
   uploads -- ``RetentionManager`` refuses to schedule itself, so
   every real invocation has to come from here.

Like ``secure_answer``, it is deliberately tolerant: if the pipeline
or its OCR dependencies are unavailable on this machine it returns a
result with ``ok`` false and a reason, and never raises into the UI.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MULTIMODAL = PROJECT_ROOT / "knowledge_base" / "multimodal"

UPLOAD_DIR = PROJECT_ROOT / "uploads"

# The pipeline rejects anything larger, so refuse it before writing.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _ensure_importable() -> None:
    """
    ``multimodal`` modules import each other by bare name, so the
    directory has to be on ``sys.path`` for them to load at all.
    """

    if MULTIMODAL.is_dir():

        entry = str(MULTIMODAL)

        if entry not in sys.path:

            sys.path.insert(0, entry)


def _unavailable(reason: str) -> Dict[str, Any]:
    """
    The shape every failure returns, so callers can read ``ok`` and
    ``reason`` without special-casing.
    """

    return {
        "ok": False,
        "allowed": False,
        "reason": reason,
        "notifications": [],
    }


def save_upload(filename: str, data: bytes) -> Path:
    """
    Write an uploaded file into ``uploads/`` under a generated name.

    The customer's filename is only used for its suffix: the stem is
    replaced with a random id, so ``../../etc/passwd`` and a name with
    a directory component both land inside ``uploads/``.
    """

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename or "").suffix.lower()

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    target.write_bytes(data)

    return target


def _extension_supported(filename: str) -> Optional[bool]:
    """
    Whether the pipeline accepts this file's extension.

    Returns ``None`` when the validator cannot be consulted, in which
    case the pipeline itself decides -- but a rejected extension is
    caught here first, so a bad file never gets written to disk.
    """

    _ensure_importable()

    try:

        from file_validator import ALLOWED_EXTENSIONS

    except Exception:  # noqa: BLE001 - let the pipeline decide

        return None

    suffix = Path(filename or "").suffix.lower()

    return suffix in ALLOWED_EXTENSIONS


def process_upload(
    filename: str,
    data: bytes,
    message: str = "",
    on_notify: Optional[Callable[[Dict[str, Any]], None]] = None,
    threshold_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Store one uploaded file and run the full multimodal pipeline.

    Returns the pipeline's result dict, or a failure dict with ``ok``
    false and a reason. Never raises -- the UI always has something to
    render.
    """

    if not data:

        return _unavailable("The uploaded file is empty.")

    if len(data) > MAX_UPLOAD_BYTES:

        return _unavailable(
            "The file is larger than the 10 MB limit."
        )

    supported = _extension_supported(filename)

    if supported is False:

        return _unavailable(
            "Unsupported file type. Attach a PDF or an image "
            "(png, jpg, jpeg, webp)."
        )

    try:

        path = save_upload(filename, data)

    except OSError as error:

        return _unavailable(f"Could not store the upload: {error}")

    _ensure_importable()

    try:

        import upload_pipeline

    except Exception as error:  # noqa: BLE001 - report, never raise

        return _unavailable(
            f"The multimodal pipeline is unavailable: {error}"
        )

    try:

        result = upload_pipeline.process_upload(
            str(path),
            message=message,
            on_notify=on_notify,
            threshold_seconds=threshold_seconds,
        )

    except Exception as error:  # noqa: BLE001 - report, never raise

        return _unavailable(f"Processing failed: {error}")

    if not isinstance(result, dict):

        return _unavailable("The pipeline returned no result.")

    return result


def run_retention_cleanup() -> Dict[str, Any]:
    """
    Delete expired uploads.

    ``RetentionManager`` deliberately does not schedule its own
    cleanup, so this is the real invocation: the app calls it on
    startup and after every upload, which is what makes retention
    actually happen rather than being a method nothing calls.
    """

    _ensure_importable()

    try:

        from retention_manager import RetentionManager

    except Exception as error:  # noqa: BLE001 - report, never raise

        return {"deleted_count": 0, "error": str(error)}

    try:

        manager = RetentionManager(storage_dir=str(UPLOAD_DIR))

        return manager.cleanup_expired()

    except Exception as error:  # noqa: BLE001 - report, never raise

        return {"deleted_count": 0, "error": str(error)}


def has_notifications(result: Dict[str, Any]) -> bool:

    notifications = result.get("notifications")

    return bool(isinstance(notifications, list) and notifications)
