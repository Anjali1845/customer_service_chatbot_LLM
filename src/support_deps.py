"""
SUPPORT OPS - SHARED DEPENDENCY ADAPTER

Task 7 (sentiment + escalation) and Task 8 (ticketing) both reuse
helpers that already exist in knowledge_base/. This module is the
single place where that reuse is wired up.

The reused modules are loaded by file path under private names
(``_kbm_*``) rather than by adding knowledge_base/ to sys.path.
Importing them as bare ``extractor`` / ``security`` / ``comparator``
would put very generic names into sys.modules, where they could one
day collide with a module in src/. Loading by path keeps the imports
isolated and the reusable surface explicit.

Public surface re-exported here:

    mask_sensitive_text        - PII/payment masking for handoff + logs
    mask_sensitive_fields      - same, for dict-shaped contact records
    extract_structured_data    - order id / date / amount / product / error
    extract_message_fields     - conservative extraction from a chat message
    detect_prompt_injection    - instruction-like text in untrusted content
    sanitize_document_content  - strip instruction-like lines
    has_substantive_evidence   - conservative evidence-support gate
    as_utc                     - normalise any datetime to aware UTC
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"
_MULTIMODAL = _KNOWLEDGE_BASE / "multimodal"


def _load_module(name: str, path: Path):
    """
    Load a module from an explicit file path.

    Kept private: callers should import the re-exported names below,
    never the ``_kbm_*`` modules themselves.
    """

    if not path.is_file():
        raise ImportError(
            f"Required support dependency is missing: {path}"
        )

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


_kbm_pii_masker = _load_module(
    "_kbm_pii_masker", _MULTIMODAL / "pii_masker.py"
)

_kbm_extractor = _load_module(
    "_kbm_extractor", _MULTIMODAL / "extractor.py"
)

_kbm_comparator = _load_module(
    "_kbm_comparator", _MULTIMODAL / "comparator.py"
)

_kbm_security = _load_module(
    "_kbm_security", _KNOWLEDGE_BASE / "security.py"
)


mask_sensitive_text = _kbm_pii_masker.mask_sensitive_text
mask_sensitive_fields = _kbm_pii_masker.mask_sensitive_fields

extract_structured_data = _kbm_extractor.extract_structured_data

extract_message_fields = _kbm_comparator.extract_message_fields

detect_prompt_injection = _kbm_security.detect_prompt_injection
sanitize_document_content = _kbm_security.sanitize_document_content
has_substantive_evidence = _kbm_security.has_substantive_evidence


def as_utc(moment: Optional[datetime]) -> datetime:
    """
    Normalise any datetime to timezone-aware UTC.

    Naive timestamps are assumed to already be UTC. This mirrors the
    private helper in task6_multilingual so that simulated time travel
    and mixed naive/aware callers behave identically across modules.
    """

    if moment is None:

        return datetime.now(timezone.utc)

    if moment.tzinfo is None:

        return moment.replace(tzinfo=timezone.utc)

    return moment.astimezone(timezone.utc)


__all__ = [
    "as_utc",
    "detect_prompt_injection",
    "extract_message_fields",
    "extract_structured_data",
    "has_substantive_evidence",
    "mask_sensitive_fields",
    "mask_sensitive_text",
    "sanitize_document_content",
]
