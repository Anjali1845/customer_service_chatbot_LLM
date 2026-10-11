"""
SECURE ANSWER BRIDGE

The seam between a chat turn and the authorised RAG layer.

Before this module the application answered through a FAISS
RetrievalQA over ``dataset/dataset.csv`` -- a chain with no metadata,
no access control, no effective-date logic and no citations. The
whole RAG layer in ``knowledge_base/`` (authorised, date-aware,
citation-carrying) existed but nothing in ``src/`` could reach it.

This bridge is what reaches it. It is deliberately tolerant: if the
knowledge-base index or its retrieval dependencies are not available
on this machine, it returns ``None`` and the caller falls back to the
legacy chain, so the app still starts on a bare checkout.

``answer_fn`` is the shape ``SupportFlow`` expects:
``Callable[[question, tone_instruction], answer]`` (see
``support_flow.AnswerFn``).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Callable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"


def _ensure_importable() -> None:
    """
    ``knowledge_base/`` modules import each other by bare name
    (``from authorized_retriever import ...``), so the directory has
    to be on ``sys.path`` for them to load at all.
    """

    if KNOWLEDGE_BASE.is_dir():

        entry = str(KNOWLEDGE_BASE)

        if entry not in sys.path:

            sys.path.insert(0, entry)


def _supported(module, name: str, *candidates: str) -> dict:
    """
    The subset of ``candidates`` that ``module.name`` actually accepts.

    Lets this bridge pass ``tone_instruction``/``product``/``region``
    when the retrieval layer supports them and quietly omit them when
    it does not, instead of depending on one exact signature.
    """

    function = getattr(module, name, None)

    if function is None:

        return {}

    try:

        accepted = inspect.signature(function).parameters

    except (TypeError, ValueError):

        return {}

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in accepted.values()
    ):

        return {candidate: True for candidate in candidates}

    return {
        candidate: candidate in accepted for candidate in candidates
    }


def answer_with_citations(
    question: str,
    tone_instruction: str = "",
    user_access_level: str = "public",
    k: int = 3,
    product: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[str]:
    """
    Answer through the authorised, citation-carrying RAG layer.

    Returns ``None`` -- never raises -- when the layer is unavailable,
    so the caller can fall back rather than fail the turn.
    """

    _ensure_importable()

    try:

        import qa_with_citations

    except Exception:  # noqa: BLE001 - unavailable, fall back

        return None

    extras = _supported(
        qa_with_citations,
        "answer_question",
        "tone_instruction",
        "product",
        "region",
    )

    kwargs = {
        "user_access_level": user_access_level,
        "k": k,
    }

    if extras.get("tone_instruction") and tone_instruction:

        kwargs["tone_instruction"] = tone_instruction

    if extras.get("product") and product:

        kwargs["product"] = product

    if extras.get("region") and region:

        kwargs["region"] = region

    try:

        result = qa_with_citations.answer_question(question, **kwargs)

    except Exception:  # noqa: BLE001 - unavailable, fall back

        return None

    if not isinstance(result, dict):

        return None

    answer = result.get("answer") or ""

    sources = result.get("sources")

    # The citation block may already be embedded in the answer by the
    # retrieval layer; only append it when it is not.
    if isinstance(sources, str) and sources.strip():

        if sources.strip() not in answer:

            answer = f"{answer}\n\n{sources}"

    elif isinstance(sources, list) and sources:

        rendered = "\n".join(str(item) for item in sources)

        if rendered not in answer:

            answer = f"{answer}\n\nSources:\n{rendered}"

    return answer or None


def build_answer_fn(
    user_access_level: str = "public",
    k: int = 3,
    product: Optional[str] = None,
    region: Optional[str] = None,
) -> Callable[[str, str], str]:
    """
    The ``answer_fn`` for ``SupportFlow``.

    Tries the authorised RAG first and falls back to the legacy CSV
    chain, so an answer is always produced -- but a grounded,
    authorised, cited one whenever the knowledge base is reachable.
    """

    def _answer(question: str, tone_instruction: str = "") -> str:

        secured = answer_with_citations(
            question,
            tone_instruction=tone_instruction,
            user_access_level=user_access_level,
            k=k,
            product=product,
            region=region,
        )

        if secured is not None:

            return secured

        from langchain_helper import get_qa_chain

        return get_qa_chain(
            tone_instruction=tone_instruction
        ).invoke({"query": question})["result"]

    return _answer
