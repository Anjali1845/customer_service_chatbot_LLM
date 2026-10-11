"""
Quality gate for knowledge documents.

``validate_document`` used to check only that a file was non-empty, at
least 50 characters and not spamming the word "Refund". None of that
says anything about *accuracy*: an update could quietly replace a
grounded policy with fluent nonsense and sail through.

The gate now has a second stage, *grounding*. Given the serving corpus
(the documents the knowledge base actually answers from, or any other
source of truth) it extracts the candidate's key terms and requires
them to be represented in that corpus, and rejects an update whose
coverage falls below a baseline -- so an update that reduces what the
knowledge base can support is rejectable, not just imaginable.

Everything is deterministic token arithmetic -- no models, no network,
no dependencies beyond the standard library.

Call shapes (all optional, keyword-only, so the historical two-argument
call keeps working unchanged):

    validate_document(path)                      # structural checks only
    validate_document(path, corpus=[Path(...)])  # + grounding vs corpus
    validate_document(path, corpus=[...], baseline=0.8)   # + floor
    validate_document(path, corpus=[...],
                      baseline=Path("old.txt"))  # + previous version

``corpus`` entries are ``Path`` objects (read from disk) or ``str``
(literal text). ``baseline`` is a coverage floor as a number, or the
previous version's content as ``Path`` / ``str``.
"""

from pathlib import Path
import hashlib
import re
from typing import Iterable, Optional, Tuple, Union

MIN_LENGTH = 50

# The share of a candidate's key terms that must be represented in the
# corpus before the update is considered grounded.
GROUNDING_MIN_COVERAGE = 0.6

# Words too common to carry meaning for the grounding comparison.
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "so", "if", "then", "than",
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "into",
    "per", "via", "as", "is", "are", "was", "were", "be", "been",
    "being", "am", "has", "have", "had", "do", "does", "did", "will",
    "would", "shall", "should", "must", "may", "might", "can", "could",
    "not", "no", "nor", "only", "own", "same", "too", "very", "just",
    "also", "again", "once", "here", "there", "about", "above",
    "below", "over", "under", "between", "through", "during",
    "before", "after", "against", "up", "down", "out", "off", "all",
    "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "it", "its", "this", "that", "these", "those", "you",
    "your", "yours", "we", "our", "ours", "they", "them", "their",
    "he", "him", "his", "she", "her", "hers", "who", "whom", "whose",
    "which", "what", "when", "where", "why", "how", "i", "me", "my",
})

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def generate_hash(file_path):
    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            sha.update(chunk)

    return sha.hexdigest()


# --------------------------------------------------
# key-term extraction and grounding
# --------------------------------------------------

def _normalise_term(token: str) -> str:
    """
    One deterministic fold per token: lowercase, and a trailing plural
    "s" dropped ("refunds" and "refund" are the same claim word).
    """

    if token.isdigit() or len(token) <= 3:
        return token

    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def extract_key_terms(text: str) -> list:
    """
    The content words of ``text``, in order of first appearance.

    Deterministic and dependency-free: lowercase word runs, minus
    stopwords, single characters and lone digits, lightly de-pluralised.
    Numbers of two or more digits are kept -- "30" vs "60" days is
    exactly the kind of claim that has to stay grounded.
    """

    terms = []
    seen = set()

    for token in _TOKEN_PATTERN.findall(text.lower()):

        if token in STOPWORDS:
            continue

        if token.isdigit() and len(token) < 2:
            continue

        if not token.isdigit() and len(token) < 3:
            continue

        term = _normalise_term(token)

        if term not in seen:
            seen.add(term)
            terms.append(term)

    return terms


def _corpus_texts(corpus) -> list:
    """
    Read a corpus into plain texts.

    ``Path`` entries are read from disk (missing or unreadable ones
    are skipped); ``str`` entries are literal text.
    """

    if corpus is None:
        return []

    if isinstance(corpus, (str, Path)):
        corpus = [corpus]

    texts = []

    for entry in corpus:

        if isinstance(entry, Path):

            try:
                texts.append(entry.read_text(encoding="utf-8"))
            except OSError:
                continue

        elif isinstance(entry, str):
            texts.append(entry)

    return texts


def grounding_coverage(content: str, corpus) -> float:
    """
    The share of ``content``'s key terms represented in ``corpus``.

    1.0 when there is nothing to ground (no key terms) or no corpus to
    ground against (no usable text): those cases are decided by the
    caller, not silently scored.
    """

    terms = extract_key_terms(content)

    corpus_text = "\n".join(_corpus_texts(corpus))

    corpus_terms = set(extract_key_terms(corpus_text))

    if not terms or not corpus_terms:
        return 1.0

    supported = sum(1 for term in terms if term in corpus_terms)

    return supported / len(terms)


def _missing_terms(content: str, corpus) -> list:
    """The key terms of ``content`` absent from ``corpus``, in order."""

    terms = extract_key_terms(content)

    corpus_terms = set(extract_key_terms("\n".join(_corpus_texts(corpus))))

    return [term for term in terms if term not in corpus_terms]


def _as_text(source) -> str:
    """Read a baseline given as Path (file) or str (literal text)."""

    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")

    return source


# --------------------------------------------------
# the gate
# --------------------------------------------------

def validate_document(
    file_path,
    *,
    corpus=None,
    baseline=None,
) -> Tuple[bool, str]:
    """
    Structural quality checks, then grounding.

    Returns ``(passed, message)``. With neither ``corpus`` nor
    ``baseline`` this is exactly the historical two-argument gate.

    ``corpus``: the serving corpus / source of truth, as ``Path``s
    (files) or ``str``s (literal text). When given, the candidate's key
    terms must be represented in it to at least
    ``GROUNDING_MIN_COVERAGE``.

    ``baseline``: a required coverage floor as a number, or the
    content being replaced (``Path`` / ``str``). An update whose
    coverage falls below the baseline is rejected -- that is the
    "reduce accuracy or grounding" case.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        return False, "File not found"

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, "Document is not valid UTF-8 text"

    if len(content.strip()) == 0:
        return False, "Empty document"

    if len(content) < MIN_LENGTH:
        return False, "Document too short"

    if content.count("Refund") > 20:
        return False, "Possible duplicated content"

    # --------------------------------------------------
    # grounding stage
    # --------------------------------------------------

    texts = _corpus_texts(corpus)
    corpus_terms = set(extract_key_terms("\n".join(texts)))

    if corpus_terms:

        coverage = grounding_coverage(content, texts)

        if coverage < GROUNDING_MIN_COVERAGE:

            missing = _missing_terms(content, texts)

            return False, (
                "Insufficient grounding: "
                f"{len(missing)} of {len(extract_key_terms(content))} "
                "key terms are not represented in the serving corpus "
                f"(coverage {coverage:.0%}, minimum "
                f"{GROUNDING_MIN_COVERAGE:.0%}); missing: "
                + ", ".join(missing[:8])
            )

    else:
        coverage = 1.0

    if baseline is not None:

        if isinstance(baseline, (int, float)):
            floor = float(baseline)
            floor_label = f"{floor:.0%}"

        else:
            previous = _as_text(baseline)
            floor = grounding_coverage(previous, texts)
            floor_label = f"{floor:.0%} (previous version)"

        if coverage < floor:
            return False, (
                "Update reduces grounding coverage: "
                f"{coverage:.0%} is below the required {floor_label}"
            )

    if corpus is not None or baseline is not None:
        return True, (
            "Quality validation passed "
            f"(grounding coverage {coverage:.0%})"
        )

    return True, "Quality validation passed"


if __name__ == "__main__":
    path = input("Document path: ")

    status, message = validate_document(path)

    if status:
        print("✅ PASS:", message)
    else:
        print("❌ FAIL:", message)
