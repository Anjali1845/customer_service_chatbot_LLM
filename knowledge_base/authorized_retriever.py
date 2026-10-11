import hashlib
from pathlib import Path
from datetime import date
import re

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingest import load_documents


BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_PATH = BASE_DIR / "faiss_index"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


ACCESS_LEVELS = {
    "public": 1,
    "internal": 2,
    "restricted": 3,
}


# Document metadata values that apply to every product / region
# when a specific one is requested.

PRODUCT_WILDCARDS = {
    "all products",
    "all product",
    "any product",
    "all",
    "any",
    "*",
}

REGION_WILDCARDS = {
    "global",
    "all regions",
    "all region",
    "any region",
    "worldwide",
    "all",
    "any",
    "*",
}


def is_authorized(
    document_access_level: str,
    user_access_level: str
) -> bool:

    document_level = document_access_level.lower()
    user_level = user_access_level.lower()

    if document_level not in ACCESS_LEVELS:
        return False

    if user_level not in ACCESS_LEVELS:
        return False

    return ACCESS_LEVELS[user_level] >= ACCESS_LEVELS[document_level]


def parse_date(value):

    if not value:
        return None

    return date.fromisoformat(value)


def is_active_on_date(
    document,
    target_date: date
) -> bool:

    effective_date = parse_date(
        document.metadata.get("effective_date")
    )

    expiry_date = parse_date(
        document.metadata.get("expiry_date")
    )

    if effective_date is None:
        return False

    if target_date < effective_date:
        return False

    if expiry_date is not None and target_date > expiry_date:
        return False

    return True


# =========================================================
# DATE EXTRACTION FROM QUERIES
# =========================================================

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _safe_date(year: int, month: int, day: int):
    """
    Build a date, returning None when the values are not a real
    calendar date (e.g. 31 February).
    """

    try:
        return date(year, month, day)
    except ValueError:
        return None


def _month_number(name: str):
    """
    Resolve a month name or abbreviation ("March", "Mar",
    "sept.") to its number.
    """

    if not name:
        return None

    cleaned = name.strip().rstrip(".").lower()

    return MONTHS.get(cleaned)


# Recognised query date formats, tried in order. Each entry is
# (pattern, builder) where builder turns the match into a date
# (or None when the match is not a valid calendar date).

_QUERY_DATE_PATTERNS = [
    # 2026-03-15 (ISO, original behaviour)
    (
        r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",
        lambda m: _safe_date(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
        ),
    ),
    # 2026/03/15
    (
        r"\b(20\d{2})/(\d{1,2})/(\d{1,2})\b",
        lambda m: _safe_date(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
        ),
    ),
    # 15/03/2026, 15-03-2026, 03/15/2026 (US order also
    # accepted when the day-first reading is impossible)
    (
        r"\b(\d{1,2})[/](\d{1,2})[/](20\d{2})\b",
        lambda m: (
            _safe_date(
                int(m.group(3)),
                int(m.group(2)),
                int(m.group(1)),
            )
            or _safe_date(
                int(m.group(3)),
                int(m.group(1)),
                int(m.group(2)),
            )
        ),
    ),
    (
        r"\b(\d{1,2})-(\d{1,2})-(20\d{2})\b",
        lambda m: (
            _safe_date(
                int(m.group(3)),
                int(m.group(2)),
                int(m.group(1)),
            )
            or _safe_date(
                int(m.group(3)),
                int(m.group(1)),
                int(m.group(2)),
            )
        ),
    ),
    # 15 March 2026, 15th March 2026, 15 Mar 2026
    (
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?\s*,?\s*(20\d{2})\b",
        lambda m: _safe_date(
            int(m.group(3)),
            _month_number(m.group(2)) or 0,
            int(m.group(1)),
        ),
    ),
    # March 15, 2026, Mar 15 2026, March 15th, 2026
    (
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(20\d{2})\b",
        lambda m: _safe_date(
            int(m.group(3)),
            _month_number(m.group(1)) or 0,
            int(m.group(2)),
        ),
    ),
]


def extract_date_from_query(query: str):
    """
    Extract a calendar date mentioned in a query.

    Supported formats (matching is case-insensitive):

        2026-03-15              ISO, original behaviour
        2026/03/15
        15/03/2026              day-first, slashes
        15-03-2026              day-first, dashes
        03/15/2026              month-first, accepted when
                                the day-first reading is not a
                                valid calendar date
        15 March 2026
        15th March 2026
        15 Mar 2026
        March 15, 2026
        Mar 15 2026
        March 15th, 2026

    Returns a datetime.date, or None when the query contains no
    recognisable date.
    """

    if not query:
        return None

    text = query.strip()

    for pattern, build in _QUERY_DATE_PATTERNS:

        match = re.search(pattern, text, flags=re.IGNORECASE)

        if not match:
            continue

        parsed = build(match)

        if parsed is not None:
            return parsed

    return None


# =========================================================
# AUTHORIZATION + DATE + PRODUCT/REGION FILTERING
# =========================================================

def _normalize(value):
    """
    Normalise a metadata value for case-insensitive comparison.
    """

    return (value or "").strip().lower()


def _matches_metadata(
    requested_value,
    document_value,
    wildcards
):
    """
    Whether a document's product/region value covers a requested
    product/region filter.

    A document applies when its own value matches the request
    exactly (case-insensitive) or when its value is a wildcard
    such as "All Products" or "GLOBAL". A filter of None means
    "no filtering" and always matches.
    """

    if not requested_value:
        return True

    document_value_norm = _normalize(document_value)

    if document_value_norm in wildcards:
        return True

    return document_value_norm == _normalize(requested_value)


def get_authorized_documents(
    user_access_level: str = "public"
):

    all_documents = load_documents()

    return [
        document
        for document in all_documents
        if is_authorized(
            document.metadata.get("access_level", ""),
            user_access_level
        )
    ]


def get_applicable_documents(
    user_access_level: str = "public",
    target_date: date | None = None,
    product: str | None = None,
    region: str | None = None
):
    """
    Documents the user may see that are in force on the target
    date and that match the optional product/region filters.

    ``product``/``region`` are case-insensitive; documents whose
    own product/region is a wildcard ("All Products", "GLOBAL")
    match any request. Both default to None, meaning "no
    filter" (the original behaviour).
    """

    if target_date is None:
        target_date = date.today()

    authorized_documents = get_authorized_documents(
        user_access_level
    )

    return [
        document
        for document in authorized_documents
        if is_active_on_date(
            document,
            target_date
        )
        and _matches_metadata(
            product,
            document.metadata.get("product", ""),
            PRODUCT_WILDCARDS
        )
        and _matches_metadata(
            region,
            document.metadata.get("region", ""),
            REGION_WILDCARDS
        )
    ]


def _select_latest_by_key(documents):
    """
    Original conflict resolution: keep the latest-effective
    version per (title, product, region).
    """

    selected = {}

    for document in documents:

        metadata = document.metadata

        key = (
            metadata.get("title", "").lower(),
            metadata.get("product", "").lower(),
            metadata.get("region", "").lower(),
        )

        effective_date = parse_date(
            metadata.get("effective_date")
        )

        if key not in selected:
            selected[key] = document
            continue

        existing_effective_date = parse_date(
            selected[key].metadata.get("effective_date")
        )

        if (
            effective_date is not None
            and (
                existing_effective_date is None
                or effective_date > existing_effective_date
            )
        ):
            selected[key] = document

    return list(selected.values())


def select_latest_applicable_documents(
    documents,
    preferred_product: str | None = None,
    preferred_region: str | None = None
):
    """
    Resolve conflicting versions of a document title.

    Without preferences this is the original behaviour: the
    latest-effective version per (title, product, region) key.

    When a ``preferred_product``/``preferred_region`` is given
    and a title exists in both a wildcard and a matching
    specific variant (for example an "All Products" refund
    policy and an Aurora-specific one), the document matching
    the requested product/region wins. Documents that match
    neither exactly are still kept if nothing matches exactly,
    so a preference never drops evidence it cannot improve.
    """

    if not preferred_product and not preferred_region:
        return _select_latest_by_key(documents)

    by_title = {}

    for document in documents:

        title = document.metadata.get("title", "").lower()

        by_title.setdefault(title, []).append(document)

    selected = []

    for title, candidates in by_title.items():

        narrowed = candidates

        if preferred_product:

            exact = [
                document
                for document in narrowed
                if _normalize(
                    document.metadata.get("product", "")
                ) == _normalize(preferred_product)
            ]

            if exact:
                narrowed = exact

        if preferred_region:

            exact = [
                document
                for document in narrowed
                if _normalize(
                    document.metadata.get("region", "")
                ) == _normalize(preferred_region)
            ]

            if exact:
                narrowed = exact

        selected.extend(
            _select_latest_by_key(narrowed)
        )

    return selected


# =========================================================
# FAISS INDEX CACHE
# =========================================================

# Cache of FAISS indexes over the applicable document set, so
# the index is not rebuilt on every query. Keyed on the
# applicable document set identity: access level, target date
# and product/region filters. The stored fingerprint (a hash of
# the selected documents' ids, versions, dates, product,
# region and content) makes the entry self-invalidating when
# the underlying documents change, and invalidate_cache()
# forces a full refresh for update pipelines that activate or
# rewrite documents.

_INDEX_CACHE = {}
_INDEX_CACHE_MAX_ENTRIES = 64
_INDEX_CACHE_STATS = {"builds": 0, "hits": 0, "invalidations": 0}


def _documents_fingerprint(documents):
    """
    Stable fingerprint of an applicable document set.
    """

    hasher = hashlib.sha256()

    for document in documents:

        metadata = document.metadata

        for field in (
            "document_id",
            "document_version",
            "effective_date",
            "product",
            "region",
            "access_level",
        ):
            hasher.update(
                str(metadata.get(field, "")).encode("utf-8")
            )
            hasher.update(b"\x1f")

        hasher.update(
            document.page_content.encode("utf-8")
        )
        hasher.update(b"\x1e")

    return hasher.hexdigest()


def _get_cached_index(
    cache_key,
    applicable_documents
):
    """
    Return a FAISS index over the applicable documents, reusing
    the cached one when the applicable document set is
    unchanged. Retrieval results are identical to an uncached
    rebuild; only the rebuild is avoided.
    """

    fingerprint = _documents_fingerprint(
        applicable_documents
    )

    cached = _INDEX_CACHE.get(cache_key)

    if cached is not None and cached[0] == fingerprint:
        _INDEX_CACHE_STATS["hits"] += 1
        return cached[1]

    index = FAISS.from_documents(
        documents=applicable_documents,
        embedding=embeddings
    )

    if len(_INDEX_CACHE) >= _INDEX_CACHE_MAX_ENTRIES:
        _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))

    _INDEX_CACHE[cache_key] = (fingerprint, index)

    _INDEX_CACHE_STATS["builds"] += 1

    return index


def invalidate_cache():
    """
    Force the per-query FAISS index cache to be rebuilt.

    Call this after the knowledge base changes on disk (an
    update pipeline activating, expiring, rewriting or deleting
    documents). The next retrieve_authorized_documents call
    rebuilds its index from the freshly loaded documents. The
    cache also self-invalidates whenever a cached entry's
    document-set fingerprint no longer matches, so this call is
    a safety net rather than a requirement.

    Returns the number of cache entries that were dropped.
    """

    dropped = len(_INDEX_CACHE)

    _INDEX_CACHE.clear()

    _INDEX_CACHE_STATS["invalidations"] += dropped

    return dropped


def get_cache_stats():
    """
    Cache observability for tests and the update pipeline:

        builds        index constructions (cache misses)
        hits          queries served from the cache
        invalidations entries dropped by invalidate_cache()
        entries       cache entries currently held
    """

    return {
        "builds": _INDEX_CACHE_STATS["builds"],
        "hits": _INDEX_CACHE_STATS["hits"],
        "invalidations": _INDEX_CACHE_STATS["invalidations"],
        "entries": len(_INDEX_CACHE),
    }


# =========================================================
# VECTOR DATABASE
# =========================================================

def create_authorized_vector_db():

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No active knowledge documents were found."
        )

    db = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    db.save_local(
        str(VECTOR_DB_PATH)
    )

    print(
        f"Knowledge-base vector database created at: "
        f"{VECTOR_DB_PATH}"
    )

    print(
        f"Documents indexed: {len(documents)}"
    )


def load_vector_db():

    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            "Task 4 FAISS database does not exist."
        )

    return FAISS.load_local(
        str(VECTOR_DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_authorized_documents(
    query: str,
    user_access_level: str = "public",
    k: int = 3,
    target_date: date | None = None,
    product: str | None = None,
    region: str | None = None
):
    """
    Authorised, date-aware, product/region-aware semantic
    retrieval.

    ``product`` and ``region`` are optional case-insensitive
    filters (default None = no filtering). A document matches a
    filter when its own product/region equals the requested
    value or is a wildcard such as "All Products" or "GLOBAL".
    When both a wildcard and a specific document share a title,
    the one matching the requested product/region is preferred.
    """

    if not query or not query.strip():
        return []

    user_access_level = user_access_level.lower()

    if user_access_level not in ACCESS_LEVELS:
        raise ValueError(
            f"Unknown user access level: "
            f"{user_access_level}"
        )

    if k < 1:
        raise ValueError("k must be at least 1.")

    # -------------------------------------------------
    # Detect historical date from the question
    # -------------------------------------------------

    query_date = extract_date_from_query(query)

    if query_date is not None:
        target_date = query_date

    if target_date is None:
        target_date = date.today()

    # -------------------------------------------------
    # SECURITY:
    # Authorization happens FIRST.
    # -------------------------------------------------

    applicable_documents = get_applicable_documents(
        user_access_level=user_access_level,
        target_date=target_date,
        product=product,
        region=region
    )

    # -------------------------------------------------
    # Select latest applicable policy version, preferring
    # the requested product/region on conflicts.
    # -------------------------------------------------

    applicable_documents = (
        select_latest_applicable_documents(
            applicable_documents,
            preferred_product=product,
            preferred_region=region
        )
    )

    if not applicable_documents:
        return []

    # -------------------------------------------------
    # Semantic retrieval ONLY over documents that passed
    # authorization + date + product/region filtering.
    # The FAISS index is cached per applicable document
    # set, so repeated queries do not rebuild it.
    # -------------------------------------------------

    cache_key = (
        user_access_level,
        target_date,
        _normalize(product),
        _normalize(region),
    )

    authorized_db = _get_cached_index(
        cache_key,
        applicable_documents
    )

    results = authorized_db.similarity_search(
        query,
        k=min(k, len(applicable_documents))
    )

    return results


if __name__ == "__main__":

    print("=" * 60)
    print("TASK 4.4 - DATE-AWARE POLICY RETRIEVAL")
    print("=" * 60)

    create_authorized_vector_db()

    # -------------------------------------------------
    # TEST 1: CURRENT POLICY
    # -------------------------------------------------

    query = "What is the refund policy?"

    print("\nTEST 1 - CURRENT POLICY")
    print(f"Query: {query}")

    results = retrieve_authorized_documents(
        query=query,
        user_access_level="public",
        k=3,
        target_date=date(2026, 10, 1)
    )

    for document in results:

        print(
            f"  {document.metadata['document_id']} | "
            f"Version "
            f"{document.metadata['document_version']} | "
            f"Effective "
            f"{document.metadata['effective_date']}"
        )

    # -------------------------------------------------
    # TEST 2: HISTORICAL POLICY
    # -------------------------------------------------

    query = (
        "What was the refund policy on 2026-03-15?"
    )

    print("\nTEST 2 - HISTORICAL POLICY")
    print(f"Query: {query}")

    results = retrieve_authorized_documents(
        query=query,
        user_access_level="public",
        k=3
    )

    for document in results:

        print(
            f"  {document.metadata['document_id']} | "
            f"Version "
            f"{document.metadata['document_version']} | "
            f"Effective "
            f"{document.metadata['effective_date']} | "
            f"Expiry "
            f"{document.metadata['expiry_date']}"
        )

    # -------------------------------------------------
    # TEST 3: DATE FILTERING
    # -------------------------------------------------

    print("\nTEST 3 - DATE FILTERING")

    results = get_applicable_documents(
        user_access_level="public",
        target_date=date(2026, 5, 1)
    )

    print(
        "Applicable documents on 2026-05-01:"
    )

    for document in results:

        print(
            f"  {document.metadata['document_id']} | "
            f"{document.metadata['title']} | "
            f"effective="
            f"{document.metadata['effective_date']} | "
            f"expiry="
            f"{document.metadata['expiry_date']}"
        )

    print("\n" + "=" * 60)
    print("TASK 4.4 TEST COMPLETE")
    print("=" * 60)
