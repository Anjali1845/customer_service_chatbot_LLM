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


def extract_date_from_query(query: str):

    match = re.search(
        r"\b(20\d{2}-\d{2}-\d{2})\b",
        query
    )

    if not match:
        return None

    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


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
    target_date: date | None = None
):

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
    ]


def select_latest_applicable_documents(
    documents
):

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
    target_date: date | None = None
):

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
        target_date=target_date
    )

    # -------------------------------------------------
    # Select latest applicable policy version.
    # -------------------------------------------------

    applicable_documents = (
        select_latest_applicable_documents(
            applicable_documents
        )
    )

    if not applicable_documents:
        return []

    # -------------------------------------------------
    # Semantic retrieval ONLY over documents that
    # passed authorization + date filtering.
    # -------------------------------------------------

    authorized_db = FAISS.from_documents(
        documents=applicable_documents,
        embedding=embeddings
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
