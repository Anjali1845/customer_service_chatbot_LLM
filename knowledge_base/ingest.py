import json
from pathlib import Path

from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
METADATA_FILE = BASE_DIR / "metadata" / "documents.json"


REQUIRED_FIELDS = [
    "document_id",
    "filename",
    "title",
    "document_version",
    "product",
    "region",
    "access_level",
    "effective_date",
    "expiry_date",
    "status",
]


def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError(
            "documents.json must contain a JSON list."
        )

    return metadata


def validate_metadata(metadata):
    for index, item in enumerate(metadata):

        missing_fields = [
            field for field in REQUIRED_FIELDS
            if field not in item
        ]

        if missing_fields:
            raise ValueError(
                f"Document #{index + 1} is missing: "
                f"{', '.join(missing_fields)}"
            )

        for field in [
            "filename",
            "document_version",
            "product",
            "region",
            "access_level",
        ]:
            if not item[field]:
                raise ValueError(
                    f"Document #{index + 1} has no {field}."
                )

    return True


def load_documents():
    metadata_list = load_metadata()
    validate_metadata(metadata_list)

    documents = []

    for metadata in metadata_list:

        filename = metadata["filename"]
        document_path = DOCUMENTS_DIR / filename

        if not document_path.exists():
            raise FileNotFoundError(
                f"Document not found: {document_path}"
            )

        if metadata["status"].lower() != "active":
            print(f"Skipping inactive document: {filename}")
            continue

        with open(
            document_path,
            "r",
            encoding="utf-8"
        ) as file:
            content = file.read().strip()

        if not content:
            print(f"Skipping empty document: {filename}")
            continue

        document = Document(
            page_content=content,
            metadata={
                "document_id": metadata["document_id"],
                "filename": metadata["filename"],
                "title": metadata["title"],
                "document_version": metadata["document_version"],
                "product": metadata["product"],
                "region": metadata["region"],
                "access_level": metadata["access_level"],
                "effective_date": metadata["effective_date"],
                "expiry_date": metadata["expiry_date"],
                "status": metadata["status"],
            },
        )

        documents.append(document)

    return documents


if __name__ == "__main__":

    print("=" * 60)
    print("TASK 4.2 - KNOWLEDGE DOCUMENT INGESTION")
    print("=" * 60)

    try:
        documents = load_documents()

        print(
            f"\nSuccessfully loaded "
            f"{len(documents)} document(s).\n"
        )

        for index, document in enumerate(documents, start=1):

            print(f"Document {index}")
            print("-" * 40)

            for key, value in document.metadata.items():
                print(f"{key:17}: {value}")

            print(
                f"\nContent preview:\n"
                f"{document.page_content[:200]}..."
            )

            print()

        print("=" * 60)
        print("INGESTION SUCCESSFUL")
        print("=" * 60)

    except Exception as error:
        print("\n❌ INGESTION FAILED")
        print(f"Error: {error}")
