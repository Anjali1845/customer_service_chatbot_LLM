from pathlib import Path
import hashlib

MIN_LENGTH = 50


def generate_hash(file_path):
    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            sha.update(chunk)

    return sha.hexdigest()


def validate_document(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        return False, "File not found"

    content = file_path.read_text(encoding="utf-8")

    if len(content.strip()) == 0:
        return False, "Empty document"

    if len(content) < MIN_LENGTH:
        return False, "Document too short"

    if content.count("Refund") > 20:
        return False, "Possible duplicated content"

    return True, "Quality validation passed"


if __name__ == "__main__":
    path = input("Document path: ")

    status, message = validate_document(path)

    if status:
        print("✅ PASS:", message)
    else:
        print("❌ FAIL:", message)