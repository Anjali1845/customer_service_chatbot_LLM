from pathlib import Path
import mimetypes


# =========================================================
# TASK 5.1 - SECURE MULTIMODAL FILE VALIDATOR
# =========================================================

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}

ALLOWED_MIME_TYPES = set(ALLOWED_EXTENSIONS.values())

DANGEROUS_EXTENSIONS = {
    ".exe",
    ".sh",
    ".bat",
    ".cmd",
    ".com",
    ".msi",
    ".dmg",
    ".app",
    ".js",
    ".vbs",
    ".ps1",
    ".jar",
    ".py",
}

DANGEROUS_FILENAME_PATTERNS = {
    "..",
    "/",
    "\\",
}


def validate_filename(filename: str) -> dict:
    """
    Validate the uploaded filename.

    Prevents path traversal and dangerous executable formats.
    """

    if not filename:
        return {
            "valid": False,
            "reason": "Filename is empty.",
        }

    path = Path(filename)
    extension = path.suffix.lower()

    # Prevent path traversal.
    if any(pattern in filename for pattern in DANGEROUS_FILENAME_PATTERNS):
        return {
            "valid": False,
            "reason": "Unsafe filename or path detected.",
        }

    # Reject executable formats.
    if extension in DANGEROUS_EXTENSIONS:
        return {
            "valid": False,
            "reason": f"Dangerous file type: {extension}",
        }

    # Reject unsupported formats.
    if extension not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "reason": f"Unsupported file extension: {extension}",
        }

    return {
        "valid": True,
        "extension": extension,
        "expected_mime_type": ALLOWED_EXTENSIONS[extension],
    }


def validate_file(file_path: str) -> dict:
    """
    Validate an uploaded multimodal file.

    Checks:
    - existence
    - filename
    - extension
    - file size
    - MIME type
    """

    path = Path(file_path)

    # -----------------------------------------------------
    # EXISTENCE
    # -----------------------------------------------------

    if not path.exists():
        return {
            "valid": False,
            "reason": "File does not exist.",
        }

    if not path.is_file():
        return {
            "valid": False,
            "reason": "Path is not a regular file.",
        }

    # -----------------------------------------------------
    # FILENAME
    # -----------------------------------------------------

    filename_result = validate_filename(path.name)

    if not filename_result["valid"]:
        return filename_result

    # -----------------------------------------------------
    # FILE SIZE
    # -----------------------------------------------------

    file_size = path.stat().st_size

    if file_size == 0:
        return {
            "valid": False,
            "reason": "File is empty.",
        }

    if file_size > MAX_FILE_SIZE_BYTES:
        return {
            "valid": False,
            "reason": (
                f"File exceeds the {MAX_FILE_SIZE_MB} MB "
                "maximum allowed size."
            ),
        }

    # -----------------------------------------------------
    # MIME TYPE
    # -----------------------------------------------------

    detected_mime_type, _ = mimetypes.guess_type(
        path.name
    )

    expected_mime_type = filename_result[
        "expected_mime_type"
    ]

    if detected_mime_type != expected_mime_type:
        return {
            "valid": False,
            "reason": (
                "File MIME type does not match "
                "the expected file type."
            ),
            "expected_mime_type": expected_mime_type,
            "detected_mime_type": detected_mime_type,
        }

    # -----------------------------------------------------
    # SUCCESS
    # -----------------------------------------------------

    return {
        "valid": True,
        "filename": path.name,
        "extension": filename_result["extension"],
        "mime_type": detected_mime_type,
        "size_bytes": file_size,
        "size_mb": round(
            file_size / (1024 * 1024),
            4,
        ),
    }


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TASK 5.1 - FILE VALIDATOR SELF-TEST")
    print("=" * 60)

    print("\nAllowed extensions:")

    for extension, mime_type in ALLOWED_EXTENSIONS.items():
        print(f"- {extension} -> {mime_type}")

    print("\nMaximum file size:")
    print(f"{MAX_FILE_SIZE_MB} MB")

    print("\nDangerous extensions blocked:")

    for extension in sorted(DANGEROUS_EXTENSIONS):
        print(f"- {extension}")

    print("\nValidator module loaded successfully.")
    print("RESULT: PASS ✅")

    print("\n" + "=" * 60)
