from pathlib import Path

from file_validator import validate_filename, validate_file


TEST_DIR = Path("validator_test_files")
TEST_DIR.mkdir(exist_ok=True)


print("=" * 65)
print("TASK 5.1 - SECURE FILE VALIDATION TEST")
print("=" * 65)


# =========================================================
# TEST 1 - VALID PNG
# =========================================================

print("\nTEST 1 - VALID PNG")

png_file = TEST_DIR / "receipt.png"

# Minimal PNG signature/content for validation testing.
png_file.write_bytes(
    b"\x89PNG\r\n\x1a\n"
)

result = validate_file(str(png_file))

print(result)

assert result["valid"] is True
assert result["mime_type"] == "image/png"

print("Valid PNG accepted: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 2 - VALID PDF
# =========================================================

print("\nTEST 2 - VALID PDF")

pdf_file = TEST_DIR / "invoice.pdf"

pdf_file.write_bytes(
    b"%PDF-1.4\n"
)

result = validate_file(str(pdf_file))

print(result)

assert result["valid"] is True
assert result["mime_type"] == "application/pdf"

print("Valid PDF accepted: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 3 - DANGEROUS EXECUTABLE
# =========================================================

print("\nTEST 3 - DANGEROUS EXECUTABLE")

exe_file = TEST_DIR / "malware.exe"

exe_file.write_bytes(
    b"fake executable"
)

result = validate_file(str(exe_file))

print(result)

assert result["valid"] is False

print("Executable rejected: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 4 - PYTHON FILE
# =========================================================

print("\nTEST 4 - PYTHON FILE")

python_file = TEST_DIR / "payload.py"

python_file.write_text(
    "print('malicious payload')"
)

result = validate_file(str(python_file))

print(result)

assert result["valid"] is False

print("Python file rejected: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 5 - UNSUPPORTED FILE TYPE
# =========================================================

print("\nTEST 5 - UNSUPPORTED FILE TYPE")

txt_file = TEST_DIR / "document.txt"

txt_file.write_text(
    "This is a text file."
)

result = validate_file(str(txt_file))

print(result)

assert result["valid"] is False

print("Unsupported type rejected: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 6 - EMPTY FILE
# =========================================================

print("\nTEST 6 - EMPTY FILE")

empty_file = TEST_DIR / "empty.pdf"

empty_file.write_bytes(b"")

result = validate_file(str(empty_file))

print(result)

assert result["valid"] is False

print("Empty file rejected: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 7 - PATH TRAVERSAL
# =========================================================

print("\nTEST 7 - PATH TRAVERSAL")

result = validate_filename(
    "../secret.pdf"
)

print(result)

assert result["valid"] is False

print("Path traversal rejected: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 8 - DANGEROUS FILENAME
# =========================================================

print("\nTEST 8 - DANGEROUS FILENAME")

result = validate_filename(
    "payload.sh"
)

print(result)

assert result["valid"] is False

print("Dangerous filename rejected: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 9 - NON-EXISTENT FILE
# =========================================================

print("\nTEST 9 - NON-EXISTENT FILE")

result = validate_file(
    str(TEST_DIR / "does_not_exist.png")
)

print(result)

assert result["valid"] is False

print("Missing file rejected: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# CLEANUP
# =========================================================

for file in TEST_DIR.iterdir():
    file.unlink()

TEST_DIR.rmdir()


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 65)
print("TASK 5.1 FILE VALIDATION TEST COMPLETE")
print("=" * 65)

print("ALL FILE VALIDATION TESTS PASSED ✅")
print("=" * 65)
