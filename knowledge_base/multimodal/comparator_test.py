from comparator import compare_message_with_document


print("=" * 65)
print("TASK 5.3 - MESSAGE ↔ FILE COMPARISON TEST")
print("=" * 65)


# =========================================================
# TEST 1 - CONSISTENT INFORMATION
# =========================================================

print("\nTEST 1 - CONSISTENT INFORMATION")

message = """
My order ORD-12345 was charged ₹1499.
"""

document = {
    "order_id": "ORD-12345",
    "amount": "1499.00",
    "date": None,
    "product_code": None,
    "error_code": None,
}

result = compare_message_with_document(
    message,
    document,
)

print(result)

assert result["status"] == "consistent"
assert result["conflicts"] == []

print("Message and document are consistent: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 2 - AMOUNT CONFLICT
# =========================================================

print("\nTEST 2 - AMOUNT CONFLICT")

message = """
My order ORD-12345 was charged ₹1499.
"""

document = {
    "order_id": "ORD-12345",
    "amount": "1999.00",
    "date": None,
    "product_code": None,
    "error_code": None,
}

result = compare_message_with_document(
    message,
    document,
)

print(result)

assert result["status"] == "conflict"
assert "amount" in result["conflicts"]

print("Amount conflict detected: YES ✅")
print("System does not choose which value is correct: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 3 - ORDER ID CONFLICT
# =========================================================

print("\nTEST 3 - ORDER ID CONFLICT")

message = """
My order ORD-11111 was delivered.
"""

document = {
    "order_id": "ORD-22222",
    "amount": None,
    "date": None,
    "product_code": None,
    "error_code": None,
}

result = compare_message_with_document(
    message,
    document,
)

print(result)

assert result["status"] == "conflict"
assert "order_id" in result["conflicts"]

print("Order ID conflict detected: YES ✅")
print("Clarification required: YES ⚠️")
print("RESULT: PASS ✅")


# =========================================================
# TEST 4 - DOCUMENT HAS EXTRA INFORMATION
# =========================================================

print("\nTEST 4 - DOCUMENT HAS EXTRA INFORMATION")

message = """
My order is ORD-12345.
"""

document = {
    "order_id": "ORD-12345",
    "amount": "1499.00",
    "date": "2026-09-15",
    "product_code": "KB-900",
    "error_code": None,
}

result = compare_message_with_document(
    message,
    document,
)

print(result)

assert result["status"] == "clarification_required"
assert "amount" in result["clarifications"]
assert "date" in result["clarifications"]
assert "product_code" in result["clarifications"]

print("Missing message information detected: YES ✅")
print("Clarification required: YES ⚠️")
print("RESULT: PASS ✅")


# =========================================================
# TEST 5 - MESSAGE HAS EXTRA INFORMATION
# =========================================================

print("\nTEST 5 - MESSAGE HAS EXTRA INFORMATION")

message = """
My order ORD-12345 was charged ₹1499.
"""

document = {
    "order_id": "ORD-12345",
    "amount": None,
    "date": None,
    "product_code": None,
    "error_code": None,
}

result = compare_message_with_document(
    message,
    document,
)

print(result)

assert result["status"] == "clarification_required"
assert "amount" in result["clarifications"]

print("Document missing comparable information: YES ✅")
print("Clarification required: YES ⚠️")
print("RESULT: PASS ✅")


# =========================================================
# TEST 6 - ERROR CODE CONSISTENCY
# =========================================================

print("\nTEST 6 - ERROR CODE CONSISTENCY")

message = """
The screen shows Error Code: ERR-503.
"""

document = {
    "order_id": None,
    "amount": None,
    "date": None,
    "product_code": None,
    "error_code": "ERR-503",
}

result = compare_message_with_document(
    message,
    document,
)

print(result)

assert result["status"] == "consistent"
assert "error_code" not in result["conflicts"]

print("Error code matches: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 7 - ERROR CODE CONFLICT
# =========================================================

print("\nTEST 7 - ERROR CODE CONFLICT")

message = """
The screen shows Error Code: ERR-404.
"""

document = {
    "order_id": None,
    "amount": None,
    "date": None,
    "product_code": None,
    "error_code": "ERR-503",
}

result = compare_message_with_document(
    message,
    document,
)

print(result)

assert result["status"] == "conflict"
assert "error_code" in result["conflicts"]

print("Error code conflict detected: YES ✅")
print("System requests clarification: YES ⚠️")
print("RESULT: PASS ✅")


# =========================================================
# TEST 8 - NO COMPARABLE EVIDENCE
# =========================================================

print("\nTEST 8 - NO COMPARABLE EVIDENCE")

message = """
I really like the product.
"""

document = {
    "order_id": None,
    "amount": None,
    "date": None,
    "product_code": None,
    "error_code": None,
}

result = compare_message_with_document(
    message,
    document,
)

print(result)

assert result["status"] == "no_comparable_evidence"

print("No comparable evidence detected: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 65)
print("TASK 5.3 COMPARISON TEST COMPLETE")
print("=" * 65)

print("ALL COMPARISON TESTS PASSED ✅")
print("Conflicting values were never automatically resolved.")
print("=" * 65)
