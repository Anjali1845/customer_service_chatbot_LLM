from extractor import extract_structured_data


print("=" * 65)
print("TASK 5.2 - STRUCTURED EXTRACTION TEST")
print("=" * 65)


# =========================================================
# TEST 1 - COMPLETE INVOICE
# =========================================================

print("\nTEST 1 - COMPLETE INVOICE")

text = """
Invoice
Order ID: ORD-78901
Date: 15/09/2026
Product Name: Wireless Mouse
Product Code: WM-500
Total Amount: ₹2,499.00
"""

result = extract_structured_data(text)

print(result)

assert result["order_id"] == "ORD-78901"
assert result["date"] == "2026-09-15"
assert result["amount"] == "2499.00"
assert result["product_name"] == "Wireless Mouse"
assert result["product_code"] == "WM-500"
assert result["error_code"] is None

print("RESULT: PASS ✅")


# =========================================================
# TEST 2 - ERROR CODE
# =========================================================

print("\nTEST 2 - ERROR CODE")

text = """
Support Screenshot

Order ID: ORD-45678
Error Code: ERR-503
"""

result = extract_structured_data(text)

print(result)

assert result["order_id"] == "ORD-45678"
assert result["error_code"] == "ERR-503"

print("Actual error code extracted correctly: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 3 - MISSING INFORMATION
# =========================================================

print("\nTEST 3 - MISSING INFORMATION")

text = """
Customer Support Message

Order ID: ORD-11111
Product Name: Laptop Stand
"""

result = extract_structured_data(text)

print(result)

assert result["order_id"] == "ORD-11111"
assert result["product_name"] == "Laptop Stand"
assert result["date"] is None
assert result["amount"] is None
assert result["product_code"] is None
assert result["error_code"] is None

print("Missing fields returned as None: YES ✅")
print("No information invented: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 4 - ISO DATE
# =========================================================

print("\nTEST 4 - ISO DATE")

text = """
Order ID: ORD-22222
Date: 2026-10-01
"""

result = extract_structured_data(text)

print(result)

assert result["date"] == "2026-10-01"

print("ISO date extracted correctly: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 5 - DIFFERENT CURRENCY
# =========================================================

print("\nTEST 5 - CURRENCY")

text = """
Order ID: ORD-33333
Total: $49.99
"""

result = extract_structured_data(text)

print(result)

assert result["amount"] == "49.99"

print("Currency amount extracted correctly: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 6 - PRODUCT CODE MUST NOT BECOME ERROR CODE
# =========================================================

print("\nTEST 6 - PRODUCT CODE VS ERROR CODE")

text = """
Product Name: Keyboard
Product Code: KB-900
"""

result = extract_structured_data(text)

print(result)

assert result["product_code"] == "KB-900"
assert result["error_code"] is None

print("Product code correctly separated from error code: YES ✅")
print("RESULT: PASS ✅")


# =========================================================
# TEST 7 - IRRELEVANT TEXT
# =========================================================

print("\nTEST 7 - IRRELEVANT TEXT")

text = """
Hello, I received my package yesterday.
The keyboard feels comfortable.
I like the product very much.
"""

result = extract_structured_data(text)

print(result)

assert result["order_id"] is None
assert result["date"] is None
assert result["amount"] is None
assert result["product_name"] is None
assert result["product_code"] is None
assert result["error_code"] is None

print("Unstructured text did not produce invented fields: YES ✅")
print("RESULT: PASS ✅")


print("\n" + "=" * 65)
print("TASK 5.2 STRUCTURED EXTRACTION TEST COMPLETE")
print("=" * 65)

print("ALL EXTRACTION TESTS PASSED ✅")
print("=" * 65)
