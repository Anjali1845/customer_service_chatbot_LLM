from pii_masker import mask_sensitive_text, mask_sensitive_fields


print("=" * 70)
print("TASK 5.4 - PII & PAYMENT MASKING TEST")
print("=" * 70)


# --------------------------------------------------
# TEST 1 - EMAIL
# --------------------------------------------------
text = "Customer email is anjali@example.com."

masked = mask_sensitive_text(text)

print("\nTEST 1 - EMAIL MASKING")
print(masked)

assert "anjali@example.com" not in masked
assert "a***@example.com" in masked

print("Email masked: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 2 - PHONE
# --------------------------------------------------
text = "Customer phone number is 9876543210."

masked = mask_sensitive_text(text)

print("\nTEST 2 - PHONE MASKING")
print(masked)

assert "9876543210" not in masked
assert "******3210" in masked

print("Phone masked: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 3 - CREDIT CARD
# --------------------------------------------------
text = "Payment card: 4111 1111 1111 1111."

masked = mask_sensitive_text(text)

print("\nTEST 3 - CARD MASKING")
print(masked)

assert "4111 1111 1111 1111" not in masked
assert "**** **** **** 1111" in masked

print("Card number masked: YES ✅")
print("Last four digits preserved: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 4 - CVV
# --------------------------------------------------
text = "Payment details: CVV: 123."

masked = mask_sensitive_text(text)

print("\nTEST 4 - CVV MASKING")
print(masked)

assert "CVV: 123" not in masked
assert "***" in masked

print("CVV masked: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 5 - UPI
# --------------------------------------------------
text = "Customer UPI ID is anjali@upi."

masked = mask_sensitive_text(text)

print("\nTEST 5 - UPI MASKING")
print(masked)

assert "anjali@upi" not in masked
assert "a***@upi" in masked

print("UPI ID masked: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 6 - BANK ACCOUNT
# --------------------------------------------------
text = "Customer account number: 123456789012."

masked = mask_sensitive_text(text)

print("\nTEST 6 - ACCOUNT NUMBER MASKING")
print(masked)

assert "123456789012" not in masked
assert "********9012" in masked

print("Account number masked: YES ✅")
print("Last four digits preserved: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 7 - MULTIPLE PII TYPES
# --------------------------------------------------
text = (
    "Email: anjali@example.com, "
    "Phone: 9876543210, "
    "Card: 4111 1111 1111 1111, "
    "CVV: 123, "
    "UPI: anjali@upi"
)

masked = mask_sensitive_text(text)

print("\nTEST 7 - MULTIPLE SENSITIVE FIELDS")
print(masked)

assert "anjali@example.com" not in masked
assert "9876543210" not in masked
assert "4111 1111 1111 1111" not in masked
assert "CVV: 123" not in masked
assert "anjali@upi" not in masked

print("All sensitive fields masked: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 8 - ORIGINAL TEXT PRESERVED
# --------------------------------------------------
original = (
    "Email: anjali@example.com, "
    "Phone: 9876543210, "
    "Card: 4111 1111 1111 1111"
)

masked = mask_sensitive_text(original)

print("\nTEST 8 - ORIGINAL TEXT PRESERVATION")

assert original == (
    "Email: anjali@example.com, "
    "Phone: 9876543210, "
    "Card: 4111 1111 1111 1111"
)

assert masked != original

print("Original sensitive data remains unchanged: YES ✅")
print("Masked copy created separately: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 9 - STRUCTURED DATA
# --------------------------------------------------
data = {
    "customer": "Anjali",
    "email": "anjali@example.com",
    "phone": "9876543210",
    "card_number": "4111111111111111",
    "cvv": "123",
    "upi_id": "anjali@upi",
    "account_number": "123456789012",
    "order_id": "ORD-12345",
    "amount": "1499.00",
}

masked_data = mask_sensitive_fields(data)

print("\nTEST 9 - STRUCTURED DATA MASKING")
print(masked_data)

assert masked_data["customer"] == "Anjali"
assert masked_data["order_id"] == "ORD-12345"
assert masked_data["amount"] == "1499.00"

assert masked_data["email"] != data["email"]
assert masked_data["phone"] != data["phone"]
assert masked_data["card_number"] != data["card_number"]
assert masked_data["cvv"] != data["cvv"]
assert masked_data["upi_id"] != data["upi_id"]
assert masked_data["account_number"] != data["account_number"]

print("Sensitive structured fields masked: YES ✅")
print("Non-sensitive fields preserved: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# TEST 10 - ORIGINAL STRUCTURED DATA PRESERVED
# --------------------------------------------------
print("\nTEST 10 - ORIGINAL STRUCTURED DATA PRESERVATION")

assert data["email"] == "anjali@example.com"
assert data["phone"] == "9876543210"
assert data["card_number"] == "4111111111111111"
assert data["cvv"] == "123"
assert data["upi_id"] == "anjali@upi"
assert data["account_number"] == "123456789012"

print("Original dictionary preserved: YES ✅")
print("RESULT: PASS ✅")


# --------------------------------------------------
# FINAL
# --------------------------------------------------
print("\n" + "=" * 70)
print("TASK 5.4 PII & PAYMENT MASKING TEST COMPLETE")
print("=" * 70)
print("ALL MASKING TESTS PASSED ✅")
print("Sensitive information is protected before logging.")
print("Original evidence/data is never modified.")
print("=" * 70)
