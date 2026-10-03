"""
Task 5.4 - PII and Payment Information Masking

Masks sensitive information before data is written to logs.
The original evidence/data is never modified.
"""

import re
from typing import Dict, Any


# -----------------------------
# Masking helpers
# -----------------------------

def mask_email(value: str) -> str:
    """Mask the local part of an email address."""
    if not value:
        return value

    parts = value.split("@", 1)
    if len(parts) != 2:
        return value

    local, domain = parts

    if len(local) <= 1:
        masked_local = "*"
    elif len(local) == 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[0] + "***"

    return f"{masked_local}@{domain}"


def mask_phone(value: str) -> str:
    """Keep only the last four digits of a phone number."""
    digits = re.sub(r"\D", "", value)

    if len(digits) < 4:
        return "*" * len(digits)

    return "*" * (len(digits) - 4) + digits[-4:]


def mask_card(value: str) -> str:
    """Keep only the last four digits of a payment card."""
    digits = re.sub(r"\D", "", value)

    if len(digits) < 12:
        return "*" * len(digits)

    masked = "*" * (len(digits) - 4) + digits[-4:]

    # Preserve grouped card formatting where possible.
    if " " in value:
        return " ".join(
            masked[i:i + 4] for i in range(0, len(masked), 4)
        )

    if "-" in value:
        return "-".join(
            masked[i:i + 4] for i in range(0, len(masked), 4)
        )

    return masked


def mask_cvv(value: str) -> str:
    """Mask CVV/security code completely."""
    return "*" * len(re.sub(r"\D", "", value))


def mask_upi(value: str) -> str:
    """Mask the user portion of a UPI ID."""
    if not value or "@" not in value:
        return value

    user, provider = value.split("@", 1)

    if len(user) <= 1:
        masked_user = "*"
    else:
        masked_user = user[0] + "***"

    return f"{masked_user}@{provider}"


def mask_account_number(value: str) -> str:
    """Mask bank/account numbers while preserving the last four digits."""
    digits = re.sub(r"\D", "", value)

    if len(digits) <= 4:
        return "*" * len(digits)

    return "*" * (len(digits) - 4) + digits[-4:]


# -----------------------------
# Text masking
# -----------------------------

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
)

CARD_PATTERN = re.compile(
    r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
)

CVV_PATTERN = re.compile(
    r"(?i)\b(?:cvv|cvc|security\s*code)\s*[:#-]?\s*\d{3,4}\b"
)

UPI_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z][a-zA-Z0-9._-]{1,}\b"
)

ACCOUNT_PATTERN = re.compile(
    r"(?i)\b(?:account\s*(?:number|no)|a/c)\s*[:#-]?\s*\d{8,18}\b"
)


def mask_sensitive_text(text: str) -> str:
    """
    Mask sensitive information in text.

    This function returns a NEW string.
    The original text remains unchanged.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    masked = text

    # CVV first so its digits are not processed as other fields.
    def replace_cvv(match):
        original = match.group(0)
        digits = re.search(r"\d{3,4}", original)
        if not digits:
            return original

        return original[:digits.start()] + "*" * len(digits.group()) + original[digits.end():]

    masked = CVV_PATTERN.sub(replace_cvv, masked)

    # Account numbers with labels.
    def replace_account(match):
        original = match.group(0)
        digits = re.search(r"\d{8,18}", original)
        if not digits:
            return original

        masked_number = mask_account_number(digits.group())
        return original[:digits.start()] + masked_number + original[digits.end():]

    masked = ACCOUNT_PATTERN.sub(replace_account, masked)

    # Email addresses.
    masked = EMAIL_PATTERN.sub(
        lambda m: mask_email(m.group(0)),
        masked
    )

    # UPI IDs.
    masked = UPI_PATTERN.sub(
        lambda m: mask_upi(m.group(0)),
        masked
    )

    # Indian phone numbers.
    masked = PHONE_PATTERN.sub(
        lambda m: mask_phone(m.group(0)),
        masked
    )

    # Payment cards.
    masked = CARD_PATTERN.sub(
        lambda m: mask_card(m.group(0)),
        masked
    )

    return masked


# -----------------------------
# Structured-data masking
# -----------------------------

SENSITIVE_FIELDS = {
    "email",
    "phone",
    "phone_number",
    "mobile",
    "mobile_number",
    "card",
    "card_number",
    "credit_card",
    "debit_card",
    "cvv",
    "cvc",
    "security_code",
    "upi",
    "upi_id",
    "account_number",
    "bank_account",
}


def mask_sensitive_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a masked COPY of a dictionary.

    Original dictionary is never modified.
    """

    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    masked = {}

    for key, value in data.items():
        normalized_key = str(key).lower().strip()

        if normalized_key in SENSITIVE_FIELDS:
            if value is None:
                masked[key] = None
            elif normalized_key in {"email"}:
                masked[key] = mask_email(str(value))
            elif normalized_key in {
                "phone",
                "phone_number",
                "mobile",
                "mobile_number",
            }:
                masked[key] = mask_phone(str(value))
            elif normalized_key in {
                "card",
                "card_number",
                "credit_card",
                "debit_card",
            }:
                masked[key] = mask_card(str(value))
            elif normalized_key in {"cvv", "cvc", "security_code"}:
                masked[key] = "*" * len(str(value))
            elif normalized_key in {"upi", "upi_id"}:
                masked[key] = mask_upi(str(value))
            elif normalized_key in {
                "account_number",
                "bank_account",
            }:
                masked[key] = mask_account_number(str(value))
            else:
                masked[key] = mask_sensitive_text(str(value))
        else:
            masked[key] = value

    return masked


# -----------------------------
# Self-test
# -----------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("TASK 5.4 - PII & PAYMENT MASKING SELF-TEST")
    print("=" * 65)

    sample = (
        "Customer email: anjali@example.com, "
        "phone: 9876543210, "
        "card: 4111 1111 1111 1111, "
        "CVV: 123, "
        "UPI: anjali@upi"
    )

    print("\nOriginal:")
    print(sample)

    masked = mask_sensitive_text(sample)

    print("\nMasked:")
    print(masked)

    assert "anjali@example.com" not in masked
    assert "9876543210" not in masked
    assert "4111 1111 1111 1111" not in masked
    assert "123" not in masked
    assert "anjali@upi" not in masked

    # Verify original remains unchanged.
    assert "anjali@example.com" in sample
    assert "9876543210" in sample

    structured = {
        "customer": "Anjali",
        "email": "anjali@example.com",
        "phone": "9876543210",
        "card_number": "4111111111111111",
        "cvv": "123",
        "upi_id": "anjali@upi",
        "order_id": "ORD-12345",
    }

    masked_structured = mask_sensitive_fields(structured)

    print("\nStructured data:")
    print(structured)

    print("\nMasked structured data:")
    print(masked_structured)

    assert masked_structured["customer"] == "Anjali"
    assert masked_structured["order_id"] == "ORD-12345"
    assert masked_structured["email"] != structured["email"]
    assert masked_structured["phone"] != structured["phone"]
    assert masked_structured["card_number"] != structured["card_number"]
    assert masked_structured["cvv"] != structured["cvv"]
    assert masked_structured["upi_id"] != structured["upi_id"]

    # Original dictionary must remain unchanged.
    assert structured["email"] == "anjali@example.com"
    assert structured["phone"] == "9876543210"
    assert structured["card_number"] == "4111111111111111"

    print("\nOriginal data preserved: YES")
    print("Sensitive information masked: YES")
    print("Task 5.4 masking self-test: PASS")
    print("=" * 65)
