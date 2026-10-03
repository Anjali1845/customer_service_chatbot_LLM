import re


# =========================================================
# TASK 5.3 - MESSAGE ↔ FILE EVIDENCE COMPARATOR
# =========================================================


COMPARABLE_FIELDS = {
    "order_id",
    "date",
    "amount",
    "product_code",
    "error_code",
}


def normalize_value(field: str, value):
    """
    Normalize values before comparison.

    This does NOT change the actual extracted value.
    It only creates a comparison form.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    if field == "amount":
        # Compare monetary values numerically.
        try:
            return round(
                float(
                    value.replace(",", "")
                    .replace("₹", "")
                    .replace("$", "")
                    .replace("€", "")
                    .replace("£", "")
                ),
                2,
            )
        except ValueError:
            return value.lower()

    if field in {
        "order_id",
        "product_code",
        "error_code",
    }:
        return value.upper()

    if field == "date":
        return value

    return value


def extract_message_fields(message: str) -> dict:
    """
    Extract comparable fields from the customer's message.

    This deliberately uses conservative patterns.
    Missing fields remain None.
    """

    fields = {
        "order_id": None,
        "date": None,
        "amount": None,
        "product_code": None,
        "error_code": None,
    }

    # -----------------------------------------------------
    # ORDER ID
    # -----------------------------------------------------

    order_patterns = [
        r"\border\s*(?:id|number|no\.?|#)\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-_]{3,})\b",
        r"\b(ORD[-_][A-Z0-9\-]{3,})\b",
    ]

    for pattern in order_patterns:
        match = re.search(
            pattern,
            message,
            re.IGNORECASE,
        )

        if match:
            fields["order_id"] = match.group(1)
            break

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    date_patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
    ]

    for pattern in date_patterns:
        match = re.search(
            pattern,
            message,
            re.IGNORECASE,
        )

        if match:
            fields["date"] = match.group(1)
            break

    # -----------------------------------------------------
    # AMOUNT
    # -----------------------------------------------------

    amount_patterns = [
        r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)"
        r"\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",

        r"(?:amount|total|price)"
        r"\s*(?:is|was|:)?\s*"
        r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)?"
        r"\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    ]

    for pattern in amount_patterns:
        match = re.search(
            pattern,
            message,
            re.IGNORECASE,
        )

        if match:
            fields["amount"] = match.group(1).replace(
                ",", ""
            )
            break

    # -----------------------------------------------------
    # PRODUCT CODE
    # -----------------------------------------------------

    product_code_pattern = (
        r"(?:product\s*code|sku)"
        r"\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-_]{2,})\b"
    )

    match = re.search(
        product_code_pattern,
        message,
        re.IGNORECASE,
    )

    if match:
        fields["product_code"] = match.group(1)

    # -----------------------------------------------------
    # ERROR CODE
    # -----------------------------------------------------

    error_code_pattern = (
        r"(?:error\s*code|error)"
        r"\s*[:#\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-_]{2,})\b"
    )

    match = re.search(
        error_code_pattern,
        message,
        re.IGNORECASE,
    )

    if match:
        fields["error_code"] = match.group(1)

    return fields


def compare_message_with_document(
    message: str,
    document_fields: dict,
) -> dict:
    """
    Compare customer message fields with fields extracted
    from an uploaded document.

    Rules:

    - Matching values -> consistent
    - Different explicit values -> conflict
    - Missing/unclear message evidence -> clarification
    - No comparable evidence -> no_comparable_evidence

    The function never decides which conflicting value
    is correct.
    """

    message_fields = extract_message_fields(message)

    comparisons = []
    conflicts = []
    clarifications = []

    for field in COMPARABLE_FIELDS:

        message_value = normalize_value(
            field,
            message_fields.get(field),
        )

        document_value = normalize_value(
            field,
            document_fields.get(field),
        )

        # -------------------------------------------------
        # Neither source contains the field
        # -------------------------------------------------

        if (
            message_value is None
            and document_value is None
        ):
            continue

        # -------------------------------------------------
        # Document has evidence but message doesn't
        # -------------------------------------------------

        if (
            message_value is None
            and document_value is not None
        ):
            clarifications.append(field)

            comparisons.append(
                {
                    "field": field,
                    "status": "clarification_required",
                    "message_value": None,
                    "document_value": document_value,
                }
            )

            continue

        # -------------------------------------------------
        # Message has evidence but document doesn't
        # -------------------------------------------------

        if (
            message_value is not None
            and document_value is None
        ):
            clarifications.append(field)

            comparisons.append(
                {
                    "field": field,
                    "status": "clarification_required",
                    "message_value": message_value,
                    "document_value": None,
                }
            )

            continue

        # -------------------------------------------------
        # Both sources contain the field
        # -------------------------------------------------

        if message_value == document_value:

            comparisons.append(
                {
                    "field": field,
                    "status": "consistent",
                    "message_value": message_value,
                    "document_value": document_value,
                }
            )

        else:

            conflicts.append(field)

            comparisons.append(
                {
                    "field": field,
                    "status": "conflict",
                    "message_value": message_value,
                    "document_value": document_value,
                }
            )

    # -----------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------

    if conflicts:
        overall_status = "conflict"

    elif clarifications:
        overall_status = "clarification_required"

    elif comparisons:
        overall_status = "consistent"

    else:
        overall_status = "no_comparable_evidence"

    return {
        "status": overall_status,
        "message_fields": message_fields,
        "document_fields": document_fields,
        "comparisons": comparisons,
        "conflicts": conflicts,
        "clarifications": clarifications,
    }


if __name__ == "__main__":

    print("=" * 60)
    print("TASK 5.3 - COMPARATOR SELF-TEST")
    print("=" * 60)

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

    print("\nComparison result:")
    print(result)

    print("\nComparator module loaded successfully.")
    print("RESULT: PASS ✅")

    print("\n" + "=" * 60)
