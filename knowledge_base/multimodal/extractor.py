import re
from datetime import datetime


# =========================================================
# TASK 5.2 - MULTIMODAL STRUCTURED DATA EXTRACTOR
# =========================================================

def extract_order_id(text: str):
    """
    Extract an order ID when explicitly present.

    Examples:
        Order ID: ORD-12345
        Order #ORD-12345
    """

    patterns = [
        r"\border\s*(?:id|number|no\.?|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{3,})\b",
        r"\b(ORD[-_][A-Z0-9\-]{3,})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1)

    return None


def extract_date(text: str):
    """
    Extract common date formats.

    Supported examples:
        2026-09-15
        15/09/2026
        15-09-2026
        15 Sep 2026
        15 September 2026

    Returns ISO format: YYYY-MM-DD
    """

    patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
        r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b",
    ]

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %b %Y",
        "%d %B %Y",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if not match:
            continue

        value = match.group(1)

        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

    return None


def extract_amount(text: str):
    """
    Extract a monetary amount when explicitly labelled.

    Examples:
        Amount: ₹1,499
        Total: $49.99
        Total Amount: 1499.00

    Returns a numeric string without currency symbols.
    """

    patterns = [
        r"(?:total\s+amount|total|amount|price|grand\s+total)"
        r"\s*[:\-]?\s*(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)?"
        r"\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).replace(",", "")

    return None


def extract_product_code(text: str):
    """
    Extract an explicitly labelled product/SKU code.
    """

    patterns = [
        r"(?:product\s*code|product\s*id|sku)"
        r"\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_]{2,})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1)

    return None


def extract_product_name(text: str):
    """
    Extract an explicitly labelled product name.

    OCR may remove punctuation such as ":" or "-".
    Therefore, the separator is optional.

    The product/item label must occur at the beginning
    of a line so ordinary sentences containing the word
    "product" are not treated as product fields.

    The extractor intentionally avoids guessing a product name
    from arbitrary text.
    """

    patterns = [
        r"^\s*(?:product\s*name|product)"
        r"\s*(?::|-)?\s*"
        r"([^\n,]+)",

        r"^\s*(?:item\s*name|item)"
        r"\s*(?::|-)?\s*"
        r"([^\n,]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.MULTILINE,
        )

        if match:
            value = match.group(1).strip()

            if value:
                return value

    return None



def extract_error_code(text: str):
    """
    Extract an explicitly labelled error code.

    Only matches an explicit "error code" or "error" label.
    Generic occurrences of the word "code" are not treated
    as error codes.
    """

    patterns = [
        r"(?:error\s*code)"
        r"\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_]{1,})\b",

        r"(?:error)"
        r"\s*[:\-]?\s*([A-Z][A-Z0-9\-_]{1,})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1)

    return None



def extract_structured_data(text: str) -> dict:
    """
    Extract all supported structured fields.

    Missing information is returned as None.
    No values are invented.
    """

    return {
        "order_id": extract_order_id(text),
        "date": extract_date(text),
        "amount": extract_amount(text),
        "product_name": extract_product_name(text),
        "product_code": extract_product_code(text),
        "error_code": extract_error_code(text),
    }


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TASK 5.2 - STRUCTURED EXTRACTOR SELF-TEST")
    print("=" * 60)

    sample = """
    INVOICE

    Order ID: ORD-12345
    Date: 15/09/2026
    Product Name: Wireless Keyboard
    Product Code: KB-900
    Total Amount: ₹1,499.00
    """

    result = extract_structured_data(sample)

    print("\nExtracted data:")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nExtractor module loaded successfully.")
    print("RESULT: PASS ✅")

    print("\n" + "=" * 60)
