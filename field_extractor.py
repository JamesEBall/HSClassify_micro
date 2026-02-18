"""
Structured field extraction from OCR text of trade documents.

Extracts:
- Made in (country of origin)
- Ship to (destination country)
- Item price (numeric value)
- Currency (USD, EUR, etc.)
- Product description
- Email addresses
"""

import re
from typing import Optional

# --- Country Matching ---

COUNTRIES = {
    # Major trading nations
    "china": "CN", "peoples republic of china": "CN", "prc": "CN", "中国": "CN",
    "united states": "US", "usa": "US", "u.s.a.": "US", "united states of america": "US",
    "japan": "JP", "日本": "JP",
    "germany": "DE", "deutschland": "DE",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "england": "GB",
    "france": "FR",
    "italy": "IT", "italia": "IT",
    "south korea": "KR", "korea": "KR", "republic of korea": "KR", "한국": "KR",
    "india": "IN",
    "canada": "CA",
    "australia": "AU",
    "brazil": "BR",
    "mexico": "MX",
    "indonesia": "ID",
    "thailand": "TH", "ไทย": "TH",
    "vietnam": "VN", "viet nam": "VN", "việt nam": "VN",
    "malaysia": "MY",
    "singapore": "SG",
    "taiwan": "TW", "chinese taipei": "TW",
    "netherlands": "NL", "holland": "NL",
    "spain": "ES", "españa": "ES",
    "turkey": "TR", "türkiye": "TR",
    "switzerland": "CH",
    "saudi arabia": "SA",
    "united arab emirates": "AE", "uae": "AE",
    "poland": "PL",
    "sweden": "SE",
    "belgium": "BE",
    "argentina": "AR",
    "austria": "AT",
    "norway": "NO",
    "ireland": "IE",
    "israel": "IL",
    "denmark": "DK",
    "philippines": "PH",
    "colombia": "CO",
    "pakistan": "PK",
    "chile": "CL",
    "finland": "FI",
    "bangladesh": "BD",
    "egypt": "EG",
    "czech republic": "CZ", "czechia": "CZ",
    "portugal": "PT",
    "romania": "RO",
    "new zealand": "NZ",
    "greece": "GR",
    "peru": "PE",
    "south africa": "ZA",
    "hungary": "HU",
    "sri lanka": "LK",
    "cambodia": "KH",
    "myanmar": "MM", "burma": "MM",
    "nigeria": "NG",
    "kenya": "KE",
    "ghana": "GH",
    "ethiopia": "ET",
    "tanzania": "TZ",
    "morocco": "MA",
    "hong kong": "HK",
}

# Reverse map: code -> name
COUNTRY_CODE_TO_NAME = {}
for name, code in COUNTRIES.items():
    if code not in COUNTRY_CODE_TO_NAME:
        COUNTRY_CODE_TO_NAME[code] = name.title()

# Fix some names
COUNTRY_CODE_TO_NAME["US"] = "United States"
COUNTRY_CODE_TO_NAME["GB"] = "United Kingdom"
COUNTRY_CODE_TO_NAME["CN"] = "China"
COUNTRY_CODE_TO_NAME["KR"] = "South Korea"
COUNTRY_CODE_TO_NAME["AE"] = "United Arab Emirates"
COUNTRY_CODE_TO_NAME["NZ"] = "New Zealand"
COUNTRY_CODE_TO_NAME["ZA"] = "South Africa"
COUNTRY_CODE_TO_NAME["CZ"] = "Czech Republic"
COUNTRY_CODE_TO_NAME["HK"] = "Hong Kong"
COUNTRY_CODE_TO_NAME["TW"] = "Taiwan"
COUNTRY_CODE_TO_NAME["SA"] = "Saudi Arabia"
COUNTRY_CODE_TO_NAME["NL"] = "Netherlands"

# All country names for dropdown
ALL_COUNTRIES = sorted(set(COUNTRY_CODE_TO_NAME.values()))

# --- Currency Matching ---

CURRENCIES = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "CNY": "Chinese Yuan",
    "RMB": "Chinese Yuan",
    "KRW": "Korean Won",
    "THB": "Thai Baht",
    "VND": "Vietnamese Dong",
    "INR": "Indian Rupee",
    "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar",
    "SGD": "Singapore Dollar",
    "MYR": "Malaysian Ringgit",
    "IDR": "Indonesian Rupiah",
    "PHP": "Philippine Peso",
    "BRL": "Brazilian Real",
    "MXN": "Mexican Peso",
    "CHF": "Swiss Franc",
    "SEK": "Swedish Krona",
    "NOK": "Norwegian Krone",
    "DKK": "Danish Krone",
    "HKD": "Hong Kong Dollar",
    "TWD": "Taiwan Dollar",
    "AED": "UAE Dirham",
    "SAR": "Saudi Riyal",
    "ZAR": "South African Rand",
    "NZD": "New Zealand Dollar",
    "TRY": "Turkish Lira",
    "PLN": "Polish Zloty",
}

CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "฿": "THB",
    "₫": "VND",
    "₩": "KRW",
    "R$": "BRL",
}


def find_country(text: str, context_keywords: list[str]) -> Optional[str]:
    """Find a country name near context keywords in the text."""
    text_lower = text.lower()
    
    # Try to find country near context keywords
    for keyword in context_keywords:
        # Search for keyword in text
        pattern = re.compile(
            rf'{keyword}\s*[:\-]?\s*(.{{2,50}})',
            re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            fragment = match.group(1).strip().lower()
            # Check if any country name is in the fragment
            for country_name, code in sorted(COUNTRIES.items(), key=lambda x: -len(x[0])):
                if country_name in fragment:
                    return code
            # Also check for ISO country codes (2 letters)
            code_match = re.match(r'^([A-Z]{2})\b', match.group(1).strip())
            if code_match:
                c = code_match.group(1)
                if c in COUNTRY_CODE_TO_NAME:
                    return c
    
    return None


def extract_fields(ocr_text: str) -> dict:
    """
    Extract structured fields from OCR text of a trade document.
    
    Returns dict with:
        - email: str or None
        - made_in: country code or None
        - ship_to: country code or None
        - item_price: float or None
        - currency: currency code or None
        - product_description: str or None
        - raw_text: the original OCR text
        - confidence: dict with confidence scores for each field
    """
    result = {
        "email": None,
        "made_in": None,
        "made_in_name": None,
        "ship_to": None,
        "ship_to_name": None,
        "item_price": None,
        "currency": None,
        "product_description": None,
        "raw_text": ocr_text,
        "confidence": {},
    }
    
    if not ocr_text or not ocr_text.strip():
        return result
    
    text = ocr_text.strip()
    
    # --- Extract Email ---
    email_pattern = re.compile(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        re.IGNORECASE
    )
    email_match = email_pattern.search(text)
    if email_match:
        result["email"] = email_match.group(0)
        result["confidence"]["email"] = 0.95
    
    # --- Extract Country of Origin (Made in) ---
    origin_keywords = [
        "made in", "manufactured in", "produced in", "origin",
        "country of origin", "country of manufacture",
        "mfg country", "mfg. country", "fabricated in",
        "assembled in", "place of origin", "product of",
        "sourced from", "shipped from", "exporting country",
        "from"
    ]
    
    origin_code = find_country(text, origin_keywords)
    if origin_code:
        result["made_in"] = origin_code
        result["made_in_name"] = COUNTRY_CODE_TO_NAME.get(origin_code, origin_code)
        result["confidence"]["made_in"] = 0.85
    
    # --- Extract Destination (Ship to) ---
    dest_keywords = [
        "ship to", "shipped to", "deliver to", "delivery to",
        "destination", "consignee", "import to", "importing country",
        "port of discharge", "port of destination", "final destination",
        "to country", "dest", "buyer country",
        "bill to", "sold to"
    ]
    
    dest_code = find_country(text, dest_keywords)
    if dest_code:
        result["ship_to"] = dest_code
        result["ship_to_name"] = COUNTRY_CODE_TO_NAME.get(dest_code, dest_code)
        result["confidence"]["ship_to"] = 0.80
    
    # --- Extract Currency ---
    # First check for currency symbols
    for symbol, curr_code in sorted(CURRENCY_SYMBOLS.items(), key=lambda x: -len(x[0])):
        if symbol in text:
            result["currency"] = curr_code
            result["confidence"]["currency"] = 0.90
            break
    
    # Then check for explicit currency codes
    if not result["currency"]:
        for curr_code in CURRENCIES:
            pattern = re.compile(rf'\b{curr_code}\b', re.IGNORECASE)
            if pattern.search(text):
                result["currency"] = curr_code
                result["confidence"]["currency"] = 0.95
                break
    
    # --- Extract Price ---
    price_patterns = [
        # "price: $123.45" or "amount: 123.45 USD"
        re.compile(
            r'(?:price|amount|total|value|unit price|item price|cost|fob value|cif value|invoice value)\s*[:\-]?\s*'
            r'(?:[A-Z]{3}\s*)?'
            r'[\$€£¥₹฿₫₩]?\s*'
            r'([\d,]+\.?\d*)',
            re.IGNORECASE
        ),
        # "$123.45" or "€99.99"
        re.compile(
            r'[\$€£¥₹฿₫₩]\s*([\d,]+\.?\d*)'
        ),
        # "123.45 USD" or "99.99 EUR"
        re.compile(
            r'([\d,]+\.?\d*)\s*(?:USD|EUR|GBP|JPY|CNY|RMB|THB|VND|INR|CAD|AUD|SGD|MYR)',
            re.IGNORECASE
        ),
    ]
    
    for pattern in price_patterns:
        match = pattern.search(text)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                price = float(price_str)
                if 0 < price < 1e12:  # Sanity check
                    result["item_price"] = price
                    result["confidence"]["item_price"] = 0.80
                    break
            except ValueError:
                continue
    
    # --- Extract Product Description ---
    desc_keywords = [
        "description", "product description", "item description",
        "goods description", "description of goods",
        "commodity", "product name", "item name",
        "goods", "merchandise", "articles"
    ]
    
    for keyword in desc_keywords:
        pattern = re.compile(
            rf'{keyword}\s*[:\-]?\s*(.{{10,300}}?)(?:\n|$)',
            re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            desc = match.group(1).strip()
            # Clean up
            desc = re.sub(r'\s+', ' ', desc)
            if len(desc) > 10:
                result["product_description"] = desc
                result["confidence"]["product_description"] = 0.75
                break
    
    # If no structured description found, use the longest non-header line
    if not result["product_description"]:
        lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 15]
        # Filter out lines that look like headers/labels
        content_lines = [
            l for l in lines
            if not re.match(r'^(invoice|bill|date|ref|no\.|number|email|phone|fax|tel|address)', l, re.IGNORECASE)
            and not re.match(r'^[A-Z\s]{2,20}:$', l)
        ]
        if content_lines:
            # Pick the longest line as likely description
            best = max(content_lines, key=len)
            result["product_description"] = best[:300]
            result["confidence"]["product_description"] = 0.40
    
    return result


def get_all_countries() -> list[dict]:
    """Return list of all countries for dropdowns."""
    return [
        {"code": code, "name": name}
        for code, name in sorted(COUNTRY_CODE_TO_NAME.items(), key=lambda x: x[1])
    ]


def get_all_currencies() -> list[dict]:
    """Return list of all currencies for dropdowns."""
    return [
        {"code": code, "name": name}
        for code, name in sorted(CURRENCIES.items(), key=lambda x: x[1])
    ]
