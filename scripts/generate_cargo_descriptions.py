#!/usr/bin/env python3
"""
Synthetic Customs Cargo Description Generator.

Reads HS codes from HTS CSV files and the harmonized-system dataset,
generates realistic cargo descriptions with chapter-specific augmentation,
and outputs in training-compatible format.

Target: ~100K samples in data/cargo_descriptions.csv
"""

import csv
import os
import random
import re
import sys
from pathlib import Path

random.seed(42)

PROJECT_DIR = Path(__file__).parent.parent
HS_CSV_PATH = PROJECT_DIR / "data" / "harmonized-system" / "harmonized-system.csv"
HTS_DIR = PROJECT_DIR / "data" / "hts"
OUTPUT_PATH = PROJECT_DIR / "data" / "cargo_descriptions.csv"

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

PACKAGING = {
    "food": [
        "IN CARTONS", "IN BOXES", "IN BAGS", "IN SACKS", "IN CRATES",
        "IN POLYBAGS", "IN VACUUM PACKS", "IN TINS", "IN CANS",
        "IN JARS", "IN BOTTLES", "IN PAILS", "IN POUCHES",
        "SHRINK WRAPPED", "IN REEFER CONTAINER", "IN STYROFOAM BOXES",
        "IN BULK BAGS", "IN FIBC", "IN WOVEN BAGS",
    ],
    "liquid": [
        "IN DRUMS", "IN IBC", "IN IBCS", "IN FLEXI TANKS", "IN FLEXITANK",
        "IN BARRELS", "IN TANK CONTAINER", "IN ISO TANK", "IN JERRYCANS",
        "IN TOTES", "IN PAILS", "IN BOTTLES", "IN TANKER",
    ],
    "general": [
        "IN CARTONS", "IN BOXES", "IN CRATES", "IN PALLETS", "ON PALLETS",
        "IN WOODEN CASES", "IN PLYWOOD CASES", "IN BUNDLES", "IN BALES",
        "IN ROLLS", "IN DRUMS", "IN BAGS", "IN BULK", "LOOSE",
        "IN CONTAINER", "IN PACKAGES", "IN PKGS", "PALLETIZED",
        "SHRINK WRAPPED ON PALLETS", "IN WOODEN CRATES",
    ],
    "fragile": [
        "IN WOODEN CRATES", "IN FOAM LINED BOXES", "IN DOUBLE WALL CARTONS",
        "ON WOODEN PALLETS", "IN CUSTOM PACKAGING", "IN ANTI-STATIC BAGS",
        "IN BUBBLE WRAPPED CARTONS", "IN REINFORCED CARTONS",
    ],
}

COUNTRIES_OF_ORIGIN = [
    "CHINA", "INDIA", "VIETNAM", "THAILAND", "INDONESIA", "BANGLADESH",
    "TURKEY", "BRAZIL", "MEXICO", "SOUTH KOREA", "JAPAN", "TAIWAN",
    "PAKISTAN", "PHILIPPINES", "MALAYSIA", "CAMBODIA", "SRI LANKA",
    "EGYPT", "MOROCCO", "ETHIOPIA", "KENYA", "COLOMBIA", "PERU",
    "CHILE", "ARGENTINA", "ECUADOR", "HONDURAS", "GUATEMALA",
    "MYANMAR", "LAOS", "NEPAL", "URUGUAY", "PARAGUAY",
    "SOUTH AFRICA", "NIGERIA", "GHANA", "IVORY COAST",
    "ITALY", "GERMANY", "FRANCE", "SPAIN", "PORTUGAL", "POLAND",
    "NETHERLANDS", "BELGIUM", "GREECE", "SWITZERLAND", "AUSTRIA",
    "USA", "CANADA", "AUSTRALIA", "NEW ZEALAND", "UK",
]

ABBREVIATIONS = {
    "STAINLESS STEEL": "S/S",
    "STAINLESS": "SS",
    "PIECES": "PCS",
    "PIECE": "PC",
    "KILOGRAM": "KG",
    "KILOGRAMS": "KGS",
    "METRIC TON": "MT",
    "METRIC TONS": "MTS",
    "CONTAINER": "CNTR",
    "CONTAINERS": "CNTRS",
    "APPROXIMATELY": "APPROX",
    "APPROXIMATELY": "APPROX.",
    "CERTIFICATE": "CERT",
    "CERTIFICATE OF": "C/O",
    "TEMPERATURE": "TEMP",
    "QUANTITY": "QTY",
    "NUMBER": "NO.",
    "PACKAGE": "PKG",
    "PACKAGES": "PKGS",
    "MANUFACTURING": "MFG",
    "MANUFACTURER": "MFR",
    "SPECIFICATION": "SPEC",
    "SPECIFICATIONS": "SPECS",
    "DIAMETER": "DIA",
    "THICKNESS": "THK",
    "LENGTH": "LEN",
    "WIDTH": "W",
    "HEIGHT": "H",
    "WEIGHT": "WT",
    "GROSS WEIGHT": "GW",
    "NET WEIGHT": "NW",
    "CUBIC METER": "CBM",
    "SQUARE METER": "SQM",
    "SQUARE METERS": "SQM",
    "MILLIMETER": "MM",
    "CENTIMETER": "CM",
    "REFERENCE": "REF",
    "INVOICE": "INV",
    "PURCHASE ORDER": "PO",
    "BILL OF LADING": "B/L",
    "COMMERCIAL": "COMM",
    "INDUSTRIAL": "IND",
    "INTERNATIONAL": "INTL",
}


# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

def _maybe(prob):
    return random.random() < prob

def _pick(lst):
    return random.choice(lst)

def _weight():
    unit = _pick(["KG", "KGS", "MT", "MTS", "LBS", "TON", "TONS"])
    if unit in ["MT", "MTS", "TON", "TONS"]:
        val = round(random.uniform(0.5, 500), 1)
    elif unit in ["LBS"]:
        val = random.randint(10, 50000)
    else:
        val = round(random.uniform(1, 25000), 1)
    return f"{val} {unit}"

def _qty():
    unit = _pick(["PCS", "UNITS", "SETS", "PAIRS", "ROLLS", "SHEETS",
                   "BAGS", "CTNS", "CARTONS", "BOXES", "CASES", "PKGS",
                   "BALES", "DRUMS", "BUNDLES", "COILS", "REELS", "LOTS"])
    val = random.randint(1, 50000)
    return f"{val} {unit}"


def _apply_abbreviations(text):
    if not _maybe(0.3):
        return text
    for full, abbr in ABBREVIATIONS.items():
        if full in text and _maybe(0.5):
            text = text.replace(full, abbr, 1)
    return text


def _add_noise(text):
    # Random extra space
    if _maybe(0.1):
        words = text.split()
        if len(words) > 3:
            idx = random.randint(1, len(words) - 2)
            words[idx] = words[idx] + " "
            text = " ".join(words)
    # Minor typo (swap two adjacent chars)
    if _maybe(0.05) and len(text) > 10:
        idx = random.randint(2, len(text) - 3)
        text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
    # Truncation (cut description short)
    if _maybe(0.08) and len(text) > 40:
        words = text.split()
        cut = random.randint(len(words) // 2, len(words) - 1)
        text = " ".join(words[:cut])
    return text


# ─────────────────────────────────────────────────────────
# CHAPTER-SPECIFIC GENERATORS
# ─────────────────────────────────────────────────────────

def gen_animal_products(base_desc, hs_code):
    """Chapters 01-05: Live animals, meat, fish, dairy, animal products."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    if ch == 1:  # Live animals
        if _maybe(0.5): parts.append(_pick(["FOR BREEDING", "FOR SLAUGHTER", "FOR EXHIBITION", "FOR ZOOLOGICAL GARDENS"]))
        if _maybe(0.3): parts.append(_pick(["HEALTH CERT ATTACHED", "WITH VETERINARY CERTIFICATE", "CITES PERMIT NO. " + str(random.randint(10000, 99999))]))
    elif ch == 2:  # Meat
        cuts = ["BONELESS", "BONE-IN", "SKINLESS", "SKIN-ON", "TRIMMED", "UNTRIMMED"]
        states = ["FROZEN", "FRESH CHILLED", "FRESH", "DEEP FROZEN AT -18C"]
        grades = ["GRADE A", "GRADE B", "PRIME", "CHOICE", "SELECT", "HALAL", "HALAL CERTIFIED", "KOSHER"]
        if _maybe(0.7): parts.insert(0, _pick(states))
        if _maybe(0.5): parts.append(_pick(cuts))
        if _maybe(0.4): parts.append(_pick(grades))
        parts.append(_pick(PACKAGING["food"]))
    elif ch == 3:  # Fish/seafood
        states = ["FROZEN", "FRESH", "CHILLED", "DRIED", "SALTED", "SMOKED", "LIVE"]
        forms = ["WHOLE", "HEADLESS", "HEAD-ON", "SHELL-ON", "PEELED", "DEVEINED",
                 "FILLETS", "GUTTED", "IQF", "BLOCK FROZEN", "SEMI-IQF"]
        sizes = ["6/8", "8/12", "13/15", "16/20", "21/25", "26/30", "31/40", "41/50", "U/10", "U/15"]
        if _maybe(0.8): parts.insert(0, _pick(states))
        if _maybe(0.6): parts.append(_pick(forms))
        if _maybe(0.4): parts.append(_pick(sizes) + " COUNT")
        if _maybe(0.3): parts.append(_pick(["PRODUCT OF " + _pick(COUNTRIES_OF_ORIGIN),
                                             "ORIGIN: " + _pick(COUNTRIES_OF_ORIGIN)]))
        parts.append(_pick(PACKAGING["food"]))
    elif ch == 4:  # Dairy, eggs, honey
        if _maybe(0.5): parts.append(_pick(["FAT CONTENT " + str(random.choice([1.5, 2, 3.2, 3.5, 4, 25, 30, 33, 45, 48, 55])) + "%",
                                             str(random.choice([1.5, 2, 3.2, 3.5, 25, 30, 45])) + "% FAT"]))
        if _maybe(0.4): parts.append(_pick(["PASTEURIZED", "UHT", "UNPASTEURIZED", "RAW", "POWDERED", "SPRAY DRIED"]))
        parts.append(_pick(PACKAGING["food"]))

    if _maybe(0.3): parts.append(_qty())
    if _maybe(0.3): parts.append(_weight())
    return " ".join(parts)


def gen_vegetable_products(base_desc, hs_code):
    """Chapters 06-14: Plants, vegetables, fruit, cereals, oil seeds."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    states = ["FRESH", "FROZEN", "DRIED", "PRESERVED", "DEHYDRATED", "ORGANIC", "CONVENTIONAL"]
    if ch in [7, 8]:  # Veg and fruit
        varieties = {
            7: ["ICEBERG", "ROMAINE", "RED ONION", "WHITE ONION", "CHERRY TOMATO", "RUSSET", "YUKON GOLD"],
            8: ["CAVENDISH", "HASS", "ATAULFO", "ALPHONSO", "MEDJOOL", "DEGLET NOOR", "THOMPSON SEEDLESS", "CARABAO"],
        }
        if _maybe(0.7): parts.insert(0, _pick(states))
        if ch in varieties and _maybe(0.4): parts.append(_pick(varieties[ch]) + " VARIETY")
        if _maybe(0.3): parts.append(_pick(["CLASS I", "CLASS II", "GRADE A", "GRADE B", "EXTRA CLASS"]))
        if _maybe(0.3): parts.append(_pick(["SIZE " + str(random.randint(40, 90)) + "MM",
                                             str(random.choice([1, 2, 5, 10, 20, 25])) + "KG BOXES"]))
    elif ch == 9:  # Coffee, tea, spices
        grades = ["GRADE AA", "GRADE A", "GRADE B", "PREMIUM", "SPECIALTY", "COMMERCIAL GRADE",
                  "SHB", "EP", "FAQ", "TGFOP", "BOPF", "ASTA " + str(random.choice([100, 120, 160, 200]))]
        if _maybe(0.6): parts.append(_pick(grades))
        if _maybe(0.5): parts.append(_pick(["GREEN BEANS", "ROASTED", "GROUND", "WHOLE LEAF", "CTC", "BROKEN"]))
        if _maybe(0.4): parts.append("CROP " + str(random.randint(2023, 2025)) + "/" + str(random.randint(24, 26)))
    elif ch == 10:  # Cereals
        if _maybe(0.5): parts.append(_pick(["MILLING QUALITY", "FEED GRADE", "FOOD GRADE", "SEED QUALITY",
                                             "MAX MOISTURE " + str(random.choice([12, 13, 14, 14.5])) + "%"]))
    elif ch == 12:  # Oil seeds
        if _maybe(0.4): parts.append(_pick(["NON-GMO", "GMO", "ORGANIC CERTIFIED", "IP CERTIFIED"]))
        if _maybe(0.3): parts.append("OIL CONTENT MIN " + str(random.choice([18, 20, 22, 38, 40, 42, 44])) + "%")

    parts.append(_pick(PACKAGING["food"]))
    if _maybe(0.3): parts.append(_weight())
    return " ".join(parts)


def gen_food_products(base_desc, hs_code):
    """Chapters 15-24: Prepared foods, beverages, tobacco."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    if ch == 15:  # Fats and oils
        if _maybe(0.5): parts.append(_pick(["REFINED", "CRUDE", "VIRGIN", "EXTRA VIRGIN", "RBD", "COLD PRESSED"]))
        if _maybe(0.4): parts.append("FFA MAX " + str(random.choice([0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])) + "%")
        parts.append(_pick(PACKAGING["liquid"]))
    elif ch in [16, 17, 18, 19, 20, 21]:  # Prepared foods
        if _maybe(0.4): parts.append(_pick(["READY TO EAT", "INSTANT", "SHELF STABLE", "REQUIRES REFRIGERATION"]))
        if _maybe(0.3): parts.append("BBD " + str(random.randint(1, 12)).zfill(2) + "/" + str(random.randint(2026, 2028)))
        if _maybe(0.3): parts.append(_pick(["HALAL CERTIFIED", "KOSHER", "ORGANIC", "GLUTEN FREE", "VEGAN"]))
        parts.append(_pick(PACKAGING["food"]))
    elif ch == 22:  # Beverages
        if _maybe(0.5): parts.append(_pick(["ALC " + str(random.choice([4.5, 5.0, 5.5, 7.0, 11.5, 12.5, 13.5, 14.0, 40.0, 43.0])) + "% VOL",
                                             "NON-ALCOHOLIC", "0.0% ABV"]))
        if _maybe(0.4): parts.append(str(random.choice([250, 330, 355, 500, 750, 1000, 1500])) + "ML")
        if _maybe(0.3): parts.append(str(random.choice([6, 12, 24, 30, 48])) + " BOTTLES/CASE")
        parts.append(_pick(PACKAGING["food"]))
    elif ch == 24:  # Tobacco
        if _maybe(0.4): parts.append(_pick(["VIRGINIA BLEND", "BURLEY", "ORIENTAL", "FLUE-CURED", "AIR-CURED"]))

    if _maybe(0.3): parts.append(_qty())
    if _maybe(0.3): parts.append(_weight())
    return " ".join(parts)


def gen_chemicals(base_desc, hs_code):
    """Chapters 28-38: Chemicals, pharmaceuticals, fertilizers, plastics precursors."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    if ch in [28, 29]:  # Inorganic/organic chemicals
        purities = ["PURITY 99%", "PURITY 99.5%", "PURITY 99.9%", "PURITY 99.99%",
                    "MIN 98%", "MIN 99%", "TECH GRADE", "REAGENT GRADE", "ACS GRADE",
                    "INDUSTRIAL GRADE", "PHARMA GRADE", "EP GRADE", "USP GRADE", "BP GRADE"]
        if _maybe(0.6): parts.append(_pick(purities))
        if _maybe(0.4): parts.append("CAS NO. " + "-".join([str(random.randint(10, 99999)) for _ in range(3)]))
        un_numbers = ["UN1219", "UN1230", "UN1263", "UN1294", "UN1307", "UN1760",
                      "UN1789", "UN1791", "UN1824", "UN1830", "UN2031", "UN2672",
                      "UN2789", "UN2790", "UN2794", "UN2796", "UN3082", "UN3264", "UN3266"]
        if _maybe(0.3): parts.append(_pick(un_numbers))
        if _maybe(0.3): parts.append(_pick(["CLASS " + str(random.choice([3, 5.1, 6.1, 8, 9])),
                                             "PG " + _pick(["I", "II", "III"]),
                                             "DG CLASS " + str(random.choice([3, 5, 6, 8, 9]))]))
        if _maybe(0.3): parts.append("NON-HAZARDOUS")
        parts.append(_pick(PACKAGING["liquid"] + PACKAGING["general"]))

    elif ch == 30:  # Pharmaceuticals
        forms = ["TABLETS", "CAPSULES", "POWDER", "INJECTION", "SOLUTION", "SUSPENSION",
                 "CREAM", "OINTMENT", "SYRUP", "GRANULES", "API BULK", "AMPOULES"]
        standards = ["BP", "USP", "EP", "IP", "JP", "WHO PREQUALIFIED"]
        if _maybe(0.6): parts.append(_pick(forms))
        if _maybe(0.5): parts.append(_pick(standards))
        if _maybe(0.4): parts.append(str(random.choice([5, 10, 20, 25, 50, 100, 250, 500])) + "MG")
        if _maybe(0.3): parts.append("BATCH NO. " + "".join([str(random.randint(0,9)) for _ in range(8)]))
        parts.append(_pick(PACKAGING["general"]))

    elif ch in [31, 32, 33, 34, 35, 36, 37, 38]:
        uses = {
            31: ["NPK " + str(random.randint(10,20)) + "-" + str(random.randint(5,20)) + "-" + str(random.randint(5,20)),
                 "UREA 46%", "DAP 18-46-0", "GRANULAR", "PRILLED", "FOR AGRICULTURAL USE"],
            32: ["FOR INDUSTRIAL USE", "RAL " + str(random.randint(1000, 9016)),
                 "PANTONE " + str(random.randint(100, 7000)) + "C",
                 "WATER BASED", "SOLVENT BASED", "UV CURABLE"],
            33: ["FOR COSMETIC USE", "FRAGRANCE OIL", "ESSENTIAL OIL",
                 "DERMATOLOGICALLY TESTED", "PARABEN FREE", "CRUELTY FREE"],
            34: ["BIODEGRADABLE", "CONCENTRATED", "READY TO USE",
                 "PH " + str(random.choice([5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]))],
            38: ["CATALYST", "FLUX", "INDUSTRIAL GRADE", "BIODIESEL", "ANTIFREEZE",
                 "COOLANT", "NOT RESTRICTED", "NON-HAZARDOUS", "MSDS ATTACHED"],
        }
        if ch in uses and _maybe(0.5):
            parts.append(_pick(uses[ch]))
        parts.append(_pick(PACKAGING["liquid"] + PACKAGING["general"]))

    if _maybe(0.3): parts.append(_weight())
    return " ".join(parts)


def gen_plastics_rubber(base_desc, hs_code):
    """Chapters 39-40: Plastics, rubber."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    if ch == 39:  # Plastics
        types = ["HDPE", "LDPE", "LLDPE", "PP", "PET", "PS", "ABS", "PVC", "PC",
                 "PMMA", "PA6", "PA66", "POM", "PEEK", "PTFE", "PBT"]
        forms = ["GRANULES", "PELLETS", "RESIN", "POWDER", "SHEET", "FILM", "PIPE",
                 "TUBE", "ROD", "PROFILE", "SCRAP", "REGRIND", "VIRGIN"]
        if _maybe(0.5): parts.append(_pick(types))
        if _maybe(0.5): parts.append(_pick(forms))
        if _maybe(0.3): parts.append("MFI " + str(random.choice([0.3, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 35.0])))
        if _maybe(0.3): parts.append("DENSITY " + str(random.choice([0.91, 0.92, 0.94, 0.95, 0.96, 1.04, 1.14, 1.20, 1.38])))
    elif ch == 40:  # Rubber
        types = ["NATURAL RUBBER", "NR", "SBR", "NBR", "EPDM", "SILICONE", "CR", "FKM"]
        forms = ["SHEETS", "BLOCKS", "BALES", "COMPOUND", "VULCANIZED", "UNVULCANIZED",
                 "TYRES", "TUBES", "HOSES", "GASKETS", "O-RINGS", "SEALS"]
        if _maybe(0.5): parts.append(_pick(types))
        if _maybe(0.5): parts.append(_pick(forms))
        if _maybe(0.3): parts.append("HARDNESS " + str(random.choice([40, 50, 55, 60, 65, 70, 75, 80, 85, 90])) + " SHORE A")

    parts.append(_pick(PACKAGING["general"]))
    if _maybe(0.3): parts.append(_qty())
    if _maybe(0.3): parts.append(_weight())
    return " ".join(parts)


def gen_textiles(base_desc, hs_code):
    """Chapters 50-63: Textiles and garments."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    fibres = ["100% COTTON", "100% POLYESTER", "65/35 POLY/COTTON", "80/20 COTTON/POLY",
              "100% SILK", "100% LINEN", "100% WOOL", "100% NYLON", "100% VISCOSE",
              "60% COTTON 40% POLYESTER", "95% COTTON 5% ELASTANE", "70% WOOL 30% ACRYLIC",
              "100% ACRYLIC", "50/50 COTTON POLYESTER"]

    if ch in range(50, 56):  # Fibres and fabrics
        if _maybe(0.6): parts.append(_pick(fibres))
        if _maybe(0.4): parts.append(_pick(["WOVEN", "KNITTED", "NON-WOVEN", "PRINTED", "DYED", "BLEACHED",
                                             "UNBLEACHED", "GREY FABRIC", "MERCERIZED"]))
        if _maybe(0.4): parts.append(str(random.choice([90, 110, 120, 140, 150, 160, 180, 200, 220, 240, 280, 300])) + "CM WIDTH")
        if _maybe(0.3): parts.append(str(random.choice([80, 100, 120, 150, 180, 200, 250, 300])) + " GSM")
    elif ch in range(61, 64):  # Garments
        sizes = ["S-XL", "S-XXL", "XS-XXL", "S/M/L/XL", "ONE SIZE", "FREE SIZE",
                 "SIZES 2-14", "SIZES 6-16", "EU 36-44", "US 4-12"]
        styles = ["MENS", "WOMENS", "LADIES", "BOYS", "GIRLS", "CHILDRENS", "UNISEX", "INFANT", "TODDLER"]
        colors = ["ASSORTED COLOURS", "ASSTD COLORS", "BLACK", "WHITE", "NAVY", "MULTI-COLOR",
                  "AS PER PO", "AS PER BUYER SPECIFICATION"]
        if _maybe(0.7): parts.insert(0, _pick(styles))
        if _maybe(0.6): parts.append(_pick(fibres))
        if _maybe(0.5): parts.append(_pick(sizes))
        if _maybe(0.4): parts.append(_pick(colors))
        if _maybe(0.3): parts.append("STYLE NO. " + "".join([chr(random.randint(65, 90)) for _ in range(2)]) + str(random.randint(100, 9999)))

    parts.append(_pick(PACKAGING["general"]))
    if _maybe(0.3): parts.append(_qty())
    return " ".join(parts)


def gen_metals(base_desc, hs_code):
    """Chapters 72-83: Iron, steel, metals, metal products."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    if ch in [72, 73]:  # Iron and steel
        grades = ["ASTM A36", "ASTM A572 GR50", "ASTM A106 GR.B", "ASTM A312 TP304",
                  "ASTM A312 TP316L", "ASTM A53 GR.B", "API 5L GR.B", "API 5CT J55",
                  "SS304", "SS316", "SS316L", "EN10025 S275JR", "EN10025 S355JR",
                  "JIS G3101 SS400", "JIS G3141 SPCC", "SAE 1020", "SAE 4140",
                  "IS 2062 E250", "AISI 304", "AISI 316L"]
        forms = ["PLATE", "SHEET", "COIL", "HR COIL", "CR COIL", "PIPE", "TUBE", "SEAMLESS PIPE",
                 "WELDED PIPE", "ERW PIPE", "BAR", "ROD", "WIRE ROD", "ANGLE", "CHANNEL",
                 "I-BEAM", "H-BEAM", "FLANGE", "FITTING", "ELBOW", "TEE", "REDUCER",
                 "REBAR", "FLAT BAR", "ROUND BAR", "HEX BAR", "SQUARE TUBE"]
        if _maybe(0.6): parts.append(_pick(grades))
        if _maybe(0.6): parts.append(_pick(forms))
        dims = [
            "OD " + str(random.choice([21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73.0, 88.9, 114.3, 168.3, 219.1, 273.0, 323.8])) + "MM",
            "THK " + str(random.choice([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0])) + "MM",
            str(random.choice([4, 6, 8, 10, 12, 14, 16, 18, 20, 24])) + " INCH",
            "DN" + str(random.choice([15, 20, 25, 32, 40, 50, 65, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600])),
            "SCH " + str(random.choice([10, 20, 40, 80, 120, 160, "STD", "XS", "XXS"])),
            str(random.choice([1200, 1219, 1250, 1500, 1524, 2000, 2438])) + "MM X " + str(random.choice([2400, 2438, 3000, 3048, 6000, 6096, 12000])) + "MM",
            "ANSI 150#", "ANSI 300#", "ANSI 600#", "PN10", "PN16", "PN25", "PN40",
            "CLASS " + str(random.choice([150, 300, 600, 900, 1500, 2500])),
        ]
        if _maybe(0.6): parts.append(_pick(dims))
        if _maybe(0.3): parts.append(_pick(dims))
        finishes = ["HOT DIP GALVANIZED", "HDG", "GALV", "ELECTRO GALVANIZED",
                    "PAINTED", "BARE", "BLACK", "PICKLED AND OILED",
                    "2B FINISH", "BA FINISH", "NO.4 FINISH", "MIRROR FINISH",
                    "MILL FINISH", "EPOXY COATED", "3LPE COATED"]
        if _maybe(0.4): parts.append(_pick(finishes))

    elif ch == 74:  # Copper
        if _maybe(0.5): parts.append(_pick(["C11000", "C12200", "C26000", "C27000", "C36000",
                                             "C51000", "C71500", "CW004A", "CW024A"]))
        if _maybe(0.4): parts.append(_pick(["PIPE", "TUBE", "SHEET", "STRIP", "WIRE", "ROD", "BAR", "CATHODE"]))
    elif ch == 76:  # Aluminum
        if _maybe(0.5): parts.append(_pick(["6061-T6", "6063-T5", "5052-H32", "5083-H111",
                                             "7075-T6", "1100-H14", "3003-H14", "2024-T3"]))
        if _maybe(0.4): parts.append(_pick(["SHEET", "PLATE", "COIL", "EXTRUSION", "PROFILE",
                                             "INGOT", "BILLET", "WIRE", "FOIL"]))

    parts.append(_pick(PACKAGING["general"]))
    if _maybe(0.3): parts.append(_weight())
    return " ".join(parts)


def gen_machinery(base_desc, hs_code):
    """Chapters 84-85: Machinery, electrical equipment."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    if ch == 84:  # Machinery
        if _maybe(0.5): parts.append("MODEL " + "".join([chr(random.randint(65, 90)) for _ in range(random.randint(1, 3))]) + "-" + str(random.randint(100, 9999)))
        if _maybe(0.4): parts.append(_pick(["NEW", "BRAND NEW", "USED", "REFURBISHED", "RECONDITIONED"]))
        if _maybe(0.3): parts.append(_pick(["SINGLE PHASE", "THREE PHASE", "3-PHASE",
                                             str(random.choice([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10, 15, 20, 30, 50, 75, 100])) + "HP",
                                             str(random.choice([0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 110])) + "KW"]))
        if _maybe(0.3): parts.append(_pick(["S/N " + str(random.randint(10000, 999999)),
                                             "SERIAL NO. " + "".join([str(random.randint(0,9)) for _ in range(10)])]))
        if _maybe(0.3): parts.append(_pick(["COMPLETE WITH ACCESSORIES", "WITH SPARE PARTS",
                                             "WITH TOOLING", "CKD", "SKD", "FULLY ASSEMBLED"]))
    elif ch == 85:  # Electrical
        if _maybe(0.5): parts.append(_pick([
            str(random.choice([3.7, 5, 7.4, 11.1, 12, 14.4, 24, 36, 48])) + "V",
            str(random.choice([110, 120, 220, 230, 240, 380, 400, 415, 440, 480, 690])) + "V",
            str(random.choice([50, 60])) + "HZ",
            str(random.choice([500, 1000, 2000, 2200, 2600, 3000, 3350, 4000, 5000, 10000, 20000])) + "MAH",
            str(random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])) + "GB",
        ]))
        if _maybe(0.4): parts.append("MODEL " + "".join([chr(random.randint(65, 90)) for _ in range(1, 3)]) + str(random.randint(100, 9999)))
        if _maybe(0.3): parts.append(_pick(["CE MARKED", "UL LISTED", "FCC CERTIFIED",
                                             "ROHS COMPLIANT", "IP65", "IP67", "IP68"]))

    parts.append(_pick(PACKAGING["general"] + PACKAGING["fragile"]))
    if _maybe(0.3): parts.append(_qty())
    if _maybe(0.2): parts.append(_weight())
    return " ".join(parts)


def gen_vehicles(base_desc, hs_code):
    """Chapters 86-89: Vehicles, ships, aircraft."""
    ch = int(hs_code[:2])
    parts = [base_desc.upper()]

    if ch == 87:  # Vehicles
        if _maybe(0.5): parts.append(_pick(["NEW", "USED", "SECOND HAND", "BRAND NEW"]))
        if _maybe(0.4): parts.append(_pick([
            str(random.choice([1.0, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0])) + "L",
            str(random.choice([660, 998, 1197, 1332, 1498, 1598, 1798, 1998, 2494, 2998, 3498, 4999])) + "CC",
            _pick(["DIESEL", "PETROL", "GASOLINE", "HYBRID", "ELECTRIC", "EV", "PHEV", "LPG", "CNG"]),
        ]))
        if _maybe(0.4): parts.append("VIN: " + "".join([chr(random.choice(list(range(48, 58)) + list(range(65, 91)))) for _ in range(17)]))
        if _maybe(0.3): parts.append("YEAR " + str(random.randint(2018, 2026)))
        if _maybe(0.3): parts.append(_pick(["LHD", "RHD", "LEFT HAND DRIVE", "RIGHT HAND DRIVE",
                                             "4WD", "4X4", "4X2", "6X4", "AUTOMATIC", "MANUAL"]))
    elif ch == 89:  # Ships/boats
        if _maybe(0.5): parts.append(_pick(["FIBREGLASS", "FIBERGLASS", "ALUMINUM HULL", "STEEL HULL",
                                             "INFLATABLE", "RIB"]))
        if _maybe(0.4): parts.append(str(random.choice([3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30])) + "M LOA")

    if _maybe(0.3): parts.append(_qty())
    return " ".join(parts)


def gen_instruments(base_desc, hs_code):
    """Chapter 90: Optical, medical, measuring instruments."""
    parts = [base_desc.upper()]

    if _maybe(0.5): parts.append("MODEL " + "".join([chr(random.randint(65, 90)) for _ in range(random.randint(1, 3))]) + "-" + str(random.randint(100, 9999)))
    if _maybe(0.4): parts.append(_pick(["CE MARKED", "FDA CLEARED", "ISO 13485", "CLASS I MEDICAL DEVICE",
                                         "CLASS II MEDICAL DEVICE", "IVD", "NOT FOR HUMAN USE",
                                         "FOR LABORATORY USE ONLY", "RESEARCH USE ONLY"]))
    if _maybe(0.3): parts.append(_pick(["WITH ACCESSORIES", "COMPLETE SET", "REPLACEMENT PARTS",
                                         "CONSUMABLES", "CALIBRATION KIT INCLUDED"]))
    if _maybe(0.3): parts.append(_pick(PACKAGING["fragile"]))
    else: parts.append(_pick(PACKAGING["general"]))

    if _maybe(0.3): parts.append(_qty())
    return " ".join(parts)


def gen_generic(base_desc, hs_code):
    """Fallback for chapters without specific generators."""
    parts = [base_desc.upper()]

    if _maybe(0.4): parts.append(_pick(["NEW", "BRAND NEW", "USED"]))
    if _maybe(0.3): parts.append(_pick(PACKAGING["general"]))
    if _maybe(0.3): parts.append(_qty())
    if _maybe(0.2): parts.append(_weight())
    return " ".join(parts)


# ─────────────────────────────────────────────────────────
# VAGUE DESCRIPTIONS
# ─────────────────────────────────────────────────────────

VAGUE_DESCRIPTIONS = [
    "GENERAL CARGO", "GENERAL MERCHANDISE", "SUNDRY GOODS",
    "PERSONAL EFFECTS", "HOUSEHOLD GOODS", "PERSONAL BELONGINGS",
    "SAID TO CONTAIN {base}", "STC {base}",
    "CONSOLIDATED CARGO", "MIXED CARGO",
    "VARIOUS GOODS", "AS PER INVOICE", "AS PER PROFORMA INVOICE",
    "AS PER ATTACHED LIST", "SEE ATTACHED PACKING LIST",
    "FAK", "FREIGHT ALL KINDS",
    "{base} AND RELATED ITEMS", "{base} AND ACCESSORIES",
    "PARTS AND ACCESSORIES FOR {base}",
    "SAMPLES OF {base}", "SAMPLE SHIPMENT {base}",
    "{base} NOS", "{base} NESOI", "{base} NOT ELSEWHERE SPECIFIED",
]


# ─────────────────────────────────────────────────────────
# DESCRIPTION CLEANING AND ROUTING
# ─────────────────────────────────────────────────────────

def clean_hs_description(raw_desc):
    """Clean the hierarchical HS description into a usable base description."""
    desc = re.sub(r'<\d+>', '', raw_desc)
    desc = re.sub(r'[*+]+', '', desc)
    parts = [p.strip() for p in desc.split(':') if p.strip()]
    if len(parts) >= 2:
        specific = parts[-1]
        if specific.lower().startswith('other') and len(parts) >= 3:
            specific = parts[-2] + " " + parts[-1]
        if len(specific.split()) <= 3 and len(parts) >= 2:
            context = parts[0]
            if context.lower() != specific.lower():
                specific = context + " - " + specific
        return specific.strip()
    return parts[0].strip() if parts else raw_desc.strip()


def get_generator(hs_code):
    """Route to the appropriate chapter-specific generator."""
    ch = int(hs_code[:2])
    if ch <= 5:
        return gen_animal_products
    elif ch <= 14:
        return gen_vegetable_products
    elif ch <= 24:
        return gen_food_products
    elif ch <= 38:
        return gen_chemicals
    elif ch <= 40:
        return gen_plastics_rubber
    elif ch <= 63:
        return gen_textiles
    elif ch <= 71:
        return gen_generic
    elif ch <= 83:
        return gen_metals
    elif ch <= 85:
        return gen_machinery
    elif ch <= 89:
        return gen_vehicles
    elif ch == 90:
        return gen_instruments
    else:
        return gen_generic


def generate_descriptions(hs_code, raw_desc, n_variants=3):
    """Generate n variant cargo descriptions for a given HS code."""
    base = clean_hs_description(raw_desc)
    generator = get_generator(hs_code)
    results = []

    for i in range(n_variants):
        detail_roll = random.random()
        if detail_roll < 0.08:
            template = _pick(VAGUE_DESCRIPTIONS)
            short_base = base.upper().split(' - ')[-1] if ' - ' in base else base.upper()
            desc = template.format(base=short_base)
            detail_level = "vague"
        elif detail_roll < 0.30:
            desc = base.upper()
            if _maybe(0.5):
                desc += " " + _pick(PACKAGING["general"])
            detail_level = "minimal"
        else:
            desc = generator(base, hs_code)
            detail_level = "detailed"

        desc = _apply_abbreviations(desc)
        desc = _add_noise(desc)
        desc = re.sub(r'\s+', ' ', desc).strip()
        desc = desc.upper()

        results.append({
            "hs_code": hs_code,
            "cargo_description": desc,
            "hs_description": raw_desc,
            "detail_level": detail_level,
        })

    return results


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────

def load_hs_seeds():
    """Load unique (hs6, description) pairs from harmonized-system.csv and HTS CSVs."""
    seeds = {}  # hs6 -> description

    # 1. Official harmonized-system.csv (6-digit codes)
    if HS_CSV_PATH.exists():
        with open(HS_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row["hscode"].strip().replace(".", "").replace(" ", "")
                desc = row["description"].strip()
                level = int(row["level"])
                if level == 6 and len(code) == 6:
                    seeds[code] = desc
        print(f"  Loaded {len(seeds)} 6-digit codes from harmonized-system.csv")

    # 2. HTS CSVs — extract unique 6-digit prefixes with descriptions
    hts_count = 0
    for fname in sorted(os.listdir(HTS_DIR)):
        if not fname.endswith(".csv") or fname.endswith(".json"):
            continue
        filepath = HTS_DIR / fname
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hts_raw = row.get("HTS Number", "").strip()
                if not hts_raw:
                    continue
                hts_clean = hts_raw.replace(".", "").replace(" ", "")
                # Only look at 8-10 digit codes to extract hs6 prefix
                if not re.match(r"^\d{8,10}$", hts_clean):
                    continue
                hs6 = hts_clean[:6]
                desc = row.get("Description", "").strip()
                if hs6 not in seeds and desc and desc.lower() not in ("other", "other:"):
                    seeds[hs6] = desc
                    hts_count += 1

    print(f"  Added {hts_count} additional HS6 codes from HTS CSVs")
    print(f"  Total unique HS6 seeds: {len(seeds)}")
    return seeds


def get_chapter_info(hs_code):
    """Get chapter code and name from harmonized-system.csv."""
    ch_code = hs_code[:2]
    # Load chapter names from HS dataset
    if not hasattr(get_chapter_info, "_cache"):
        get_chapter_info._cache = {}
        if HS_CSV_PATH.exists():
            with open(HS_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = row["hscode"].strip().replace(".", "").replace(" ", "")
                    if int(row["level"]) == 2 and len(code) == 2:
                        get_chapter_info._cache[code] = row["description"].strip()
    return ch_code, get_chapter_info._cache.get(ch_code, "Unknown")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print("Loading HS code seeds...")
    seeds = load_hs_seeds()

    # Calculate variants per seed to target ~100K
    target = 100_000
    n_seeds = len(seeds)
    base_variants = max(2, target // n_seeds)  # ~4 per seed for ~5600 seeds
    print(f"\n  Targeting {target} samples from {n_seeds} seeds ({base_variants} variants/seed)")

    all_records = []
    for i, (hs_code, raw_desc) in enumerate(sorted(seeds.items())):
        # Vary count: fewer for generic "Other" entries, more for specific ones
        if raw_desc.strip().lower() in ("other", "other:") or raw_desc.strip().endswith(":Other"):
            n = max(1, base_variants // 2)
        else:
            n = random.choice([base_variants - 1, base_variants, base_variants, base_variants + 1, base_variants + 2])

        variants = generate_descriptions(hs_code, raw_desc, n_variants=n)

        ch_code, ch_name = get_chapter_info(hs_code)
        for v in variants:
            all_records.append({
                "text": v["cargo_description"],
                "hs_code": hs_code,
                "hs_chapter": f"Chapter {int(ch_code)}",
                "hs_chapter_code": ch_code,
                "hs_chapter_name": ch_name,
                "hs_desc": v["hs_description"],
                "language": "en",
            })

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{n_seeds} codes, {len(all_records)} records so far")

    # Shuffle
    random.shuffle(all_records)

    # Write output
    fieldnames = ["text", "hs_code", "hs_chapter", "hs_chapter_code", "hs_chapter_name", "hs_desc", "language"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nGenerated {len(all_records)} cargo descriptions")
    print(f"Output: {OUTPUT_PATH}")

    # Stats
    unique_codes = set(r["hs_code"] for r in all_records)
    print(f"Unique HS codes: {len(unique_codes)}")

    chapter_counts = {}
    for r in all_records:
        chapter_counts[r["hs_chapter_code"]] = chapter_counts.get(r["hs_chapter_code"], 0) + 1
    print(f"\nTop 10 chapters by record count:")
    for ch, count in sorted(chapter_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  Chapter {ch}: {count}")


if __name__ == "__main__":
    main()
