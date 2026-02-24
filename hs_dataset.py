"""
Harmonized System dataset integration.

Loads the official HS code dataset from:
https://github.com/datasets/harmonized-system

Provides:
- Full HS code lookup (2, 4, 6 digit)
- Section/chapter/heading/subheading hierarchy
- HTS extension support (country-specific 7-10 digit codes)
- Search by description
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).parent
HS_DATA_PATH = PROJECT_DIR / "data" / "harmonized-system" / "harmonized-system.csv"
US_HTS_LOOKUP_PATH = PROJECT_DIR / "data" / "hts" / "us_hts_lookup.json"


class HSDataset:
    """Harmonized System code dataset."""
    
    def __init__(self):
        self.codes = {}        # hscode -> {section, description, parent, level}
        self.sections = {}     # section number -> section name
        self.chapters = {}     # 2-digit -> description
        self.headings = {}     # 4-digit -> description
        self.subheadings = {}  # 6-digit -> description
        self._loaded = False
    
    def load(self) -> bool:
        """Load the HS dataset from CSV."""
        if self._loaded:
            return True
            
        if not HS_DATA_PATH.exists():
            print(f"HS dataset not found at {HS_DATA_PATH}")
            return False
        
        with open(HS_DATA_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hscode = row['hscode'].strip()
                desc = row['description'].strip()
                section = row['section'].strip()
                parent = row['parent'].strip()
                level = int(row['level'])
                
                self.codes[hscode] = {
                    'section': section,
                    'description': desc,
                    'parent': parent,
                    'level': level,
                }
                
                if level == 2:
                    self.chapters[hscode] = desc
                elif level == 4:
                    self.headings[hscode] = desc
                elif level == 6:
                    self.subheadings[hscode] = desc
        
        self._loaded = True
        print(f"Loaded HS dataset: {len(self.chapters)} chapters, "
              f"{len(self.headings)} headings, {len(self.subheadings)} subheadings")
        return True
    
    def lookup(self, hscode: str) -> Optional[dict]:
        """Look up an HS code and return full hierarchy."""
        hscode = hscode.strip().replace('.', '').replace(' ', '')
        
        if hscode not in self.codes:
            return None
        
        entry = self.codes[hscode].copy()
        
        # Build hierarchy
        hierarchy = []
        current = hscode
        while current and current in self.codes and current != 'TOTAL':
            hierarchy.insert(0, {
                'code': current,
                'description': self.codes[current]['description'],
                'level': self.codes[current]['level'],
            })
            current = self.codes[current]['parent']
        
        entry['hierarchy'] = hierarchy
        entry['hscode'] = hscode
        
        # Get chapter and heading descriptions
        if len(hscode) >= 2:
            ch = hscode[:2]
            entry['chapter'] = self.chapters.get(ch, '')
            entry['chapter_code'] = ch
        if len(hscode) >= 4:
            hd = hscode[:4]
            entry['heading'] = self.headings.get(hd, '')
            entry['heading_code'] = hd
        if len(hscode) == 6:
            entry['subheading'] = self.subheadings.get(hscode, '')
        
        return entry
    
    def search(self, query: str, max_results: int = 20) -> list[dict]:
        """Search HS codes by description text."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for hscode, info in self.codes.items():
            if info['level'] != 6:
                continue
            
            desc_lower = info['description'].lower()
            
            # Score by word overlap
            desc_words = set(desc_lower.split())
            overlap = query_words & desc_words
            
            if overlap:
                score = len(overlap) / len(query_words)
                # Bonus for exact substring match
                if query_lower in desc_lower:
                    score += 1.0
                
                results.append({
                    'hscode': hscode,
                    'description': info['description'],
                    'section': info['section'],
                    'score': score,
                })
        
        results.sort(key=lambda x: -x['score'])
        return results[:max_results]
    
    def get_chapter_name(self, chapter_code: str) -> str:
        """Get chapter description from 2-digit code."""
        return self.chapters.get(chapter_code.zfill(2), 'Unknown')
    
    def validate_hs_code(self, hscode: str) -> dict:
        """Validate an HS code and return info about its validity."""
        hscode = hscode.strip().replace('.', '').replace(' ', '')
        
        result = {
            'valid': False,
            'code': hscode,
            'level': None,
            'description': None,
            'message': '',
        }
        
        if not re.match(r'^\d{2,6}$', hscode):
            result['message'] = 'HS code must be 2-6 digits'
            return result
        
        if hscode in self.codes:
            info = self.codes[hscode]
            result['valid'] = True
            result['level'] = info['level']
            result['description'] = info['description']
            result['message'] = f'Valid {info["level"]}-digit HS code'
        else:
            # Check if partial code is valid
            if len(hscode) == 6:
                heading = hscode[:4]
                chapter = hscode[:2]
                if heading in self.codes:
                    result['message'] = f'Heading {heading} exists but subheading {hscode} not found'
                elif chapter in self.codes:
                    result['message'] = f'Chapter {chapter} exists but code {hscode} not found'
                else:
                    result['message'] = f'Code {hscode} not found in HS nomenclature'
        
        return result
    
    def get_all_6digit_codes(self) -> list[dict]:
        """Return all 6-digit HS codes with descriptions."""
        return [
            {'hscode': code, 'description': info['description'], 'section': info['section']}
            for code, info in self.codes.items()
            if info['level'] == 6
        ]


# --- HTS Extensions ---
# HTS (Harmonized Tariff Schedule) adds country-specific digits (7-10) after the 6-digit HS code.
# This is a simplified reference for major trading partners.

def _load_us_hts_extensions() -> dict:
    """Load US HTS extensions from the pre-built JSON lookup table."""
    if not US_HTS_LOOKUP_PATH.exists():
        return {}
    with open(US_HTS_LOOKUP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convert from build_hts_lookup format to API format
    extensions = {}
    for hs6, entries in raw.items():
        extensions[hs6] = [
            {"hts": e["hts_code"], "description": e["description"],
             "general_duty": e.get("general_duty", ""),
             "special_duty": e.get("special_duty", ""),
             "unit": e.get("unit", "")}
            for e in entries
        ]
    return extensions


# Lazy-loaded cache for US HTS data
_us_hts_cache = None


def _get_us_hts_extensions() -> dict:
    global _us_hts_cache
    if _us_hts_cache is None:
        _us_hts_cache = _load_us_hts_extensions()
    return _us_hts_cache


HTS_EXTENSIONS = {
    "US": {
        "name": "United States HTS",
        "digits": 10,
        "format": "XXXX.XX.XXXX",
        # Extensions loaded lazily from us_hts_lookup.json
        "extensions": None,  # Sentinel — resolved in get_hts_extensions()
    },
    "EU": {
        "name": "EU Combined Nomenclature (CN)",
        "digits": 8,
        "format": "XXXX.XX.XX",
        "extensions": {
            "851712": [
                {"hts": "85171200", "description": "Telephones for cellular networks; smartphones"},
            ],
            "847130": [
                {"hts": "84713000", "description": "Portable digital automatic data-processing machines, ≤ 10 kg"},
            ],
            "870380": [
                {"hts": "87038000", "description": "Other vehicles, with electric motor for propulsion"},
            ],
        }
    },
    "CN": {
        "name": "China Customs Tariff",
        "digits": 10,
        "format": "XXXX.XXXX.XX",
        "extensions": {
            "851712": [
                {"hts": "8517120010", "description": "Smartphones, 5G capable"},
                {"hts": "8517120090", "description": "Other mobile phones"},
            ],
            "847130": [
                {"hts": "8471300000", "description": "Portable digital data processing machines"},
            ],
        }
    },
    "JP": {
        "name": "Japan HS Tariff",
        "digits": 9,
        "format": "XXXX.XX.XXX",
        "extensions": {
            "851712": [
                {"hts": "851712000", "description": "Telephones for cellular networks or wireless"},
            ],
            "870380": [
                {"hts": "870380000", "description": "Electric motor vehicles for passenger transport"},
            ],
        }
    },
}


def get_hts_extensions(hs_code: str, country_code: str) -> Optional[dict]:
    """
    Get HTS (country-specific) extensions for a 6-digit HS code.
    
    Args:
        hs_code: 6-digit HS code
        country_code: 2-letter country code (US, EU, CN, JP, etc.)
    
    Returns:
        Dict with country HTS info and available extensions, or None.
    """
    hs_code = hs_code.strip().replace('.', '').replace(' ', '')
    country_code = country_code.upper().strip()
    
    if country_code not in HTS_EXTENSIONS:
        return {
            "available": False,
            "country": country_code,
            "message": f"HTS extensions not available for {country_code}. "
                       f"Available: {', '.join(HTS_EXTENSIONS.keys())}",
            "extensions": [],
        }
    
    tariff = HTS_EXTENSIONS[country_code]
    # US extensions are lazy-loaded from JSON
    if country_code == "US":
        ext_dict = _get_us_hts_extensions()
    else:
        ext_dict = tariff["extensions"]
    extensions = ext_dict.get(hs_code, [])
    
    return {
        "available": True,
        "country": country_code,
        "tariff_name": tariff["name"],
        "total_digits": tariff["digits"],
        "format": tariff["format"],
        "extensions": extensions,
        "hs_code": hs_code,
        "message": f"Found {len(extensions)} HTS extension(s)" if extensions else
                   f"No specific extensions found for {hs_code} in {tariff['name']}. "
                   f"The base HS code {hs_code} applies.",
    }


def get_available_hts_countries() -> list[dict]:
    """Return list of countries with HTS extensions available."""
    return [
        {"code": code, "name": info["name"], "digits": info["digits"]}
        for code, info in HTS_EXTENSIONS.items()
    ]


# Singleton instance
_dataset = HSDataset()


def get_dataset() -> HSDataset:
    """Get the singleton HSDataset instance, loading if necessary."""
    if not _dataset._loaded:
        _dataset.load()
    return _dataset
