"""
Bank scraping configuration.

To add a new bank:
  1. Add an entry to BANKS below
  2. Set "method" to "html", "selenium", or "api"
  3. Provide ONE root URL per category — sub-pages are discovered automatically
  4. Re-run: python scraper.py

Methods:
  - "html"     : requests + BeautifulSoup. For static/server-rendered pages.
  - "selenium" : Headless Chrome. For JS-rendered SPAs.
  - "api"      : REST API (Ardshinbank-specific). Requires "api_base".
                  Discovers pages from root URL, converts to API aliases.
"""

BANKS = {
    "Ameriabank": {
        "method": "selenium",
        "credits": "https://ameriabank.am/personal/loans",
        "deposits": "https://ameriabank.am/personal/saving",
        "branches": "https://ameriabank.am/contact-us/branches-and-atms",
    },
    "Evocabank": {
        "method": "html",
        "credits": "https://www.evoca.am/hy/loans/",
        "deposits": "https://www.evoca.am/hy/deposits/",
        "branches": "https://www.evoca.am/hy/branches-and-atms/",
    },
    "Ardshinbank": {
        "method": "api",
        "api_base": "https://ardshinbank.am/api",
        "credits": "https://www.ardshinbank.am/hy/for-you",
        "deposits": "https://www.ardshinbank.am/hy/for-you",
        "branches": "https://www.ardshinbank.am/hy/Information/branch-atm",
    },
    "Inecobank": {
        "method": "selenium",
        "credits": "https://www.inecobank.am/hy/Individual",
        "deposits": "https://www.inecobank.am/hy/Individual/deposits",
        "branches": "https://www.inecobank.am/hy/map",
    },
}

CATEGORIES = ["credits", "deposits", "branches"]
