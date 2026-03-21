"""
Bank scraping configuration.

To add a new bank:
  1. Add an entry to BANKS below
  2. Set "method" to "html", "selenium", or "api"
  3. List all URLs per category (credits, deposits, branches)
  4. Re-run: python scraper.py

Methods:
  - "html"     : Fast, uses requests + BeautifulSoup. For static/server-rendered pages.
  - "selenium" : Headless Chrome. For JS-rendered SPAs (slower but handles dynamic content).
  - "api"      : REST API with JSON extraction. Requires "api_base" and uses aliases
                  instead of full URLs. Falls back to Selenium on failure.
"""

BANKS = {
    "Ameriabank": {
        "method": "selenium",
        "credits": [
            "https://ameriabank.am/personal/loans",
            "https://ameriabank.am/personal/loans/consumer-loans/consumer-loans",
            "https://ameriabank.am/personal/loans/consumer-loans/overdraft",
            "https://ameriabank.am/personal/loans/consumer-loans/credit-line",
            "https://ameriabank.am/personal/loans/consumer-loans/consumer-finance",
            "https://ameriabank.am/personal/loans/mortgage/primary-market-loan",
            "https://ameriabank.am/personal/loans/mortgage/secondary-market",
            "https://ameriabank.am/personal/loans/mortgage/commercial-mortgage",
            "https://ameriabank.am/personal/loans/mortgage/renovation-mortgage",
            "https://ameriabank.am/personal/loans/mortgage/construction-mortgage",
            "https://ameriabank.am/personal/loans/car-loan/primary",
            "https://ameriabank.am/personal/loans/car-loan/secondary-market",
            "https://ameriabank.am/personal/loans/other-loans/investment-loan",
        ],
        "deposits": [
            "https://ameriabank.am/personal/saving",
            "https://ameriabank.am/personal/saving/deposits/ameria-deposit",
            "https://ameriabank.am/personal/saving/deposits/kids-deposit",
            "https://ameriabank.am/personal/saving/deposits/cumulative-deposit",
            "https://ameriabank.am/personal/accounts/accounts/saving-account",
        ],
        "branches": [
            "https://ameriabank.am/contact-us/branches-and-atms",
        ],
    },

    "Evocabank": {
        "method": "html",
        "credits": [
            "https://www.evoca.am/hy/loans/",
            # Mortgage loans
            "https://www.evoca.am/hy/loans/mortgage-loans/residential-mortgage-loans",
            "https://www.evoca.am/hy/loans/mortgage-loans/commercial-mortgage-loans",
            "https://www.evoca.am/hy/loans/mortgage-loans/land-acquisition-loan",
            "https://www.evoca.am/hy/loans/mortgage-loans/micro-renovation-loan-with-bank-resources",
            # Consumer loans (secured)
            "https://www.evoca.am/hy/loans/consumer-loans/property-secured-personal-loans-based-on-client-creditworthiness",
            "https://www.evoca.am/hy/loans/consumer-loans/gold-secured-pawn-loans",
            "https://www.evoca.am/hy/loans/consumer-loans/deposit-secured-credit-line",
            # Unsecured loans
            "https://www.evoca.am/hy/loans/unsecured-consumer-loans/evocapower-loan",
            "https://www.evoca.am/hy/loans/unsecured-consumer-loans/guaranteed-loans",
            "https://www.evoca.am/hy/loans/unsecured-consumer-loans/overdrafts",
            "https://www.evoca.am/hy/loans/unsecured-consumer-loans/student-loans",
            # Online loans
            "https://www.evoca.am/hy/loans/online-loans/action-loan",
            "https://www.evoca.am/hy/loans/online-loans/online-overdraft",
            # Car loans
            "https://www.evoca.am/hy/loans/car-loans/car-loans",
        ],
        "deposits": [
            "https://www.evoca.am/hy/deposits/",
            "https://www.evoca.am/hy/deposits/classical-deposit",
            "https://www.evoca.am/hy/deposits/deposit-for-children",
            "https://www.evoca.am/hy/deposits/evoca-online-deposit",
        ],
        "branches": [
            "https://www.evoca.am/hy/branches-and-atms/",
        ],
    },

    "Ardshinbank": {
        "method": "api",
        "api_base": "https://ardshinbank.am/api",
        "credits": {
            "api_aliases": [
                "for-you/loans-ardshinbank",
                "for-you/consumer-loans",
                "for-you/mortgage",
                "for-you/hipotek",
            ],
            "fallback_url": "https://www.ardshinbank.am/hy/personal/loans",
        },
        "deposits": {
            "api_aliases": [
                "for-you/avand",
                "for-you/savings-account",
                "for-you/accounts",
            ],
            "fallback_url": "https://www.ardshinbank.am/hy/personal/deposits",
        },
        "branches": {
            "api_aliases": [
                "Information/branch-atm",
            ],
            "fallback_url": "https://www.ardshinbank.am/hy/Information/branch-atm",
        },
    },

    "Inecobank": {
        "method": "selenium",
        "credits": [
            "https://www.inecobank.am/hy/Individual/consumer-loans",
            "https://www.inecobank.am/hy/Individual/car-loans",
            "https://www.inecobank.am/hy/Individual/mortgage-loans",
        ],
        "deposits": [
            "https://www.inecobank.am/hy/Individual/deposits/simple",
            "https://www.inecobank.am/hy/Individual/deposits/flexible",
            "https://www.inecobank.am/hy/Individual/deposits/accumulative",
        ],
        "branches": [
            "https://www.inecobank.am/hy/map",
        ],
    },
}

CATEGORIES = ["credits", "deposits", "branches"]
