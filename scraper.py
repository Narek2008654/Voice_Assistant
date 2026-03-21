"""
Web scraper for 4 Armenian banks: Ameriabank, Evocabank, Ardshinbank, Inecobank.
Extracts loan, deposit, and branch information in Armenian.
Saves cleaned text to data/bank_data.json.

Each bank uses a different strategy:
- Ameriabank: Selenium (JS-rendered pages) with sub-page crawling
- Evocabank: HTML (requests + BeautifulSoup) with sub-page crawling
- Ardshinbank: REST API + Selenium fallback for branches (regional pages)
- Inecobank: Selenium (Cloudflare-protected) with sub-page crawling
"""

import json
import time
import re
import logging
from pathlib import Path
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "hy-AM,hy;q=0.9,en;q=0.5",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 2
OUTPUT_PATH = Path(__file__).parent / "data" / "bank_data.json"

STRIP_TAGS = [
    "script", "style", "noscript", "iframe", "svg", "path",
    "meta", "link", "head",
]


# ─── Generic HTML helpers ─────────────────────────────────────────────

def fetch_html(url: str) -> str | None:
    """Fetch raw HTML from a URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        logger.info(f"Fetched {url} ({resp.status_code}, {len(resp.text)} chars)")
        return resp.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def clean_html(html: str, strip_nav_footer: bool = True) -> str:
    """Extract clean text from HTML, stripping boilerplate."""
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    if strip_nav_footer:
        for tag_name in ["header", "footer", "nav"]:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        for selector in [
            "[class*='cookie']", "[class*='popup']", "[class*='modal']",
            "[class*='social']", "[class*='breadcrumb']",
            "[id*='cookie']",
        ]:
            try:
                for el in soup.select(selector):
                    el.decompose()
            except Exception:
                pass

    raw_text = soup.get_text(separator="\n")
    return _clean_text(raw_text)


def _clean_text(raw_text: str) -> str:
    """Clean extracted raw text: remove blanks, JS artifacts, non-content lines."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"[\u0530-\u058F]", line):
            cleaned_lines.append(line)
        elif re.search(r"\d", line) and re.search(r"[%֏$€₽AMD|USD|EUR]", line):
            cleaned_lines.append(line)
        elif re.search(r"(\+374|@|\.am)", line):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)



def _scrape_multiple_html(urls: list[str], separator: str = "\n\n---\n\n") -> str:
    """Scrape multiple HTML pages and concatenate cleaned text."""
    parts = []
    for url in urls:
        html = fetch_html(url)
        if html:
            text = clean_html(html)
            if text.strip():
                parts.append(text)
        time.sleep(DELAY_BETWEEN_REQUESTS)
    return separator.join(parts)


# ─── Selenium helpers ────────────────────────────────────────────────

def _get_selenium_driver():
    """Create a headless Chrome driver."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(f"--user-agent={HEADERS['User-Agent']}")
    opts.add_argument("--lang=hy-AM")
    opts.add_experimental_option("prefs", {"intl.accept_languages": "hy-AM,hy"})
    return webdriver.Chrome(options=opts)


def _scrape_with_selenium(url: str, wait_seconds: int = 10) -> str:
    """Scrape a JS-rendered page using Selenium."""
    driver = None
    try:
        driver = _get_selenium_driver()
        logger.info(f"Selenium: loading {url}")
        driver.get(url)
        time.sleep(wait_seconds)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        page_source = driver.page_source
        body_text = driver.find_element("tag name", "body").text

        driver.quit()
        driver = None

        cleaned_from_source = clean_html(page_source, strip_nav_footer=True)
        cleaned_from_text = _clean_text(body_text)

        if len(cleaned_from_source) > len(cleaned_from_text):
            return cleaned_from_source
        return cleaned_from_text

    except Exception as e:
        logger.error(f"Selenium error for {url}: {e}")
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        return ""


def _scrape_multiple_selenium(urls: list[str], separator: str = "\n\n---\n\n") -> str:
    """Scrape multiple Selenium pages and concatenate."""
    parts = []
    for url in urls:
        text = _scrape_with_selenium(url)
        if text.strip():
            parts.append(text)
        time.sleep(DELAY_BETWEEN_REQUESTS)
    return separator.join(parts)


# ─── Evocabank: HTML with sub-page crawling ──────────────────────────

EVOCABANK_CREDIT_SUBPAGES = [
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
]

EVOCABANK_DEPOSIT_SUBPAGES = [
    "https://www.evoca.am/hy/deposits/classical-deposit",
    "https://www.evoca.am/hy/deposits/deposit-for-children",
    "https://www.evoca.am/hy/deposits/evoca-online-deposit",
]


def scrape_evocabank(category: str) -> str:
    """Scrape Evocabank with sub-page crawling."""
    if category == "credits":
        # Main overview + all detail sub-pages
        main_html = fetch_html("https://www.evoca.am/hy/loans/")
        main_text = clean_html(main_html) if main_html else ""
        sub_text = _scrape_multiple_html(EVOCABANK_CREDIT_SUBPAGES)
        return f"{main_text}\n\n---\n\n{sub_text}" if main_text else sub_text

    elif category == "deposits":
        # Main overview + all detail sub-pages
        main_html = fetch_html("https://www.evoca.am/hy/deposits/")
        main_text = clean_html(main_html) if main_html else ""
        sub_text = _scrape_multiple_html(EVOCABANK_DEPOSIT_SUBPAGES)
        return f"{main_text}\n\n---\n\n{sub_text}" if main_text else sub_text

    elif category == "branches":
        html = fetch_html("https://www.evoca.am/hy/branches-and-atms/")
        return clean_html(html) if html else ""

    return ""


# ─── Ameriabank: Selenium with sub-page crawling ────────────────────

AMERIABANK_CREDIT_SUBPAGES = [
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
]

AMERIABANK_DEPOSIT_SUBPAGES = [
    "https://ameriabank.am/personal/saving/deposits/ameria-deposit",
    "https://ameriabank.am/personal/saving/deposits/kids-deposit",
    "https://ameriabank.am/personal/saving/deposits/cumulative-deposit",
    "https://ameriabank.am/personal/accounts/accounts/saving-account",
]


def scrape_ameriabank(category: str) -> str:
    """Scrape Ameriabank with sub-page crawling via Selenium."""
    if category == "credits":
        main_text = _scrape_with_selenium("https://ameriabank.am/personal/loans")
        sub_text = _scrape_multiple_selenium(AMERIABANK_CREDIT_SUBPAGES)
        return f"{main_text}\n\n---\n\n{sub_text}" if main_text else sub_text

    elif category == "deposits":
        main_text = _scrape_with_selenium("https://ameriabank.am/personal/saving")
        sub_text = _scrape_multiple_selenium(AMERIABANK_DEPOSIT_SUBPAGES)
        return f"{main_text}\n\n---\n\n{sub_text}" if main_text else sub_text

    elif category == "branches":
        return _scrape_with_selenium("https://ameriabank.am/contact-us/branches-and-atms")

    return ""


# ─── Ardshinbank: REST API + Selenium for branches ──────────────────

ARDSHINBANK_API_BASE = "https://ardshinbank.am/api"

ARDSHINBANK_API_PAGES = {
    "credits": [
        "for-you/loans-ardshinbank",
        "for-you/consumer-loans",
        "for-you/mortgage",
        "for-you/hipotek",
    ],
    "deposits": [
        "for-you/avand",
        "for-you/savings-account",
        "for-you/accounts",
    ],
}

def _extract_text_from_ardshin_json(data, separator: str = "\n") -> str:
    """Recursively extract Armenian text from Ardshinbank API JSON response.
    Handles HTML embedded in JSON strings."""
    texts = []

    def _walk(obj):
        if isinstance(obj, str):
            if re.search(r"[\u0530-\u058F]", obj) or re.search(r"\+374", obj):
                if "<" in obj and ">" in obj:
                    soup = BeautifulSoup(obj, "html.parser")
                    clean = soup.get_text(separator="\n").strip()
                    # Keep lines with Armenian chars, phone numbers, or addresses
                    lines = []
                    for line in clean.splitlines():
                        line = line.strip()
                        if line and (re.search(r"[\u0530-\u058F]", line) or re.search(r"\+374", line)):
                            lines.append(line)
                    if lines:
                        texts.append("\n".join(lines))
                else:
                    texts.append(obj.strip())
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)

    seen = set()
    unique = []
    for t in texts:
        if t not in seen and len(t) > 1:
            seen.add(t)
            unique.append(t)

    return separator.join(unique)


def scrape_ardshinbank(category: str) -> str:
    """Scrape Ardshinbank via API (credits/deposits/branches)."""
    if category == "branches":
        # Branch data is in the branch-atm API page, deeply nested with HTML in JSON
        logger.info("Scraping Ardshinbank branches via API (branch-atm page)...")
        url = f"{ARDSHINBANK_API_BASE}/pages/alias/Information/branch-atm"
        try:
            resp = requests.get(
                url,
                headers={**HEADERS, "Accept": "application/json"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            # Extract from the page content (where branch data lives)
            content = data.get("page", {}).get("content", [])
            text = _extract_text_from_ardshin_json(content)
            if text.strip():
                logger.info(f"Ardshinbank branches API: {len(text)} chars")
                return text
        except Exception as e:
            logger.warning(f"Ardshinbank branches API failed: {e}")

        # Fallback to Selenium
        return _scrape_with_selenium("https://www.ardshinbank.am/hy/Information/branch-atm")

    # Credits or deposits: try multiple API page aliases
    aliases = ARDSHINBANK_API_PAGES.get(category, [])
    parts = []
    for alias in aliases:
        url = f"{ARDSHINBANK_API_BASE}/pages/alias/{alias}"
        try:
            resp = requests.get(
                url,
                headers={**HEADERS, "Accept": "application/json"},
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                text = _extract_text_from_ardshin_json(resp.json())
                if text.strip():
                    parts.append(text)
                    logger.info(f"Ardshinbank API {alias}: {len(text)} chars")
        except Exception as e:
            logger.warning(f"Ardshinbank API {alias} failed: {e}")
        time.sleep(1)

    if not parts:
        # Fallback to Selenium
        fallback_urls = {
            "credits": "https://www.ardshinbank.am/hy/personal/loans",
            "deposits": "https://www.ardshinbank.am/hy/personal/deposits",
        }
        fallback_url = fallback_urls.get(category)
        if fallback_url:
            text = _scrape_with_selenium(fallback_url)
            if text.strip():
                parts.append(text)

    return "\n\n---\n\n".join(parts)


# ─── Inecobank: Selenium with sub-page crawling ─────────────────────

INECOBANK_CREDIT_URLS = [
    "https://www.inecobank.am/hy/Individual/consumer-loans",
    "https://www.inecobank.am/hy/Individual/car-loans",
    "https://www.inecobank.am/hy/Individual/mortgage-loans",
]

INECOBANK_DEPOSIT_URLS = [
    "https://www.inecobank.am/hy/Individual/deposits/simple",
    "https://www.inecobank.am/hy/Individual/deposits/flexible",
    "https://www.inecobank.am/hy/Individual/deposits/accumulative",
]


def scrape_inecobank(category: str) -> str:
    """Scrape Inecobank via Selenium."""
    if category == "credits":
        return _scrape_multiple_selenium(INECOBANK_CREDIT_URLS)
    elif category == "deposits":
        return _scrape_multiple_selenium(INECOBANK_DEPOSIT_URLS)
    elif category == "branches":
        return _scrape_with_selenium("https://www.inecobank.am/hy/map")
    return ""


# ─── Main orchestration ──────────────────────────────────────────────

BANK_SCRAPERS = {
    "Ameriabank": scrape_ameriabank,
    "Evocabank": scrape_evocabank,
    "Ardshinbank": scrape_ardshinbank,
    "Inecobank": scrape_inecobank,
}

CATEGORIES = ["credits", "deposits", "branches"]


def scrape_all() -> list[dict]:
    """Scrape all banks and return structured data."""
    results = []

    for bank, scraper_fn in BANK_SCRAPERS.items():
        for category in CATEGORIES:
            logger.info(f"Scraping {bank} / {category}")

            text = scraper_fn(category)

            # Deduplicate lines that appear multiple times (from overlapping sub-pages)
            if text:
                seen_lines = set()
                deduped = []
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped == "---" or stripped == "":
                        deduped.append(line)
                    elif stripped not in seen_lines:
                        seen_lines.add(stripped)
                        deduped.append(line)
                text = "\n".join(deduped)

            status = "ok" if text else "empty"
            results.append({
                "bank": bank,
                "category": category,
                "text": text,
                "status": status,
                "char_count": len(text) if text else 0,
            })

            logger.info(f"  -> {status}: {len(text) if text else 0} chars")

    return results


def save_results(results: list[dict]) -> None:
    """Save scraped data to JSON."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results)} entries to {OUTPUT_PATH}")


def print_summary(results: list[dict]) -> None:
    """Print a summary of scraping results."""
    print("\n" + "=" * 60)
    print("SCRAPING SUMMARY")
    print("=" * 60)
    for r in results:
        icon = "OK" if r["char_count"] > 0 else "EMPTY"
        print(f"  [{icon:5s}] {r['bank']:15s} | {r['category']:10s} | {r['char_count']:>6d} chars")
    print("=" * 60)

    ok_count = sum(1 for r in results if r["char_count"] > 0)
    total_chars = sum(r["char_count"] for r in results)
    print(f"  Content: {ok_count}/{len(results)} pages with data")
    print(f"  Total:   {total_chars:,} chars")
    print(f"  Output:  {OUTPUT_PATH}\n")


def main():
    logger.info("Starting bank data scraper (with sub-page crawling)...")
    results = scrape_all()
    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
