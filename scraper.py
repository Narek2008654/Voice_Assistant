"""
Generic web scraper for Armenian bank data.
Driven by banks_config.py — provide one root URL per category,
sub-pages are discovered and crawled automatically.

Supports three scraping methods:
  - "html"     : requests + BeautifulSoup (fast, for static pages)
  - "selenium" : headless Chrome (for JS-rendered SPAs)
  - "api"      : REST API with JSON extraction + Selenium fallback
"""

import json
import time
import re
import logging
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from banks_config import BANKS, CATEGORIES

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
MAX_SUBPAGES = 25

STRIP_TAGS = [
    "script", "style", "noscript", "iframe", "svg", "path",
    "meta", "link", "head",
]

# Keywords to filter discovered links by category
CATEGORY_LINK_KEYWORDS = {
    "credits": [
        "loan", "credit", "mortgage", "hipotek", "consumer", "overdraft",
        "finance", "car-loan", "vark",
    ],
    "deposits": [
        "deposit", "saving", "avand", "accumul", "flexible", "simple",
        "account", "classical",
    ],
    "branches": [
        "branch", "atm", "map", "contact", "locat",
    ],
}

SKIP_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".doc", ".docx", ".xlsx", ".zip"}


# ─── Text cleaning ───────────────────────────────────────────────────

def _clean_text(raw_text: str) -> str:
    """Clean extracted raw text: keep Armenian lines, numbers with currency, contacts."""
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


def _clean_html(html: str, strip_nav_footer: bool = True) -> str:
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


def _dedup_lines(text: str) -> str:
    """Deduplicate lines that appear multiple times."""
    seen = set()
    deduped = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            deduped.append(line)
        elif stripped not in seen:
            seen.add(stripped)
            deduped.append(line)
    return "\n".join(deduped)


# ─── Sub-page discovery ─────────────────────────────────────────────

def _discover_subpages(html: str, root_url: str, category: str) -> list[str]:
    """Auto-discover sub-page links from a root page.

    Finds links that are:
      - On the same domain
      - Under the root URL path (sub-pages)
      - OR anywhere on the domain matching category keywords (siblings)
    Then filters all results by category keywords to keep only relevant pages.
    """
    soup = BeautifulSoup(html, "html.parser")
    root_parsed = urlparse(root_url)
    root_path = root_parsed.path.rstrip("/")
    keywords = CATEGORY_LINK_KEYWORDS.get(category, [])

    # Detect language prefix from root URL (e.g., /hy/, /en/, /ru/)
    root_lang = None
    lang_match = re.match(r"/(hy|en|ru|am)/", root_parsed.path)
    if lang_match:
        root_lang = lang_match.group(1)

    candidates = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        full_url = urljoin(root_url, href)
        parsed = urlparse(full_url)

        # Same domain only
        if parsed.netloc != root_parsed.netloc:
            continue

        path = parsed.path.rstrip("/")
        if not path or path == root_path:
            continue

        # Skip other language versions (e.g., /en/, /ru/ when root is /hy/)
        if root_lang:
            other_lang = re.match(r"/(hy|en|ru|am)/", path)
            if other_lang and other_lang.group(1) != root_lang:
                continue

        # Skip file downloads
        if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
            continue

        clean_url = f"{parsed.scheme}://{parsed.netloc}{path}"

        # Sub-page of root
        if path.startswith(root_path + "/"):
            candidates.add(clean_url)
        # Or same-domain link matching category keywords
        elif keywords and any(kw in path.lower() for kw in keywords):
            candidates.add(clean_url)

    # Filter all candidates by category keywords
    if keywords and candidates:
        filtered = {url for url in candidates
                    if any(kw in urlparse(url).path.lower() for kw in keywords)}
        if filtered:
            return sorted(filtered)[:MAX_SUBPAGES]

    # Fallback: return only direct sub-pages
    sub_only = {url for url in candidates
                if urlparse(url).path.startswith(root_path + "/")}
    return sorted(sub_only)[:MAX_SUBPAGES]


# ─── HTML scraping ───────────────────────────────────────────────────

def _fetch_html(url: str) -> str | None:
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


def _scrape_html_with_discovery(root_url: str, category: str) -> list[dict]:
    """Scrape root page + auto-discovered sub-pages via HTML."""
    pages = []

    html = _fetch_html(root_url)
    if not html:
        return pages

    root_text = _clean_html(html)
    if root_text.strip():
        pages.append({"url": root_url, "text": root_text})

    sub_urls = _discover_subpages(html, root_url, category)
    logger.info(f"Discovered {len(sub_urls)} sub-pages from {root_url}")

    for url in sub_urls:
        sub_html = _fetch_html(url)
        if sub_html:
            text = _clean_html(sub_html)
            if text.strip():
                pages.append({"url": url, "text": text})
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return pages


# ─── Selenium scraping ───────────────────────────────────────────────

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


def _scrape_selenium_page(url: str, wait_seconds: int = 10) -> tuple[str, str]:
    """Scrape a single JS-rendered page. Returns (cleaned_text, page_source_html)."""
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

        cleaned_from_source = _clean_html(page_source, strip_nav_footer=True)
        cleaned_from_text = _clean_text(body_text)

        best_text = cleaned_from_source if len(cleaned_from_source) > len(cleaned_from_text) else cleaned_from_text
        return best_text, page_source

    except Exception as e:
        logger.error(f"Selenium error for {url}: {e}")
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        return "", ""


def _scrape_selenium_with_discovery(root_url: str, category: str) -> list[dict]:
    """Scrape root page + auto-discovered sub-pages via Selenium."""
    pages = []

    root_text, page_source = _scrape_selenium_page(root_url)
    if root_text.strip():
        pages.append({"url": root_url, "text": root_text})

    if not page_source:
        return pages

    sub_urls = _discover_subpages(page_source, root_url, category)
    logger.info(f"Discovered {len(sub_urls)} sub-pages from {root_url}")

    for url in sub_urls:
        text, _ = _scrape_selenium_page(url)
        if text.strip():
            pages.append({"url": url, "text": text})
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return pages


# ─── API scraping (JSON with embedded HTML) ──────────────────────────

def _extract_armenian_from_json(data) -> str:
    """Recursively extract Armenian text from API JSON (handles HTML in strings)."""
    texts = []

    def _walk(obj):
        if isinstance(obj, str):
            if re.search(r"[\u0530-\u058F]", obj) or re.search(r"\+374", obj):
                if "<" in obj and ">" in obj:
                    soup = BeautifulSoup(obj, "html.parser")
                    clean = soup.get_text(separator="\n").strip()
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

    return "\n".join(unique)


def _scrape_api_with_discovery(root_url: str, api_base: str, category: str) -> list[dict]:
    """Discover sub-pages from root web URL, fetch each via API."""
    pages = []

    # Fetch root page to discover links
    html = _fetch_html(root_url)
    if not html:
        # Fallback: try Selenium for root page
        text, _ = _scrape_selenium_page(root_url)
        if text.strip():
            return [{"url": root_url, "text": text}]
        return []

    # Discover sub-page URLs
    sub_urls = _discover_subpages(html, root_url, category)
    all_urls = [root_url] + sub_urls
    logger.info(f"API: discovered {len(sub_urls)} sub-pages from {root_url}")

    # Convert each web URL to an API alias and fetch
    for url in all_urls:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        # Strip /hy/ prefix to get API alias (e.g., /hy/for-you/avand → for-you/avand)
        alias = path[3:] if path.startswith("hy/") else path

        api_url = f"{api_base}/pages/alias/{alias}"
        try:
            resp = requests.get(
                api_url,
                headers={**HEADERS, "Accept": "application/json"},
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("page", {}).get("content", [])
                text = _extract_armenian_from_json(content) if content else _extract_armenian_from_json(data)
                if text.strip():
                    pages.append({"url": api_url, "text": text})
                    logger.info(f"API {alias}: {len(text)} chars")
        except Exception as e:
            logger.warning(f"API {alias} failed: {e}")
        time.sleep(1)

    # Fallback to Selenium if API returned nothing
    if not pages:
        logger.info(f"API returned nothing, falling back to Selenium: {root_url}")
        text, _ = _scrape_selenium_page(root_url)
        if text.strip():
            pages.append({"url": root_url, "text": text})

    return pages


# ─── Main orchestration ──────────────────────────────────────────────

def _scrape_bank_category(bank_name: str, bank_config: dict, category: str) -> list[dict]:
    """Scrape one bank/category combo using the configured method."""
    method = bank_config["method"]
    root_url = bank_config.get(category)

    if not root_url:
        return []

    if method == "html":
        return _scrape_html_with_discovery(root_url, category)

    elif method == "selenium":
        return _scrape_selenium_with_discovery(root_url, category)

    elif method == "api":
        api_base = bank_config["api_base"]
        return _scrape_api_with_discovery(root_url, api_base, category)

    else:
        logger.error(f"Unknown method '{method}' for {bank_name}")
        return []


def scrape_all() -> list[dict]:
    """Scrape all banks defined in config. Returns one entry per page."""
    results = []

    for bank_name, bank_config in BANKS.items():
        for category in CATEGORIES:
            logger.info(f"Scraping {bank_name} / {category}")

            pages = _scrape_bank_category(bank_name, bank_config, category)

            if not pages:
                logger.info(f"  -> empty: 0 pages")
                continue

            for page in pages:
                text = _dedup_lines(page["text"])
                if not text.strip():
                    continue
                results.append({
                    "bank": bank_name,
                    "category": category,
                    "url": page["url"],
                    "text": text,
                    "char_count": len(text),
                })

            logger.info(f"  -> {len(pages)} pages scraped")

    return results


def save_results(results: list[dict]) -> None:
    """Save scraped data to JSON."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results)} entries to {OUTPUT_PATH}")


def print_summary(results: list[dict]) -> None:
    """Print a summary of scraping results."""
    print("\n" + "=" * 70)
    print("SCRAPING SUMMARY")
    print("=" * 70)

    groups = defaultdict(list)
    for r in results:
        groups[(r["bank"], r["category"])].append(r)

    for bank_name in BANKS:
        for cat in CATEGORIES:
            pages = groups.get((bank_name, cat), [])
            total = sum(p["char_count"] for p in pages)
            print(f"  {bank_name:15s} | {cat:10s} | {len(pages):2d} pages | {total:>6d} chars")

    print("=" * 70)
    total_chars = sum(r["char_count"] for r in results)
    print(f"  Total: {len(results)} pages, {total_chars:,} chars")
    print(f"  Output: {OUTPUT_PATH}\n")


def main():
    logger.info("Starting bank data scraper...")
    results = scrape_all()
    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
