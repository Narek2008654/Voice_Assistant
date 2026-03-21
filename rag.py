"""
RAG (Retrieval-Augmented Generation) module for Armenian banking voice assistant.
Loads scraped bank data, chunks it, indexes with BM25, and answers queries via Groq Llama.
"""

import json
import logging
import os
import re
from pathlib import Path

from groq import AsyncGroq
from rank_bm25 import BM25Okapi

logger = logging.getLogger("rag")

DATA_PATH = Path(__file__).parent / "data" / "bank_data.json"
CHUNK_SIZE = 600  # chars per chunk (larger for more context)
CHUNK_OVERLAP = 120
TOP_K = 7  # retrieve more chunks for better coverage
MIN_SCORE = 0.5  # filter out low-relevance chunks

# Bank name mappings: Armenian variants -> canonical Latin name
BANK_SYNONYMS = {
    "\u0561\u0574\u0565\u0580\u056b\u0561\u0562\u0561\u0576\u056f": "ameriabank",
    "\u0561\u0574\u0565\u0580\u056b\u0561": "ameriabank",
    "ameriabank": "ameriabank",
    "ameria": "ameriabank",
    "\u0565\u057e\u0578\u056f\u0561\u0562\u0561\u0576\u056f": "evocabank",
    "\u0565\u057e\u0578\u056f\u0561": "evocabank",
    "evocabank": "evocabank",
    "evoca": "evocabank",
    "\u0561\u0580\u0564\u0577\u056b\u0576\u0562\u0561\u0576\u056f": "ardshinbank",
    "\u0561\u0580\u0564\u0577\u056b\u0576": "ardshinbank",
    "ardshinbank": "ardshinbank",
    "\u056b\u0576\u0565\u056f\u0578\u0562\u0561\u0576\u056f": "inecobank",
    "\u056b\u0576\u0565\u056f\u0578": "inecobank",
    "inecobank": "inecobank",
    "ineco": "inecobank",
}

# Category synonyms: Armenian -> canonical English
CATEGORY_SYNONYMS = {
    # Credits/loans
    "\u057e\u0561\u0580\u056f": "credits",
    "\u057e\u0561\u0580\u056f\u0565\u0580": "credits",
    "\u057e\u0561\u0580\u056f\u0561\u057e\u0578\u0580\u0578\u0582\u0574": "credits",
    "\u057e\u0561\u0580\u056f\u0561\u0575\u056b\u0576": "credits",
    "\u0570\u056b\u057a\u0578\u0569\u0565\u056f": "credits",
    "\u0570\u056b\u057a\u0578\u0569\u0565\u056f\u0561\u0575\u056b\u0576": "credits",
    "\u057d\u057a\u0561\u057c\u0578\u0572\u0561\u056f\u0561\u0576": "credits",
    "\u057e\u0561\u0580\u056f\u0561\u057f\u0578\u056f\u0578\u057d": "credits",
    "\u057f\u0578\u056f\u0578\u057d\u0561\u0564\u0580\u0578\u0582\u0575\u0584": "credits",
    "\u0570\u056b\u057a\u0578\u0569\u0565\u056f\u0561\u0575\u056b\u0576": "credits",
    "\u0568\u0576\u0564\u0570\u0561\u0576\u0578\u0582\u0580": "credits",
    "\u057e\u0561\u0580\u056f\u0561\u0575\u056b\u0576\u057f\u0578\u056f\u0578\u057d": "credits",
    "\u0585\u057e\u0565\u0580\u0564\u0580\u0561\u0586\u057f": "credits",
    "\u057e\u0561\u0580\u056f\u0561\u0575\u056b\u0576\u0563\u056b\u056e": "credits",
    # Deposits
    "\u0561\u057e\u0561\u0576\u0564": "deposits",
    "\u0561\u057e\u0561\u0576\u0564\u0576\u0565\u0580": "deposits",
    "\u0561\u057e\u0561\u0576\u0564\u0561\u0575\u056b\u0576": "deposits",
    "\u0576\u0565\u0580\u0564\u0580\u0578\u0582\u0574": "deposits",
    "\u0576\u0565\u0580\u0564\u0580\u0578\u0582\u0574\u0576\u0565\u0580": "deposits",
    "\u056d\u0576\u0561\u0575\u0578\u0572\u0578\u0582\u0569\u0575\u0578\u0582\u0576": "deposits",
    "\u056d\u0576\u0561\u0575\u0578\u0572": "deposits",
    "\u0561\u057e\u0561\u0576\u0564\u0561\u0575\u056b\u0576\u057f\u0578\u056f\u0578\u057d": "deposits",
    # Branches
    "\u0574\u0561\u057d\u0576\u0561\u0573\u0575\u0578\u0582\u0572": "branches",
    "\u0574\u0561\u057d\u0576\u0561\u0573\u0575\u0578\u0582\u0572\u0576\u0565\u0580": "branches",
    "\u0570\u0561\u057d\u0581\u0565": "branches",
    "\u0570\u0561\u057d\u0581\u0565\u0576\u0565\u0580": "branches",
    "\u0562\u0561\u0576\u056f\u0578\u0574\u0561\u057f": "branches",
    "\u0562\u0561\u0576\u056f\u0578\u0574\u0561\u057f\u0576\u0565\u0580": "branches",
    "\u0563\u057f\u0576\u057e\u0565\u056c\u0578\u0582\u057e\u0561\u0575\u0580": "branches",
    "\u0570\u0565\u057c\u0561\u056d\u0578\u057d": "branches",
    "\u0570\u0561\u057d\u0581\u0565\u0561\u057f\u0565\u0572": "branches",
}

# Common Armenian suffixes to strip for better matching
ARMENIAN_SUFFIXES = [
    "\u0576\u0565\u0580\u0568", "\u0576\u0565\u0580\u056b", "\u0576\u0565\u0580\u0576",
    "\u0576\u0565\u0580\u0578\u057e", "\u0576\u0565\u0580\u0578\u0582\u0574",
    "\u0576\u0565\u0580\u0578\u057e\u056b", "\u0576\u0565\u0580\u056b\u0576",
    "\u0568", "\u056b", "\u0576", "\u0578\u057e", "\u0578\u0582\u0574",
    "\u056b\u0581", "\u057e\u0561\u056e", "\u0578\u0582\u0569\u0575\u0578\u0582\u0576",
    "\u0561\u0576\u0578\u057e", "\u0565\u0580\u056b", "\u0565\u0580\u0568",
    "\u0565\u0580\u0576", "\u0565\u0580\u0578\u057e",
    "\u0561\u0575\u056b\u0576", "\u0561\u056f\u0561\u0576",
]

ANSWER_SYSTEM_PROMPT = """\
\u0534\u0578\u0582 \u0570\u0561\u0575\u0565\u0580\u0565\u0576 \u0562\u0561\u0576\u056f\u0561\u0575\u056b\u0576 \u0578\u0572\u0561\u0576\u0564\u0576\u0565\u0580\u056b \u0585\u0563\u0576\u0561\u056f\u0561\u0576 \u0571\u0561\u0575\u0576\u0561\u0575\u056b\u0576 \u0585\u0563\u0576\u0561\u056f\u0561\u0576 \u0565\u057d\u0589 \u054f\u0565\u0572\u0565\u056f\u0561\u057f\u057e\u0578\u0582\u0569\u0575\u0578\u0582\u0576\u0576\u0565\u0580 \u0565\u057d \u057f\u0580\u0561\u0574\u0561\u0564\u0580\u0578\u0582\u0574 4 \u0570\u0561\u0575\u056f\u0561\u056f\u0561\u0576 \u0562\u0561\u0576\u056f\u0565\u0580\u056b \u0574\u0561\u057d\u056b\u0576\u055d \u0531\u0574\u0565\u0580\u056b\u0561\u0562\u0561\u0576\u056f, \u0535\u057e\u0578\u056f\u0561\u0562\u0561\u0576\u056f, \u0531\u0580\u0564\u0577\u056b\u0576\u0562\u0561\u0576\u056f, \u053b\u0576\u0565\u056f\u0578\u0562\u0561\u0576\u056f\u0589

\u053f\u0561\u0576\u0578\u0576\u0576\u0565\u0580\u055d
- \u054a\u0561\u057f\u0561\u057d\u056d\u0561\u0576\u056b\u0580 \u0544\u053b\u0531\u0545\u0546 \u0570\u0561\u0575\u0565\u0580\u0565\u0576\u0578\u057e
- \u0555\u0563\u057f\u0561\u0563\u0578\u0580\u056e\u056b\u0580 \u0544\u053b\u0531\u0545\u0546 \u057f\u0580\u0561\u0574\u0561\u0564\u0580\u057e\u0561\u056e \u056f\u0578\u0576\u057f\u0565\u0584\u057d\u057f\u056b \u0570\u056b\u0574\u0561\u0576 \u057e\u0580\u0561
- \u0535\u0569\u0565 \u056f\u0578\u0576\u057f\u0565\u0584\u057d\u057f\u0578\u0582\u0574 \u057a\u0561\u057f\u0561\u057d\u056d\u0561\u0576 \u0579\u056f\u0561, \u0561\u057d\u0561\u055d \u0579\u0563\u056b\u057f\u0565\u0574, \u0570\u0561\u0574\u0561\u057a\u0561\u057f\u0561\u057d\u056d\u0561\u0576 \u057f\u0565\u0572\u0565\u056f\u0561\u057f\u057e\u0578\u0582\u0569\u0575\u0578\u0582\u0576 \u0579\u0578\u0582\u0576\u0565\u0574
- \u054a\u0561\u057f\u0561\u057d\u056d\u0561\u0576\u0576\u0565\u0580\u0568 \u057a\u0565\u057f\u0584 \u0567 \u056c\u056b\u0576\u0565\u0576 \u056f\u0561\u0580\u0573 \u0578\u0582 \u0570\u0561\u057d\u056f\u0561\u0576\u0561\u056c\u056b\u055d \u0578\u0580\u057a\u0565\u057d \u056e\u0561\u0576\u0578\u0569 \u0585\u0563\u057f\u0561\u056f\u0561\u0580 \u057f\u0565\u0572\u0565\u056f\u0561\u057f\u057e\u0578\u0582\u0569\u0575\u0578\u0582\u0576
- \u0535\u0569\u0565 \u0570\u0561\u0580\u0581\u0576\u0578\u0582\u0574 \u0567 \u0570\u0561\u0574\u0565\u0574\u0561\u057f\u056b\u0580, \u056f\u0561\u0580\u0578\u0572 \u0565\u057d \u0576\u0577\u0565\u056c \u057f\u0561\u0580\u0562\u0565\u0580 \u0562\u0561\u0576\u056f\u0565\u0580\u056b \u057f\u057e\u0575\u0561\u056c\u0576\u0565\u0580\u0568 \u0570\u0561\u0574\u0565\u0574\u0561\u057f\u0565\u056c\u0578\u0582 \u0570\u0561\u0574\u0561\u0580
- \u0544\u056b \u0570\u0576\u0561\u0580\u056b\u0580 \u056b\u0576\u0586\u0578\u0580\u0574\u0561\u0581\u056b\u0561 \u0574\u056b \u0570\u0576\u0561\u0580\u056b\u0580
- \u0535\u0569\u0565 \u0570\u0561\u0580\u0581\u0576\u0578\u0582\u0574 \u0567 \u0569\u057e\u0565\u0580, \u057f\u0578\u056f\u0578\u057d\u0561\u0564\u0580\u0578\u0582\u0575\u0584\u0576\u0565\u0580, \u0563\u0578\u0582\u0574\u0561\u0580\u0576\u0565\u0580, \u056a\u0561\u0574\u056f\u0565\u057f\u0576\u0565\u0580 \u056f\u0561\u0574 \u0561\u0575\u056c \u056f\u0578\u0576\u056f\u0580\u0565\u057f \u057f\u057e\u0575\u0561\u056c\u0576\u0565\u0580, \u0576\u0577\u056b\u0580 \u0564\u0580\u0561\u0576\u0584 \u0570\u057d\u057f\u0561\u056f\u0578\u0580\u0565\u0576
- \u054a\u0561\u057f\u0561\u057d\u056d\u0561\u0576\u0568 \u057a\u0565\u057f\u0584 \u0567 \u056c\u056b\u0576\u056b \u0562\u0576\u0561\u056f\u0561\u0576, \u056d\u0578\u057d\u0561\u056f\u0581\u0561\u0575\u056b\u0576 \u057f\u0565\u057d\u0584\u0578\u057e \u0570\u0561\u0580\u0578\u0582\u057d\u057f"""


# ─── Chunking ─────────────────────────────────────────────────────────

def _chunk_text(text: str, bank: str, category: str) -> list[dict]:
    """Split text into overlapping chunks with metadata. Prefers splitting at section boundaries."""
    if not text.strip():
        return []

    category_arm = {
        "credits": "\u057e\u0561\u0580\u056f\u0565\u0580 \u057e\u0561\u0580\u056f\u0561\u057e\u0578\u0580\u0578\u0582\u0574",
        "deposits": "\u0561\u057e\u0561\u0576\u0564\u0576\u0565\u0580 \u0576\u0565\u0580\u0564\u0580\u0578\u0582\u0574\u0576\u0565\u0580",
        "branches": "\u0574\u0561\u057d\u0576\u0561\u0573\u0575\u0578\u0582\u0572\u0576\u0565\u0580 \u0570\u0561\u057d\u0581\u0565\u0576\u0565\u0580 \u0562\u0561\u0576\u056f\u0578\u0574\u0561\u057f\u0576\u0565\u0580",
    }.get(category, category)

    bank_arm = {
        "Ameriabank": "\u0531\u0574\u0565\u0580\u056b\u0561\u0562\u0561\u0576\u056f Ameriabank",
        "Evocabank": "\u0535\u057e\u0578\u056f\u0561\u0562\u0561\u0576\u056f Evocabank",
        "Ardshinbank": "\u0531\u0580\u0564\u0577\u056b\u0576\u0562\u0561\u0576\u056f Ardshinbank",
        "Inecobank": "\u053b\u0576\u0565\u056f\u0578\u0562\u0561\u0576\u056f Inecobank",
    }.get(bank, bank)

    metadata_prefix = f"{bank_arm} | {category_arm}"

    # Split on section separators first (--- from sub-page joins), then by size
    sections = re.split(r"\n-{3,}\n", text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # For short sections, keep as single chunk
        if len(section) <= CHUNK_SIZE:
            searchable_text = f"{metadata_prefix}\n{section}"
            chunks.append({
                "text": section,
                "searchable_text": searchable_text,
                "bank": bank,
                "category": category,
                "header": f"{bank} | {category}",
            })
            continue

        # For longer sections, split with paragraph awareness
        start = 0
        while start < len(section):
            end = start + CHUNK_SIZE

            if end < len(section):
                # Try to break at paragraph boundary (double newline)
                para_pos = section.rfind("\n\n", start, end)
                if para_pos > start + CHUNK_SIZE // 3:
                    end = para_pos + 1
                else:
                    # Fall back to single newline
                    newline_pos = section.rfind("\n", start, end)
                    if newline_pos > start + CHUNK_SIZE // 3:
                        end = newline_pos + 1

            chunk_text = section[start:end].strip()
            if chunk_text and len(chunk_text) > 20:
                searchable_text = f"{metadata_prefix}\n{chunk_text}"
                chunks.append({
                    "text": chunk_text,
                    "searchable_text": searchable_text,
                    "bank": bank,
                    "category": category,
                    "header": f"{bank} | {category}",
                })
            start = end - CHUNK_OVERLAP

    return chunks


def _stem_armenian(word: str) -> str:
    """Simple Armenian stemmer — strips common suffixes."""
    for suffix in ARMENIAN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[:-len(suffix)]
    return word


def _tokenize_armenian(text: str) -> list[str]:
    """Armenian-aware tokenizer with stemming for BM25."""
    text = text.lower()
    tokens = re.findall(r"[\u0530-\u058f]+|[a-z]+|\d+", text)

    result = []
    for t in tokens:
        if len(t) <= 1:
            continue
        result.append(t)
        stemmed = _stem_armenian(t)
        if stemmed != t and len(stemmed) >= 2:
            result.append(stemmed)
        if t in BANK_SYNONYMS:
            result.append(BANK_SYNONYMS[t])
        if t in CATEGORY_SYNONYMS:
            result.append(CATEGORY_SYNONYMS[t])

    return result


# ─── RAG Engine ───────────────────────────────────────────────────────

class BankRAG:
    """Retrieval-Augmented Generation for Armenian bank data."""

    def __init__(self):
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self._load_data()

    def _load_data(self):
        """Load bank data and build BM25 index."""
        if not DATA_PATH.exists():
            logger.error(f"Bank data not found at {DATA_PATH}")
            return

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            if entry.get("status") == "ok" and entry.get("text"):
                self._chunks.extend(
                    _chunk_text(entry["text"], entry["bank"], entry["category"])
                )

        if not self._chunks:
            logger.error("No chunks created from bank data")
            return

        tokenized = [_tokenize_armenian(c["searchable_text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

        logger.info(f"RAG initialized: {len(self._chunks)} chunks from {len(entries)} entries")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve the most relevant chunks for a query."""
        if not self._bm25 or not self._chunks:
            return []

        tokens = _tokenize_armenian(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            if scores[i] >= MIN_SCORE:
                results.append({**self._chunks[i], "score": float(scores[i])})

        return results

    async def answer(self, question: str) -> str:
        """Retrieve relevant context and generate an answer."""
        chunks = self.retrieve(question)

        if not chunks:
            return "\u0546\u0565\u0580\u0565\u0581\u0565\u0584, \u0561\u0575\u057d \u0570\u0561\u0580\u0581\u056b \u057e\u0565\u0580\u0561\u0562\u0565\u0580\u0575\u0561\u056c \u0570\u0561\u0574\u0561\u057a\u0561\u057f\u0561\u057d\u056d\u0561\u0576 \u057f\u0565\u0572\u0565\u056f\u0561\u057f\u057e\u0578\u0582\u0569\u0575\u0578\u0582\u0576 \u0579\u0563\u057f\u0561:"

        # Build context from retrieved chunks
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[{chunk['header']}]\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)

        try:
            response = await self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"\u053f\u0578\u0576\u057f\u0565\u0584\u057d\u057f\u055d\n{context}\n\n\u0540\u0561\u0580\u0581\u055d {question}"},
                ],
                temperature=0.3,
                max_tokens=700,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "\u0546\u0565\u0580\u0565\u0581\u0565\u0584, \u057d\u056d\u0561\u056c \u0561\u057c\u0561\u057b\u0561\u0581\u0561\u057e\u0589 \u053d\u0576\u0564\u0580\u0565\u0574 \u056f\u0580\u056f\u056b\u0576 \u0583\u0578\u0580\u0564\u0565\u0584:"
