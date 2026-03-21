"""
RAG (Retrieval-Augmented Generation) module for Armenian banking voice assistant.
Loads scraped bank data, chunks it, indexes with ChromaDB + multilingual embeddings,
and answers queries via Groq Llama.
"""

import json
import logging
import os
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import AsyncGroq

logger = logging.getLogger("rag")

DATA_PATH = Path(__file__).parent / "data" / "bank_data.json"
CHROMA_PATH = Path(__file__).parent / "data" / "chroma_db"
COLLECTION_NAME = "bank_chunks"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 7

ANSWER_SYSTEM_PROMPT = (
    "\u0534\u0578\u0582 \u0570\u0561\u0575\u0565\u0580\u0565\u0576 \u0562\u0561\u0576\u056f\u0561\u0575\u056b\u0576 \u0578\u0572\u0561\u0576\u0564\u0576\u0565\u0580\u056b \u0585\u0563\u0576\u0561\u056f\u0561\u0576 \u0571\u0561\u0575\u0576\u0561\u0575\u056b\u0576 \u0585\u0563\u0576\u0561\u056f\u0561\u0576 \u0565\u057d\u0589 "
    "\u054f\u0565\u0572\u0565\u056f\u0561\u057f\u057e\u0578\u0582\u0569\u0575\u0578\u0582\u0576\u0576\u0565\u0580 \u0565\u057d \u057f\u0580\u0561\u0574\u0561\u0564\u0580\u0578\u0582\u0574 4 \u0570\u0561\u0575\u056f\u0561\u056f\u0561\u0576 \u0562\u0561\u0576\u056f\u0565\u0580\u056b \u0574\u0561\u057d\u056b\u0576\u055d "
    "\u0531\u0574\u0565\u0580\u056b\u0561\u0562\u0561\u0576\u056f, \u0535\u057e\u0578\u056f\u0561\u0562\u0561\u0576\u056f, \u0531\u0580\u0564\u0577\u056b\u0576\u0562\u0561\u0576\u056f, \u053b\u0576\u0565\u056f\u0578\u0562\u0561\u0576\u056f\u0589\n\n"
    "\u053f\u0561\u0576\u0578\u0576\u0576\u0565\u0580\u055d\n"
    "- \u054a\u0561\u057f\u0561\u057d\u056d\u0561\u0576\u056b\u0580 \u0544\u053b\u0531\u0545\u0546 \u0570\u0561\u0575\u0565\u0580\u0565\u0576\u0578\u057e\n"
    "- \u0555\u0563\u057f\u0561\u0563\u0578\u0580\u056e\u056b\u0580 \u0544\u053b\u0531\u0545\u0546 \u057f\u0580\u0561\u0574\u0561\u0564\u0580\u057e\u0561\u056e \u056f\u0578\u0576\u057f\u0565\u0584\u057d\u057f\u056b \u0570\u056b\u0574\u0561\u0576 \u057e\u0580\u0561\n"
    "- \u0535\u0569\u0565 \u056f\u0578\u0576\u057f\u0565\u0584\u057d\u057f\u0578\u0582\u0574 \u057a\u0561\u057f\u0561\u057d\u056d\u0561\u0576 \u0579\u056f\u0561, \u0561\u057d\u0561\u055d \u0579\u0563\u056b\u057f\u0565\u0574, \u0570\u0561\u0574\u0561\u057a\u0561\u057f\u0561\u057d\u056d\u0561\u0576 \u057f\u0565\u0572\u0565\u056f\u0561\u057f\u057e\u0578\u0582\u0569\u0575\u0578\u0582\u0576 \u0579\u0578\u0582\u0576\u0565\u0574\n"
    "- \u054a\u0561\u057f\u0561\u057d\u056d\u0561\u0576\u0576\u0565\u0580\u0568 \u057a\u0565\u057f\u0584 \u0567 \u056c\u056b\u0576\u0565\u0576 \u056f\u0561\u0580\u0573 \u0578\u0582 \u0570\u0561\u057d\u056f\u0561\u0576\u0561\u056c\u056b\u055d \u0578\u0580\u057a\u0565\u057d \u056e\u0561\u0576\u0578\u0569 \u0585\u0563\u057f\u0561\u056f\u0561\u0580 \u057f\u0565\u0572\u0565\u056f\u0561\u057f\u057e\u0578\u0582\u0569\u0575\u0578\u0582\u0576\n"
    "- \u0535\u0569\u0565 \u0570\u0561\u0580\u0581\u0576\u0578\u0582\u0574 \u0567 \u0570\u0561\u0574\u0565\u0574\u0561\u057f\u056b\u0580, \u056f\u0561\u0580\u0578\u0572 \u0565\u057d \u0576\u0577\u0565\u056c \u057f\u0561\u0580\u0562\u0565\u0580 \u0562\u0561\u0576\u056f\u0565\u0580\u056b \u057f\u057e\u0575\u0561\u056c\u0576\u0565\u0580\u0568 \u0570\u0561\u0574\u0565\u0574\u0561\u057f\u0565\u056c\u0578\u0582 \u0570\u0561\u0574\u0561\u0580\n"
    "- \u0544\u056b \u0570\u0576\u0561\u0580\u056b\u0580 \u056b\u0576\u0586\u0578\u0580\u0574\u0561\u0581\u056b\u0561 \u0574\u056b \u0570\u0576\u0561\u0580\u056b\u0580\n"
    "- \u0535\u0569\u0565 \u0570\u0561\u0580\u0581\u0576\u0578\u0582\u0574 \u0567 \u0569\u057e\u0565\u0580, \u057f\u0578\u056f\u0578\u057d\u0561\u0564\u0580\u0578\u0582\u0575\u0584\u0576\u0565\u0580, \u0563\u0578\u0582\u0574\u0561\u0580\u0576\u0565\u0580, \u056a\u0561\u0574\u056f\u0565\u057f\u0576\u0565\u0580 \u056f\u0561\u0574 \u0561\u0575\u056c \u056f\u0578\u0576\u056f\u0580\u0565\u057f \u057f\u057e\u0575\u0561\u056c\u0576\u0565\u0580, \u0576\u0577\u056b\u0580 \u0564\u0580\u0561\u0576\u0584 \u0570\u057d\u057f\u0561\u056f\u0578\u0580\u0565\u0576\n"
    "- \u054a\u0561\u057f\u0561\u057d\u056d\u0561\u0576\u0568 \u057a\u0565\u057f\u0584 \u0567 \u056c\u056b\u0576\u056b \u0562\u0576\u0561\u056f\u0561\u0576, \u056d\u0578\u057d\u0561\u056f\u0581\u0561\u0575\u056b\u0576 \u057f\u0565\u057d\u0584\u0578\u057e \u0570\u0561\u0580\u0578\u0582\u057d\u057f"
)


# ─── Chunking ─────────────────────────────────────────────────────────

def _chunk_text(text: str, bank: str, category: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    if not text.strip():
        return []

    # Split on section separators first (--- from sub-page joins), then by size
    sections = re.split(r"\n-{3,}\n", text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= CHUNK_SIZE:
            chunks.append({
                "text": section,
                "bank": bank,
                "category": category,
            })
            continue

        # For longer sections, split with paragraph awareness
        start = 0
        while start < len(section):
            end = start + CHUNK_SIZE

            if end < len(section):
                para_pos = section.rfind("\n\n", start, end)
                if para_pos > start + CHUNK_SIZE // 3:
                    end = para_pos + 1
                else:
                    newline_pos = section.rfind("\n", start, end)
                    if newline_pos > start + CHUNK_SIZE // 3:
                        end = newline_pos + 1

            chunk_text = section[start:end].strip()
            if chunk_text and len(chunk_text) > 20:
                chunks.append({
                    "text": chunk_text,
                    "bank": bank,
                    "category": category,
                })
            start = end - CHUNK_OVERLAP

    return chunks


# ─── RAG Engine ───────────────────────────────────────────────────────

class BankRAG:
    """Retrieval-Augmented Generation using ChromaDB + multilingual embeddings."""

    def __init__(self):
        self._client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )
        self._chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Index data if collection is empty
        if self._collection.count() == 0:
            self._index_data()

        logger.info(f"RAG initialized: {self._collection.count()} chunks in ChromaDB")

    def _index_data(self):
        """Load bank data, chunk it, and index into ChromaDB."""
        if not DATA_PATH.exists():
            logger.error(f"Bank data not found at {DATA_PATH}. Run scraper.py first.")
            return

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)

        documents = []
        metadatas = []
        ids = []

        chunk_id = 0
        for entry in entries:
            if entry.get("status") == "ok" and entry.get("text"):
                chunks = _chunk_text(entry["text"], entry["bank"], entry["category"])
                for chunk in chunks:
                    # Prepend bank+category to the document for better embedding context
                    doc_text = f"{chunk['bank']} | {chunk['category']}\n{chunk['text']}"
                    documents.append(doc_text)
                    metadatas.append({
                        "bank": chunk["bank"],
                        "category": chunk["category"],
                    })
                    ids.append(f"chunk_{chunk_id}")
                    chunk_id += 1

        if not documents:
            logger.error("No chunks created from bank data")
            return

        # ChromaDB batch limit is ~5000, add in batches
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self._collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )
            logger.info(f"Indexed batch {i // batch_size + 1} ({min(i + batch_size, len(documents))}/{len(documents)})")

        logger.info(f"Indexed {len(documents)} chunks into ChromaDB")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve the most relevant chunks using semantic search."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                chunks.append({
                    "text": doc,
                    "bank": results["metadatas"][0][i]["bank"],
                    "category": results["metadatas"][0][i]["category"],
                    "header": f"{results['metadatas'][0][i]['bank']} | {results['metadatas'][0][i]['category']}",
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })

        return chunks

    def reindex(self):
        """Force re-index: delete collection and rebuild from bank_data.json."""
        self._chroma.delete_collection(COLLECTION_NAME)
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_data()

    async def answer(self, question: str) -> str:
        """Retrieve relevant context and generate an answer."""
        chunks = self.retrieve(question)

        if not chunks:
            return "\u0546\u0565\u0580\u0565\u0581\u0565\u0584, \u0561\u0575\u057d \u0570\u0561\u0580\u0581\u056b \u057e\u0565\u0580\u0561\u0562\u0565\u0580\u0575\u0561\u056c \u0570\u0561\u0574\u0561\u057a\u0561\u057f\u0561\u057d\u056d\u0561\u0576 \u057f\u0565\u0572\u0565\u056f\u0561\u057f\u057e\u0578\u0582\u0569\u0575\u0578\u0582\u0576 \u0579\u0563\u057f\u0561:"

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
