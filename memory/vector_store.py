"""
memory/vector_store.py

ChromaDB wrapper for forensic case similarity search.

Stores embedded case representations and retrieves
similar past cases for dynamic few-shot injection.
"""

import logging
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings


logger = logging.getLogger("forensic.memory")


class ForensicVectorStore:
    """
    Vector store for forensic case memory.

    Uses ChromaDB with default embeddings (all-MiniLM-L6-v2).
    Stores case text representations + metadata.
    Retrieves similar cases for learning.
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"description": "Forensic case embeddings"},
        )
        logger.info(
            f"Vector store initialized: {settings.CHROMA_PERSIST_DIR} "
            f"({self._collection.count()} cases)"
        )

    def add_case(
        self,
        case_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add or update a case in the vector store.

        Args:
            case_id: Unique case identifier (used as ChromaDB ID)
            text: Text representation for embedding
            metadata: Searchable metadata (scores, labels, etc.)
        """
        # ChromaDB metadata must be str/int/float/bool
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif v is None:
                clean_metadata[k] = ""
            else:
                clean_metadata[k] = str(v)

        self._collection.upsert(
            ids=[case_id],
            documents=[text],
            metadatas=[clean_metadata],
        )

        logger.debug(f"[{case_id}] Embedded in vector store")

    def search_similar(
        self,
        query_text: str,
        k: int = 5,
        min_score: float = 0.3,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases.

        Args:
            query_text: Text representation of the query case
            k: Max results
            min_score: Minimum similarity (0-1, higher = more similar)
            where: ChromaDB metadata filter

        Returns:
            List of dicts with: id, text, metadata, similarity_score
        """
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(k, self._collection.count() or 1),
                where=where,
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        similar = []
        for i, case_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances, convert to similarity
            distance = results["distances"][0][i] if results["distances"] else 1.0
            similarity = max(0, 1 - distance)

            if similarity < min_score:
                continue

            similar.append({
                "case_id": case_id,
                "text": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "similarity_score": round(similarity, 3),
            })

        logger.debug(f"Found {len(similar)} similar cases (threshold={min_score})")
        return similar

    def count(self) -> int:
        return self._collection.count()

    def has_labeled_cases(self) -> bool:
        """Check if there are any labeled cases for learning."""
        try:
            results = self._collection.get(
                where={"human_label": {"$ne": ""}},
                limit=1,
            )
            return len(results["ids"]) > 0
        except Exception:
            return False


# ── Singleton ──

_vector_store: Optional[ForensicVectorStore] = None


def get_vector_store() -> ForensicVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = ForensicVectorStore()
    return _vector_store
