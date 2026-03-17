"""Financial situation memory using BM25 for lexical similarity matching.

Uses BM25 (Best Matching 25) algorithm for retrieval - no API calls,
no token limits, works offline with any LLM provider.
"""

import re

from rank_bm25 import BM25Okapi

# Default cap per memory instance — keeps BM25 matching quality high
# and memory usage bounded (~200 entries ≈ 400KB text + ~1MB index).
DEFAULT_MAX_ENTRIES = 200


class FinancialSituationMemory:
    """Memory system for storing and retrieving financial situations using BM25.

    When the number of entries exceeds *max_entries*, the oldest entries
    are evicted (FIFO) to keep the index fresh and memory bounded.
    """

    def __init__(
        self,
        name: str,
        config: dict | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self.name = name
        self.max_entries = max_entries
        self.documents: list[str] = []
        self.recommendations: list[str] = []
        self.bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing."""
        return re.findall(r"\b\w+\b", text.lower())

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index after adding documents."""
        if self.documents:
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def _evict_oldest(self) -> None:
        """Drop oldest entries if we exceed max_entries."""
        if len(self.documents) > self.max_entries:
            excess = len(self.documents) - self.max_entries
            self.documents = self.documents[excess:]
            self.recommendations = self.recommendations[excess:]

    def add_situations(self, situations_and_advice: list[tuple[str, str]]) -> None:
        """Add financial situations and their corresponding advice.

        Args:
            situations_and_advice: List of tuples (situation, recommendation)
        """
        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)

        self._evict_oldest()
        self._rebuild_index()

    def get_memories(self, current_situation: str, n_matches: int = 1) -> list[dict]:
        """Find matching recommendations using BM25 similarity.

        Args:
            current_situation: The current financial situation to match against
            n_matches: Number of top matches to return

        Returns:
            List of dicts with matched_situation, recommendation, and similarity_score
        """
        if not self.documents or self.bm25 is None:
            return []

        query_tokens = self._tokenize(current_situation)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_matches]

        max_score = max(scores) if max(scores) > 0 else 1
        results = []
        for idx in top_indices:
            normalized_score = scores[idx] / max_score if max_score > 0 else 0
            results.append(
                {
                    "matched_situation": self.documents[idx],
                    "recommendation": self.recommendations[idx],
                    "similarity_score": normalized_score,
                }
            )
        return results

    def clear(self) -> None:
        """Clear all stored memories."""
        self.documents = []
        self.recommendations = []
        self.bm25 = None
