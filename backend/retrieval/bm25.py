"""BM25 sparse retrieval for lexical search"""
from typing import List, Dict, Any
import re


class BM25Retriever:
    """BM25-based lexical search for sparse retrieval.
    
    Tradeoff: BM25 is fast, interpretable, and works without embeddings.
    Dense-only systems miss exact keyword matches; hybrid combines strength.
    Implementation uses simplified BM25 (full algorithm in Elasticsearch).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.idf = {}
        self.avg_doc_len = 0

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """Index documents for BM25 search.
        
        Args:
            chunks: List of code chunks to index
        """
        self.documents = chunks
        self._compute_idf()

    def _compute_idf(self) -> None:
        """Compute inverse document frequency."""
        import math
        
        vocab = {}
        for chunk in self.documents:
            text = chunk.get("text", "").lower()
            tokens = self._tokenize(text)
            for token in set(tokens):
                vocab[token] = vocab.get(token, 0) + 1

        n_docs = len(self.documents)
        for term, df in vocab.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

        self.avg_doc_len = sum(
            len(self._tokenize(chunk.get("text", "")))
            for chunk in self.documents
        ) / max(1, n_docs)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search chunks by query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Top-k most relevant chunks
        """
        query_tokens = self._tokenize(query.lower())
        scores = []

        for i, chunk in enumerate(self.documents):
            text = chunk.get("text", "").lower()
            tokens = self._tokenize(text)
            doc_len = len(tokens)

            score = 0.0
            for qt in query_tokens:
                freq = tokens.count(qt)
                idf = self.idf.get(qt, 0)

                if freq > 0:
                    norm_len = 1 - self.b + self.b * (doc_len / max(1, self.avg_doc_len))
                    score += idf * (self.k1 + 1) * freq / (self.k1 * norm_len + freq)

            if score > 0:
                scores.append((i, score, chunk))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score, chunk in scores[:top_k]:
            result = chunk.copy()
            result["bm25_score"] = score
            results.append(result)

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove non-alphanumeric and split
        text = re.sub(r"[^a-z0-9_]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Filter short tokens
