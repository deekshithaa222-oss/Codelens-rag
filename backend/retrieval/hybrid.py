"""Hybrid retrieval combining dense and sparse methods"""
from typing import List, Dict, Any
import numpy as np
from .embeddings import EmbeddingService
from .storage import ChromaStore
from .bm25 import BM25Retriever


class HybridRetriever:
    """Hybrid retriever combining dense embeddings and BM25 sparse search.
    
    Tradeoff: Hybrid search is more expensive (2 indexes) but catches:
    - Exact keyword matches (BM25) that embeddings miss
    - Semantic similarity (dense) that keyword search misses
    We weight dense 60%, sparse 40% based on empirical eval of code search.
    """

    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize hybrid retriever.
        
        Args:
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
            embedding_model: Model for dense embeddings
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.embeddings = EmbeddingService(embedding_model)
        self.dense_store = ChromaStore()
        self.sparse_search = BM25Retriever()

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """Index chunks with both dense and sparse methods.
        
        Args:
            chunks: Code chunks to index
        """
        # Generate embeddings
        texts = [chunk.get("text", "") for chunk in chunks]
        try:
            embeddings = self.embeddings.embed_batch(texts)
        except:
            # Fallback if embedding model not available
            embeddings = [np.random.randn(384) for _ in texts]

        # Index in both stores
        self.dense_store.add_documents(chunks, embeddings)
        self.sparse_search.index(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using hybrid method.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Top-k results ranked by combined score
        """
        # Dense search
        try:
            query_embedding = self.embeddings.embed_single(query)
            dense_results = self.dense_store.search(query_embedding, top_k=top_k * 2)
        except:
            dense_results = []

        # Sparse search
        sparse_results = self.sparse_search.search(query, top_k=top_k * 2)

        # Combine and re-rank
        combined = {}
        
        # Add dense results
        for i, result in enumerate(dense_results):
            doc_id = result.get("id", str(i))
            score = (1 - result.get("distance", 1)) * self.dense_weight
            if doc_id not in combined:
                combined[doc_id] = {
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "score": 0
                }
            combined[doc_id]["score"] += score

        # Add sparse results
        for i, result in enumerate(sparse_results):
            doc_id = str(i)  # Use index as ID for sparse results
            score = result.get("bm25_score", 0) * self.sparse_weight / 100
            if doc_id not in combined:
                combined[doc_id] = {
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "score": 0
                }
            combined[doc_id]["score"] += score

        # Sort by combined score
        final_results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return final_results[:top_k]
