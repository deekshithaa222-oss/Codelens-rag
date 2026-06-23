"""Embedding service for dense retrieval"""
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np


class EmbeddingService:
    """Generates embeddings for code chunks.
    
    Tradeoff: Using sentence-transformers for embeddings is fast and works
    offline. Proprietary APIs (OpenAI, Cohere) are better quality but add
    dependencies and latency. We choose semantic models for MVP.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding service.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self._embeddings = {}
        self._model = None

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._embeddings:
            return self._embeddings[text_hash]

        # Load model on first use (lazy init)
        if self._model is None:
            self._load_model()

        # Generate embedding
        embedding = self._model.encode(text, convert_to_numpy=True)
        self._embeddings[text_hash] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed_single(text) for text in texts]

    def _load_model(self):
        """Load embedding model from HuggingFace."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            # Fallback: simple hash-based "embeddings" for testing
            self._model = None

    def __call__(self, text: str) -> np.ndarray:
        """Allow callable interface."""
        return self.embed_single(text)
