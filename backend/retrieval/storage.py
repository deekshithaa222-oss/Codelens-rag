"""Vector database storage with ChromaDB"""
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path


class ChromaStore:
    """Vector store using ChromaDB for dense retrieval.
    
    Tradeoff: ChromaDB provides persistence and similarity search without
    external dependencies. Pinecone/Weaviate offer better scaling but require
    cloud infrastructure. ChromaDB is pragmatic for MVP + self-hosted setups.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        """Initialize ChromaDB store.
        
        Args:
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        self._client = None
        self._collection = None

    def _init_chroma(self):
        """Initialize ChromaDB client lazily."""
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name="codelens",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            # Fallback: simple in-memory store
            self._storage = {}
            self._id_counter = 0

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> None:
        """Add documents with embeddings.
        
        Args:
            chunks: List of code chunks with metadata
            embeddings: Corresponding embeddings
        """
        if self._collection is None:
            self._init_chroma()

        ids = []
        documents = []
        metadatas = []
        embedded_vectors = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"chunk_{id(chunk)}_{i}"
            ids.append(chunk_id)
            documents.append(chunk.get("text", ""))
            metadatas.append({
                "file_path": chunk.get("file_path", ""),
                "start_line": str(chunk.get("start_line", 0)),
                "end_line": str(chunk.get("end_line", 0)),
                "type": chunk.get("type", "code"),
                "language": chunk.get("language", "")
            })
            embedded_vectors.append(embedding)

        if self._collection is not None:
            # Use ChromaDB
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedded_vectors.tolist() if isinstance(embedded_vectors, np.ndarray) else
                           [v.tolist() if isinstance(v, np.ndarray) else v for v in embedded_vectors]
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            
        Returns:
            List of most similar chunks with scores
        """
        if self._collection is None:
            self._init_chroma()

        if self._collection is not None:
            # Use ChromaDB
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist() if isinstance(query_embedding, np.ndarray)
                                  else query_embedding],
                n_results=top_k
            )

            # Format results
            output = []
            for i in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            return output
        else:
            return []

    def clear(self) -> None:
        """Clear all documents."""
        if self._collection is not None:
            self._collection.delete(where={})
