"""Tests for CodeLens retrieval and evaluation"""
import pytest
from pathlib import Path
from backend.ingest import RepositoryLoader, ASTChunker
from backend.retrieval.bm25 import BM25Retriever
from backend.eval import FaithfulnessScorer


@pytest.fixture
def sample_code():
    """Sample code for testing."""
    return """
def calculate_sum(a, b):
    '''Calculate the sum of two numbers'''
    return a + b

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        self.result = x + y
        return self.result
"""


@pytest.fixture
def chunker():
    """Initialize AST chunker."""
    return ASTChunker(target_chunk_size=256)


@pytest.fixture
def bm25():
    """Initialize BM25 retriever."""
    return BM25Retriever()


class TestChunking:
    """Test code chunking."""

    def test_chunk_creation(self, sample_code, chunker):
        """Test that chunks are created."""
        chunks = chunker.chunk(sample_code, "python", "test.py")
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("start_line" in chunk for chunk in chunks)

    def test_chunk_metadata(self, sample_code, chunker):
        """Test chunk metadata."""
        chunks = chunker.chunk(sample_code, "python", "test.py")
        for chunk in chunks:
            assert chunk.get("file_path") == "test.py"
            assert chunk.get("language") == "python"

    def test_chunk_contains_function(self, sample_code, chunker):
        """Test that chunks contain expected content."""
        chunks = chunker.chunk(sample_code, "python", "test.py")
        chunk_text = " ".join(chunk["text"] for chunk in chunks)
        assert "def calculate_sum" in chunk_text
        assert "class Calculator" in chunk_text


class TestBM25:
    """Test BM25 sparse retrieval."""

    def test_bm25_indexing(self, sample_code, chunker, bm25):
        """Test BM25 indexing."""
        chunks = chunker.chunk(sample_code, "python", "test.py")
        bm25.index(chunks)
        assert len(bm25.documents) == len(chunks)

    def test_bm25_search(self, sample_code, chunker, bm25):
        """Test BM25 search."""
        chunks = chunker.chunk(sample_code, "python", "test.py")
        bm25.index(chunks)

        results = bm25.search("calculate sum", top_k=3)
        assert len(results) > 0
        assert results[0].get("bm25_score", 0) > 0

    def test_bm25_keyword_match(self, sample_code, chunker, bm25):
        """Test that BM25 finds keyword matches."""
        chunks = chunker.chunk(sample_code, "python", "test.py")
        bm25.index(chunks)

        results = bm25.search("Calculator class", top_k=5)
        chunk_texts = [r.get("text", "") for r in results]
        assert any("class Calculator" in text for text in chunk_texts)


class TestFaithfulness:
    """Test faithfulness scoring."""

    def test_faithfulness_score_range(self):
        """Test that scores are in [0, 1]."""
        scorer = FaithfulnessScorer()
        result = scorer.score(
            "The function adds two numbers",
            "def add(a, b): return a + b",
            "What does the function do?"
        )
        assert 0 <= result["score"] <= 1

    def test_faithfulness_high_overlap(self):
        """Test high overlap -> high score."""
        scorer = FaithfulnessScorer()
        context = "The function calculates sum of numbers"
        response = "The function calculates the sum of two numbers"

        result = scorer.score(response, context, "What does the function do?")
        assert result["score"] > 0.5  # High overlap should score well

    def test_faithfulness_no_overlap(self):
        """Test no overlap -> low score."""
        scorer = FaithfulnessScorer()
        context = "The function sorts arrays"
        response = "The function deletes all data"

        result = scorer.score(response, context, "What does the function do?")
        assert result["score"] < 0.5  # No overlap should score poorly

    def test_hallucination_detection(self):
        """Test detection of hallucinated content."""
        scorer = FaithfulnessScorer()
        context = "The system logs user actions"
        response = "The system logs user actions and sends emails to Facebook and AWS"

        result = scorer.score(response, context)
        assert result["hallucination_rate"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
