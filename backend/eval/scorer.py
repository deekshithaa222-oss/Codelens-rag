"""Faithfulness evaluation for RAG responses"""
import re
from typing import Dict, List, Tuple, Any
from backend.logger import logger


class FaithfulnessScorer:
    """Scores faithfulness of RAG responses to context.
    
    Tradeoff: Offline evaluation without LLM avoids cost and latency but
    uses simple heuristics. For production, consider question decomposition
    + entailment models. This MVP uses keyword/entity overlap.
    """

    def __init__(self, overlap_threshold: float = 0.3):
        """Initialize scorer.
        
        Args:
            overlap_threshold: Min ratio of response covered by context
        """
        self.overlap_threshold = overlap_threshold

    def score(
        self,
        response: str,
        context: str,
        question: str = ""
    ) -> Dict[str, Any]:
        """Score faithfulness of response to context.
        
        Args:
            response: Generated response
            context: Retrieved context
            question: Original question
            
        Returns:
            Dict with 'score' (0-1), 'reasoning', and component scores
        """
        # Extract entities/keywords from context
        context_entities = self._extract_entities(context)
        response_entities = self._extract_entities(response)

        # Calculate overlap
        entity_overlap = len(context_entities & response_entities) / max(1, len(response_entities))

        # Check for hallucinations (entities in response but not context)
        hallucinated = response_entities - context_entities
        hallucination_rate = len(hallucinated) / max(1, len(response_entities))

        # Check if response actually answers the question
        question_coverage = self._check_question_coverage(response, question) if question else 0.5

        # Combine scores
        faithfulness_score = (
            0.5 * entity_overlap +
            0.3 * (1 - hallucination_rate) +
            0.2 * question_coverage
        )

        return {
            "score": min(1.0, max(0.0, faithfulness_score)),
            "entity_overlap": entity_overlap,
            "hallucination_rate": hallucination_rate,
            "question_coverage": question_coverage,
            "hallucinated_entities": list(hallucinated)[:5],
            "reasoning": self._explain_score(
                entity_overlap, hallucination_rate, question_coverage
            )
        }

    def _extract_entities(self, text: str) -> set:
        """Extract key entities/keywords from text."""
        # Simple: extract capitalized words and code identifiers
        words = set()
        
        # Capitalized words (likely entities/classes)
        for word in re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", text):
            if len(word) > 2:
                words.add(word.lower())

        # Code identifiers (snake_case, camelCase)
        for word in re.findall(r"\b[a-z_][a-z0-9_]*\b", text):
            if len(word) > 4 and word not in {"return", "function", "class"}:
                words.add(word.lower())

        return words

    def _check_question_coverage(self, response: str, question: str) -> float:
        """Check if response covers question topics."""
        question_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
        response_words = set(re.findall(r"\b\w{4,}\b", response.lower()))

        overlap = question_words & response_words
        return len(overlap) / max(1, len(question_words))

    def _explain_score(
        self,
        entity_overlap: float,
        hallucination_rate: float,
        question_coverage: float
    ) -> str:
        """Provide human-readable explanation."""
        reasons = []
        
        if entity_overlap > 0.7:
            reasons.append("High entity overlap with context")
        elif entity_overlap > 0.4:
            reasons.append("Moderate entity overlap with context")
        else:
            reasons.append("Low entity overlap - may not be grounded")

        if hallucination_rate > 0.3:
            reasons.append("High hallucination risk")
        elif hallucination_rate > 0.1:
            reasons.append("Some ungrounded statements")

        if question_coverage > 0.7:
            reasons.append("Addresses question comprehensively")
        elif question_coverage > 0.3:
            reasons.append("Partially addresses question")

        return "; ".join(reasons)
