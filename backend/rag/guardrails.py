"""Guardrails for RAG output quality"""
import re
from typing import Dict, Any, Tuple


class GuardrailChecker:
    """Quality checks on generated responses.
    
    Tradeoff: Guardrails prevent hallucinations but may reject valid answers.
    We use conservative checks (length, reference presence) rather than
    semantic validation which would be expensive.
    """

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 4000,
        require_code_for_code_q: bool = True
    ):
        """Initialize guardrails.
        
        Args:
            min_length: Minimum response length
            max_length: Maximum response length
            require_code_for_code_q: Require code blocks in code questions
        """
        self.min_length = min_length
        self.max_length = max_length
        self.require_code_for_code_q = require_code_for_code_q

    def check_response(
        self,
        response: str,
        question: str = "",
        context: str = ""
    ) -> Tuple[bool, str]:
        """Check if response passes guardrails.
        
        Args:
            response: Generated response
            question: Original question
            context: Retrieved context
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check length
        if len(response) < self.min_length:
            return False, f"Response too short ({len(response)} < {self.min_length})"
        if len(response) > self.max_length:
            return False, f"Response too long ({len(response)} > {self.max_length})"

        # Check for error indicators
        if "[Error" in response or "Error:" in response:
            return False, "Response contains error indicator"

        # Check if code question but no code in response
        if self.require_code_for_code_q and self._is_code_question(question):
            if not re.search(r"```", response):
                return False, "Code question but no code block in response"

        # Check if response references context
        if context and not self._has_reference_to_context(response, context):
            return False, "Response does not reference provided context"

        return True, "Response passed all checks"

    def _is_code_question(self, question: str) -> bool:
        """Detect if question is about code."""
        code_keywords = ["code", "function", "class", "method", "implementation", "how", "why"]
        return any(kw in question.lower() for kw in code_keywords)

    def _has_reference_to_context(self, response: str, context: str) -> bool:
        """Check if response references context."""
        # Extract unique words from context (length > 5)
        context_words = set(
            word.lower() for word in re.findall(r"\b\w{5,}\b", context)
        )
        response_words = set(
            word.lower() for word in re.findall(r"\b\w{5,}\b", response)
        )
        
        # Check overlap
        overlap = context_words & response_words
        return len(overlap) > 2  # At least 3 words in common
