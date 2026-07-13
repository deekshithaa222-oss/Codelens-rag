"""Prompt builder for RAG pipeline"""
import json
from typing import List, Dict, Any


class PromptBuilder:
    """Constructs prompts for code understanding tasks.
    
    Tradeoff: Few-shot examples improve output quality but increase context
    length. We include 1-2 relevant examples and document alternatives.
    """

    def __init__(self, max_context_tokens: int = 2000):
        """Initialize prompt builder.
        
        Args:
            max_context_tokens: Maximum tokens for retrieved context
        """
        self.max_context_tokens = max_context_tokens

    def build_qa_prompt(
        self,
        question: str,
        context: List[Dict[str, Any]],
        code_snippets: List[str] = None
    ) -> str:
        """Build QA prompt for code understanding.
        
        Args:
            question: User question
            context: Retrieved code chunks
            code_snippets: Optional relevant code snippets
            
        Returns:
            Formatted prompt for LLM
        """
        system = """You are an expert code analyst. Answer questions about code
based on the provided context. Be precise and reference line numbers when applicable."""

        context_str = self._format_context(context)

        prompt = f"""{system}

## Context
{context_str}

## Question
{question}

## Answer
Provide a clear, precise answer referencing the code context above."""

        return prompt

    def build_explanation_prompt(self, code: str, language: str = "python") -> str:
        """Build prompt for code explanation.
        
        Args:
            code: Code to explain
            language: Programming language
            
        Returns:
            Formatted prompt
        """
        return f"""Explain the following {language} code:

```{language}
{code}
```

Provide:
1. Purpose and high-level behavior
2. Key functions/classes and their roles
3. Any non-obvious implementation details
4. Potential issues or improvements"""

    def build_summary_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Build prompt for summarizing multiple code chunks.
        
        Args:
            chunks: Code chunks to summarize
            
        Returns:
            Formatted prompt
        """
        context = self._format_context(chunks)
        return f"""Summarize the key functionality and architecture of the following code:

{context}

Summary should cover:
1. Main purpose
2. Core components/functions
3. Data flow
4. Dependencies"""

    def build_impact_explanation_prompt(self, impact_result: Dict[str, Any]) -> str:
        """Build prompt for explaining graph-based impact analysis results."""
        result_json = json.dumps(impact_result, indent=2, sort_keys=True)
        return f"""You are explaining a code change impact analysis result.

The dependency graph analysis below is the source of truth. Do not invent
additional files, tests, risks, or dependencies. Explain only what is present
in the provided result.

## Impact Analysis Result
{result_json}

## Explanation
Write a concise explanation for a developer. Include:
1. The overall risk level and why
2. Which files are directly affected
3. Which tests should be considered
4. One practical next step"""

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        token_count = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            file_path = chunk.get("file_path") or chunk.get("metadata", {}).get("file_path", "")
            start_line = chunk.get("start_line") or chunk.get("metadata", {}).get("start_line", "?")
            end_line = chunk.get("end_line") or chunk.get("metadata", {}).get("end_line", "?")

            # Estimate tokens (rough: 4 chars per token)
            chunk_tokens = len(text) // 4
            if token_count + chunk_tokens > self.max_context_tokens:
                break

            context_parts.append(
                f"### {file_path} (lines {start_line}-{end_line})\n```\n{text}\n```"
            )
            token_count += chunk_tokens

        return "\n\n".join(context_parts) if context_parts else "[No context found]"
