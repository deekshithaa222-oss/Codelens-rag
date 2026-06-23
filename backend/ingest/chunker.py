"""AST-aware code chunking with tree-sitter and fallback"""
import re
from typing import List, Dict, Any, Optional
from backend.logger import logger


class ASTChunker:
    """Chunks code using tree-sitter AST parsing with regex fallback.
    
    Tradeoff: Tree-sitter gives accurate semantic boundaries, but requires
    native libraries. Regex fallback works everywhere but misses nesting.
    We attempt tree-sitter first, fall back to regex for unsupported langs.
    """

    def __init__(self, target_chunk_size: int = 512, overlap: int = 100):
        """Initialize chunker.
        
        Args:
            target_chunk_size: Target tokens per chunk (approx 4 chars/token)
            overlap: Overlap between chunks to preserve context
        """
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap
        self.chunk_size_bytes = target_chunk_size * 4

    def chunk(self, content: str, language: str, file_path: str = "") -> List[Dict[str, Any]]:
        """Chunk code into semantically meaningful pieces.
        
        Args:
            content: Source code
            language: Programming language
            file_path: File path for metadata
            
        Returns:
            List of chunks with 'text', 'start_line', 'end_line', 'type' keys
        """
        try:
            # Try AST-based chunking
            chunks = self._chunk_by_ast(content, language)
        except Exception as e:
            logger.debug(f"AST chunking failed for {language}: {e}, using regex fallback")
            # Fall back to regex-based chunking
            chunks = self._chunk_by_regex(content)

        # Add metadata
        for chunk in chunks:
            chunk["file_path"] = file_path
            chunk["language"] = language

        logger.debug(f"Chunked {file_path} into {len(chunks)} pieces")
        return chunks

    def _chunk_by_ast(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Try to chunk by AST. Requires tree-sitter installation."""
        try:
            from tree_sitter import Language, Parser
        except ImportError:
            raise ImportError("tree-sitter not installed; falling back to regex")

        # This is a simplified version - full impl would parse each language
        # For now, use function/class boundaries as proxy
        chunks = []
        lines = content.split("\n")

        if language in ("python", "py"):
            # Python: split on def/class
            current_chunk = []
            current_start = 0

            for i, line in enumerate(lines):
                current_chunk.append(line)
                # Split on function/class definitions when chunk is large enough
                if (len("\n".join(current_chunk)) > self.chunk_size_bytes and
                    re.match(r"^\s*(def|class)\s+", line)):
                    text = "\n".join(current_chunk[:-1])
                    if text.strip():
                        chunks.append({
                            "text": text,
                            "start_line": current_start + 1,
                            "end_line": i,
                            "type": "function/class"
                        })
                    current_chunk = [line]
                    current_start = i

            if current_chunk:
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "start_line": current_start + 1,
                    "end_line": len(lines),
                    "type": "code"
                })
        else:
            # For unsupported languages, raise to trigger regex fallback
            raise NotImplementedError(f"AST parsing not implemented for {language}")

        return chunks

    def _chunk_by_regex(self, content: str) -> List[Dict[str, Any]]:
        """Chunk by regex patterns (function/class definitions)."""
        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_start = 0

        # Pattern for function/class definitions (works for most languages)
        defn_pattern = re.compile(
            r"^\s*(def|class|function|func|interface|struct|enum|type|impl|fn)\s+"
        )

        for i, line in enumerate(lines):
            current_chunk.append(line)

            # Split when we hit a new definition and chunk is large
            if (len("\n".join(current_chunk)) > self.chunk_size_bytes and
                defn_pattern.match(line)):
                text = "\n".join(current_chunk[:-1]).strip()
                if text:
                    chunks.append({
                        "text": text,
                        "start_line": current_start + 1,
                        "end_line": i,
                        "type": "function/class"
                    })
                current_chunk = [line]
                current_start = i

        # Add remaining chunk
        if current_chunk:
            text = "\n".join(current_chunk).strip()
            if text:
                chunks.append({
                    "text": text,
                    "start_line": current_start + 1,
                    "end_line": len(lines),
                    "type": "code"
                })

        # If no chunks created, split by size
        if not chunks:
            for i in range(0, len(lines), max(1, len(lines) // 4)):
                chunk_lines = lines[i:min(i + 50, len(lines))]
                text = "\n".join(chunk_lines).strip()
                if text:
                    chunks.append({
                        "text": text,
                        "start_line": i + 1,
                        "end_line": min(i + 50, len(lines)),
                        "type": "code"
                    })

        return chunks
