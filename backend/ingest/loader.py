"""Repository loader with secret/junk filtering"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from backend.logger import logger


class RepositoryLoader:
    """Loads code from repositories with filtering for secrets and junk files."""

    # Extensions to include
    SUPPORTED_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".go",
        ".rs", ".rb", ".php", ".cs", ".swift", ".kt", ".scala", ".clj",
        ".sh", ".sql", ".html", ".css", ".yml", ".yaml", ".json"
    }

    # Patterns for secrets/junk
    SECRET_PATTERNS = [
        r"api[_-]?key",
        r"secret",
        r"password",
        r"token",
        r"auth",
        r"credential",
    ]

    # Directories to skip
    SKIP_DIRS = {
        ".git", ".github", "__pycache__", "node_modules", ".venv", "venv",
        ".env", "build", "dist", ".pytest_cache", ".vscode", ".idea",
        ".DS_Store", "*.egg-info"
    }

    def __init__(self, repo_path: str, max_file_size: int = 100000):
        """Initialize repository loader.
        
        Args:
            repo_path: Path to repository root
            max_file_size: Maximum file size in bytes (filters large binaries)
        
        Tradeoff: Filtering by size excludes large generated files but loses context
        from well-commented large files. We accept this to avoid token waste.
        """
        self.repo_path = Path(repo_path)
        self.max_file_size = max_file_size

    def load(self) -> List[Dict[str, Any]]:
        """Load all code files from repository.
        
        Returns:
            List of dicts with 'path', 'content', 'language' keys
        """
        files = []

        for file_path in self.repo_path.rglob("*"):
            if not self._should_include(file_path):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if self._contains_secrets(content):
                    logger.warning(f"Skipping {file_path}: contains potential secrets")
                    continue

                files.append({
                    "path": str(file_path.relative_to(self.repo_path)),
                    "content": content,
                    "language": self._detect_language(file_path)
                })
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(files)} files from {self.repo_path}")
        return files

    def _should_include(self, path: Path) -> bool:
        """Check if file should be included."""
        # Skip directories
        if path.is_dir():
            return False

        # Skip hidden/junk dirs
        if any(part in self.SKIP_DIRS for part in path.parts):
            return False

        # Check extension
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            return False

        # Check file size
        if path.stat().st_size > self.max_file_size:
            logger.debug(f"Skipping {path}: exceeds max size")
            return False

        return True

    def _contains_secrets(self, content: str) -> bool:
        """Check if content likely contains secrets."""
        for pattern in self.SECRET_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def _detect_language(self, path: Path) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "jsx", ".tsx": "tsx", ".java": "java", ".cpp": "cpp",
            ".c": "c", ".go": "go", ".rs": "rust", ".rb": "ruby",
            ".php": "php", ".cs": "csharp", ".swift": "swift",
            ".kt": "kotlin", ".scala": "scala", ".clj": "clojure",
            ".sh": "bash", ".sql": "sql", ".html": "html",
            ".css": "css", ".yml": "yaml", ".yaml": "yaml", ".json": "json"
        }
        return ext_to_lang.get(path.suffix, "text")
