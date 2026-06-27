"""Build a lightweight Python dependency graph for impact analysis."""
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from backend.ingest.loader import RepositoryLoader
from backend.logger import logger


class CodeGraphBuilder:
    """Parses Python files and extracts imports, definitions, and calls.

    This is intentionally lightweight: it gives useful engineering signals
    without requiring a full language server or project-specific build.
    """

    def __init__(self, repo_path: str, max_file_size: int = 100000):
        self.repo_path = Path(repo_path).resolve()
        self.max_file_size = max_file_size
        self.cache_path = self.repo_path / ".codelens" / "code_graph.json"

    def build(self) -> Dict[str, Any]:
        """Build a graph for Python files in the repository."""
        files: Dict[str, Dict[str, Any]] = {}
        module_to_file: Dict[str, str] = {}
        symbol_to_files: Dict[str, List[str]] = {}

        for path in self._iter_python_files():
            rel_path = self._relative_path(path)
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(content)
            except SyntaxError as exc:
                logger.warning(f"Skipping {rel_path}: syntax error: {exc}")
                continue
            except Exception as exc:
                logger.warning(f"Skipping {rel_path}: {exc}")
                continue

            node = self._extract_file_node(tree, rel_path)
            files[rel_path] = node
            module_to_file[node["module"]] = rel_path

            for symbol in node["defines"]:
                symbol_to_files.setdefault(symbol, []).append(rel_path)

        self._resolve_edges(files, module_to_file, symbol_to_files)

        return {
            "repo_path": str(self.repo_path),
            "files": files,
            "module_to_file": module_to_file,
            "symbol_to_files": symbol_to_files,
            "cache": {
                "enabled": False,
                "source": "full_rebuild",
                "parsed_files": len(files),
                "reused_files": 0,
                "deleted_files": 0,
            },
        }

    def build_cached(self) -> Dict[str, Any]:
        """Build or incrementally update the cached graph.

        File hashes are used as a cheap gate: unchanged files reuse their
        previous AST extraction, while new or changed files are parsed again.
        """
        cached = self.load_cached_graph() or {}
        cached_files = cached.get("files", {})
        files: Dict[str, Dict[str, Any]] = {}
        parsed_files = 0
        reused_files = 0

        for path in self._iter_python_files():
            rel_path = self._relative_path(path)
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.warning(f"Skipping {rel_path}: {exc}")
                continue

            content_hash = self._hash_content(content)
            cached_node = cached_files.get(rel_path)

            if cached_node and cached_node.get("hash") == content_hash:
                files[rel_path] = cached_node
                reused_files += 1
                continue

            try:
                tree = ast.parse(content)
            except SyntaxError as exc:
                logger.warning(f"Skipping {rel_path}: syntax error: {exc}")
                continue
            except Exception as exc:
                logger.warning(f"Skipping {rel_path}: {exc}")
                continue

            node = self._extract_file_node(tree, rel_path)
            node["hash"] = content_hash
            files[rel_path] = node
            parsed_files += 1

        graph = self._assemble_graph(files)
        graph["cache"] = {
            "enabled": True,
            "source": "incremental_update",
            "path": str(self.cache_path),
            "parsed_files": parsed_files,
            "reused_files": reused_files,
            "deleted_files": len(set(cached_files) - set(files)),
        }
        self.save_graph(graph)
        return graph

    def load_cached_graph(self) -> Optional[Dict[str, Any]]:
        """Load a previously persisted graph, if available."""
        if not self.cache_path.exists():
            return None
        try:
            with self.cache_path.open() as f:
                graph = json.load(f)
            if graph.get("repo_path") != str(self.repo_path):
                return None
            graph.setdefault("cache", {})
            graph["cache"].update({
                "enabled": True,
                "source": "cache",
                "path": str(self.cache_path),
                "parsed_files": 0,
                "reused_files": len(graph.get("files", {})),
                "deleted_files": 0,
            })
            return graph
        except Exception as exc:
            logger.warning(f"Failed to load graph cache {self.cache_path}: {exc}")
            return None

    def save_graph(self, graph: Dict[str, Any]) -> None:
        """Persist graph metadata for fast impact analysis."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w") as f:
                json.dump(graph, f, indent=2, sort_keys=True)
        except Exception as exc:
            logger.warning(f"Failed to save graph cache {self.cache_path}: {exc}")

    def _assemble_graph(self, files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        module_to_file: Dict[str, str] = {}
        symbol_to_files: Dict[str, List[str]] = {}

        for rel_path, node in files.items():
            module_to_file[node["module"]] = rel_path
            for symbol in node["defines"]:
                symbol_to_files.setdefault(symbol, []).append(rel_path)

        self._resolve_edges(files, module_to_file, symbol_to_files)

        return {
            "repo_path": str(self.repo_path),
            "files": files,
            "module_to_file": module_to_file,
            "symbol_to_files": symbol_to_files,
        }

    def _iter_python_files(self):
        for path in self.repo_path.rglob("*.py"):
            if not self._should_include(path):
                continue
            yield path

    def _should_include(self, path: Path) -> bool:
        if path.is_dir():
            return False
        if any(part in RepositoryLoader.SKIP_DIRS for part in path.parts):
            return False
        try:
            if path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False
        return True

    def _relative_path(self, path: Path) -> str:
        return str(path.relative_to(self.repo_path))

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _module_name(self, rel_path: str) -> str:
        module = rel_path[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        return module

    def _extract_file_node(self, tree: ast.AST, rel_path: str) -> Dict[str, Any]:
        imports: List[Dict[str, Any]] = []
        functions: List[str] = []
        classes: List[str] = []
        calls: Set[str] = set()

        for child in ast.iter_child_nodes(tree):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(child.name)
            elif isinstance(child, ast.ClassDef):
                classes.append(child.name)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "module": alias.name,
                        "name": alias.asname or alias.name.split(".")[0],
                        "symbol": None,
                        "level": 0,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = self._resolve_import_from_module(rel_path, node.module, node.level)
                for alias in node.names:
                    imports.append({
                        "module": module,
                        "name": alias.asname or alias.name,
                        "symbol": alias.name,
                        "level": node.level,
                    })
            elif isinstance(node, ast.Call):
                call_name = self._call_name(node.func)
                if call_name:
                    calls.add(call_name)

        defines = sorted(set(functions + classes))
        return {
            "path": rel_path,
            "module": self._module_name(rel_path),
            "imports": imports,
            "imported_files": [],
            "functions": sorted(functions),
            "classes": sorted(classes),
            "defines": defines,
            "calls": sorted(calls),
            "is_test": self._is_test_file(rel_path),
        }

    def _resolve_import_from_module(
        self,
        rel_path: str,
        module: Optional[str],
        level: int,
    ) -> str:
        if level == 0:
            return module or ""

        current_parts = self._module_name(rel_path).split(".")
        package_parts = current_parts[:-1]
        base_parts = package_parts[: max(0, len(package_parts) - level + 1)]
        if module:
            base_parts.extend(module.split("."))
        return ".".join(part for part in base_parts if part)

    def _call_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    def _is_test_file(self, rel_path: str) -> bool:
        parts = rel_path.split("/")
        name = parts[-1]
        if name == "__init__.py":
            return False
        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or "tests" in parts
        )

    def _resolve_edges(
        self,
        files: Dict[str, Dict[str, Any]],
        module_to_file: Dict[str, str],
        symbol_to_files: Dict[str, List[str]],
    ) -> None:
        for node in files.values():
            imported_files: Set[str] = set()
            for import_item in node["imports"]:
                module = import_item["module"]
                symbol = import_item["symbol"]

                module_file = self._find_module_file(module, module_to_file)
                if module_file:
                    imported_files.add(module_file)

                if symbol:
                    symbol_module = f"{module}.{symbol}" if module else symbol
                    symbol_file = self._find_module_file(symbol_module, module_to_file)
                    if symbol_file:
                        imported_files.add(symbol_file)

                    for file_path in symbol_to_files.get(symbol, []):
                        imported_files.add(file_path)

            imported_files.discard(node["path"])
            node["imported_files"] = sorted(imported_files)

    def _find_module_file(
        self,
        module: str,
        module_to_file: Dict[str, str],
    ) -> Optional[str]:
        if not module:
            return None
        if module in module_to_file:
            return module_to_file[module]

        parts = module.split(".")
        while len(parts) > 1:
            parts.pop()
            candidate = ".".join(parts)
            if candidate in module_to_file:
                return module_to_file[candidate]
        return None
