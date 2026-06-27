"""Change impact analysis built on the lightweight code graph."""
from pathlib import Path
from typing import Any, Dict, List, Set

from .graph_builder import CodeGraphBuilder


class ImpactAnalyzer:
    """Find likely blast radius for changed Python files."""

    def analyze(
        self,
        repo_path: str,
        changed_files: List[str],
        changed_symbols: List[str] = None,
    ) -> Dict[str, Any]:
        changed_symbols = changed_symbols or []
        graph = CodeGraphBuilder(repo_path).build()
        normalized_changes = self._normalize_changed_files(
            repo_path,
            changed_files,
            set(graph["files"].keys()),
        )

        affected = self._find_affected_files(
            graph,
            normalized_changes,
            set(changed_symbols),
        )
        tests = self._suggest_tests(graph, normalized_changes, affected)
        risk = self._risk_level(normalized_changes, affected, tests, graph)

        return {
            "repo_path": str(Path(repo_path).resolve()),
            "changed_files": sorted(normalized_changes),
            "changed_symbols": sorted(changed_symbols),
            "risk": risk["level"],
            "risk_reasons": risk["reasons"],
            "direct_dependents": sorted(affected["direct_dependents"]),
            "symbol_dependents": sorted(affected["symbol_dependents"]),
            "related_files": sorted(affected["related_files"]),
            "suggested_tests": sorted(tests),
            "summary": self._summary(normalized_changes, affected, tests, risk["level"]),
            "graph_stats": {
                "python_files_analyzed": len(graph["files"]),
                "changed_files_found": len(normalized_changes),
                "related_files_found": len(affected["related_files"]),
            },
        }

    def _normalize_changed_files(
        self,
        repo_path: str,
        changed_files: List[str],
        known_files: Set[str],
    ) -> Set[str]:
        repo = Path(repo_path).resolve()
        normalized: Set[str] = set()

        for file_path in changed_files:
            path = Path(file_path)
            try:
                if path.is_absolute():
                    rel = str(path.resolve().relative_to(repo))
                else:
                    rel = str(path)
            except ValueError:
                rel = str(path)

            rel = rel.replace("\\", "/").lstrip("./")
            if rel in known_files:
                normalized.add(rel)

        return normalized

    def _find_affected_files(
        self,
        graph: Dict[str, Any],
        changed_files: Set[str],
        changed_symbols: Set[str],
    ) -> Dict[str, Set[str]]:
        direct_dependents: Set[str] = set()
        symbol_dependents: Set[str] = set()
        related_files: Set[str] = set()

        changed_defines = set(changed_symbols)
        for changed_file in changed_files:
            node = graph["files"].get(changed_file)
            if node:
                changed_defines.update(node.get("defines", []))

        for file_path, node in graph["files"].items():
            if file_path in changed_files:
                continue

            imports = set(node.get("imported_files", []))
            if imports & changed_files:
                direct_dependents.add(file_path)

            calls = set(node.get("calls", []))
            called_names = {call.split(".")[-1] for call in calls}
            if changed_defines and called_names & changed_defines:
                symbol_dependents.add(file_path)

        related_files.update(direct_dependents)
        related_files.update(symbol_dependents)

        # Include files the changed file imports: they are useful review context.
        for changed_file in changed_files:
            node = graph["files"].get(changed_file, {})
            related_files.update(node.get("imported_files", []))

        related_files.difference_update(changed_files)
        return {
            "direct_dependents": direct_dependents,
            "symbol_dependents": symbol_dependents,
            "related_files": related_files,
        }

    def _suggest_tests(
        self,
        graph: Dict[str, Any],
        changed_files: Set[str],
        affected: Dict[str, Set[str]],
    ) -> Set[str]:
        tests: Set[str] = set()
        target_names = {
            Path(file_path).stem.replace("test_", "")
            for file_path in changed_files | affected["related_files"]
            if Path(file_path).stem != "__init__"
        }

        for file_path, node in graph["files"].items():
            if not node.get("is_test"):
                continue

            test_stem = Path(file_path).stem.replace("test_", "")
            imports = set(node.get("imported_files", []))
            if imports & (changed_files | affected["related_files"]):
                tests.add(file_path)
            elif test_stem in target_names:
                tests.add(file_path)
            elif any(name in file_path for name in target_names):
                tests.add(file_path)

        return tests

    def _risk_level(
        self,
        changed_files: Set[str],
        affected: Dict[str, Set[str]],
        tests: Set[str],
        graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = 0
        reasons: List[str] = []
        related_count = len(affected["related_files"])

        if not changed_files:
            return {
                "level": "unknown",
                "reasons": ["No changed files were found in the analyzed Python graph."],
            }

        if related_count >= 8:
            score += 3
            reasons.append("Many related files depend on this change.")
        elif related_count >= 3:
            score += 2
            reasons.append("Several related files may be affected.")
        elif related_count > 0:
            score += 1
            reasons.append("At least one related file may be affected.")

        changed_nodes = [graph["files"].get(path, {}) for path in changed_files]
        if any(not node.get("is_test") for node in changed_nodes) and not tests:
            score += 1
            reasons.append("No directly relevant tests were detected.")

        if any(path.startswith(("backend/main", "backend/rag", "backend/retrieval")) for path in changed_files):
            score += 1
            reasons.append("The change touches a core backend pipeline area.")

        if score >= 4:
            level = "high"
        elif score >= 2:
            level = "medium"
        else:
            level = "low"

        if not reasons:
            reasons.append("Limited dependency impact was detected.")

        return {"level": level, "reasons": reasons}

    def _summary(
        self,
        changed_files: Set[str],
        affected: Dict[str, Set[str]],
        tests: Set[str],
        risk_level: str,
    ) -> str:
        if not changed_files:
            return "No matching Python files were found for the requested change."

        direct_count = len(affected["direct_dependents"])
        related_count = len(affected["related_files"])
        test_count = len(tests)
        files = ", ".join(sorted(changed_files))
        return (
            f"Change impact for {files}: {risk_level} risk. "
            f"Found {direct_count} direct dependent file(s), "
            f"{related_count} related file(s), and {test_count} suggested test file(s)."
        )
