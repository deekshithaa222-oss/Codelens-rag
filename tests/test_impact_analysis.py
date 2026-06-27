"""Tests for change impact analysis."""
from pathlib import Path

from backend.analysis import CodeGraphBuilder, ImpactAnalyzer


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_graph_builder_resolves_imported_symbols(tmp_path):
    write_file(
        tmp_path / "pkg" / "service.py",
        """
class Service:
    pass

def build_service():
    return Service()
""",
    )
    write_file(
        tmp_path / "pkg" / "api.py",
        """
from pkg.service import Service

def handler():
    return Service()
""",
    )

    graph = CodeGraphBuilder(str(tmp_path)).build()

    assert "pkg/api.py" in graph["files"]
    assert "pkg/service.py" in graph["files"]["pkg/api.py"]["imported_files"]
    assert "Service" in graph["files"]["pkg/service.py"]["classes"]


def test_impact_analyzer_finds_dependents_and_tests(tmp_path):
    write_file(
        tmp_path / "backend" / "rag" / "llm.py",
        """
class LLMClient:
    def generate(self):
        return "ok"
""",
    )
    write_file(
        tmp_path / "backend" / "main.py",
        """
from backend.rag.llm import LLMClient

def query_code():
    client = LLMClient()
    return client.generate()
""",
    )
    write_file(
        tmp_path / "tests" / "test_llm.py",
        """
from backend.rag.llm import LLMClient

def test_generate():
    assert LLMClient().generate() == "ok"
""",
    )

    result = ImpactAnalyzer().analyze(
        str(tmp_path),
        ["backend/rag/llm.py"],
        ["generate"],
    )

    assert result["risk"] in {"medium", "high"}
    assert "backend/main.py" in result["direct_dependents"]
    assert "tests/test_llm.py" in result["suggested_tests"]
    assert result["graph_stats"]["python_files_analyzed"] == 3
    assert result["graph_cache"]["enabled"] is True


def test_graph_builder_uses_file_hash_cache(tmp_path):
    service_path = tmp_path / "pkg" / "service.py"
    api_path = tmp_path / "pkg" / "api.py"
    write_file(
        service_path,
        """
class Service:
    pass
""",
    )
    write_file(
        api_path,
        """
from pkg.service import Service

def handler():
    return Service()
""",
    )

    builder = CodeGraphBuilder(str(tmp_path))
    first_graph = builder.build_cached()
    assert first_graph["cache"]["parsed_files"] == 2
    assert first_graph["cache"]["reused_files"] == 0

    second_graph = builder.build_cached()
    assert second_graph["cache"]["parsed_files"] == 0
    assert second_graph["cache"]["reused_files"] == 2

    write_file(
        service_path,
        """
class Service:
    def run(self):
        return "ok"
""",
    )

    updated_graph = builder.build_cached()
    assert updated_graph["cache"]["parsed_files"] == 1
    assert updated_graph["cache"]["reused_files"] == 1
    assert "Service" in updated_graph["files"]["pkg/service.py"]["classes"]
