"""MCP server exposing CodeLens code intelligence tools.

Run with:
    python -m backend.mcp_server
"""
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from backend.analysis import ImpactAnalyzer
from backend.eval import BatchEvalRunner, FaithfulnessScorer
from backend.ingest import ASTChunker, RepositoryLoader
from backend.rag import GuardrailChecker, LLMClient, PromptBuilder
from backend.retrieval.hybrid import HybridRetriever


mcp = FastMCP("CodeLens")

_retriever = HybridRetriever()
_prompt_builder = PromptBuilder()
_llm_client = LLMClient()
_guardrails = GuardrailChecker()
_impact_analyzer = ImpactAnalyzer()
_faithfulness_scorer = FaithfulnessScorer()
_eval_runner = BatchEvalRunner()
_indexed_repo_path: str | None = None


def _load_chunks(repo_path: str) -> List[Dict[str, Any]]:
    """Load and chunk a repository using the existing CodeLens ingestion stack."""
    loader = RepositoryLoader(repo_path)
    files = loader.load()
    chunker = ASTChunker()
    chunks: List[Dict[str, Any]] = []

    for file_obj in files:
        chunks.extend(
            chunker.chunk(
                file_obj["content"],
                file_obj["language"],
                file_obj["path"],
            )
        )

    return chunks


def _ensure_index(repo_path: str) -> Dict[str, Any]:
    """Index a repo once per MCP server process."""
    global _indexed_repo_path

    resolved_repo_path = str(RepositoryLoader.resolve_repository_path(repo_path))

    if _indexed_repo_path == resolved_repo_path:
        return {
            "status": "ready",
            "source": repo_path,
            "repo_path": resolved_repo_path,
            "message": "Repository is already indexed in this MCP server session.",
        }

    chunks = _load_chunks(resolved_repo_path)
    _retriever.index(chunks)
    _indexed_repo_path = resolved_repo_path
    return {
        "status": "success",
        "source": repo_path,
        "repo_path": resolved_repo_path,
        "chunks_created": len(chunks),
        "message": f"Indexed {len(chunks)} code chunks.",
    }


@mcp.tool()
def ingest_repository(repo_path: str = ".") -> Dict[str, Any]:
    """Index a repository for later MCP search and question answering."""
    return _ensure_index(repo_path)


@mcp.tool()
def search_code(
    question: str,
    repo_path: str = ".",
    top_k: int = 5,
    auto_ingest: bool = True,
) -> Dict[str, Any]:
    """Search code with CodeLens hybrid dense+sparse retrieval."""
    if auto_ingest:
        _ensure_index(repo_path)

    results = _retriever.search(question, top_k=top_k)
    sources = []
    for result in results:
        metadata = result.get("metadata", {})
        sources.append(
            {
                "file_path": metadata.get("file_path", "unknown"),
                "start_line": metadata.get("start_line", "?"),
                "end_line": metadata.get("end_line", "?"),
                "score": result.get("score", 0),
                "text": result.get("text", ""),
            }
        )

    return {
        "question": question,
        "repo_path": repo_path,
        "sources": sources,
    }


@mcp.tool()
def answer_question(
    question: str,
    repo_path: str = ".",
    top_k: int = 5,
    auto_ingest: bool = True,
) -> Dict[str, Any]:
    """Answer a codebase question using retrieval, Ollama, guardrails, and faithfulness scoring."""
    search_result = search_code(question, repo_path, top_k, auto_ingest)
    sources = search_result["sources"]
    if not sources:
        return {
            "question": question,
            "answer": "No relevant code found. Ingest the repository first or broaden the question.",
            "sources": [],
            "faithfulness_score": 0.0,
        }

    context_docs = [
        {
            "text": source["text"],
            "file_path": source["file_path"],
            "start_line": source["start_line"],
            "end_line": source["end_line"],
        }
        for source in sources
    ]
    context = "\n".join(doc["text"] for doc in context_docs)
    prompt = _prompt_builder.build_qa_prompt(question, context_docs)
    answer = _llm_client.generate(prompt)

    is_valid, reason = _guardrails.check_response(answer, question, context)
    if not is_valid:
        answer = f"[Guardrail triggered: {reason}]"

    faithfulness = _faithfulness_scorer.score(answer, context, question)
    return {
        "question": question,
        "answer": answer,
        "sources": context_docs,
        "faithfulness_score": faithfulness["score"],
        "faithfulness_details": faithfulness,
    }


@mcp.tool()
def analyze_change_impact(
    changed_files: List[str],
    repo_path: str = ".",
    changed_symbols: List[str] | None = None,
    refresh_graph: bool = False,
) -> Dict[str, Any]:
    """Analyze likely blast radius and suggested tests for changed Python files."""
    resolved_repo_path = str(RepositoryLoader.resolve_repository_path(repo_path))
    return _impact_analyzer.analyze(
        resolved_repo_path,
        changed_files,
        changed_symbols or [],
        refresh_graph,
    )


@mcp.tool()
def score_faithfulness(
    response: str,
    context: str,
    question: str = "",
) -> Dict[str, Any]:
    """Score whether a response is grounded in supplied context without external LLM calls."""
    return _faithfulness_scorer.score(response, context, question)


@mcp.tool()
def evaluation_status() -> Dict[str, Any]:
    """Report local evaluation set status."""
    eval_set = _eval_runner.load_eval_set()
    return {
        "eval_set_size": len(eval_set),
        "eval_set_path": str(_eval_runner.eval_dataset_path),
        "results_count": len(_eval_runner.results),
    }


if __name__ == "__main__":
    mcp.run()
