"""FastAPI main application for CodeLens RAG system"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from backend.ingest import RepositoryLoader, ASTChunker
from backend.retrieval.hybrid import HybridRetriever
from backend.rag import PromptBuilder, LLMClient, GuardrailChecker
from backend.eval import BatchEvalRunner
from backend.analysis import CodeGraphBuilder, ImpactAnalyzer
from backend.logger import logger

app = FastAPI(
    title="CodeLens API",
    description="AI Code Intelligence Platform",
    version="0.1.0"
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
retriever = HybridRetriever()
prompt_builder = PromptBuilder()
llm_client = LLMClient()
guardrails = GuardrailChecker()
eval_runner = BatchEvalRunner()
impact_analyzer = ImpactAnalyzer()


def add_impact_llm_explanation(result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach an optional LLM explanation without changing graph-based results."""
    if not llm_client.is_available():
        result["llm_explanation"] = None
        result["llm_explanation_status"] = "unavailable"
        return result

    prompt = prompt_builder.build_impact_explanation_prompt(result)
    explanation = llm_client.generate(prompt, max_tokens=350).strip()
    if explanation.startswith("[Error:"):
        result["llm_explanation"] = None
        result["llm_explanation_status"] = "error"
        result["llm_explanation_error"] = explanation
        return result

    result["llm_explanation"] = explanation
    result["llm_explanation_status"] = "generated"
    return result


# Pydantic models
class QueryRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    repo_path: str


class ImpactRequest(BaseModel):
    repo_path: str = "."
    changed_files: List[str]
    changed_symbols: List[str] = []
    refresh_graph: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    faithfulness_score: float = 0.0


class HealthResponse(BaseModel):
    status: str
    llm_available: bool


@app.on_event("startup")
def startup_event():
    """Initialize on startup."""
    logger.info("CodeLens API starting up...")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "llm_available": llm_client.is_available()
    }


@app.post("/ingest", response_model=Dict[str, Any])
async def ingest_repository(request: IngestRequest):
    """Ingest code from a repository.
    
    Tradeoff analysis:
    - We index synchronously (simpler API) but document how to add async
    - Tree-sitter chunking with regex fallback balances accuracy vs compat
    """
    try:
        logger.info(f"Ingesting repository: {request.repo_path}")

        # Load repository
        loader = RepositoryLoader(request.repo_path)
        files = loader.load()
        resolved_repo_path = str(loader.repo_path)

        if not files:
            raise HTTPException(status_code=400, detail="No code files found")

        # Chunk files
        chunker = ASTChunker()
        all_chunks = []

        for file_obj in files:
            chunks = chunker.chunk(
                file_obj["content"],
                file_obj["language"],
                file_obj["path"]
            )
            all_chunks.extend(chunks)

        # Index chunks
        retriever.index(all_chunks)

        # Build/update dependency graph cache for fast impact analysis.
        graph = CodeGraphBuilder(resolved_repo_path).build_cached()
        graph_cache = graph.get("cache", {})

        logger.info(f"Indexed {len(all_chunks)} chunks from {len(files)} files")

        return {
            "status": "success",
            "source": request.repo_path,
            "source_type": loader.source_type,
            "repo_path": resolved_repo_path,
            "files_ingested": len(files),
            "chunks_created": len(all_chunks),
            "graph_files_analyzed": len(graph.get("files", {})),
            "graph_parsed_files": graph_cache.get("parsed_files", 0),
            "graph_reused_files": graph_cache.get("reused_files", 0),
            "message": f"Successfully indexed {len(all_chunks)} code chunks"
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_code(request: QueryRequest):
    """Query the indexed code repository.
    
    Pipeline:
    1. Retrieve relevant chunks (hybrid search)
    2. Build prompt with context
    3. Generate answer with LLM
    4. Check guardrails
    5. Score faithfulness
    """
    try:
        logger.info(f"Processing query: {request.question}")

        # Retrieve context
        results = retriever.search(request.question, top_k=5)

        if not results:
            return {
                "question": request.question,
                "answer": "No relevant code found. Try indexing a repository first.",
                "sources": [],
                "faithfulness_score": 0.0
            }

        # Extract context
        context_docs = []
        for result in results:
            context_docs.append({
                "text": result.get("text", ""),
                "file_path": result.get("metadata", {}).get("file_path", "unknown"),
                "start_line": result.get("metadata", {}).get("start_line", "?"),
                "end_line": result.get("metadata", {}).get("end_line", "?"),
            })

        # Build prompt
        prompt = prompt_builder.build_qa_prompt(
            request.question,
            context_docs
        )

        # Generate answer
        answer = llm_client.generate(prompt)

        # Check guardrails
        is_valid, reason = guardrails.check_response(
            answer,
            request.question,
            "\n".join([doc["text"] for doc in context_docs])
        )

        if not is_valid:
            logger.warning(f"Response failed guardrails: {reason}")
            answer = f"[Guardrail triggered: {reason}]"

        # Score faithfulness
        faithfulness = 0.0
        try:
            scorer_result = eval_runner.scorer.score(
                answer,
                "\n".join([doc["text"] for doc in context_docs]),
                request.question
            )
            faithfulness = scorer_result["score"]
        except Exception as e:
            logger.debug(f"Faithfulness scoring failed: {e}")

        return {
            "question": request.question,
            "answer": answer,
            "sources": context_docs,
            "faithfulness_score": faithfulness
        }

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/impact", response_model=Dict[str, Any])
async def analyze_impact(request: ImpactRequest):
    """Analyze likely blast radius for changed Python files.

    This builds a lightweight dependency graph from imports, definitions,
    and function calls, then reports affected files and likely tests.
    """
    try:
        if not request.changed_files:
            raise HTTPException(status_code=400, detail="changed_files is required")

        logger.info(f"Analyzing impact for: {request.changed_files}")
        resolved_repo_path = str(RepositoryLoader.resolve_repository_path(request.repo_path))
        result = impact_analyzer.analyze(
            resolved_repo_path,
            request.changed_files,
            request.changed_symbols,
            request.refresh_graph,
        )
        return add_impact_llm_explanation(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Impact analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval/status", response_model=Dict[str, Any])
async def eval_status():
    """Get evaluation status."""
    eval_set = eval_runner.load_eval_set()
    return {
        "eval_set_size": len(eval_set),
        "eval_set_path": str(eval_runner.eval_dataset_path),
        "results_count": len(eval_runner.results)
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
