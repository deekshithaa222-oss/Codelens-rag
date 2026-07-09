"""Streamlit frontend for CodeLens"""
import streamlit as st
import requests
from pathlib import Path
import json
from backend.eval import BatchEvalRunner

st.set_page_config(page_title="CodeLens", page_icon="🔍", layout="wide")

# Initialize eval runner
eval_runner = BatchEvalRunner()

# Configuration
API_URL = "http://localhost:8000"
EVAL_SET_PATH = "eval_set.json"


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def ingest_repository(repo_path: str):
    """Ingest a repository."""
    try:
        response = requests.post(
            f"{API_URL}/ingest",
            json={"repo_path": repo_path},
            timeout=30
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Ingest timed out. Repository may be large."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_repository(question: str):
    """Query the indexed repository."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def analyze_impact(
    repo_path: str,
    changed_files: str,
    changed_symbols: str,
    refresh_graph: bool,
):
    """Analyze change impact for files or symbols."""
    files = [item.strip() for item in changed_files.splitlines() if item.strip()]
    symbols = [
        item.strip()
        for item in changed_symbols.replace("\n", ",").split(",")
        if item.strip()
    ]

    try:
        response = requests.post(
            f"{API_URL}/impact",
            json={
                "repo_path": repo_path,
                "changed_files": files,
                "changed_symbols": symbols,
                "refresh_graph": refresh_graph,
            },
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def load_eval_set():
    """Load evaluation set."""
    eval_path = Path(EVAL_SET_PATH)
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)
    return []


# Header
st.title("🔍 CodeLens: AI Code Intelligence")
st.markdown("Ask questions about your codebase with **AST-aware chunking**, "
            "**hybrid retrieval** (dense + sparse), and **offline faithfulness evaluation**.")

# Check API health
api_available = check_api_health()
if not api_available:
    st.warning(
        "ℹ️ **Demo Mode:** API backend unavailable. Features work with local FastAPI.\n\n"
        "To run full LLM mode locally:\n"
        "1. `docker-compose up --build`\n"
        "2. Or: `ollama serve` + `uvicorn backend.main:app --reload`\n\n"
        "This demo shows retrieval + faithfulness scoring. Click 'Learn More' tab for details."
    )
else:
    st.success("✅ API connected - Full mode active")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 Ingest",
    "🔍 Query",
    "🧭 Impact",
    "📊 Evaluate",
    "ℹ️ About"
])

# Tab 1: Ingest
with tab1:
    st.subheader("Ingest Code Repository")
    st.write("Index a code repository for semantic search.")

    col1, col2 = st.columns([4, 1])
    with col1:
        repo_path = st.text_input(
            "Repository path:",
            value=".",
            help="Absolute or relative path to repository root"
        )
    with col2:
        ingest_button = st.button("Ingest", use_container_width=True)

    if ingest_button:
        with st.spinner("Ingesting repository..."):
            result = ingest_repository(repo_path)

        if api_available:
            if result.get("status") == "success":
                st.success(
                    f"✅ Indexed {result['files_ingested']} files "
                    f"into {result['chunks_created']} chunks"
                )
            else:
                st.error(f"❌ {result.get('message', 'Unknown error')}")
        else:
            st.info("📚 **Demo Mode:** Backend required to ingest.\n\n"
                   "To index a repository:\n"
                   "1. Deploy locally: `docker-compose up --build`\n"
                   "2. Upload repo path\n"
                   "3. System chunks code using AST parser + hybrid indexing")


# Tab 2: Query
with tab2:
    st.subheader("Query Code Repository")
    st.write("Ask questions about the indexed code.")

    question = st.text_area(
        "Your question:",
        placeholder="e.g., How does authentication work in this codebase?",
        height=100
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        query_button = st.button("Search", use_container_width=True, type="primary")
    with col2:
        st.write("")  # Spacing
    with col3:
        st.write("")  # Spacing

    if query_button and question:
        with st.spinner("Querying..."):
            result = query_repository(question)

        if "error" not in result and api_available:
            # Full mode: show real results
            st.subheader("Answer")
            st.write(result.get("answer", ""))

            # Faithfulness score
            faithfulness = result.get("faithfulness_score", 0)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Faithfulness Score",
                    f"{faithfulness:.1%}",
                    help="How well-grounded the answer is in context"
                )
            with col2:
                if faithfulness > 0.7:
                    st.success("✅ High confidence")
                elif faithfulness > 0.4:
                    st.warning("⚠️ Medium confidence")
                else:
                    st.error("❌ Low confidence")

            # Sources
            st.subheader("Sources")
            sources = result.get("sources", [])
            for i, source in enumerate(sources, 1):
                with st.expander(
                    f"📄 {source.get('file_path', 'unknown')} "
                    f"(lines {source.get('start_line', '?')}-{source.get('end_line', '?')})"
                ):
                    st.code(source.get("text", ""), language="python")
        else:
            # Demo mode
            st.info("📚 **Demo Mode: Retrieval Simulation**\n\nIn full mode, CodeLens would:\n"
                   "1. Retrieve relevant code chunks (hybrid search)\n"
                   "2. Generate answer via Ollama LLM\n"
                   "3. Score faithfulness offline (no external LLM calls)\n\n"
                   "This demo mode shows the retrieval capability without a running backend.")
            
            # Show mock retrieval results
            st.subheader("Mock Retrieval Results")
            st.json({
                "question": question,
                "status": "Demo - Backend required for full results",
                "hybrid_search": "Dense (60%) + Sparse BM25 (40%)",
                "faithfulness_scorer": "Offline (no LLM calls)",
                "note": "Deploy locally or run backend for full functionality"
            })


# Tab 3: Impact
with tab3:
    st.subheader("Change Impact Analysis")
    st.write("Find likely affected files and tests before reviewing or shipping a code change.")

    impact_repo_path = st.text_input(
        "Repository path",
        value=".",
        help="Repository root used for analysis",
        key="impact_repo_path"
    )
    changed_files = st.text_area(
        "Changed files",
        placeholder="backend/rag/llm.py\nbackend/main.py",
        height=100,
        help="One file path per line, relative to the repository root"
    )
    changed_symbols = st.text_input(
        "Changed symbols",
        placeholder="LLMClient, generate",
        help="Optional function or class names if you know what changed"
    )
    refresh_graph = st.checkbox(
        "Refresh dependency graph",
        value=False,
        help="Use this if files changed since the last ingestion"
    )

    if st.button("Analyze Impact", type="primary"):
        if not changed_files.strip():
            st.warning("Add at least one changed file.")
        elif not api_available:
            st.info("🧭 **Demo Mode:** Start the FastAPI backend to run impact analysis.")
        else:
            with st.spinner("Analyzing dependency graph..."):
                result = analyze_impact(
                    impact_repo_path,
                    changed_files,
                    changed_symbols,
                    refresh_graph
                )

            if "error" in result:
                st.error(result["error"])
            else:
                risk = result.get("risk", "unknown")
                if risk == "high":
                    st.error(f"Risk: {risk.upper()}")
                elif risk == "medium":
                    st.warning(f"Risk: {risk.upper()}")
                elif risk == "low":
                    st.success(f"Risk: {risk.upper()}")
                else:
                    st.info(f"Risk: {risk.upper()}")

                st.write(result.get("summary", ""))

                stats = result.get("graph_stats", {})
                cache = result.get("graph_cache", {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Python Files", stats.get("python_files_analyzed", 0))
                with col2:
                    st.metric("Related Files", stats.get("related_files_found", 0))
                with col3:
                    st.metric("Changed Found", stats.get("changed_files_found", 0))

                st.caption(
                    f"Graph source: {cache.get('source', 'unknown')} | "
                    f"parsed: {cache.get('parsed_files', 0)} | "
                    f"reused: {cache.get('reused_files', 0)}"
                )

                reasons = result.get("risk_reasons", [])
                if reasons:
                    st.subheader("Risk Reasons")
                    for reason in reasons:
                        st.write(f"- {reason}")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Direct Dependents")
                    dependents = result.get("direct_dependents", [])
                    if dependents:
                        for file_path in dependents:
                            st.code(file_path, language="text")
                    else:
                        st.write("No direct dependents found.")

                with col2:
                    st.subheader("Suggested Tests")
                    tests = result.get("suggested_tests", [])
                    if tests:
                        for file_path in tests:
                            st.code(file_path, language="text")
                    else:
                        st.write("No directly related tests found.")

                with st.expander("Related Files"):
                    related = result.get("related_files", [])
                    if related:
                        for file_path in related:
                            st.code(file_path, language="text")
                    else:
                        st.write("No related files found.")


# Tab 4: Evaluate
with tab4:
    st.subheader("Offline Faithfulness Evaluation")
    st.write(
        "Evaluate RAG system on a test set using heuristic faithfulness scoring "
        "(no external LLM calls)."
    )

    eval_set = load_eval_set()

    if not eval_set:
        st.warning(
            "No evaluation set found. Create `eval_set.json` with structure:\n"
            "```json\n"
            "[{\"question\": \"...\", \"context\": \"...\", \"expected\": \"...\"}]\n"
            "```"
        )
    else:
        st.info(f"📋 Loaded {len(eval_set)} evaluation cases")

        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Evaluating..."):
                # Run evaluation
                cases = []
                for item in eval_set[:3]:  # Limit to 3 for demo
                    result = query_repository(item.get("question", ""))
                    cases.append({
                        "question": item.get("question"),
                        "response": result.get("answer", ""),
                        "context": item.get("context", "")
                    })

                metrics = eval_runner.evaluate(cases)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Evals", metrics["total_evals"])
            with col2:
                st.metric("Mean Faithfulness", f"{metrics['mean_faithfulness']:.1%}")
            with col3:
                st.metric("Range", f"{metrics['min_faithfulness']:.0%}–{metrics['max_faithfulness']:.0%}")
            with col4:
                st.metric("Pass Rate (>70%)", f"{metrics['pass_rate']:.0%}")

            # Detailed results
            st.subheader("Detailed Results")
            for i, result in enumerate(metrics["results"], 1):
                with st.expander(f"Case {i}: {result['question'][:50]}..."):
                    st.write(f"**Score:** {result['faithfulness']:.1%}")
                    st.write(f"**Details:** {result['details']['reasoning']}")


# Tab 5: About
with tab5:
    st.subheader("About CodeLens")

    st.markdown("""
    **CodeLens** is an AI code intelligence platform with four defensible differentiators:

    #### 1. AST-Aware Chunking
    - Uses tree-sitter for semantic code boundaries (functions, classes)
    - Regex fallback for unsupported languages
    - **Tradeoff:** Semantic boundaries > naive size-based splits, but tree-sitter requires native deps

    #### 2. Hybrid Dense+Sparse Retrieval
    - **Dense:** Sentence transformers for semantic search (60% weight)
    - **Sparse:** BM25 for exact keyword matches (40% weight)
    - **Tradeoff:** Catches both semantic + exact matches, higher index cost

    #### 3. Change Impact Analysis
    - Builds a cached Python dependency graph from imports, definitions, and calls
    - Estimates affected files and suggested tests before a change ships
    - **Tradeoff:** Fast and local, but Python-focused and not a full runtime tracer

    #### 4. Offline Faithfulness Eval
    - No external LLM calls—evaluates locally on entity overlap + hallucination checks
    - Batch eval runner for systematic quality measurement
    - **Tradeoff:** Heuristic-based, simpler than semantic entailment, but fast + repeatable

    ### MCP Value
    MCP can add value as an optional enterprise integration layer when CodeLens needs context beyond local files:
    - GitHub/GitLab issues, PRs, commits, reviews, branches, and CI status
    - Internal docs, ADRs, runbooks, service catalogs, and ownership metadata
    - Logs, incidents, deployments, feature flags, database schemas, and migration history

    Best fit: use read-only MCP servers for repository platforms and internal docs first. Add logs, databases, and deployment tools after access controls, audit logging, and secret redaction are in place.

    ### Architecture
    - **Backend:** FastAPI (ingest, retrieval, RAG, evaluation)
    - **Frontend:** Streamlit (UI + eval runner)
    - **LLM:** Ollama (self-hosted, offline, quantized models)
    - **Storage:** ChromaDB (vector) + BM25 (sparse)

    ### Running Locally
    ```bash
    # Install dependencies
    pip install -r requirements.txt

    # Pull a model (e.g., codellama)
    ollama pull codellama

    # Start Ollama service
    ollama serve

    # In a new terminal, start backend
    uvicorn backend.main:app --reload

    # In another terminal, start frontend
    streamlit run frontend/app.py
    ```

    ### Key Tradeoffs (see README for full table)
    - Chunking: AST accuracy vs regex compatibility
    - Retrieval: Hybrid cost vs quality gains
    - Evaluation: Heuristic speed vs semantic precision
    """)
