# STAR Story - CodeLens (WEHack 2026)

## Situation

During WEHack 2026, I built CodeLens, an AI-powered code intelligence platform designed to help developers and non-technical users understand unfamiliar GitHub repositories more quickly.

Normally, users have to search through many files, trace functions manually, and repeatedly provide code context to general-purpose AI chatbots. With larger repositories, the model cannot keep the entire codebase in its context window, which can lead to incomplete or unsupported answers. General AI chat tools can help for small repos or one-time questions, but I wanted a more repeatable system that could index a repository once, retrieve the correct code context for every question, and make repository knowledge available through both a simple web interface and AI-enabled developer tools.

## Task

My goal was to build an end-to-end developer assistant during the hackathon.

The platform needed to:

- Ingest a GitHub repository.
- Understand functions, classes, methods, imports, and code structure.
- Perform both semantic and exact keyword search.
- Answer repository-specific questions with source-grounded context.
- Analyze the likely impact of code changes.
- Provide a simple interface for non-technical users.
- Expose the same capabilities to developers inside IDE or AI-chat workflows.
- Run locally in a consistent environment without requiring paid cloud services.

## Action

I built the backend using FastAPI and the frontend using Streamlit. FastAPI exposed endpoints for repository ingestion, question answering, health checks, evaluation status, and change-impact analysis. I used Pydantic request models so each endpoint received structured input, such as a question, repository path, changed files, or changed symbols. Streamlit gave non-technical users a simple UI to ingest a repository, ask questions, view source snippets, and run impact analysis without using curl, Postman, or API documentation.

When a user submitted a GitHub repository URL, the backend cloned it into a local cache under `.codelens/repos`. If the same repository was requested again, the backend reused the cached clone and attempted a fast-forward Git update instead of recloning from scratch. I did not use a fixed TTL in the hackathon version; the cached repository stays as long as the backend filesystem or mounted volume persists. For production, I would add a cleanup policy, such as a 24-hour TTL for demos or a 7-day TTL for authenticated users, depending on disk limits and usage patterns.

After cloning or loading a local repository, I filtered out junk folders, `.git`, `.codelens`, virtual environments, large files, unsupported extensions, and files that looked like they might contain secrets. Rather than splitting the code into arbitrary text blocks by character count, I used AST-aware parsing and fallback definition-based chunking to preserve meaningful units like functions, classes, and methods. This matters because arbitrary chunks can split a function in half and make retrieved context incomplete.

For semantic search, I generated embeddings for those code chunks using the `sentence-transformers` Python library with the pretrained `sentence-transformers/all-MiniLM-L6-v2` model. I chose MiniLM because it is lightweight, free, fast enough to run locally, and practical for a hackathon demo without requiring a GPU or paid embedding API. I considered alternatives such as OpenAI embeddings, Cohere Embed, FAISS, Pinecone, Weaviate, Qdrant, Milvus, pgvector, BGE/E5 models, and code-specific embedding models. I chose ChromaDB for vector storage because it is open-source, easy to run locally, persists embeddings with minimal setup, and avoids the cost and infrastructure overhead of a managed vector database like Pinecone. FAISS would have been fast, but it is lower-level and would require more custom metadata and persistence work.

This indexing step allowed the repository to be processed once and searched repeatedly. Instead of rereading every file, rechunking the code, and regenerating embeddings for every question, CodeLens embeds the user question, searches the existing index, and retrieves only the most relevant code sections. This reduced repeated computation and helped control token cost because the LLM only receives a small set of relevant chunks instead of the whole repository.

I also implemented BM25 keyword search because semantic retrieval can miss exact technical identifiers such as class names, function names, API routes, variables, and environment keys. BM25 tokenizes the query and code chunks, computes IDF scores so rare tokens like `JWT_SECRET` receive more weight than common tokens like `return`, and ranks chunks by exact lexical relevance. For example, if a user asks where `JWT_SECRET` is used, BM25 can find the exact variable even if semantic search returns broader authentication-related code. I combined ChromaDB semantic results and BM25 keyword results into a hybrid retriever, with dense retrieval weighted at 60% and sparse retrieval weighted at 40%, so CodeLens could handle both natural-language questions and precise technical searches.

When a user asked a repository question, CodeLens retrieved the top relevant code chunks, built a prompt with file paths and line ranges, and sent that context to a local LLM through Ollama. In the FastAPI endpoint, the retrieval layer currently uses the top 5 chunks to keep the prompt focused. In the MCP tools, `top_k` is configurable and defaults to 5. The LLM is not the search engine; it is the answer generator. Retrieval decides what context the model sees, which reduces hallucination risk, although it does not eliminate it completely.

I also built a lightweight change-impact analysis feature. This is separate from normal RAG question answering. Instead of asking the LLM to guess what might break from a few retrieved chunks, I used a graph-based approach. I parsed Python files with the standard `ast` module and extracted imports, function definitions, class definitions, function calls, module names, and test-file signals. Each file became a node in a lightweight dependency graph, and relationships such as imports and calls became edges. When a user provided changed files or symbols, the backend used the graph to find direct dependents, symbol dependents, related files, suggested tests, and a risk level.

The risk summary is generated programmatically from deterministic rules. For example, risk increases when many related files depend on the changed code, when no directly relevant tests are detected, or when the change touches core backend areas such as retrieval, RAG, or the API layer. After that graph-based result is computed, I added the local LLM only as a presentation layer. The graph remains the source of truth, and the LLM receives the structured impact result with instructions not to invent files, tests, risks, or dependencies. This gives users a clearer explanation while keeping the actual impact calculation explainable and deterministic.

While building the project, I also thought about developer experience. Streamlit worked well for general users, but developers usually do not want to leave their IDE, open a separate web app, manually call FastAPI endpoints, or copy JSON results back into an AI chat. I added MCP so AI-enabled IDEs and chat clients could call CodeLens as a tool layer. Without MCP, users can still use the Streamlit UI or manually call FastAPI endpoints like `/ingest`, `/query`, and `/impact`. With MCP, an AI client can call tools such as `ingest_repository`, `search_code`, `answer_question`, and `analyze_change_impact` directly using structured parameters. This lets developers ask natural-language questions inside their IDE or AI chatbot while the client invokes the CodeLens tools in the background.

Finally, I containerized the application with Docker so the full system could run consistently in a local environment. Docker Compose starts the FastAPI backend, Streamlit frontend, and Ollama service together. This made the demo easier to reproduce because users did not have to manually align Python versions, dependencies, backend commands, frontend commands, and local LLM setup across different machines.

## Result

The final application could ingest GitHub repositories, build a persistent searchable code index, perform hybrid semantic and keyword retrieval, answer repository-specific questions, and identify likely change-impact areas.

Non-technical users could access the core workflow through Streamlit. Developers could access the same workflow directly from MCP-compatible IDEs or AI chat clients without needing to learn custom API endpoints or manually format JSON payloads. The project stood out by combining AST-aware parsing, hybrid retrieval, local embeddings, ChromaDB vector storage, BM25 keyword search, local LLM inference through Ollama, graph-based change-impact analysis, optional LLM explanations, Dockerized deployment, and MCP-based extensibility.

For evaluation, I used public GitHub repositories and manually created a small set of 25 repository-specific questions. For each question, I recorded the expected file, function, class, or route that should be retrieved. I then checked whether the expected result appeared within the top five retrieved chunks. The hybrid ChromaDB and BM25 retrieval pipeline returned the expected result for 22 of 25 questions, giving an 88% top-five retrieval success rate. I also checked answer grounding with offline faithfulness scoring, and 21 of 25 answers were fully supported by the retrieved source context, giving an 84% grounded-answer rate.

For a hackathon, this evaluation was enough to demonstrate that the core RAG pipeline worked, but I would not treat it as a full production benchmark. A stronger evaluation would include more repositories, more programming languages, larger question sets, baseline comparisons against BM25-only and embedding-only retrieval, latency measurements, hallucination tracking, and human review of answer quality.

CodeLens won second place at WEHack 2026.

## Reflection

The project taught me that an effective AI application depends on more than the LLM. The LLM is useful for generating explanations, but the quality of the system depends heavily on how context is selected and how much deterministic structure surrounds the model. AST-aware chunking preserved code structure, hybrid retrieval handled both semantic meaning and exact identifiers, ChromaDB made the index reusable, BM25 improved exact technical search, and graph-based impact analysis provided more reliable dependency reasoning than asking the LLM to guess.

I also learned the importance of separating core business logic from access layers. The same backend services power the Streamlit UI, FastAPI endpoints, and MCP tools. That made the system easier to extend without duplicating logic. The main value of CodeLens is not that it replaces ChatGPT or Claude for every use case. For a small repo or a one-time question, a general chatbot may be enough. CodeLens is valuable when users need a reusable, local, source-grounded code intelligence layer that can index once, search repeatedly, reduce token costs, support developer workflows, and explain change impact more reliably.
