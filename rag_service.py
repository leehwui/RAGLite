from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import os
import uuid
import json

# Import from the new modular structure
from raglite.api.endpoints import (
    list_datasets,
    get_task_status_endpoint,
    test_endpoint,
    generate_rag_response,
    health_check,
    rag_service  # Import the global RAGService instance
)
from raglite.api.models import RAGRequest

app = FastAPI(title="RAG Web Service", description="Retrieval-Augmented Generation API", version="0.2.0")

# CORS configuration: allow origins used by the frontend (comma-separated env var)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,  # set True if you use cookies or credentials in the browser
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers
)

@app.get("/datasets")
async def list_datasets_endpoint():
    """List all available dataset indices with embeddings"""
    return await list_datasets()

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    return await get_task_status_endpoint(task_id)

@app.get("/test/{param}")
async def test_endpoint_handler(param: str):
    """Test endpoint"""
    return await test_endpoint(param)

@app.post("/rag/generate")
async def generate_rag_response_endpoint(request: RAGRequest, background_tasks: BackgroundTasks):
    """
    Generate RAG response for a query

    - **query**: The search query (required)
    - **index_name**: Elasticsearch index name to search in (required)
    - **embedding_server**: Ollama server URL for embeddings (optional, uses env var OLLAMA_HOST)
    - **llm_server**: Ollama server URL for LLM (optional, uses env var OLLAMA_HOST)
    - **dataset_server**: Elasticsearch server URL (optional, uses env var ELASTICSEARCH_HOST)
    - **embedding_model**: Embedding model name (optional, uses env var EMBEDDING_MODEL)
    - **llm_model**: LLM model name (optional, uses env var LLM_MODEL)
    - **es_username**: Elasticsearch username (optional, uses env var ELASTIC_USERNAME)
    - **es_password**: Elasticsearch password (optional, uses env var ELASTICSEARCH_PASSWORD)
    - **top_k**: Number of documents to retrieve (default: 3)
    - **kb_id**: Optional filter by knowledge base ID to search only specific datasets
    - **filename_pattern**: Optional filter by filename pattern (supports wildcards like *.xlsx)
    - **hybrid_boost**: Whether to use hybrid search combining semantic and keyword matching (default: true)
    - **stream**: Whether to stream the response (default: true)
    - **format**: Response format - "json" or "sse" (default: sse, only used when stream=true)

    When stream=true: Returns a task ID for tracking progress. Use GET /task/{task_id} to check status.
    When stream=false: Returns immediate synchronous response.
    """
    return await generate_rag_response(request, background_tasks)

@app.post("/rag/stream")
async def stream_rag_response_post(request: RAGRequest):
    """
    Stream RAG response for a query (POST endpoint for frontend compatibility)

    - **query**: The search query (required)
    - **index_name**: Elasticsearch index name to search in (required)
    - **top_k**: Number of documents to retrieve (default: 3)
    - **embedding_model**: Embedding model name (optional, uses env var EMBEDDING_MODEL)
    - **llm_model**: LLM model name (optional, uses env var LLM_MODEL)
    - **kb_id**: Optional filter by knowledge base ID
    - **filename_pattern**: Optional filter by filename pattern
    - **reranker_type**: Reranker type ("cross_encoder" or "none")
    - **reranker_model**: Cross-encoder model name
    - **reranker_top_k**: Number of docs for reranker to consider
    """
    try:
        # Use environment variables as defaults
        embedding_server = request.embedding_server or os.getenv('EMBEDDING_HOST', os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
        llm_server = request.llm_server or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        dataset_server = request.dataset_server or os.getenv('ELASTICSEARCH_HOST', 'http://localhost:1200')
        embedding_model = request.embedding_model or os.getenv('EMBEDDING_MODEL', 'qwen3-embedding:4b')
        llm_model = request.llm_model or os.getenv('LLM_MODEL', 'qwen3:32b')
        es_username = request.es_username or os.getenv('ELASTIC_USERNAME', 'elastic')
        es_password = request.es_password or os.getenv('ELASTICSEARCH_PASSWORD', 'infini_rag_flow')

        # Connect to servers (use the global rag_service instance)
        rag_service.connect_embedding_server(embedding_server)
        rag_service.connect_llm_server(llm_server)
        rag_service.connect_dataset_server(dataset_server, es_username, es_password)

        # Validate index
        available_datasets = rag_service.list_available_datasets()
        dataset_names = [d['index_name'] for d in available_datasets]
        if request.index_name not in dataset_names:
            available = ", ".join(dataset_names) if dataset_names else "none"
            raise HTTPException(status_code=400, detail=f"Dataset index '{request.index_name}' not found. Available: {available}")

        # Resolve reranking settings
        reranker_type = request.reranker_type or os.getenv('RERANKER_TYPE', 'none')
        reranker_model = request.reranker_model or os.getenv('RERANKER_MODEL')
        reranker_top_k = request.reranker_top_k or int(os.getenv('RERANKER_TOP_K', request.top_k))

        # Perform semantic search
        search_results = rag_service.semantic_search(
            request.query,
            request.index_name,
            embedding_model,
            request.top_k,
            request.kb_id,
            request.filename_pattern,
            request.hybrid_boost
        )

        # Apply reranking
        search_results = rag_service.rerank_search_results(
            request.query,
            search_results,
            reranker_type,
            reranker_model,
            reranker_top_k
        )

        # Stream the response
        return StreamingResponse(
            rag_service.generate_sse_stream(request.query, search_results, llm_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@app.get("/rag/stream")
async def stream_rag_response(
    query: str,
    index_name: str,
    top_k: int = 3,
    embedding_model: str = None,
    llm_model: str = None,
    kb_id: str = None,
    filename_pattern: str = None,
    reranker_type: str = None,
    reranker_model: str = None,
    reranker_top_k: int = None
):
    """
    Stream RAG response for a query (GET endpoint for frontend compatibility)

    - **query**: The search query (required)
    - **index_name**: Elasticsearch index name to search in (required)
    - **top_k**: Number of documents to retrieve (default: 3)
    - **embedding_model**: Embedding model name (optional, uses env var EMBEDDING_MODEL)
    - **llm_model**: LLM model name (optional, uses env var LLM_MODEL)
    - **kb_id**: Optional filter by knowledge base ID
    - **filename_pattern**: Optional filter by filename pattern
    - **reranker_type**: Reranker type ("cross_encoder" or "none")
    - **reranker_model**: Cross-encoder model name
    - **reranker_top_k**: Number of docs for reranker to consider
    """
    try:
        # Use environment variables as defaults
        embedding_server = os.getenv('EMBEDDING_HOST', os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
        llm_server = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        dataset_server = os.getenv('ELASTICSEARCH_HOST', 'http://localhost:1200')
        embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'qwen3-embedding:4b')
        llm_model = llm_model or os.getenv('LLM_MODEL', 'qwen3:32b')
        es_username = os.getenv('ELASTIC_USERNAME', 'elastic')
        es_password = os.getenv('ELASTICSEARCH_PASSWORD', 'infini_rag_flow')

        # Connect to servers (use the global rag_service instance)
        rag_service.connect_embedding_server(embedding_server)
        rag_service.connect_llm_server(llm_server)
        rag_service.connect_dataset_server(dataset_server, es_username, es_password)

        # Validate index
        available_datasets = rag_service.list_available_datasets()
        dataset_names = [d['index_name'] for d in available_datasets]
        if index_name not in dataset_names:
            available = ", ".join(dataset_names) if dataset_names else "none"
            raise HTTPException(status_code=400, detail=f"Dataset index '{index_name}' not found. Available: {available}")

        # Resolve reranking settings
        reranker_type = reranker_type or os.getenv('RERANKER_TYPE', 'none')
        reranker_model = reranker_model or os.getenv('RERANKER_MODEL')
        reranker_top_k = reranker_top_k or int(os.getenv('RERANKER_TOP_K', top_k))

        # Perform semantic search
        search_results = rag_service.semantic_search(
            query,
            index_name,
            embedding_model,
            top_k,
            kb_id,
            filename_pattern,
            True  # hybrid_boost
        )

        # Apply reranking
        search_results = rag_service.rerank_search_results(
            query,
            search_results,
            reranker_type,
            reranker_model,
            reranker_top_k
        )

        # Stream the response
        return StreamingResponse(
            rag_service.generate_sse_stream(query, search_results, llm_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@app.get("/health")
async def health_check_endpoint():
    """Health check endpoint"""
    return await health_check()

if __name__ == "__main__":
    # ASCII Art Banner
    banner = r"""
     ________  ________  ________  ___       ___  _________  _______
    |\   __  \|\   __  \|\   ____\|\  \     |\  \|\___   ___\\  ___ \
    \ \  \|\  \ \  \|\  \ \  \___|\ \  \    \ \  \|___ \  \_\ \   __/|
     \ \   _  _\ \   __  \ \  \  __\ \  \    \ \  \   \ \  \ \ \  \_|/__
      \ \  \\  \\ \  \ \  \ \  \|\  \ \  \____\ \  \   \ \  \ \ \  \_|\ \
       \ \__\\ _\\ \__\ \__\ \_______\ \_______\ \__\   \ \__\ \ \_______\
        \|__|\|__|\|__|\|__|\|_______|\|_______|\|__|    \|__|  \|_______|
    """
    print(banner)
    port = int(os.getenv("PORT", 8000))
    print(f"Starting RAG Web Service on port {port}")
    print("API Documentation available at: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)