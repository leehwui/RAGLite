from fastapi import FastAPI, HTTPException, BackgroundTasks
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
    health_check
)

app = FastAPI(title="RAG Web Service", description="Retrieval-Augmented Generation API", version="0.2.0")

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
async def generate_rag_response_endpoint(request, background_tasks: BackgroundTasks):
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