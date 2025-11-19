from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import os
import uuid
import json
from typing import Dict

from raglite.core.rag_service import RAGService
from raglite.api.models import RAGRequest
from raglite.storage.redis_store import set_task_status, get_task_status_from_storage, get_all_task_statuses, redis_client

# Global service instance
rag_service = RAGService()

def process_rag_request(task_id: str, request: RAGRequest):
    """Background task to process RAG requests (synchronous for FastAPI BackgroundTasks)"""
    try:
        set_task_status(task_id, {"status": "processing", "progress": "Initializing..."})

        # Use environment variables as defaults for optional parameters
        embedding_server = request.embedding_server or os.getenv('EMBEDDING_HOST', os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
        llm_server = request.llm_server or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        dataset_server = request.dataset_server or os.getenv('ELASTICSEARCH_HOST', 'http://localhost:1200')
        embedding_model = request.embedding_model or os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')
        llm_model = request.llm_model or os.getenv('LLM_MODEL', 'qwen3:32b')
        es_username = request.es_username or os.getenv('ELASTIC_USERNAME', 'elastic')
        es_password = request.es_password or os.getenv('ELASTICSEARCH_PASSWORD', 'infini_rag_flow')

        set_task_status(task_id, {"status": "processing", "progress": "Connecting to servers..."})

        # Connect to servers
        rag_service.connect_embedding_server(embedding_server)
        rag_service.connect_llm_server(llm_server)
        rag_service.connect_dataset_server(dataset_server, es_username, es_password)

        set_task_status(task_id, {"status": "processing", "progress": "Validating dataset..."})

        # Validate the specified index
        available_datasets = rag_service.list_available_datasets()
        dataset_names = [d['index_name'] for d in available_datasets]
        if request.index_name not in dataset_names:
            available = ", ".join(dataset_names) if dataset_names else "none"
            set_task_status(task_id, {
                "status": "failed",
                "error": f"Dataset index '{request.index_name}' not found or does not contain embeddings. Available datasets: {available}"
            })
            return

        set_task_status(task_id, {"status": "processing", "progress": "Performing semantic search..."})

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

        set_task_status(task_id, {"status": "processing", "progress": "Generating response..."})

        # Generate response (background tasks always return complete responses)
        response = rag_service.generate_response(request.query, search_results, llm_model)

        set_task_status(task_id, {
            "status": "completed",
            "response": response,
            "sources": len(search_results['hits']['hits']),
            "dataset": request.index_name,
            "note": "Background tasks return complete responses. Use streaming endpoints for real-time token delivery."
        })

    except Exception as e:
        set_task_status(task_id, {
            "status": "failed",
            "error": str(e)
        })

async def list_datasets():
    """List all available dataset indices with embeddings"""
    # Connect to dataset server with environment variable defaults (same as standalone script)
    try:
        dataset_server = os.getenv('ELASTICSEARCH_HOST', 'http://localhost:1200')
        es_username = os.getenv('ELASTIC_USERNAME', 'elastic')
        es_password = os.getenv('ELASTICSEARCH_PASSWORD', 'infini_rag_flow')

        rag_service.connect_dataset_server(dataset_server, es_username, es_password)
        datasets = rag_service.list_available_datasets()
        return {"datasets": datasets, "count": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

async def get_task_status_endpoint(task_id: str):
    """Get the status of a background task"""
    status = get_task_status_from_storage(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        **status
    }

async def test_endpoint(param: str):
    """Test endpoint"""
    return {"param": param, "message": "Test successful", "timestamp": str(uuid.uuid4())}

async def generate_rag_response(request: RAGRequest, background_tasks: BackgroundTasks):
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
    # Basic validation
    if not request.query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    if not request.index_name:
        raise HTTPException(status_code=400, detail="index_name parameter is required")

    # Handle streaming vs synchronous based on request.stream
    if request.stream:
        # Streaming mode: Use background tasks
        task_id = str(uuid.uuid4())

        # Initialize task status
        set_task_status(task_id, {
            "status": "queued",
            "message": "Request queued for processing"
        })

        # Add background task
        background_tasks.add_task(process_rag_request, task_id, request)

        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Request has been queued for processing. Use GET /task/{task_id} to check status.",
            "estimated_time": "10-30 seconds depending on query complexity"
        }
    else:
        # Synchronous mode: Process immediately and return response
        try:
            # Use environment variables as defaults for optional parameters
            embedding_server = request.embedding_server or os.getenv('EMBEDDING_HOST', os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
            llm_server = request.llm_server or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            dataset_server = request.dataset_server or os.getenv('ELASTICSEARCH_HOST', 'http://localhost:1200')
            embedding_model = request.embedding_model or os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')
            llm_model = request.llm_model or os.getenv('LLM_MODEL', 'qwen3:32b')
            es_username = request.es_username or os.getenv('ELASTIC_USERNAME', 'elastic')
            es_password = request.es_password or os.getenv('ELASTICSEARCH_PASSWORD', 'infini_rag_flow')

            # Connect to servers
            rag_service.connect_embedding_server(embedding_server)
            rag_service.connect_llm_server(llm_server)
            rag_service.connect_dataset_server(dataset_server, es_username, es_password)

            # Validate the specified index
            available_datasets = rag_service.list_available_datasets()
            dataset_names = [d['index_name'] for d in available_datasets]
            if request.index_name not in dataset_names:
                available = ", ".join(dataset_names) if dataset_names else "none"
                raise HTTPException(status_code=400, detail=f"Dataset index '{request.index_name}' not found or does not contain embeddings. Available datasets: {available}")

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

            # Generate response synchronously
            response = rag_service.generate_response(request.query, search_results, llm_model)

            return {
                "response": response,
                "sources": len(search_results['hits']['hits']) if search_results and 'hits' in search_results else 0,
                "dataset": request.index_name,
                "mode": "synchronous"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

async def health_check():
    """Health check endpoint"""
    all_tasks = get_all_task_statuses()
    active_tasks = sum(1 for status in all_tasks.values() if status.get("status") in ["processing", "queued"])
    completed_tasks = sum(1 for status in all_tasks.values() if status.get("status") == "completed")
    failed_tasks = sum(1 for status in all_tasks.values() if status.get("status") == "failed")

    return {
        "status": "healthy",
        "service": "RAG Web Service",
        "active_tasks": active_tasks,
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "total_tasks": len(all_tasks),
        "storage_backend": "redis" if redis_client else "memory"
    }