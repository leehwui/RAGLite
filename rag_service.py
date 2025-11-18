from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import ollama
import numpy as np
from elasticsearch import Elasticsearch
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="RAG Web Service", description="Retrieval-Augmented Generation API", version="1.0.0")

from pydantic import BaseModel, Field, field_validator

class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query (required, non-empty)")
    index_name: str = Field(..., min_length=1, description="Elasticsearch index name to search in (required, non-empty)")
    embedding_server: Optional[str] = Field(None, description="Ollama server URL for embeddings")
    llm_server: Optional[str] = Field(None, description="Ollama server URL for LLM")
    dataset_server: Optional[str] = Field(None, description="Elasticsearch server URL")
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    llm_model: Optional[str] = Field(None, description="LLM model name")
    es_username: Optional[str] = Field(None, description="Elasticsearch username")
    es_password: Optional[str] = Field(None, description="Elasticsearch password")
    top_k: int = Field(3, ge=1, le=20, description="Number of documents to retrieve (1-20)")
    kb_id: Optional[str] = Field(None, description="Optional filter by knowledge base ID")
    filename_pattern: Optional[str] = Field(None, description="Optional filter by filename pattern")
    hybrid_boost: bool = Field(True, description="Whether to use hybrid search")

class RAGService:
    def __init__(self):
        self.embedding_client = None
        self.llm_client = None
        self.es_client = None

    def connect_embedding_server(self, server_url: str):
        """Connect to embedding server (Ollama)"""
        try:
            self.embedding_client = ollama.Client(host=server_url)
            # Test connection
            self.embedding_client.list()
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to embedding server: {str(e)}")

    def connect_llm_server(self, server_url: str):
        """Connect to LLM server (Ollama)"""
        try:
            self.llm_client = ollama.Client(host=server_url)
            # Test connection
            self.llm_client.list()
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to LLM server: {str(e)}")

    def connect_dataset_server(self, server_url: str, username: str, password: str):
        """Connect to dataset server (Elasticsearch)"""
        try:
            self.es_client = Elasticsearch(
                server_url,
                basic_auth=(username, password)
            )
            # Test connection
            self.es_client.info()
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to dataset server: {str(e)}")

    def get_embedding(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get embedding from Ollama"""
        try:
            response = self.embedding_client.embeddings(model=model, prompt=text)
            return np.array(response['embedding'])
        except Exception as e:
            print(f"Failed to get embedding: {e}")
            return None

    def list_available_datasets(self) -> list:
        """List all available dataset indices with embeddings"""
        try:
            indices = self.es_client.cat.indices(format="json")
            available_datasets = []
            
            for index in indices:
                doc_count = int(index['docs.count'])
                if doc_count > 0:  # Has documents
                    # Check if it has embedding fields
                    try:
                        mapping = self.es_client.indices.get_mapping(index=index['index'])
                        properties = mapping[index['index']]['mappings']['properties']
                        
                        # Look for embedding fields
                        embedding_field = None
                        for field_name in ['q_1024_vec', 'embedding', 'q_4096_vec', 'vector']:
                            if field_name in properties:
                                field_mapping = properties[field_name]
                                if field_mapping.get('type') == 'dense_vector':
                                    embedding_field = field_name
                                    break
                        
                        if embedding_field:
                            available_datasets.append({
                                "index_name": index['index'],
                                "document_count": doc_count,
                                "embedding_field": embedding_field,
                                "dimensions": properties[embedding_field].get('dims', 'unknown')
                            })
                    except:
                        continue
            
            # Sort by document count (largest first) for consistency
            available_datasets.sort(key=lambda x: x['document_count'], reverse=True)
            return available_datasets
            
        except Exception as e:
            print(f"Failed to list datasets: {e}")
            return []

    def find_dataset_index(self) -> Optional[str]:
        """Auto-detect the dataset index with embeddings (returns the largest by document count)"""
        datasets = self.list_available_datasets()
        if datasets:
            return datasets[0]['index_name']  # Return the largest dataset
        return None

    def semantic_search(self, query_string: str, index_name: str, embedding_model: str, size: int = 3, kb_id: Optional[str] = None, filename_pattern: Optional[str] = None, hybrid_boost: bool = True):
        """Perform semantic search using embeddings - EXACT REPLICATION of standalone script"""
        # Check for 1024-dim embedding field (same as standalone script)
        try:
            mapping = self.es_client.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings']['properties']

            # Look for 1024-dim embedding field for bge-m3 (same logic as standalone)
            embedding_field = None
            for field_name in ['q_1024_vec', 'embedding', 'vector']:
                if field_name in properties:
                    field_mapping = properties[field_name]
                    if field_mapping.get('type') == 'dense_vector' and field_mapping.get('dims') == 1024:
                        embedding_field = field_name
                        break

            if not embedding_field:
                raise HTTPException(status_code=400, detail=f"No 1024-dimensional dense_vector embedding field found in index '{index_name}'. Make sure your documents are indexed with bge-m3 embeddings")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to check mapping: {str(e)}")

        # Build filters for prefiltering (same as standalone)
        filters = [{"exists": {"field": embedding_field}}]  # Always require embedding field
        if kb_id:
            filters.append({"term": {"kb_id": kb_id}})
        if filename_pattern:
            filters.append({"wildcard": {"docnm_kwd": filename_pattern}})

        # Use filtered query
        base_query = {"bool": {"filter": filters}}

        # Get query embedding
        query_embedding = self.get_embedding(query_string, embedding_model)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")

        # Convert to list for Elasticsearch
        query_embedding = query_embedding.tolist()

        # ORIGINAL SIMPLE APPROACH - EXACT REPLICATION (same as standalone)
        # Just pure cosine similarity with match_all - no filters, no boosting
        search_body = {
            "size": size,
            "_source": True,
            "query": {
                "script_score": {
                    "query": base_query,
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{embedding_field}')",
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        }

        try:
            search_result = self.es_client.search(index=index_name, body=search_body)
            return search_result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

    def generate_response(self, query: str, search_results, llm_model: str):
        """Generate response using LLM with retrieved context"""
        if not search_results or 'hits' not in search_results or not search_results['hits']['hits']:
            return "No relevant information found for your query."

        # Extract top retrieved documents
        retrieved_docs = []
        for i, hit in enumerate(search_results['hits']['hits'][:3], 1):  # Use top 3
            content = ""
            if '_source' in hit:
                # Try to find text content in common fields
                for field in ['content', 'text', 'chunk', 'message', 'content_with_weight']:
                    if field in hit['_source']:
                        content = str(hit['_source'][field])
                        break
                if not content and hit['_source']:
                    content = str(hit['_source'])

            retrieved_docs.append(f"Document {i}:\n{content}")

        # Combine retrieved documents into context
        context = "\n\n".join(retrieved_docs)

        # Create prompt with context and query
        prompt = f"""基于以下检索到的相关信息，请回答用户的问题。

检索到的信息：
{context}

用户问题：{query}

请基于上述信息提供准确、有用的回答。如果信息不足以完全回答问题，请说明。"""

        try:
            response = self.llm_client.generate(
                model=llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            )

            return response['response']

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    async def generate_streaming_response(self, query: str, search_results, llm_model: str):
        """Generate streaming response using LLM with retrieved context"""
        if not search_results or 'hits' not in search_results or not search_results['hits']['hits']:
            yield "No relevant information found for your query."
            return

        # Extract top retrieved documents
        retrieved_docs = []
        for i, hit in enumerate(search_results['hits']['hits'][:3], 1):
            content = ""
            if '_source' in hit:
                for field in ['content', 'text', 'chunk', 'message', 'content_with_weight']:
                    if field in hit['_source']:
                        content = str(hit['_source'][field])
                        break
                if not content and hit['_source']:
                    content = str(hit['_source'])

            retrieved_docs.append(f"Document {i}:\n{content}")

        context = "\n\n".join(retrieved_docs)

        prompt = f"""基于以下检索到的相关信息，请回答用户的问题。

检索到的信息：
{context}

用户问题：{query}

请基于上述信息提供准确、有用的回答。如果信息不足以完全回答问题，请说明。"""

        try:
            # Use streaming generation
            stream = self.llm_client.generate(
                model=llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 512
                },
                stream=True
            )

            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']

        except Exception as e:
            yield f"Error during generation: {str(e)}"

# Global service instance
rag_service = RAGService()

@app.get("/datasets")
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

@app.post("/rag/generate")
async def generate_rag_response(request: RAGRequest, stream: bool = False):
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
    - **stream**: Whether to stream the response (default: false)
    """

    # Explicit validation for required fields (now handled by Pydantic)
    if not request.query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    if not request.index_name:
        raise HTTPException(status_code=400, detail="index_name parameter is required")

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
        raise HTTPException(
            status_code=400, 
            detail=f"Dataset index '{request.index_name}' not found or does not contain embeddings. Available datasets: {available}"
        )

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

    if stream:
        # Return streaming response
        return StreamingResponse(
            rag_service.generate_streaming_response(request.query, search_results, llm_model),
            media_type="text/plain"
        )
    else:
        # Return regular response
        response = rag_service.generate_response(request.query, search_results, llm_model)
        return {"response": response, "sources": len(search_results['hits']['hits']), "dataset": request.index_name}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Web Service"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting RAG Web Service on port {port}")
    print("API Documentation available at: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)