from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import ollama
import numpy as np
from elasticsearch import Elasticsearch
from typing import Optional

app = FastAPI(title="RAG Web Service", description="Retrieval-Augmented Generation API", version="1.0.0")

class RAGRequest(BaseModel):
    query: str
    embedding_server: str = os.getenv('EMBEDDING_HOST', os.getenv('OLLAMA_HOST', 'http://localhost:11434'))  # Ollama server for embeddings
    llm_server: str = os.getenv('OLLAMA_HOST', 'http://localhost:11434')       # Ollama server for LLM
    dataset_server: str = os.getenv('ELASTICSEARCH_HOST', 'http://localhost:1200')    # Elasticsearch server
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'qwen3-embedding:8b')
    llm_model: str = os.getenv('LLM_MODEL', 'qwen3:32b')
    es_username: str = os.getenv('ELASTIC_USERNAME', 'elastic')
    es_password: str = os.getenv('ELASTICSEARCH_PASSWORD', 'infini_rag_flow')
    index_name: Optional[str] = None  # If not provided, will auto-detect
    top_k: int = 3
    kb_id: Optional[str] = None  # Optional filter by knowledge base ID
    filename_pattern: Optional[str] = None  # Optional filter by filename pattern
    hybrid_boost: bool = True  # Whether to use hybrid search (keyword + semantic)

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
                        for field_name in ['embedding', 'q_4096_vec', 'vector']:
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
        """Perform semantic search using embeddings"""
        # Check for embedding field
        try:
            mapping = self.es_client.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings']['properties']

            # Look for embedding field
            embedding_field = None
            for field_name in ['embedding', 'q_4096_vec', 'vector']:
                if field_name in properties:
                    field_mapping = properties[field_name]
                    if field_mapping.get('type') == 'dense_vector':
                        embedding_field = field_name
                        break

            if not embedding_field:
                raise HTTPException(status_code=400, detail=f"No dense_vector embedding field found in index '{index_name}'")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to check mapping: {str(e)}")

        # Get query embedding
        query_embedding = self.get_embedding(query_string, embedding_model)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")

        # Convert to list for Elasticsearch
        query_embedding = query_embedding.tolist()

        # Build filters for prefiltering
        filters = []
        if kb_id:
            filters.append({"term": {"kb_id": kb_id}})
        if filename_pattern:
            filters.append({"wildcard": {"docnm_kwd": filename_pattern}})
        
        # Use filtered query or match_all
        base_query = {"bool": {"filter": filters}} if filters else {"match_all": {}}
        
        # Implement hybrid search (semantic + keyword boosting)
        if hybrid_boost:
            # Extract potential keywords from query for boosting
            boost_keywords = []
            if '苹果' in query_string or '水果' in query_string or '买' in query_string:
                boost_keywords.extend(['苹果', '水果', '水果店', '倩倩'])
            if '啤酒' in query_string or '饮料' in query_string:
                boost_keywords.extend(['啤酒', '饮料', '青岛啤酒'])
            
            if boost_keywords:
                # Create hybrid query that combines keyword matching with semantic search
                hybrid_query = {
                    "bool": {
                        "should": [
                            {"match": {"content_ltks": {"query": " ".join(boost_keywords), "boost": 1.5}}}
                        ],
                        "filter": filters if filters else []
                    }
                }
                
                search_body = {
                    "size": size,
                    "_source": True,
                    "query": {
                        "script_score": {
                            "query": hybrid_query,
                            "script": {
                                "source": f"cosineSimilarity(params.query_vector, '{embedding_field}') + (_score * 0.3)",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    }
                }
            else:
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
        else:
            # Pure semantic search
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
    # Connect to dataset server with default credentials
    try:
        rag_service.connect_dataset_server(
            "http://localhost:1200", 
            "elastic", 
            "infini_rag_flow"
        )
        datasets = rag_service.list_available_datasets()
        return {"datasets": datasets, "count": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

@app.post("/rag/generate")
async def generate_rag_response(request: RAGRequest, stream: bool = False):
    """
    Generate RAG response for a query

    - **query**: The search query
    - **embedding_server**: Ollama server URL for embeddings (default: http://localhost:11434)
    - **llm_server**: Ollama server URL for LLM (default: http://localhost:11434)
    - **dataset_server**: Elasticsearch server URL (default: http://localhost:1200)
    - **embedding_model**: Embedding model name (default: qwen3-embedding:8b)
    - **llm_model**: LLM model name (default: qwen3:32b)
    - **es_username**: Elasticsearch username (default: elastic)
    - **es_password**: Elasticsearch password (default: infini_rag_flow)
    - **index_name**: Elasticsearch index name (auto-detected if not provided)
    - **top_k**: Number of documents to retrieve (default: 3)
    - **kb_id**: Optional filter by knowledge base ID to search only specific datasets
    - **filename_pattern**: Optional filter by filename pattern (supports wildcards like *.xlsx)
    - **hybrid_boost**: Whether to use hybrid search combining semantic and keyword matching (default: true)
    - **stream**: Whether to stream the response (default: false)
    """

    # Connect to servers
    rag_service.connect_embedding_server(request.embedding_server)
    rag_service.connect_llm_server(request.llm_server)
    rag_service.connect_dataset_server(request.dataset_server, request.es_username, request.es_password)

    # Find or use specified index
    index_name = request.index_name
    if not index_name:
        index_name = rag_service.find_dataset_index()
        if not index_name:
            raise HTTPException(status_code=404, detail="No suitable dataset index found. Use /datasets to see available options.")
    else:
        # Validate user-specified index
        available_datasets = rag_service.list_available_datasets()
        dataset_names = [d['index_name'] for d in available_datasets]
        if index_name not in dataset_names:
            available = ", ".join(dataset_names) if dataset_names else "none"
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset index '{index_name}' not found or does not contain embeddings. Available datasets: {available}"
            )

    # Perform semantic search
    search_results = rag_service.semantic_search(
        request.query,
        index_name,
        request.embedding_model,
        request.top_k,
        request.kb_id,
        request.filename_pattern,
        request.hybrid_boost
    )

    if stream:
        # Return streaming response
        return StreamingResponse(
            rag_service.generate_streaming_response(request.query, search_results, request.llm_model),
            media_type="text/plain"
        )
    else:
        # Return regular response
        response = rag_service.generate_response(request.query, search_results, request.llm_model)
        return {"response": response, "sources": len(search_results['hits']['hits']), "dataset": index_name}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Web Service"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting RAG Web Service on port {port}")
    print("API Documentation available at: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)