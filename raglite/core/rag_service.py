from fastapi import HTTPException
import ollama
import numpy as np
from elasticsearch import Elasticsearch
from typing import Optional, Dict, Any
import os

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
                elif 'thinking' in chunk:
                    # Also yield thinking tokens for models that use thinking field
                    yield chunk['thinking']

        except Exception as e:
            yield f"Error during generation: {str(e)}"

    async def generate_sse_stream(self, query: str, search_results, llm_model: str):
        """Generate Server-Sent Events (SSE) streaming response"""
        # Send initial event with search results info
        sources_count = len(search_results['hits']['hits']) if search_results and 'hits' in search_results else 0
        yield f"event: search_complete\ndata: {json.dumps({'sources': sources_count})}\n\n"

        if not search_results or 'hits' not in search_results or not search_results['hits']['hits']:
            yield f"event: message\ndata: No relevant information found for your query.\n\n"
            yield f"event: end\ndata: [DONE]\n\n"
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
                    # Send each token as an SSE message event
                    token_data = json.dumps({
                        'token': chunk['response'],
                        'timestamp': chunk.get('created_at', '')
                    })
                    yield f"event: token\ndata: {token_data}\n\n"
                elif 'thinking' in chunk:
                    # Also send thinking tokens
                    token_data = json.dumps({
                        'token': chunk['thinking'],
                        'type': 'thinking',
                        'timestamp': chunk.get('created_at', '')
                    })
                    yield f"event: token\ndata: {token_data}\n\n"

            # Send completion event
            yield f"event: end\ndata: [DONE]\n\n"

        except Exception as e:
            error_data = json.dumps({'error': str(e)})
            yield f"event: error\ndata: {error_data}\n\n"