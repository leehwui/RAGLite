from fastapi import HTTPException
import ollama
import numpy as np
from elasticsearch import Elasticsearch
from typing import Optional, Dict, Any
import os
import json
import logging
from raglite.core.reranking import get_reranker
from raglite.config.settings import settings as app_settings

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
        """Perform semantic search using embeddings"""
        # Check for dense_vector embedding fields (more flexible than hardcoding dims)
        try:
            mapping = self.es_client.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings']['properties']

            # Look for dense_vector field, prioritizing common dimensions for qwen3-embedding:4b (2560)
            embedding_field = None
            embedding_dims = None
            for field_name in ['q_2560_vec', 'q_1024_vec', 'embedding', 'q_4096_vec', 'vector']:
                if field_name in properties:
                    field_mapping = properties[field_name]
                    if field_mapping.get('type') == 'dense_vector':
                        embedding_field = field_name
                        embedding_dims = field_mapping.get('dims')
                        break

            if not embedding_field:
                raise HTTPException(status_code=400, detail=f"No dense_vector embedding field found in index '{index_name}'. Make sure your documents are indexed with embeddings")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to check mapping: {str(e)}")

        # Build filters for prefiltering
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

        # Check if embedding dimensions match
        if len(query_embedding) != embedding_dims:
            raise HTTPException(status_code=400, detail=f"Embedding model '{embedding_model}' produces {len(query_embedding)}-dimensional vectors, but index '{index_name}' has {embedding_dims}-dimensional vectors. Re-index with matching embeddings.")

        # Convert to list for Elasticsearch
        query_embedding = query_embedding.tolist()

        # Pure cosine similarity search
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

    def rerank_search_results(self, query_string: str, search_results: Dict[str, Any], reranker_type: str, reranker_model: Optional[str], reranker_top_k: int):
        """Apply a reranker to the search hits before generation"""
        if not search_results or 'hits' not in search_results:
            return search_results

        hits = search_results['hits'].get('hits', [])
        if not hits:
            return search_results

        reranker = get_reranker(reranker_type or "cross_encoder", model_name=reranker_model)
        rerank_size = reranker_top_k or len(hits)

        try:
            reranked_hits = reranker.rerank(query_string, hits, top_k=rerank_size)
            search_results['hits']['hits'] = reranked_hits
            return search_results
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Reranking failed: {e}")
            return search_results

    def generate_response(self, query: str, search_results, llm_model: str, llm_num_predict: Optional[int] = None, include_thinking: bool = False):
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
                    "num_predict": llm_num_predict or app_settings.llm_num_predict
                }
            )

            # Some LLMs return the final text in 'response', others use 'thinking'.
            # Prefer 'response' when available, but fall back to 'thinking' if empty.
            try:
                    resp_text = ''
                    # Prefer 'response' first
                    if hasattr(response, 'response') and response.response:
                        resp_text = response.response
                    elif isinstance(response, dict) and response.get('response'):
                        resp_text = response.get('response')
                    # If response is empty, fall back to 'thinking' only when requested
                    elif include_thinking and hasattr(response, 'thinking') and response.thinking:
                        resp_text = response.thinking
                    elif include_thinking and isinstance(response, dict) and response.get('thinking'):
                        resp_text = response.get('thinking')
                    else:
                        resp_text = ''

                    # Log done_reason if available for diagnostics (e.g., length truncation)
                    try:
                        done_reason = getattr(response, 'done_reason', None) or response.get('done_reason') if isinstance(response, dict) else None
                        if done_reason:
                            logging.getLogger(__name__).info(f"LLM generate done_reason={done_reason}")
                    except Exception:
                        pass

                    return resp_text
            except Exception:
                    # Fallback if the object doesn't support attributes
                    if isinstance(response, dict):
                        if response.get('response'):
                            return response.get('response')
                        if include_thinking and response.get('thinking'):
                            return response.get('thinking')
                        return ''
                    return str(response)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    async def generate_streaming_response(self, query: str, search_results, llm_model: str, llm_num_predict: Optional[int] = None, include_thinking: bool = False):
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
                    "num_predict": llm_num_predict or app_settings.llm_num_predict
                },
                stream=True
            )

            # Use a buffer to aggregate small token chunks into larger messages
            token_buffer = ''
            last_timestamp = ''
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
                elif 'thinking' in chunk:
                    # Also yield thinking tokens for models that use thinking field
                    if include_thinking:
                        yield chunk['thinking']
                # Log done_reason if present
                try:
                    if isinstance(chunk, dict) and chunk.get('done'):
                        done_reason = chunk.get('done_reason') or chunk.get('reason')
                        logging.getLogger(__name__).info(f"LLM streaming done_reason={done_reason}")
                except Exception:
                    pass

        except Exception as e:
            yield f"Error during generation: {str(e)}"

    async def generate_sse_stream(self, query: str, search_results, llm_model: str, llm_num_predict: Optional[int] = None, include_thinking: bool = False):
        """Generate Server-Sent Events (SSE) streaming response"""
        # Send initial event with search results info
        sources_count = len(search_results['hits']['hits']) if search_results and 'hits' in search_results else 0
        yield f"event: search_complete\ndata: {json.dumps({'sources': sources_count})}\n\n"

        if not search_results or 'hits' not in search_results or not search_results['hits']['hits']:
            yield f"event: message\ndata: No relevant information found for your query.\n\n"
            end_payload = {'done_reason': None, 'truncated': False}
            yield f"event: end\ndata: {json.dumps(end_payload)}\n\n"
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

        # Buffer initialization moved outside try for robust scoping
        token_buffer = ''
        buffer_source = None
        last_timestamp = ''
        last_done_reason = None
        last_source = None
        token_seq = 1
        try:
            # Use streaming generation
            stream = self.llm_client.generate(
                model=llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": llm_num_predict or app_settings.llm_num_predict
                },
                stream=True
            )

            for chunk in stream:
                # Debug: print chunk keys so we can inspect streaming structure
                try:
                    logging.getLogger(__name__).debug(f"LLM chunk keys={list(chunk.keys())} len_response={len(chunk.get('response','')) if 'response' in chunk else 0} len_thinking={len(chunk.get('thinking','')) if 'thinking' in chunk else 0}")
                except Exception:
                    logging.getLogger(__name__).debug(f"LLM chunk (non-mapping)={chunk}")

                # Prefer response tokens, fallback to thinking
                token_text = ''
                token_source = 'unknown'
                if 'response' in chunk and chunk.get('response'):
                    token_text = str(chunk.get('response')).strip()
                    token_source = 'response'
                elif 'thinking' in chunk and chunk.get('thinking'):
                    token_text = str(chunk.get('thinking')).strip()
                    token_source = 'thinking'

                # If thinking tokens are not requested, skip them in streaming
                if token_source == 'thinking' and not include_thinking:
                    continue

                # Skip empty tokens to avoid spamming the frontend
                if not token_text:
                    continue

                # If the token source changed, flush existing buffer first
                if buffer_source and token_source != buffer_source and token_buffer:
                    token_data = json.dumps({
                        'token': token_buffer,
                        'timestamp': last_timestamp,
                        'source': buffer_source
                    }, ensure_ascii=False)
                    logging.getLogger(__name__).debug(f"SSE token flush: {token_data}")
                    # Attach sequence number for token events
                    token_obj = json.loads(token_data)
                    token_obj['seq'] = token_seq
                    token_seq += 1
                    token_data = json.dumps(token_obj, ensure_ascii=False)
                    yield f"event: token\ndata: {token_data}\n\n"
                    token_buffer = ''
                    last_timestamp = ''
                    buffer_source = None

                # Aggregate into buffer
                token_buffer += token_text
                buffer_source = token_source
                last_source = token_source
                last_timestamp = chunk.get('created_at', '')

                # If buffer is long enough or token ends with a sentence terminator, flush it
                # Chinese sentence terminators: 。！？, include ascii punctuation .!? as well
                terminators = ('。', '！', '？', '.', '!', '?')
                if (len(token_buffer) >= 32) or (token_text.endswith(terminators)):
                    token_data = json.dumps({
                        'token': token_buffer,
                        'timestamp': last_timestamp,
                        'source': buffer_source
                    }, ensure_ascii=False)
                    logging.getLogger(__name__).debug(f"SSE token terminator flush: {token_data}")
                    token_obj = json.loads(token_data)
                    token_obj['seq'] = token_seq
                    token_seq += 1
                    token_data = json.dumps(token_obj, ensure_ascii=False)
                    yield f"event: token\ndata: {token_data}\n\n"
                    token_buffer = ''
                    last_timestamp = ''

                # Log done_reason if present and update last_done_reason
                try:
                    if isinstance(chunk, dict) and chunk.get('done'):
                        done_reason = chunk.get('done_reason') or chunk.get('reason')
                        last_done_reason = done_reason
                        logging.getLogger(__name__).info(f"LLM streaming done_reason={done_reason}")
                except Exception:
                    pass

            # Flush any remaining buffered tokens before ending
            if token_buffer:
                token_data = json.dumps({
                    'token': token_buffer,
                    'timestamp': last_timestamp
                    ,
                    'source': buffer_source
                }, ensure_ascii=False)
                token_obj = json.loads(token_data)
                token_obj['seq'] = token_seq
                token_seq += 1
                token_data = json.dumps(token_obj, ensure_ascii=False)
                logging.getLogger(__name__).debug(f"SSE token final flush: {token_data}")
                yield f"event: token\ndata: {token_data}\n\n"

            # Build end payload including done_reason and truncated flag for the client
            end_payload = {
                'done_reason': last_done_reason,
                'truncated': last_done_reason == 'length' if last_done_reason else False,
                'final_source': last_source
            }
            yield f"event: end\ndata: {json.dumps(end_payload)}\n\n"

        except Exception as e:
            error_data = json.dumps({'error': str(e)})
            yield f"event: error\ndata: {error_data}\n\n"