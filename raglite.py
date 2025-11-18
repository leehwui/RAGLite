from elasticsearch import Elasticsearch
import os
import ollama
import numpy as np

# Elasticsearch configuration
HOST_URL = os.getenv('ELASTICSEARCH_HOST', 'http://localhost:1200')
ES_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD', 'infini_rag_flow')

# Ollama configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
EMBEDDING_HOST = os.getenv('EMBEDDING_HOST', OLLAMA_HOST)  # Default to same as OLLAMA_HOST

# Model configuration
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3:32b')

# Authentication
username = os.getenv('ELASTIC_USERNAME', 'elastic')
password = os.getenv('ELASTIC_PASSWORD', ES_PASSWORD)  # Use env var or fallback to hardcoded

client = Elasticsearch(
    HOST_URL,
    basic_auth=(username, password)
)

# Ollama client for embeddings
embedding_client = ollama.Client(host=EMBEDDING_HOST)

# Ollama client for LLM generation
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Test the connection
try:
    info = client.info()
    print("Elasticsearch connection successful!")
    print(f"Cluster name: {info['cluster_name']}")
    print(f"Version: {info['version']['number']}")
except Exception as e:
    print(f"Connection failed: {e}")

# Test get operation
try:
    result = client.get(index="test-index", id="1")
    print("Get operation result:")
    print(result)
except Exception as e:
    print(f"Get operation failed: {e}")

# Function to list all available datasets with embeddings
def list_available_datasets(client):
    """List all available dataset indices with embeddings"""
    try:
        indices = client.cat.indices(format="json")
        available_datasets = []
        
        for index in indices:
            doc_count = int(index['docs.count'])
            if doc_count > 0:  # Has documents
                # Check if it has embedding fields
                try:
                    mapping = client.indices.get_mapping(index=index['index'])
                    properties = mapping[index['index']]['mappings']['properties']
                    
                    # Look for embedding fields
                    embedding_field = None
                    for field_name in ['embedding', 'q_4096_vec', 'q_1024_vec', 'vector']:
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
        
        print(f"\n📊 Available datasets with embeddings ({len(available_datasets)} found):")
        for i, dataset in enumerate(available_datasets, 1):
            print(f"{i}. {dataset['index_name']}: {dataset['document_count']} documents")
            print(f"   Embedding field: {dataset['embedding_field']} ({dataset['dimensions']} dimensions)")
        
        return available_datasets
        
    except Exception as e:
        print(f"Failed to list datasets: {e}")
        return []

# Function to auto-detect dataset index with embeddings
def find_dataset_index(client):
    """Auto-detect the dataset index with embeddings (prefers indices with 1024-dim bge-m3 embeddings)"""
    datasets = list_available_datasets(client)
    if datasets:
        # Check which indices have documents with 1024-dim embeddings
        for dataset in datasets:
            try:
                # Check if any document has q_1024_vec populated
                result = client.search(index=dataset['index_name'], body={
                    'query': {'exists': {'field': 'q_1024_vec'}},
                    'size': 1,
                    '_source': ['q_1024_vec']
                })
                if result['hits']['hits']:
                    doc = result['hits']['hits'][0]
                    vec = doc['_source'].get('q_1024_vec')
                    if vec and len(vec) == 1024:
                        print(f"✓ Found index with 1024-dim embeddings: '{dataset['index_name']}' ({dataset['document_count']} documents)")
                        return dataset['index_name']
            except:
                continue
        
        # Fallback to largest dataset
        selected = max(datasets, key=lambda x: x['document_count'])
        print(f"✓ Auto-selected dataset: '{selected['index_name']}' ({selected['document_count']} documents, {selected['dimensions']} dims)")
        return selected['index_name']
    return None

# # Check if _source is enabled for the index - commented out
# try:
#     mapping = client.indices.get_mapping(index=indices[0]['index'])
#     source_enabled = mapping[indices[0]['index']]['mappings'].get('_source', {}).get('enabled', True)
#     print(f"_source enabled for index '{indices[0]['index']}': {source_enabled}")
# except Exception as e:
#     print(f"Failed to check mapping: {e}")

# # To list all document IDs in an index: - commented out
# search_result = client.search(
#     index = indices[0]['index'],
#     body = {
#         "query": {"match_all": {}}, 
#         "size": 100  # Adjust size as needed
#     }
# )

# # Print document IDs - commented out
# for hit in search_result['hits']['hits']:
#     print(f"Document ID: {hit['_id']}")
#     # print(f"Full source: {hit['_source']}")  # Show all fields
#     content = hit.get('_source', {}).get('content_with_weight', 'No content field')
#     print(f"Content: {content}")
#     print("---")

# # Search for documents in an index (if indices exist) - commented out for brevity
# if indices:
#     sample_index = indices[0]['index']  # Use the first index as example
#     try:
#         search_result = client.search(index=sample_index, body={"query": {"match_all": {}}, "size": 5})
#         print(f"\nSample documents from index '{sample_index}':")
#         for hit in search_result['hits']['hits']:
#             print(f"  ID: {hit['_id']}, Score: {hit['_score']}")
#             print(f"  Source: {hit['_source']}")
#     except Exception as e:
#         print(f"Failed to search index '{sample_index}': {e}")


# Function to get embedding from Ollama
def get_embedding(text, model=None):
    if model is None:
        model = EMBEDDING_MODEL
    try:
        response = embedding_client.embeddings(model=model, prompt=text)
        return np.array(response['embedding'])
    except Exception as e:
        print(f"Failed to get embedding: {e}")
        return None

# Function to perform semantic search using embeddings
def semantic_search(query_string, index_name, embedding_model=None, size=3, kb_id=None, filename_pattern=None, hybrid_boost=True):
    """
    Perform semantic search using qwen3-embedding:8b (4096 dimensions)
    
    Args:
        query_string: The search query
        index_name: Elasticsearch index name
        embedding_model: Ollama embedding model name
        size: Number of results to return (reduced back to 3 for efficiency)
        kb_id: Optional filter by knowledge base ID
        filename_pattern: Optional filter by filename pattern (supports wildcards)
        hybrid_boost: Whether to use hybrid search (keyword + semantic) for better results
    """
    if embedding_model is None:
        embedding_model = EMBEDDING_MODEL
    
    # Check for 4096-dim embedding field
    try:
        mapping = client.indices.get_mapping(index=index_name)
        properties = mapping[index_name]['mappings']['properties']
        
        # Look for 1024-dim embedding field for bge-m3
        embedding_field = None
        for field_name in ['q_1024_vec', 'embedding', 'vector']:
            if field_name in properties:
                field_mapping = properties[field_name]
                if field_mapping.get('type') == 'dense_vector' and field_mapping.get('dims') == 1024:
                    embedding_field = field_name
                    print(f"✓ Found 1024-dim dense_vector field: '{field_name}'")
                    break
        
        if not embedding_field:
            print(f"❌ No 1024-dimensional dense_vector embedding field found in index '{index_name}'")
            print("Make sure your documents are indexed with bge-m3 embeddings")
            return None
            
    except Exception as e:
        print(f"Failed to check mapping: {e}")
        return None
    
    # Build filters for prefiltering
    filters = [{"exists": {"field": embedding_field}}]  # Always require embedding field
    if kb_id:
        filters.append({"term": {"kb_id": kb_id}})
    if filename_pattern:
        filters.append({"wildcard": {"docnm_kwd": filename_pattern}})
    
    # Use filtered query
    base_query = {"bool": {"filter": filters}}
    
    # Get query embedding
    query_embedding = get_embedding(query_string, embedding_model)
    if query_embedding is None:
        return None
    
    # Convert to list for Elasticsearch
    query_embedding = query_embedding.tolist()
    
    print(f"Query embedding dimensions: {len(query_embedding)}")
    # Check query embedding values
    query_non_zero = sum(1 for x in query_embedding if x != 0)
    print(f"Query non-zero values: {query_non_zero}/{len(query_embedding)}")
    print(f"Query first 5 values: {query_embedding[:5]}")
    
    # ORIGINAL SIMPLE APPROACH - EXACT REPLICATION
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
        search_result = client.search(index=index_name, body=search_body)
        
        print(f"\nSemantic search results for query: '{query_string}'")
        print(f"Using embedding model: {embedding_model}")
        print(f"Total hits: {search_result['hits']['total']['value']}")
        print("-" * 50)
        
        if search_result['hits']['total']['value'] == 0:
            print("❌ No documents found. Possible issues:")
            print("1. The document doesn't have the expected embedding field")
            print("2. The embedding field is not mapped as dense_vector")
            print("3. The embedding dimensions don't match")
            print("4. The document content doesn't match the query semantically")
            
            # Let's check the actual document
            try:
                doc_check = client.search(index=index_name, body={"query": {"match_all": {}}, "size": 1, "_source": True})
                if doc_check['hits']['hits']:
                    doc = doc_check['hits']['hits'][0]
                    print(f"\nActual document in index '{index_name}':")
                    print(f"ID: {doc['_id']}")
                    print(f"Fields: {list(doc['_source'].keys())}")
                    if embedding_field in doc['_source']:
                        emb = doc['_source'][embedding_field]
                        if isinstance(emb, list):
                            print(f"✓ {embedding_field} exists: {len(emb)} dimensions")
                            # Check if vector has valid values
                            non_zero = sum(1 for x in emb if x != 0)
                            print(f"  Non-zero values: {non_zero}/{len(emb)}")
                            print(f"  First 5 values: {emb[:5]}")
                            if all(x == 0 for x in emb):
                                print("  ⚠️  WARNING: All values are zero!")
                        else:
                            print(f"✗ {embedding_field} is not a list: {type(emb)}")
                    else:
                        print(f"✗ {embedding_field} field not found in document")
                else:
                    print("No documents found in index at all!")
            except Exception as e:
                print(f"Failed to check document: {e}")
            return None
        
        for i, hit in enumerate(search_result['hits']['hits'], 1):
            print(f"{i}. Document ID: {hit['_id']}")
            print(f"   Similarity score: {hit['_score']:.4f}")
            if '_source' in hit:
                # Try to find text content in common fields
                content = ""
                for field in ['content', 'text', 'chunk', 'message', 'content_with_weight']:
                    if field in hit['_source']:
                        content = str(hit['_source'][field])
                        break
                if not content and hit['_source']:
                    content = str(hit['_source'])
                
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"   Content preview: {preview}")
            print()
            
        return search_result
    except Exception as e:
        print(f"Semantic search failed: {e}")
        return None

# Function to generate response using LLM with retrieved context
def generate_with_context(query, search_results, model=None):
    """
    Generate response using qwen3:32b with retrieved context
    """
    if model is None:
        model = LLM_MODEL
    if not search_results or 'hits' not in search_results or not search_results['hits']['hits']:
        print("No search results to use for generation")
        return None
    
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
        print(f"\n🤖 Sending to LLM ({model})...")
        print(f"Context length: {len(context)} characters")
        print(f"Query: {query}")
        print("-" * 50)
        
        response = ollama_client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Lower temperature for more factual responses
                "top_p": 0.9,
                "num_predict": 512  # Reasonable response length
            }
        )
        
        print("LLM Response:")
        print("=" * 50)
        print(response['response'])
        print("=" * 50)
        
        return response['response']
        
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return None

# Example usage - RAG pipeline
print("\n🔍 Looking for dataset index with embeddings...")
# dataset_index = find_dataset_index(client)
dataset_index = "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4"  # Hardcode for testing

if dataset_index:
    sample_query = "买苹果"
    print(f"\n🤖 Using embedding model: {EMBEDDING_MODEL}")
    
    # Step 1: Semantic search
    search_results = semantic_search(sample_query, dataset_index)
    
    # Step 2: Generate response with retrieved context
    if search_results:
        print("\n🔄 Generating response with retrieved context...")
        generate_with_context(sample_query, search_results)
    else:
        print("❌ No search results to generate response from")
else:
    print("❌ No suitable dataset index found. Please ensure you have indexed documents with embeddings.")