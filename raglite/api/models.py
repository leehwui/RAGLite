from pydantic import BaseModel, Field
from typing import Optional

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
    # Streaming / response format preferences (optional)
    stream: bool = Field(True, description="Whether to stream the response (default: true)")
    format: str = Field("sse", description='Response format: "json" or "sse" (default: sse)')