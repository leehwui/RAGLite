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
    reranker_type: Optional[str] = Field(
        None,
        description='Optional reranker to apply to search hits ("cross_encoder" or "none", defaults to the configured value)',
    )
    reranker_model: Optional[str] = Field(
        None,
        description='Optional HuggingFace cross-encoder model override (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")',
    )
    reranker_top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Number of documents to rerank (defaults to top_k)",
    )
    llm_num_predict: Optional[int] = Field(None, ge=1, description="Optional override for number of tokens (predictions) to request from the LLM")
    include_thinking: Optional[bool] = Field(None, description="Whether to include 'thinking' (chain-of-thought) tokens in streaming responses. False by default.")