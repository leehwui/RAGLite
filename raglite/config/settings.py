import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables"""

    # Elasticsearch configuration
    elasticsearch_host: str = os.getenv('ELASTICSEARCH_HOST', 'http://localhost:1200')
    elasticsearch_username: str = os.getenv('ELASTIC_USERNAME', 'elastic')
    elasticsearch_password: str = os.getenv('ELASTICSEARCH_PASSWORD', 'infini_rag_flow')

    # Ollama configuration
    ollama_host: str = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    embedding_host: str = os.getenv('EMBEDDING_HOST', ollama_host)  # Default to same as OLLAMA_HOST

    # Model configuration
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')
    llm_model: str = os.getenv('LLM_MODEL', 'qwen3:32b')

    # Redis configuration
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', 6379))
    redis_db: int = int(os.getenv('REDIS_DB', 0))
    redis_password: str = os.getenv('REDIS_PASSWORD', '')

    # Server configuration
    port: int = int(os.getenv('PORT', 8000))

# Global settings instance
settings = Settings()