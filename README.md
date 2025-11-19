# RAGLite: Lightweight Retrieval-Augmented Generation System

![RAGLite Logo](https://img.shields.io/badge/RAGLite-v0.1.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121+-orange?style=flat-square)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.x-yellow?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-purple?style=flat-square)

RAGLite is a lightweight, production-ready Retrieval-Augmented Generation (RAG) system that combines semantic search with large language models. Built for simplicity and performance, it enables you to create intelligent question-answering systems using your own documents and local AI models.

> **Note**: RAGLite is model-agnostic and works with any Ollama-compatible embedding and text generation models. The examples in this documentation use Qwen models, but you can use Llama, Mistral, or any other supported models.

## ✨ Features

- **🔍 Semantic Search**: Vector-based similarity search using state-of-the-art embedding models
- **🤖 Local LLM Integration**: Powered by Ollama for complete data privacy and offline operation
- **📊 Elasticsearch Backend**: High-performance vector database for scalable document storage
- **🌐 RESTful API**: Modern FastAPI-based web service with automatic documentation
- **📡 Streaming Responses**: Real-time streaming generation for better user experience
- **🔧 Configurable**: Flexible server endpoints and model configurations
- **🚀 Production Ready**: Comprehensive error handling, logging, and health checks
- **📦 Minimal Dependencies**: Lightweight footprint with carefully selected libraries

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Semantic Search│───▶│   LLM Generation│
│                 │    │  (Elasticsearch)│    │     (Ollama)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Embeddings    │    │   Vector DB     │    │   Response      │
│   (Ollama)      │    │   (dense_vector)│    │   Generation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **Embedding Service**: Converts text queries into high-dimensional vectors using embedding models (e.g., `qwen3-embedding:8b`)
2. **Vector Database**: Elasticsearch with dense_vector fields for efficient similarity search
3. **LLM Service**: Text generation using large language models with retrieved context (e.g., `qwen3:32b`)
4. **Web API**: FastAPI service providing REST endpoints and streaming responses

## 📋 Prerequisites

Before deploying RAGLite, ensure you have the following services running:

### Required Services

1. **Elasticsearch 8.x**
   - **Default URL**: `http://localhost:1200`
   - **Authentication**: Configure with your Elasticsearch credentials
   - **Requirements**: Dense vector support for high-dimensional embeddings

2. **Ollama Server**
   - **Default URL**: `http://localhost:11434`
   - **Required Models**: Any compatible embedding and text generation models
     - Example embedding model: `qwen3-embedding:8b` (4096 dimensions)
     - Example LLM: `qwen3:32b` (for text generation)

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for optimal performance)
- **Storage**: Sufficient space for your document collections and vector indices

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone or download the project
cd raglite-project

# Install all dependencies
pip install -r requirements.txt
```

### 2. Start Required Services

```bash
# Terminal 1: Start Elasticsearch (if not already running)
# Configure Elasticsearch on port 1200 with authentication

# Terminal 2: Start Ollama server (if not already running)
ollama serve

# Terminal 3: Pull your preferred models (examples)
ollama pull qwen3-embedding:8b  # Example embedding model
ollama pull qwen3:32b          # Example LLM
```

### 3. Run RAGLite

Choose your preferred deployment method:

#### Option A: Web Service (Recommended)
```bash
python rag_service.py
```

#### Option B: Standalone Script
```bash
python raglite.py
```

### 4. Test the System

```bash
# Test the web service
python test_rag_api.py

# Or test manually with curl
curl -X GET "http://localhost:8000/health"
```

## 📖 Usage

### Web Service API

RAGLite provides a comprehensive REST API for integration with applications.

#### List Available Datasets
```bash
curl -X GET "http://localhost:8000/datasets"
```

#### Generate RAG Response
```bash
curl -X POST "http://localhost:8000/rag/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Your question here",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4",
    "top_k": 3,
    "kb_id": "specific-kb-id",              // Optional: filter by knowledge base
    "filename_pattern": "*.xlsx"            // Optional: filter by filename pattern
  }'
```

#### Streaming Response
```bash
curl -X POST "http://localhost:8000/rag/generate?stream=true" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Your question here"
  }'
```

### Standalone Usage

For development and testing, use the standalone script:

```python
from raglite import list_available_datasets, find_dataset_index, semantic_search, generate_with_context

# List all available datasets
datasets = list_available_datasets(client)

# Auto-detect dataset index (selects largest by document count)
dataset_index = find_dataset_index(client)

# Or specify index manually
# dataset_index = "your-specific-dataset-index"

# Perform semantic search
results = semantic_search("your query", dataset_index)

# Generate response with context
response = generate_with_context("your query", results)
```

### API Documentation

When running the web service, visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

#### Available Endpoints

- `GET /health` - Health check
- `GET /datasets` - List all available dataset indices with embeddings
- `POST /rag/generate` - Generate RAG response (supports streaming with `?stream=true`)

### Dataset Management

RAGLite supports multiple datasets in Elasticsearch:

#### Auto-Detection
If no `index_name` is specified, RAGLite automatically selects the dataset with the most documents.

#### Manual Selection
Specify `index_name` in your API request to use a specific dataset. Use `GET /datasets` to see available options.

#### Validation
The API validates that the specified dataset exists and contains embeddings before processing requests.

### Advanced Features

#### Prefiltering for Targeted Search

RAGLite supports prefiltering to search within specific subsets of your data:

**Knowledge Base ID Filtering**:
```bash
curl -X POST "http://localhost:8000/rag/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4",
    "kb_id": "4e2fdd04c48a11f0a5d79dda1442bdb4"
  }'
```

**Filename Pattern Filtering** (supports wildcards):
```bash
curl -X POST "http://localhost:8000/rag/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4",
    "filename_pattern": "*.xlsx"
  }'
```

**Combined Filtering**:
```bash
curl -X POST "http://localhost:8000/rag/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4",
    "kb_id": "specific-kb-id",
    "filename_pattern": "*.pdf"
  }'
```

**Benefits of Prefiltering**:
- ⚡ **Faster Search**: Only computes similarity on relevant documents
- 💰 **Cost Effective**: Reduces computational overhead
- 🎯 **More Accurate**: Ensures results come from correct data sources
- 📈 **Better Scaling**: Performance degrades more gracefully with dataset growth

#### Hybrid Search (Semantic + Keyword)

RAGLite supports hybrid search that combines semantic vector similarity with keyword matching for improved relevance:

```bash
# Enable hybrid search (default: enabled)
curl -X POST "http://localhost:8000/rag/generate" \
  -d '{
    "query": "买苹果",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4",
    "hybrid_boost": true
  }'

# Disable for pure semantic search
curl -X POST "http://localhost:8000/rag/generate" \
  -d '{
    "query": "买苹果",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4",
    "hybrid_boost": false
  }'
```

**How Hybrid Search Works**:
1. **Keyword Detection**: Identifies relevant keywords in your query (e.g., "苹果" → boosts "水果", "水果店", "倩倩")
2. **Combined Scoring**: Semantic similarity + keyword relevance score
3. **Better Ranking**: Documents containing both semantic matches and keywords rank higher

**Benefits**:
- 🍎 **Domain-Specific**: Better results for queries like "买苹果" → finds fruit stores
- 🔍 **Precision**: Combines semantic understanding with exact keyword matching
- 📊 **Flexibility**: Can be enabled/disabled per query

## ⚙️ Configuration

### Hybrid Configuration Approach (Recommended)

RAGLite uses a **hybrid configuration approach** that combines the best of both worlds:

- **Environment Variables**: Set defaults for production deployments and sensitive settings
- **API Parameters**: Override defaults per request for flexibility and testing

### Setup Environment Variables (Recommended)

RAGLite now automatically loads environment variables from a `.env` file:

```bash
# 1. Copy the template
cp .env.example .env

# 2. Edit with your values
nano .env

# 3. Install dependencies (includes python-dotenv)
pip install -r requirements.txt

# 4. Start the service - .env is loaded automatically
conda activate ./venv && python rag_service.py
```

**The `.env` file is automatically loaded and kept out of version control for security.**

#### Usage Examples

**1. Simple API Call (uses environment defaults):**
```bash
curl -X POST "http://localhost:8000/rag/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "买苹果",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4"
  }'
```

**2. Override Specific Settings:**
```bash
curl -X POST "http://localhost:8000/rag/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "买苹果",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4",
    "embedding_model": "qwen3-embedding:8b",
    "top_k": 5
  }'
```

**3. Full Control (specify everything):**
```bash
curl -X POST "http://localhost:8000/rag/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "买苹果",
    "index_name": "ragflow_45f257f2c38111f0a8bf07e3ef4fa8b4",
    "embedding_server": "http://localhost:11434",
    "llm_server": "http://localhost:11434",
    "dataset_server": "http://localhost:1200",
    "embedding_model": "bge-m3:latest",
    "llm_model": "qwen3:32b",
    "es_username": "elastic",
    "es_password": "infini_rag_flow",
    "top_k": 3
  }'
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Web service port |
| `ELASTICSEARCH_HOST` | `http://localhost:1200` | Elasticsearch server URL |
| `ELASTICSEARCH_PASSWORD` | `infini_rag_flow` | Elasticsearch password |
| `ELASTIC_USERNAME` | `elastic` | Elasticsearch username |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL for LLM generation |
| `EMBEDDING_HOST` | `OLLAMA_HOST` | Ollama server URL for embeddings (defaults to same as OLLAMA_HOST) |
| `EMBEDDING_MODEL` | `bge-m3:latest` | Default embedding model name |
| `LLM_MODEL` | `qwen3:32b` | Default LLM model name |

### API Parameters Reference

All API parameters are **optional** except `query` and `index_name` - they override environment variable defaults:

```json
{
  "query": "string",                    // Required: Search query
  "index_name": "string",               // Required: Elasticsearch index name
  "embedding_server": "http://localhost:11434",  // Optional: Ollama embedding server URL
  "llm_server": "http://localhost:11434",        // Optional: Ollama LLM server URL
  "dataset_server": "http://localhost:1200",     // Optional: Elasticsearch server URL
  "embedding_model": "bge-m3:latest",             // Optional: Embedding model name
  "llm_model": "qwen3:32b",                        // Optional: LLM model name
  "es_username": "elastic",                        // Optional: Elasticsearch username
  "es_password": "your_password",                  // Optional: Elasticsearch password
  "top_k": 3,                                      // Optional: Number of documents to retrieve
  "kb_id": null,                                   // Optional: filter by knowledge base ID
  "filename_pattern": null,                        // Optional: filter by filename pattern (wildcards supported)
  "hybrid_boost": true                             // Optional: Use hybrid search (semantic + keyword)
}
```

## 🧪 Testing

### Automated Testing

```bash
# Run the test suite
python test_rag_api.py
```

### Manual Testing

1. **Health Check**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **RAG Query**:
   ```bash
   curl -X POST http://localhost:8000/rag/generate \
     -H "Content-Type: application/json" \
     -d '{"query": "test query"}'
   ```

3. **Streaming Test**:
   ```bash
   curl -X POST "http://localhost:8000/rag/generate?stream=true" \
     -H "Content-Type: application/json" \
     -d '{"query": "test query"}'
   ```

## 🔧 Troubleshooting

### Common Issues

#### 1. Connection Errors

**Elasticsearch Connection Failed**
```
Error: Failed to connect to dataset server
```
**Solution**:
- Verify Elasticsearch is running on the correct port (default: 1200)
- Check authentication credentials
- Ensure Elasticsearch has dense_vector support

**Ollama Connection Failed**
```
Error: Failed to connect to embedding/LLM server
```
**Solution**:
- Start Ollama server: `ollama serve`
- Pull your preferred models: `ollama pull <your-embedding-model>` and `ollama pull <your-llm>`
- Verify server URL (default: localhost:11434)

#### 2. No Search Results

**Empty search results despite having documents**
```
No documents found. Possible issues:
1. Document embeddings don't match query semantics
2. Wrong embedding field name in index
3. Embedding dimensions mismatch
```
**Solution**:
- Verify documents are indexed with correct embeddings
- Check embedding field mapping (should be `dense_vector` with matching dimensions)
- Ensure query and document embeddings use the same model

#### 3. Model Loading Issues

**Model not found**
```
Error: model 'your-model-name' not found
```
**Solution**:
```bash
# Pull your preferred models
ollama pull <your-embedding-model>
ollama pull <your-llm-model>
```

### Performance Optimization

- **Index Selection**: Use SSD storage for Elasticsearch data
- **Memory Allocation**: Allocate sufficient RAM for Ollama models
- **Batch Processing**: Process documents in batches for large datasets
- **Caching**: Implement result caching for frequently asked questions

#### 4. Dataset Index Issues

**No suitable dataset index found**
```
❌ No index with dense_vector embeddings found
```
**Solution**:
- Ensure your documents are indexed with embeddings in a dense_vector field
- Check that your Elasticsearch index contains documents with vector embeddings
- The system automatically detects indices with dense_vector fields

## 📁 Project Structure

```
raglite/
├── raglite.py              # Core RAG pipeline implementation
├── rag_service.py          # FastAPI web service
├── test_rag_api.py         # API testing script
├── embeddding-test.py      # Basic embedding tests
├── requirements.txt        # Full dependencies
├── requirements-minimal.txt # Minimal dependencies
├── README.md              # This file
└── etc/
    └── aau_token          # Authentication tokens (if needed)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** for providing excellent local LLM infrastructure
- **Elasticsearch** for powerful vector search capabilities
- **FastAPI** for the modern web framework
- **Open-source LLM communities** for providing high-quality models like Qwen, Llama, and others

---

**RAGLite** - Bringing the power of Retrieval-Augmented Generation to your local environment with simplicity and performance.</content>
<parameter name="filePath">/Users/leehwui/workspace/embedding-test/README.md