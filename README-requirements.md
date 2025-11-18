# RAG Pipeline Requirements

This project implements a Retrieval-Augmented Generation (RAG) system with semantic search using Ollama embeddings and Elasticsearch.

## Installation

### Option 1: Full Installation (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Minimal Installation
```bash
pip install -r requirements-minimal.txt
```

## Core Libraries Used

### RAG Pipeline
- **elasticsearch** (8.19.2): Client for Elasticsearch vector database
- **ollama** (0.6.1): API client for Ollama LLM and embedding models
- **numpy** (2.3.5): Vector operations and numerical computations

### Web Service
- **fastapi** (0.121.2): Modern web framework for REST API
- **pydantic** (2.12.4): Data validation and serialization
- **uvicorn** (0.38.0): ASGI server for running FastAPI

### Testing/Utilities
- **requests** (2.32.5): HTTP client for API testing

## Prerequisites

Before running this project, ensure you have:

1. **Elasticsearch** running on `http://localhost:1200` (or configure custom URL)
2. **Ollama** running on `http://localhost:11434` with models:
   - `qwen3-embedding:8b` (for embeddings)
   - `qwen3:32b` (for text generation)

## Usage

### Start the Web Service
```bash
python rag_service.py
```

### Test the API
```bash
python test_rag_api.py
```

### Access API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## Project Files

- `raglite.py`: Original RAG pipeline implementation
- `rag_service.py`: FastAPI web service wrapper
- `test_rag_api.py`: API testing script
- `embeddding-test.py`: Basic embedding testing
- `requirements.txt`: Full requirements with tested versions
- `requirements-minimal.txt`: Minimal requirements only