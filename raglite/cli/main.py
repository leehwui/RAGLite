import argparse
import os
from dotenv import load_dotenv

from raglite.core.rag_service import RAGService
from raglite.config.settings import Settings

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="RAG CLI Tool")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--index", "-i", help="Elasticsearch index name")
    parser.add_argument("--embedding-model", default=None, help="Embedding model name")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--embedding-server", default=None, help="Ollama server URL for embeddings")
    parser.add_argument("--llm-server", default=None, help="Ollama server URL for LLM")
    parser.add_argument("--dataset-server", default=None, help="Elasticsearch server URL")
    parser.add_argument("--es-username", default=None, help="Elasticsearch username")
    parser.add_argument("--es-password", default=None, help="Elasticsearch password")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--kb-id", default=None, help="Filter by knowledge base ID")
    parser.add_argument("--filename-pattern", default=None, help="Filter by filename pattern")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets and exit")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search")

    args = parser.parse_args()

    # Load settings
    settings = Settings()

    # Initialize RAG service
    rag_service = RAGService()

    # Use command line args or environment defaults
    embedding_server = args.embedding_server or settings.embedding_host or settings.ollama_host
    llm_server = args.llm_server or settings.ollama_host
    dataset_server = args.dataset_server or settings.elasticsearch_host
    embedding_model = args.embedding_model or settings.embedding_model
    llm_model = args.llm_model or settings.llm_model
    es_username = args.es_username or settings.elastic_username
    es_password = args.es_password or settings.elasticsearch_password

    try:
        # Connect to servers
        print("🔗 Connecting to servers...")
        rag_service.connect_embedding_server(embedding_server)
        rag_service.connect_llm_server(llm_server)
        rag_service.connect_dataset_server(dataset_server, es_username, es_password)
        print("✅ Connected successfully")

        # List datasets if requested
        if args.list_datasets:
            datasets = rag_service.list_available_datasets()
            print(f"\n📊 Available datasets ({len(datasets)} found):")
            for i, dataset in enumerate(datasets, 1):
                print(f"{i}. {dataset['index_name']}: {dataset['document_count']} documents")
                print(f"   Embedding field: {dataset['embedding_field']} ({dataset['dimensions']} dimensions)")
            return

        # Determine index name
        index_name = args.index
        if not index_name:
            index_name = rag_service.find_dataset_index()
            if not index_name:
                print("❌ No suitable dataset index found. Please specify --index or ensure you have indexed documents with embeddings.")
                return

        print(f"📚 Using dataset: {index_name}")

        # Perform semantic search
        print(f"\n🔍 Performing semantic search for: '{args.query}'")
        search_results = rag_service.semantic_search(
            args.query,
            index_name,
            embedding_model,
            args.top_k,
            args.kb_id,
            args.filename_pattern,
            not args.no_hybrid
        )

        if not search_results or not search_results['hits']['hits']:
            print("❌ No relevant documents found.")
            return

        print(f"📄 Found {len(search_results['hits']['hits'])} relevant documents")

        # Generate response
        print("\n🤖 Generating response...")
        response = rag_service.generate_response(args.query, search_results, llm_model)

        if response:
            print("\n" + "="*50)
            print("RESPONSE:")
            print("="*50)
            print(response)
            print("="*50)
        else:
            print("❌ Failed to generate response")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()