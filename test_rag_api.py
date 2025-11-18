import requests
import json

# Test the RAG web service
def test_rag_service():
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Test RAG generation endpoint
    print("\nTesting RAG generation...")
    payload = {
        "query": "买香蕉",
        "top_k": 2
    }

    try:
        response = requests.post(
            "http://localhost:8000/rag/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"RAG generation: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"RAG generation failed: {e}")

if __name__ == "__main__":
    test_rag_service()