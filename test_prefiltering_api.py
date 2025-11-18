#!/usr/bin/env python3
"""
Test script for RAG API with prefiltering capabilities
"""

import requests
import json

def test_rag_api():
    """Test the RAG API with different prefiltering options"""

    base_url = "http://localhost:8000"
    test_query = "买香蕉"

    print("🧪 Testing RAG API with Prefiltering")
    print("=" * 50)

    # Test 1: Basic query without filters
    print("\n1️⃣ Testing basic query (no filters):")
    payload = {
        "query": test_query,
        "top_k": 2
    }

    try:
        response = requests.post(f"{base_url}/rag/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success: {result['response'][:100]}...")
            print(f"  Sources: {result['sources']}, Dataset: {result['dataset']}")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Query with kb_id filter
    print("\n2️⃣ Testing with kb_id filter:")
    payload_kb = {
        "query": test_query,
        "kb_id": "4e2fdd04c48a11f0a5d79dda1442bdb4",  # From our test data
        "top_k": 2
    }

    try:
        response = requests.post(f"{base_url}/rag/generate", json=payload_kb)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success with kb_id filter: {result['response'][:100]}...")
            print(f"  Sources: {result['sources']}, Dataset: {result['dataset']}")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 3: Query with filename filter
    print("\n3️⃣ Testing with filename filter:")
    payload_file = {
        "query": test_query,
        "filename_pattern": "*.xlsx",
        "top_k": 2
    }

    try:
        response = requests.post(f"{base_url}/rag/generate", json=payload_file)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success with filename filter: {result['response'][:100]}...")
            print(f"  Sources: {result['sources']}, Dataset: {result['dataset']}")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 4: Query with both filters
    print("\n4️⃣ Testing with both kb_id and filename filters:")
    payload_both = {
        "query": test_query,
        "kb_id": "4e2fdd04c48a11f0a5d79dda1442bdb4",
        "filename_pattern": "*.xlsx",
        "top_k": 2
    }

    try:
        response = requests.post(f"{base_url}/rag/generate", json=payload_both)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success with both filters: {result['response'][:100]}...")
            print(f"  Sources: {result['sources']}, Dataset: {result['dataset']}")
        else:
            print(f"✓ No results (expected if no matches): {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n🎯 Prefiltering API Tests Complete!")

if __name__ == "__main__":
    test_rag_api()