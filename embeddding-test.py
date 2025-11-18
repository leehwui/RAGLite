import ollama
import numpy as np

# Configuration for Ollama connection
OLLAMA_HOST = "http://localhost:11434"  # Change this if your Ollama is running on a different host/port
client = ollama.Client(host=OLLAMA_HOST)

# Configuration for output
PRINT_EMBEDDINGS = False  # Set to True to see the actual embedding vectors (they are long numerical arrays)

# Function to get embedding from Ollama
def get_embedding(text, model):
    response = client.embeddings(model=model, prompt=text)
    return np.array(response['embedding'])

# Test cases for different languages
test_cases = [
    {
        'language': 'English',
        'query': 'buy apple',
        'document': """
Our company specializes in supplying fresh fruits to retailers and wholesalers.
We offer a wide range of fruits.
For bulk purchases and special orders, please contact our sales team.
We ensure high quality and timely delivery for all our products.
"""
    },
    {
        'language': 'Chinese',
        'query': '买香蕉',
        'document': """
 3 青岛青饮国际贸易有限公司 崂矿大桶水 傅鹏 13589227170 季度结算 4 倩倩水果店 水果 倩 倩 15192592597 现结 要水果单价和账单（免费送货上门） 5 恩丽源餐饮公司（食堂） 茶歇点心 大厨-王恩峰 13698699488 对公月结 经济、实惠、健康、安全
"""
    }
]

# List of embedding models to test (assuming they are available in Ollama)
models = ['qwen3-embedding:8b', 'bge-m3:latest', 'shaw/dmeta-embedding-zh:latest']

for test_case in test_cases:
    language = test_case['language']
    query = test_case['query']
    document = test_case['document']
    
    print(f"\n=== Testing in {language} ===")
    print(f"Connecting to Ollama at: {OLLAMA_HOST}")
    print(f"Query: '{query}'")
    print("Document: {}".format(document.strip()))
    print(f"Print embeddings: {PRINT_EMBEDDINGS}")
    print("\n" + "="*50)
    
    for model in models:
        try:
            # Get embeddings
            doc_embedding = get_embedding(document, model)
            query_embedding = get_embedding(query, model)

            # Calculate cosine similarity
            similarity = np.dot(doc_embedding, query_embedding) / (
                np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding)
            )

            print(f"Model: {model}")
            print(f"Similarity: {similarity:.4f}")
            if PRINT_EMBEDDINGS:
                print(f"Query embedding (first 10 values): {query_embedding[:10]}")
                print(f"Document embedding (first 10 values): {doc_embedding[:10]}")
                print(f"Embedding dimensions: {len(query_embedding)}")
            print("-"*30)
        except Exception as e:
            print(f"Error with model {model}: {e}")
            print("-"*30)

print("\nNote: Higher similarity scores indicate better semantic matching.")
print("This test measures how well the models bridge the semantic gap between")
print("the queries and documents in both English and Chinese.")