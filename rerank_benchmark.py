import requests
import time
import json

API_URL = "http://localhost:8000/search"

TEST_QUERIES = [
    "red car",
    "person with bag",
    "magenta scene",
    "empty room"
]

def run_benchmark():
    print("\n" + "="*60)
    print(f"{'Query':<20} | {'Mode':<15} | {'Latency':<10} | {'Top Score':<10}")
    print("-" * 60)

    for q in TEST_QUERIES:
        # 1. FAISS Only
        start = time.time()
        res_faiss = requests.post(API_URL, json={"query": q, "top_k": 5, "rerank": False}).json()
        latency_faiss = round((time.time() - start) * 1000, 2)
        score_faiss = res_faiss['results'][0]['score'] if res_faiss['results'] else 0
        print(f"{q:<20} | {'FAISS Only':<15} | {latency_faiss:<10} | {score_faiss:.4f}")

        # 2. FAISS + Rerank
        start = time.time()
        res_rerank = requests.post(API_URL, json={"query": q, "top_k": 5, "rerank": True}).json()
        latency_rerank = round((time.time() - start) * 1000, 2)
        score_rerank = res_rerank['results'][0]['score'] if res_rerank['results'] else 0
        print(f"{q:<20} | {'FAISS+Rerank':<15} | {latency_rerank:<10} | {score_rerank:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Benchmark failed: {e}. Is the API running?")
