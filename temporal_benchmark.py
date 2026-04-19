import requests
import time
import json
import numpy as np

API_URL = "http://localhost:8000/search"

TEST_QUERIES = [
    "red car",
    "person moving",
    "magenta screen",
    "empty background"
]

def run_temporal_benchmark():
    print("\n" + "="*80)
    print(f"{'Query':<20} | {'Mode':<10} | {'Latency':<10} | {'Top Score':<10} | {'Timestamp'}")
    print("-" * 80)

    for q in TEST_QUERIES:
        for mode in ["frame", "temporal"]:
            start = time.time()
            try:
                res = requests.post(API_URL, json={"query": q, "top_k": 3, "mode": mode}).json()
                latency = round((time.time() - start) * 1000, 2)
                
                if res.get('results'):
                    top = res['results'][0]
                    print(f"{q:<20} | {mode:<10} | {latency:<10} | {top['score']:.4f}     | {top['timestamp']}")
                else:
                    print(f"{q:<20} | {mode:<10} | {latency:<10} | No results  | N/A")
            except Exception as e:
                print(f"{q:<20} | {mode:<10} | Error: {e}")
        print("-" * 80)

if __name__ == "__main__":
    run_temporal_benchmark()
