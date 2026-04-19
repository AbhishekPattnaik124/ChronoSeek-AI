import os
import time
import json
import torch
import psutil
import requests
import numpy as np
import platform
from tqdm import tqdm
from datetime import datetime
from embedder import build_indexes
from logger import logger

# Configuration
TEST_QUERIES = [
    "person near entrance carrying a bag",
    "two people talking near server rack",
    "red vehicle parked in zone 3",
    "anything unusual in corridor",
    "person sitting at desk",
    "empty room",
    "group of people",
    "someone running",
    "door opening",
    "person looking at camera"
]

API_URL = "http://localhost:8000/search"

def get_sys_info():
    return {
        "os": platform.system(),
        "processor": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

def measure_indexing(metadata_path):
    logger.info("Starting Indexing Benchmark...")
    start_time = time.time()
    
    # Simple peak RAM tracking
    initial_mem = psutil.Process().memory_info().rss / (1024**2)
    build_indexes(metadata_path)
    peak_mem = psutil.Process().memory_info().rss / (1024**2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    with open(metadata_path, 'r') as f:
        num_frames = len(json.load(f))
        
    return {
        "frames_processed": num_frames,
        "total_time_sec": round(total_time, 2),
        "throughput_fps": round(num_frames / total_time, 2),
        "peak_ram_mb": round(peak_mem, 2),
        "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0
    }

def measure_queries(num_iterations=2):
    logger.info("Starting Query Latency Benchmark...")
    latencies = []
    
    # Check if API is up
    try:
        requests.get("http://localhost:8000/health", timeout=2)
    except:
        logger.error("API is not running! Start with: uvicorn api:app --host 0.0.0.0 --port 8000")
        return None

    peak_ram = 0
    for _ in range(num_iterations):
        for q in tqdm(TEST_QUERIES, desc="Benchmarking Queries"):
            payload = {"query": q, "top_k": 5}
            start = time.time()
            try:
                response = requests.post(API_URL, json=payload)
                latencies.append((time.time() - start) * 1000)
                
                # Update peak RAM during query (basic check)
                curr_ram = psutil.Process().memory_info().rss / (1024**2)
                if curr_ram > peak_ram: peak_ram = curr_ram
            except Exception as e:
                logger.error(f"Query failed: {e}")

    return {
        "avg_latency_ms": round(np.mean(latencies), 2),
        "min_latency_ms": round(np.min(latencies), 2),
        "max_latency_ms": round(np.max(latencies), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "peak_ram_query_mb": round(peak_ram, 2)
    }

def run_benchmark():
    metadata_path = "frames_metadata.json"
    if not os.path.exists(metadata_path):
        logger.error("Cannot benchmark: frames_metadata.json not found. Run ingestion first.")
        return

    report = {
        "timestamp": datetime.now().isoformat(),
        "system": get_sys_info(),
        "indexing": measure_indexing(metadata_path),
        "querying": measure_queries()
    }

    # Save Report
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Print Summary Table
    print("\n" + "="*50)
    print("      CHRONOSEEK AI BENCHMARK")
    print("="*50)
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Hardware':<30} | {report['system']['gpu'] or 'CPU'}")
    print(f"{'Indexing Throughput':<30} | {report['indexing']['throughput_fps']} FPS")
    print(f"{'Indexing Peak RAM':<30} | {report['indexing']['peak_ram_mb']} MB")
    
    if report['querying']:
        print(f"{'Avg Query Latency':<30} | {report['querying']['avg_latency_ms']} ms")
        print(f"{'P95 Query Latency':<30} | {report['querying']['p95_latency_ms']} ms")
        print(f"{'Query Peak RAM':<30} | {report['querying']['peak_ram_query_mb']} MB")
    
    if torch.cuda.is_available():
        print(f"{'Peak GPU Memory':<30} | {report['indexing']['gpu_mem_mb']} MB")
    
    print("="*50)
    print(f"Full report saved to benchmark_report.json\n")

if __name__ == "__main__":
    run_benchmark()
