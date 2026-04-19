import json
import requests
import time
import numpy as np
import math
from typing import List, Dict

API_URL = "http://localhost:8000/search"

def calculate_ndcg(relevant_indices: List[int], k: int) -> float:
    if not relevant_indices:
        return 0.0
    dcg = sum(1.0 / math.log2(idx + 2) for idx in relevant_indices if idx < k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_indices), k)))
    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(gt_path: str):
    if not os.path.exists(gt_path):
        print(f"Ground truth file {gt_path} not found. Run create_ground_truth.py first.")
        return

    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    report_entries = []
    
    print("\n" + "="*85)
    print(f"{'Query':<30} | {'P@1':<5} | {'P@5':<5} | {'P@10':<5} | {'MRR':<5} | {'NDCG@10':<7}")
    print("-" * 85)

    all_metrics = []

    for entry in ground_truth:
        query = entry["query"]
        gt_timestamps = set(entry["relevant_timestamps"])
        
        start_time = time.time()
        try:
            # We use 'temporal' mode as default for highest hypothetical accuracy
            response = requests.post(API_URL, json={"query": query, "top_k": 10, "mode": "temporal"}).json()
            latency = (time.time() - start_time) * 1000
            
            results = response.get("results", [])
            
            # Find relevant ranks
            rel_indices = []
            for idx, res in enumerate(results):
                if res["timestamp"] in gt_timestamps:
                    rel_indices.append(idx)
            
            # Metrics
            p1 = 1.0 if 0 in rel_indices else 0.0
            p5 = len([i for i in rel_indices if i < 5]) / 5.0
            p10 = len(rel_indices) / 10.0
            mrr = 1.0 / (rel_indices[0] + 1) if rel_indices else 0.0
            ndcg = calculate_ndcg(rel_indices, 10)
            
            metrics = {
                "query": query,
                "p@1": p1, "p@5": p5, "p@10": p10,
                "mrr": mrr, "ndcg@10": ndcg,
                "latency_ms": latency
            }
            all_metrics.append(metrics)
            
            print(f"{query[:30]:<30} | {p1:<5.2f} | {p5:<5.2f} | {p10:<5.2f} | {mrr:<5.2f} | {ndcg:<7.2f}")
            
        except Exception as e:
            print(f"Error evaluating '{query}': {e}")

    # Averages
    if all_metrics:
        avg_p1 = np.mean([m["p@1"] for m in all_metrics])
        avg_p5 = np.mean([m["p@5"] for m in all_metrics])
        avg_p10 = np.mean([m["p@10"] for m in all_metrics])
        avg_mrr = np.mean([m["mrr"] for m in all_metrics])
        avg_ndcg = np.mean([m["ndcg@10"] for m in all_metrics])
        avg_latency = np.mean([m["latency_ms"] for m in all_metrics])

        print("-" * 85)
        print(f"{'AVERAGE':<30} | {avg_p1:<5.2f} | {avg_p5:<5.2f} | {avg_p10:<5.2f} | {avg_mrr:<5.2f} | {avg_ndcg:<7.2f}")
        print("="*85)

        # Save Report
        report = {
            "summary": {
                "avg_precision_1": avg_p1,
                "avg_precision_5": avg_p5,
                "avg_precision_10": avg_p10,
                "avg_mrr": avg_mrr,
                "avg_ndcg_10": avg_ndcg,
                "avg_latency_ms": avg_latency
            },
            "queries": all_metrics
        }
        
        with open("evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=4)
        print("\nEvaluation report saved to evaluation_report.json")

if __name__ == "__main__":
    import os
    run_evaluation("ground_truth.json")
