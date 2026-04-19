import json
import csv
import os
from datetime import datetime, timezone

JSON_FILE = "results.json"
CSV_FILE = "results.csv"

def save_search_result(query_data):
    """
    Saves search metadata and results to both JSON and CSV files.
    Calculates UTC timestamps automatically.
    """
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    
    # 1. Update JSON
    history = []
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    
    entry = {
        "query": query_data["query"],
        "timestamp_utc": timestamp_utc,
        "filters": query_data["filters"],
        "results": query_data["results"],
        "latency_ms": query_data["latency_ms"]
    }
    history.append(entry)
    
    with open(JSON_FILE, 'w') as f:
        json.dump(history, f, indent=4)

    # 2. Update CSV
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["query", "rank", "video_name", "timestamp", "score", "frame_path", "query_time_utc"])
        
        for res in query_data["results"]:
            writer.writerow([
                query_data["query"],
                res["rank"],
                res["video_name"],
                res["timestamp"],
                res["score"],
                res["frame_path"],
                timestamp_utc
            ])
