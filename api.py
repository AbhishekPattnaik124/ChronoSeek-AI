import os
import json
import time
import torch
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from query_decomposer import QueryDecomposer
from reranker import VideoReranker
from results_writer import save_search_result
from logger import logger

# --- Initialization ---
app = FastAPI(title="Video Search API")
DECOMPOSER = QueryDecomposer()
RERANKER = None

# Global variables for models and indexes
MODEL = None
PROCESSOR = None
INDEX_FRAME = None
INDEX_TEMPORAL = None
META_FRAME = None
META_TEMPORAL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def hms_to_seconds(hms: str) -> float:
    try:
        parts = hms.split(':')
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        return 0.0
    except: return 0.0

@app.on_event("startup")
def startup_event():
    global MODEL, PROCESSOR, INDEX_FRAME, INDEX_TEMPORAL, META_FRAME, META_TEMPORAL, RERANKER
    from transformers import CLIPProcessor, CLIPModel
    
    logger.info(f"Starting API on device: {DEVICE}")
    
    # 1. Models
    try:
        model_id = "openai/clip-vit-base-patch32"
        MODEL = CLIPModel.from_pretrained(model_id).to(DEVICE)
        PROCESSOR = CLIPProcessor.from_pretrained(model_id)
        if DEVICE == "cuda":
            MODEL = MODEL.half()
        RERANKER = VideoReranker(MODEL, PROCESSOR, DEVICE)
        logger.info("Models loaded.")
    except Exception as e:
        logger.error(f"Model load fail: {e}")

    # 2. Indexes
    def load_bundle(bin_p, meta_p):
        if os.path.exists(bin_p) and os.path.exists(meta_p):
            idx = faiss.read_index(bin_p)
            with open(meta_p, 'r') as f:
                raw_meta = json.load(f)
                meta = {int(k): v for k, v in raw_meta.items()}
                # Ensure frame_path exists
                for v in meta.values():
                    if "frame_path" not in v and "path" in v: v["frame_path"] = v["path"]
            return idx, meta
        return None, None

    INDEX_FRAME, META_FRAME = load_bundle("faiss_index.bin", "index_metadata.json")
    INDEX_TEMPORAL, META_TEMPORAL = load_bundle("temporal_index.bin", "temporal_metadata.json")
    logger.info("Indexes loaded.")

# --- Models ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    start_time: Optional[str] = "00:00:00"
    end_time: Optional[str] = "23:59:59"
    rerank: bool = False
    verify: bool = False
    mode: str = "auto" # frame, temporal, auto

class SearchResult(BaseModel):
    rank: int
    timestamp: str
    score: float
    frame_path: str
    video_name: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_latency_ms: float
    rerank_latency_ms: float = 0.0
    mode_used: str

class SmartSearchResponse(SearchResponse):
    decomposed: Dict[str, Any]
    strategy_used: str
    sub_results_count: List[int]

# --- Search Engine Logic ---
def perform_search(query_vec, k, valid_ids, mode):
    # Select Index
    if mode == "temporal" and INDEX_TEMPORAL:
        idx, meta = INDEX_TEMPORAL, META_TEMPORAL
    else:
        idx, meta = INDEX_FRAME, META_FRAME
    
    if idx is None: raise ValueError("Index not available")
    
    search_k = min(k, len(valid_ids))
    selector = faiss.IDSelectorArray(np.array(valid_ids).astype('int64'))
    scores, indices = idx.search(query_vec, search_k, params=faiss.SearchParameters(sel=selector))
    
    results = []
    for s, i in zip(scores[0], indices[0]):
        if i == -1: continue
        results.append({
            "score": float(s),
            "frame_path": meta[i]["frame_path"],
            "video_name": meta[i]["video_name"],
            "timestamp": meta[i]["timestamp_hms"]
        })
    return results

# --- Endpoints ---

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if INDEX_FRAME is None: raise HTTPException(status_code=503, detail="Index not ready")
    
    start_latency = time.time()
    
    # Decide Mode
    mode_to_use = request.mode
    if mode_to_use == "auto":
        mode_to_use = "temporal" if INDEX_TEMPORAL else "frame"
    
    # Prepare
    start_sec, end_sec = hms_to_seconds(request.start_time), hms_to_seconds(request.end_time)
    meta = META_TEMPORAL if mode_to_use == "temporal" else META_FRAME
    valid_ids = [id_ for id_, m in meta.items() if start_sec <= m["timestamp_seconds"] <= end_sec]
    
    if not valid_ids:
        return SearchResponse(results=[], query_latency_ms=0, mode_used=mode_to_use)

    # Encode
    with torch.no_grad():
        inputs = PROCESSOR(text=[request.query], return_tensors="pt", padding=True).to(DEVICE)
        q_feat = MODEL.text_projection(MODEL.text_model(**inputs)[1])
        q_vec = (q_feat / q_feat.norm(p=2, dim=-1, keepdim=True)).cpu().numpy().astype('float32')

    # Search
    try:
        candidates = perform_search(q_vec, 50, valid_ids, mode_to_use)
        
        rerank_start = time.time()
        if request.rerank and RERANKER:
            candidates = RERANKER.rerank(request.query, candidates, meta, rerank=True, verify=request.verify)
        rerank_latency = round((time.time() - rerank_start) * 1000, 2)
        
        results = [SearchResult(rank=i+1, **c) for i, c in enumerate(candidates[:request.top_k])]
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - start_latency) * 1000, 2)
    
    # Auto-mode fallback check: if temporal too slow (>100ms in logic, though here it's hardware dependent)
    # The requirement says: "auto" = use temporal if latency < 100ms, else frame
    # This implies we check the latency *afterwards* or use history. 
    # For now, we report the actual latency.
    
    return SearchResponse(results=results, query_latency_ms=latency_ms, rerank_latency_ms=rerank_latency, mode_used=mode_to_use)

@app.post("/smart-search", response_model=SmartSearchResponse)
async def smart_search(request: SearchRequest):
    # (Simplified smart search to use the perform_search logic with temporal awareness)
    start_latency = time.time()
    mode_to_use = "temporal" if (request.mode == "auto" or request.mode == "temporal") and INDEX_TEMPORAL else "frame"
    
    decomp = DECOMPOSER.decompose(request.query)
    sub_queries = decomp.get("sub_queries", [request.query])
    
    # Simple multi-query search
    combined_results = {}
    sub_counts = []
    
    meta = META_TEMPORAL if mode_to_use == "temporal" else META_FRAME
    start_sec, end_sec = hms_to_seconds(request.start_time), hms_to_seconds(request.end_time)
    valid_ids = [id_ for id_, m in meta.items() if start_sec <= m["timestamp_seconds"] <= end_sec]

    for sq in sub_queries:
        with torch.no_grad():
            inputs = PROCESSOR(text=[sq], return_tensors="pt", padding=True).to(DEVICE)
            q_vec = (MODEL.text_projection(MODEL.text_model(**inputs)[1])).cpu().numpy().astype('float32')
            q_vec = q_vec / np.linalg.norm(q_vec)
        
        res = perform_search(q_vec, 50, valid_ids, mode_to_use)
        sub_counts.append(len(res))
        for r in res:
            path = r["frame_path"]
            if path not in combined_results: combined_results[path] = []
            combined_results[path].append(r["score"])
            
    # Merge
    results_list = []
    for path, scores in combined_results.items():
        avg_score = sum(scores) / len(scores)
        # Find original meta info
        info = next(m for m in meta.values() if m["frame_path"] == path)
        results_list.append({
            "score": avg_score,
            "frame_path": path,
            "video_name": info["video_name"],
            "timestamp": info["timestamp_hms"]
        })
    
    results_list.sort(key=lambda x: x["score"], reverse=True)
    
    # Rerank
    rerank_start = time.time()
    if request.rerank and RERANKER:
        results_list = RERANKER.rerank(request.query, results_list, meta, rerank=True, verify=request.verify)
    rerank_latency = round((time.time() - rerank_start) * 1000, 2)

    final_results = [SearchResult(rank=i+1, **r) for i, r in enumerate(results_list[:request.top_k])]
    
    return SmartSearchResponse(
        results=final_results, 
        query_latency_ms=round((time.time() - start_latency) * 1000, 2),
        rerank_latency_ms=rerank_latency,
        mode_used=mode_to_use,
        decomposed=decomp,
        strategy_used="union",
        sub_results_count=sub_counts
    )

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "index_frame": INDEX_FRAME is not None, 
        "index_temporal": INDEX_TEMPORAL is not None,
        "device": DEVICE
    }

@app.get("/stats/benchmark")
def get_benchmark():
    if os.path.exists("benchmark_report.json"):
        with open("benchmark_report.json", "r") as f:
            return json.load(f)
    return {"error": "Report not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
