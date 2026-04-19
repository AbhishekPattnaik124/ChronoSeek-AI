import os
import json
import time
import torch
import faiss
import psutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from logger import logger
from temporal_embedder import TemporalEmbedder

def build_indexes(metadata_path: str, batch_size: int = 32):
    """
    Builds both raw frame and temporal-aware FAISS indexes.
    """
    logger.info(f"Indexing started for metadata: {metadata_path}")
    start_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load CLIP
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    if device == "cuda":
        model = model.half()
    
    # 2. Load Metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f) # This is a list
    
    # 3. Generate Raw Embeddings
    raw_embeddings = []
    
    logger.info(f"Generating raw embeddings for {len(metadata)} frames...")
    for i in tqdm(range(0, len(metadata), batch_size), desc="Embedding Batches"):
        batch = metadata[i : i + batch_size]
        batch_images = []
        for meta in batch:
            img_path = meta["frame_path"] if "frame_path" in meta else meta["path"]
            batch_images.append(Image.open(img_path).convert("RGB"))
        
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        if device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
            
        with torch.no_grad():
            vision_outputs = model.vision_model(**inputs)
            pooled_output = vision_outputs[1]
            visual_features = model.visual_projection(pooled_output)
            visual_features = visual_features / visual_features.norm(p=2, dim=-1, keepdim=True)
            raw_embeddings.append(visual_features.cpu().numpy().astype('float32'))
            
    raw_embeddings = np.vstack(raw_embeddings)
    
    # 4. Generate Temporal Embeddings
    temp_embedder = TemporalEmbedder()
    temporal_embeddings = temp_embedder.fuse_embeddings(raw_embeddings, metadata)
    
    # 5. Build and Save FAISS Indexes
    def save_index(embeddings, bin_name, meta_name):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, bin_name)
        
        # Save metadata mapping (FAISS ID -> Metadata)
        meta_map = {i: metadata[i] for i in range(len(metadata))}
        with open(meta_name, 'w') as f:
            json.dump(meta_map, f, indent=4)
        logger.info(f"Saved {bin_name} with {index.ntotal} vectors.")

    save_index(raw_embeddings, "faiss_index.bin", "index_metadata.json")
    save_index(temporal_embeddings, "temporal_index.bin", "temporal_metadata.json")
    
    # 6. Report Resources
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 2)
    
    total_time = time.time() - start_all
    logger.info(f"Indexing Complete | Total Time: {total_time:.2f}s | FPS: {len(metadata)/total_time:.2f} | Peak RAM: {ram_usage:.2f}MB")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_metadata.json"
    build_indexes(path)
