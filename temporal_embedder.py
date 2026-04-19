import numpy as np
import torch
from typing import List, Dict
from logger import logger

class TemporalEmbedder:
    def __init__(self, window_size: int = 5, weights: List[float] = None):
        self.window_size = window_size
        self.weights = weights or [0.1, 0.2, 0.4, 0.2, 0.1]
        assert len(self.weights) == self.window_size, f"Weights length must match window_size ({self.window_size})"
        
    def fuse_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> np.ndarray:
        """
        Applies sliding window temporal fusion on a sequence of embeddings.
        Assumes embeddings are ordered chronologically within each video.
        """
        logger.info(f"Fusing {len(embeddings)} embeddings with window_size={self.window_size}...")
        
        num_frames = len(embeddings)
        dim = embeddings.shape[1]
        fused_embeddings = np.zeros_like(embeddings)
        
        # Group by video to prevent temporal leakage between different videos
        video_groups = {}
        for i, meta in enumerate(metadata):
            v_name = meta["video_name"]
            if v_name not in video_groups:
                video_groups[v_name] = []
            video_groups[v_name].append(i)
            
        for v_name, indices in video_groups.items():
            v_embeddings = embeddings[indices]
            v_len = len(indices)
            
            for i in range(v_len):
                # Calculate window boundaries relative to current index i
                context_indices = []
                current_weights = []
                
                offset = self.window_size // 2
                for j, w in enumerate(self.weights):
                    idx = i - offset + j
                    if 0 <= idx < v_len:
                        context_indices.append(idx)
                        current_weights.append(w)
                
                # Re-normalize weights if window is truncated (start/end of video)
                weight_sum = sum(current_weights)
                norm_weights = [w / weight_sum for w in current_weights]
                
                # Weighted average
                weighted_sum = np.zeros(dim, dtype='float32')
                for idx, w in zip(context_indices, norm_weights):
                    weighted_sum += v_embeddings[idx] * w
                
                # L2 Normalize the fused vector
                norm = np.linalg.norm(weighted_sum)
                if norm > 0:
                    weighted_sum /= norm
                
                # Map back to original index
                fused_embeddings[indices[i]] = weighted_sum
                
        return fused_embeddings

    def fuse_with_captions(self, visual_embeddings: np.ndarray, caption_embeddings: np.ndarray) -> np.ndarray:
        """
        Combines visual and caption embeddings with a 70/30 weighting.
        """
        logger.info("Fusing visual embeddings with caption embeddings (70/30)...")
        fused = 0.7 * visual_embeddings + 0.3 * caption_embeddings
        
        # Re-normalize
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        return fused / (norms + 1e-10)
