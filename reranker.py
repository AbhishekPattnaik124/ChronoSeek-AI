import torch
import numpy as np
import os
from PIL import Image
from typing import List, Dict, Any
from logger import logger

class VideoReranker:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.spatial_keywords = ["near", "next to", "behind", "beside", "on top of", "under"]

    def _get_quadrants(self, image: Image.Image) -> List[Image.Image]:
        """Crops an image into 4 quadrants."""
        w, h = image.size
        mid_w, mid_h = w // 2, h // 2
        
        return [
            image.crop((0, 0, mid_w, mid_h)),     # Top-left
            image.crop((mid_w, 0, w, mid_h)),     # Top-right
            image.crop((0, mid_h, mid_w, h)),     # Bottom-left
            image.crop((mid_w, mid_h, w, h))      # Bottom-right
        ]

    def rerank(self, query: str, candidates: List[Dict], metadata: Dict, rerank: bool = True, verify: bool = False) -> List[Dict]:
        """
        Applies multi-stage re-ranking including fine-grained CLIP scoring 
        and temporal consistency checks.
        """
        if not rerank:
            return candidates

        logger.info(f"Re-ranking {len(candidates)} candidates...")
        
        # 1. Encode Query once
        with torch.no_grad():
            text_inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            q_feat = self.model.text_projection(self.model.text_model(**text_inputs)[1])
            query_features = q_feat / q_feat.norm(p=2, dim=-1, keepdim=True)

        reranked_results = []
        is_spatial_query = any(k in query.lower() for k in self.spatial_keywords)

        candidate_paths = {c["frame_path"]: i for i, c in enumerate(candidates)}

        for cand in candidates:
            frame_path = cand["frame_path"]
            if not os.path.exists(frame_path):
                continue
            
            try:
                img = Image.open(frame_path).convert("RGB")
                
                # A. Fine-Grained CLIP Scoring
                quadrants = self._get_quadrants(img)
                quad_inputs = self.processor(images=quadrants, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    vision_outputs = self.model.vision_model(**quad_inputs)
                    pooled_output = vision_outputs[1]
                    quad_features = self.model.visual_projection(pooled_output)
                    quad_features = quad_features / quad_features.norm(p=2, dim=-1, keepdim=True)
                    
                    # Similarities for all quadrants
                    similarities = (quad_features @ query_features.T).squeeze()
                    max_quad_score = float(similarities.max())
                
                full_frame_score = cand["score"]
                fine_grained_score = max_quad_score * 0.4 + full_frame_score * 0.6
                
                # B. Temporal Consistency Bonus
                # Identify frame index to check neighbors
                # Assuming frame_path has frame_XXXX.jpg format
                temporal_bonus = 0.0
                try:
                    video_folder = os.path.dirname(frame_path)
                    frame_num = int(os.path.basename(frame_path).split('_')[1].split('.')[0])
                    
                    prev_frame = os.path.join(video_folder, f"frame_{frame_num-1:04d}.jpg")
                    next_frame = os.path.join(video_folder, f"frame_{frame_num+1:04d}.jpg")
                    
                    if prev_frame in candidate_paths or next_frame in candidate_paths:
                        temporal_bonus = 0.05
                except:
                    pass

                # C. Spatial Penalty
                spatial_penalty = 0.0
                if is_spatial_query:
                    # Heuristic: If max_quad_score is much higher than others, 
                    # it's likely a centered object, penalize it for spatial queries
                    # which usually require multiple items/context.
                    if max_quad_score > float(similarities.mean()) * 1.5:
                        spatial_penalty = 0.1

                # Final Formula
                final_score = (
                    fine_grained_score * 0.6 +
                    temporal_bonus * 0.2 +
                    full_frame_score * 0.2
                ) - spatial_penalty
                
                cand["final_score"] = final_score
                reranked_results.append(cand)
                
            except Exception as e:
                logger.error(f"Error re-ranking frame {frame_path}: {e}")
                cand["final_score"] = cand["score"]
                reranked_results.append(cand)

        # Sort by final score
        reranked_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 2. Optional LLM Verification for top 5
        if verify and reranked_results:
            top_5 = reranked_results[:5]
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                for res in top_5:
                    img = Image.open(res["frame_path"])
                    prompt = f"Does this frame show exactly: '{query}'? Answer with 'yes' or 'no' followed by a confidence score between 0 and 1. Format: 'yes, 0.9' or 'no, 0.2'."
                    
                    response = model.generate_content([prompt, img])
                    answer_parts = response.text.strip().lower().split(',')
                    
                    if len(answer_parts) >= 2:
                        confidence = float(answer_parts[1].strip())
                        res["final_score"] = res["final_score"] * confidence
                        logger.info(f"LLM Verified {res['frame_path']}: {answer_parts[0]} with {confidence}")
            except Exception as e:
                logger.error(f"LLM Verification Error: {e}")

        # Final Sort after potential LLM verification
        if verify:
            reranked_results.sort(key=lambda x: x["final_score"], reverse=True)

        # Update scores and rank
        for i, res in enumerate(reranked_results):
            res["score"] = res["final_score"]
            res["rank"] = i + 1

        return reranked_results
