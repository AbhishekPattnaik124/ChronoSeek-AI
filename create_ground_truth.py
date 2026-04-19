import cv2
import json
import os
from typing import List, Dict

TEST_QUERIES = [
    "person near entrance carrying a bag",
    "two people talking",
    "empty corridor",
    "person sitting at desk",
    "group of people standing",
    "door opening or closing",
    "person looking at camera",
    "someone running or moving fast",
    "person using phone",
    "bright light or flash"
]

def build_ground_truth(frames_dir: str):
    """
    Interactive script to build ground truth. 
    Select a subfolder in frames/ to review.
    """
    if not os.path.exists(frames_dir):
        print(f"Directory {frames_dir} not found.")
        return

    # Find the first available video folder
    video_folders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
    if not video_folders:
        print("No video frame folders found.")
        return
    
    target_folder = os.path.join(frames_dir, video_folders[0])
    frame_files = sorted([f for f in os.listdir(target_folder) if f.endswith(".jpg")])
    
    # We load metadata to get timestamps
    metadata_path = "index_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = {v["frame_path"]: v for v in json.load(f).values()}

    ground_truth = []
    
    print(f"Building Ground Truth for: {video_folders[0]}")
    print("Controls: 'r' = Relevant, 'n' = Not Relevant, 'q' = Quit/Skip Query")

    for query in TEST_QUERIES:
        relevant_timestamps = []
        print(f"\nQUERY: {query}")
        
        for f_name in frame_files:
            full_path = os.path.join(target_folder, f_name)
            
            # Map path back to metadata to get timestamp
            # Convert slash consistency for lookup
            lookup_path = full_path.replace("/", "\\") # Match Windows style in JSON
            if lookup_path not in metadata:
                # Try relative or matching
                match = next((m for p, m in metadata.items() if p.endswith(f_name)), None)
                if not match: continue
                ts = match["timestamp_hms"]
            else:
                ts = metadata[lookup_path]["timestamp_hms"]

            img = cv2.imread(full_path)
            cv2.putText(img, f"Query: {query}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Time: {ts} | [r]el / [n]ext / [q]uit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Ground Truth Builder", img)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('r'):
                relevant_timestamps.append(ts)
                print(f"  Added relevant: {ts}")
            elif key == ord('q'):
                break
            elif key == ord('n'):
                continue
        
        ground_truth.append({
            "query": query,
            "relevant_timestamps": relevant_timestamps,
            "total_frames_reviewed": len(frame_files)
        })
        
    # Save output
    output_path = "ground_truth.json"
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=4)
    
    cv2.destroyAllWindows()
    print(f"\nGround Truth saved to {output_path}")

if __name__ == "__main__":
    build_ground_truth("frames")
