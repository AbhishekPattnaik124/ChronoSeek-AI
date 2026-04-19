import cv2
import os
import json
import time
from datetime import timedelta
from tqdm import tqdm
import numpy as np

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

def process_video(video_path, output_base_dir, scene_threshold=30.0, fallback_seconds=2.0):
    """
    Processes a single video file, extracting frames based on scene changes 
    or a fallback interval.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    metadata = []
    last_frame_gray = None
    last_saved_time = -fallback_seconds  # Ensure the first frame is captured
    frame_count = 0
    extracted_count = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_seconds = frame_count / fps
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        is_scene_change = False
        if last_frame_gray is not None:
            # Simple scene change detection: Mean Absolute Difference
            diff = cv2.absdiff(current_gray, last_frame_gray)
            score = np.mean(diff)
            if score > scene_threshold:
                is_scene_change = True

        # Fallback: Capture if scene changed OR if 2 seconds passed since last save
        if is_scene_change or (timestamp_seconds - last_saved_time >= fallback_seconds):
            frame_filename = f"frame_{extracted_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save frame to disk
            cv2.imwrite(frame_path, frame)
            
            # Record metadata
            metadata.append({
                "frame_path": frame_path,
                "timestamp_seconds": round(timestamp_seconds, 2),
                "timestamp_hms": format_timestamp(timestamp_seconds),
                "video_name": video_name
            })
            
            last_saved_time = timestamp_seconds
            extracted_count += 1

        last_frame_gray = current_gray
        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    return metadata

def extract_frames(input_path: str):
    """
    Main entry point for frame extraction.
    Accepts a single video file or a directory containing videos.
    """
    start_time = time.time()
    output_base_dir = os.path.join(os.getcwd(), "frames")
    all_metadata = []
    
    # Identify video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
    video_files = []

    if os.path.isfile(input_path):
        if input_path.lower().endswith(video_extensions):
            video_files = [input_path]
    elif os.path.isdir(input_path):
        video_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(video_extensions)
        ]

    if not video_files:
        print(f"No valid video files found in: {input_path}")
        return

    # Process each video
    for video_file in video_files:
        video_metadata = process_video(video_file, output_base_dir)
        all_metadata.extend(video_metadata)

    # Save consolidated metadata
    metadata_file = os.path.join(os.getcwd(), "frames_metadata.json")
    
    # If file exists, merge or overwrite? Prompt implies creating it for the module.
    # We will write/overwrite for this ingestion run.
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=4)

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n--- Extraction Summary ---")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Total frames extracted: {len(all_metadata)}")
    print(f"Metadata saved to:    {metadata_file}")
    print(f"Time taken:           {total_time:.2f} seconds")
    print("--------------------------\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extract_frames(sys.argv[1])
    else:
        print("Usage: python ingestion.py <video_file_or_folder>")
