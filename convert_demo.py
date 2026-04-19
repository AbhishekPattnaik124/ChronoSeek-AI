from PIL import Image, ImageSequence
import cv2
import numpy as np
import os

input_path = r"C:\Users\DELL\.gemini\antigravity\brain\b9be9b59-3c34-42e4-b434-5eca2374c071\system_walkthrough_demo_1776585589522.webp"
output_path = r"d:\vgi-ai\demo_walkthrough.mp4"

def convert_webp_to_mp4(webp_path, out_path):
    print(f"Opening WebP: {webp_path}")
    img = Image.open(webp_path)
    
    # Get total frames
    frames = []
    print("Extracting frames...")
    for frame in ImageSequence.Iterator(img):
        # Convert PIL to RGB then to BGR for OpenCV
        frame_rgb = frame.convert("RGB")
        frame_np = np.array(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)
    
    if not frames:
        print("No frames found!")
        return

    height, width, layers = frames[0].shape
    print(f"Video Size: {width}x{height} | Total Frames: {len(frames)}")

    # Use XVID or mp4v for maximum compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, 15.0, (width, height)) # 15 FPS for faster walkthrough

    for f in frames:
        video.write(f)

    video.release()
    print(f"Successfully converted to: {out_path}")

if __name__ == "__main__":
    convert_webp_to_mp4(input_path, output_path)
