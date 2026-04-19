import cv2
import numpy as np

def create_sample_video(filename="test_sample.mp4", duration_sec=10, fps=30):
    """Generates a sample video with color-changing 'scenes' for testing."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    colors = [
        (255, 0, 0),   # Blue
        (0, 255, 0),   # Green
        (0, 0, 255),   # Red
        (255, 255, 0), # Cyan
        (255, 0, 255)  # Magenta
    ]

    print(f"Generating {filename}...")
    for i in range(duration_sec):
        color = colors[i % len(colors)]
        # Add some text to vary frames slightly
        for frame_num in range(fps):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            cv2.putText(frame, f"Scene {i+1} - Time {i}s", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            out.write(frame)

    out.release()
    print(f"Video saved: {filename}")

if __name__ == "__main__":
    create_sample_video()
