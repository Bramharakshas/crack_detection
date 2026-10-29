import io
import base64
import requests
import cv2
import numpy as np
from PIL import Image

# Configuration
API_URL = "http://127.0.0.1:8000/predict-tiled"  # Switch to /predict-tiled if you want tiled video processing
INPUT_VIDEO_PATH = r"./vid/vid_1.mp4"    # Replace with your input video filename
OUTPUT_VIDEO_PATH = "output_1.mp4"

def decode_base64_mask(base64_str: str) -> np.ndarray:
    """Decodes base64 PNG string back into a NumPy binary mask array."""
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data))
    return np.array(image)

def draw_predictions(frame: np.ndarray, bboxes: list, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlays translucent red mask and draws green bounding boxes on the frame."""
    output = frame.copy()
    
    # Create translucent red overlay for cracks where mask > 0
    red_overlay = frame.copy()
    red_overlay[mask > 0] = [0, 0, 255]  # BGR Red
    output = cv2.addWeighted(output, 1 - alpha, red_overlay, alpha, 0)
    
    # Draw bounding boxes
    for box in bboxes:
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
        cv2.putText(output, "Crack", (xmin, max(ymin - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
    return output

def run_video_inference():
    if not cv2.os.path.exists(INPUT_VIDEO_PATH):
        print(f"[-] Input video not found at: {INPUT_VIDEO_PATH}")
        print("[*] Please update INPUT_VIDEO_PATH with a valid video file.")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("[-] Error opening video stream.")
        return

    # Extract video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"[+] Processing video: {INPUT_VIDEO_PATH} ({width}x{height} @ {fps}fps, {total_frames} frames)")

    frame_idx = 0
    frame_skip_interval = 3

    # Cache for lingering predictions across skipped frames
    bboxes = []
    mask = None
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            

            # FRAME SKIPPING LOGIC: 
            # Only send to API if it's the designated frame index
            # if frame_idx % frame_skip_interval != 0:
            #     # For skipped frames, just write the raw frame to keep video continuity
            #     out.write(frame)
            #     frame_idx += 1
            #     continue


            # Encode OpenCV frame to JPEG memory buffer
            success, encoded_img = cv2.imencode(".jpg", frame)
            if not success:
                print(f"[-] Failed to encode frame {frame_idx}")
                continue
                
            img_bytes = io.BytesIO(encoded_img.tobytes())

            # Send frame to FastAPI API
            files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                data = response.json()
                bboxes = data.get("bounding_boxes", [])
                mask_b64 = data.get("mask_base64", "")

                # Decode mask and overlay visuals
                mask = decode_base64_mask(mask_b64)
                #processed_frame = draw_predictions(frame, bboxes, mask)
            else:
                print(f"[-] Frame {frame_idx} API error ({response.status_code}): {response.text}")
                #processed_frame = frame  # Fallback to raw frame if API errors out


            if mask is not None and len(mask) > 0:
                processed_frame = draw_predictions(frame, bboxes, mask)
            else:
                processed_frame = frame
            # Write processed frame to output video
            out.write(processed_frame)
            frame_idx += 1

            if frame_idx % 10 == 0 or frame_idx == total_frames:
                print(f"[+] Processed {frame_idx}/{total_frames} frames...")

    finally:
        cap.release()
        out.release()
        print(f"[+] Finished! Processed video saved securely to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    run_video_inference()