import io
import base64
import requests
import cv2
import numpy as np
from PIL import Image

# Path to an actual image on your computer
IMAGE_PATH = r"./img/crack_5.png"  # <-- Replace with your image file
URL = "http://127.0.0.1:8000/predict-tiled"

print(f"[+] Sending {IMAGE_PATH} to FastAPI endpoint...")

with open(IMAGE_PATH, "rb") as f:
    files = {"file": (IMAGE_PATH, f, "image/jpeg")}
    response = requests.post(URL, files=files)

if response.status_code == 200:
    data = response.json()
    bboxes = data.get("bounding_boxes", [])
    mask_b64 = data.get("mask_base64", "")
    
    print(f"[+] Success! Found {len(bboxes)} crack detection(s).")

    # 1. Decode base64 mask back into a NumPy array
    mask_bytes = base64.b64decode(mask_b64)
    mask_pil = Image.open(io.BytesIO(mask_bytes))
    mask_np = np.array(mask_pil)

    # 2. Draw mask and bounding boxes on top of original image
    orig_img = cv2.imread(IMAGE_PATH)
    overlay = orig_img.copy()
    overlay[mask_np > 0] = [0, 0, 255]  # Red highlight for crack
    visualized = cv2.addWeighted(orig_img, 0.6, overlay, 0.4, 0)

    for box in bboxes:
        cv2.rectangle(visualized, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 255, 0), 2)

    # 3. Save output image
    output_filename = "result_output_5.jpg"
    cv2.imwrite(output_filename, visualized)
    print(f"[+] Output visual saved to {output_filename}")
else:
    print(f"[-] Request failed ({response.status_code}): {response.text}")