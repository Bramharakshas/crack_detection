from fastapi import FastAPI, UploadFile, File, Depends
import cv2
import numpy as np
from inference import CrackDetector

app = FastAPI(title="Crack Detection API")

# Initialize the detector globally on startup
detector = CrackDetector("./model/deeplabv3_crack_fp16.onnx")

@app.post("/predict")
async def predict_crack(file: UploadFile = File(...)):
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run prediction via the wrapper class
    prob_map = detector.predict_sliding_window(image_rgb)
    mask = (prob_map > 0.5).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 50:
            boxes.append({"x": x, "y": y, "w": w, "h": h})
            
    return {"detected_cracks": len(boxes), "bounding_boxes": boxes}