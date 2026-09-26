import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import os
import sys
from pathlib import Path

# Import your validated CrackDetector
from inference import CrackDetector

app = FastAPI(
    title="Real-Time UAV Crack Detection API",
    description="FastAPI service for real-time crack detection using DeepLabV3 and TensorRT/ONNX Runtime.",
    version="1.0.0"
)

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

BASE_DIR = API_DIR.parent
MODEL_PATH = str(BASE_DIR / "model" / "deeplabv3_crack_fp16.onnx")
detector = CrackDetector(MODEL_PATH)

# Pydantic schemas for clean documentation & structured validation
class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

class DetectionResponse(BaseModel):
    bounding_boxes: List[BoundingBox]
    mask_base64: str  # PNG binary mask encoded in Base64


def extract_bounding_boxes(binary_mask: np.ndarray, min_area: int = 10) -> List[dict]:
    """
    Finds connected components (cracks) in the binary mask and extracts 
    bounding box coordinates. Filters out tiny noise using a minimum area threshold.
    """
    # Ensure mask is in uint8 format (0 or 255)
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    
    # Find contours of the detected cracks
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out negligible noise (e.g., single-pixel misclassifications)
        if (w * h) >= min_area:
            bboxes.append({
                "xmin": int(x),
                "ymin": int(y),
                "xmax": int(x + w),
                "ymax": int(y + h)
            })
            
    return bboxes


def mask_to_base64(binary_mask: np.ndarray) -> str:
    """Converts a binary mask array to a Base64 encoded PNG string."""
    visual_mask = (binary_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(visual_mask)
    
    buffer = io.BytesIO()
    mask_pil.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


def reconstruct_from_patches(image: Image.Image, patch_size: int = 512) -> np.ndarray:
    """
    Helper to slice a high-res image, run inference on patches, 
    and stitch them back into a single full-resolution prediction mask.
    """
    img_arr = np.array(image)
    h, w, _ = img_arr.shape
    
    # Create an empty canvas for the reconstructed binary mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Simple non-overlapping sliding window (can be optimized with overlap/stride if needed)
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            # Extract patch (handle boundaries gracefully)
            patch = img_arr[y:min(y + patch_size, h), x:min(x + patch_size, w)]
            
            # Pad patch if it's smaller than patch_size (e.g., at the edges)
            ph, pw, _ = patch.shape
            if ph < patch_size or pw < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_patch[:ph, :pw] = patch
                patch_input = padded_patch
            else:
                patch_input = patch
                
            # Run inference on the single patch
            patch_mask = detector.predict(patch_input)  # Expected shape: (512, 512)
            
            # Place the predicted patch back into the full mask (crop padding if applied)
            full_mask[y:min(y + patch_size, h), x:min(x + patch_size, w)] = patch_mask[:ph, :pw]
            
    return full_mask


@app.get("/health")
def health_check():
    """Lightweight endpoint for container/orchestrator health checks."""
    return {
        "status": "healthy",
        "model_loaded": detector.session is not None,
        "execution_provider": detector.session.get_providers()[0] if detector.session else None
    }


@app.post("/predict", response_model=DetectionResponse)
async def predict_single_frame(file: UploadFile = File(...)):
    """
    Predicts cracks on a single frame.
    Returns bounding box coordinates and the base64-encoded binary mask in JSON.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)
        
        # 1. Run inference
        prediction_mask = detector.predict(img_array)
        
        # 2. Extract bounding boxes
        bboxes = extract_bounding_boxes(prediction_mask)
        
        # 3. Convert mask to Base64
        mask_b64 = mask_to_base64(prediction_mask)
        
        return {
            "bounding_boxes": bboxes,
            "mask_base64": mask_b64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict-tiled", response_model=DetectionResponse)
async def predict_high_res_tiled(file: UploadFile = File(...), patch_size: int = 512):
    """
    Tiled inference for high-res images. Slices, runs inference, stitches, 
    extracts coordinates from the complete stitched mask, and returns the unified JSON payloads.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. Tile, infer, and stitch
        full_mask = reconstruct_from_patches(image, patch_size=patch_size)
        
        # 2. Extract global bounding boxes from the reconstructed mask
        bboxes = extract_bounding_boxes(full_mask)
        
        # 3. Get Base64 mask
        mask_b64 = mask_to_base64(full_mask)
        
        return {
            "bounding_boxes": bboxes,
            "mask_base64": mask_b64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tiled inference failed: {str(e)}")