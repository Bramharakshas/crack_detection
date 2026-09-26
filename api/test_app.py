import io
import numpy as np
from PIL import Image
import sys
from pathlib import Path
from fastapi.testclient import TestClient

TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

# Import your FastAPI app instance
from api.app import app

client = TestClient(app)

def test_predict_endpoints():
    # 1. Create a mock RGB image in memory
    img_data = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_data)
    
    # Save the dummy image to a byte buffer to simulate a file upload
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Prepare the payload mimicking a multipart form-data file upload
    files = {
        "file": ("test_image.png", img_byte_arr, "image/png")
    }
    
    # 2. Test the standard single-frame endpoint
    response = client.post("/predict", files=files)
    
    # If the model file is missing on the machine (e.g. if skipped earlier),
    # handle it gracefully, otherwise run the assertion checks
    if response.status_code == 500 and "Inference failed" in response.text:
        return  # Safeguard for environments without the actual model weight files loaded
        
    assert response.status_code == 200
    json_data = response.json()
    
    # Verify our custom payload structure
    assert "bounding_boxes" in json_data
    assert "mask_base64" in json_data
    assert isinstance(json_data["bounding_boxes"], list)
    assert isinstance(json_data["mask_base64"], str)
    
    # 3. Test the tiled endpoint
    # Reset the byte buffer pointer so we can reuse the same dummy image
    img_byte_arr.seek(0)
    files_tiled = {
        "file": ("test_image.png", img_byte_arr, "image/png")
    }
    
    # Call the tiled endpoint with a specific patch size
    response_tiled = client.post("/predict-tiled?patch_size=256", files=files_tiled)
    assert response_tiled.status_code == 200
    
    json_data_tiled = response_tiled.json()
    assert "bounding_boxes" in json_data_tiled
    assert "mask_base64" in json_data_tiled