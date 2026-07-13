import pytest
import numpy as np
import os
from inference import CrackDetector

def test_inference_wrapper_fallback():
    # 1. Ensure a dummy model exists or create a tiny dummy file for mocking 
    # (In real CI/CD, you can pull your real ONNX file or a lightweight version)
    model_path = "./model/deeplabv3_crack.onnx"
    
    if not os.path.exists(model_path):
        pytest.skip("ONNX model file not found, skipping integration testing.")
        
    # 2. Instantiate the wrapper class
    detector = CrackDetector(model_path)
    
    # 3. Create a fake RGB image (e.g., 512x512)
    dummy_image = np.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # 4. Process image
    prob_map = detector.predict_sliding_window(dummy_image, window_size=256, stride=128)
    
    # 5. Assertions
    assert prob_map.shape == (512, 512)
    assert isinstance(prob_map, np.ndarray)
    assert prob_map.min() >= 0.0 and prob_map.max() <= 1.0