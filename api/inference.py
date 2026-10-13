import numpy as np
import onnxruntime as ort

class CrackDetector:
    def __init__(self, model_path: str = "./model/deeplabv3_crack_fp16.onnx"):
        available_providers = ort.get_available_providers()
        self.providers = []
        
        if 'TensorRTExecutionProvider' in available_providers:
            # We pass the ONNX file structure, but tell the provider to 
            # explicitly look for our optimized engine configurations in the cache
            self.providers.append(('TensorRTExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './cache',  # Path where engines live
                'trt_engine_cache_prefix': 'deeplabv3_crack_fp16' 
            }))
            
        if 'CUDAExecutionProvider' in available_providers:
            self.providers.append(('CUDAExecutionProvider', {'device_id': 0}))
            
        self.providers.append('CPUExecutionProvider')
        
        # ONNX Runtime reads the graph structure, then instantly maps 
        # execution to the pre-compiled engine weights if found in the cache path
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        
        print(f"CrackDetector initialized using providers: {self.session.get_providers()}")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Performs inference on a single image or patch (e.g., 512x512).
        Returns a binary mask array (0 or 1).
        """
        # 1. Normalize and convert to CHW format
        norm_img = (image.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = norm_img.transpose(2, 0, 1)  # HWC to CHW
        
        # 2. Add batch dimension -> (1, C, H, W)
        batch_input = np.expand_dims(norm_img, axis=0).astype(np.float32)
        
        # 3. Run ONNX session
        output = self.session.run(None, {self.input_name: batch_input})[0]
        
        # 4. Apply Sigmoid and threshold at 0.5 to get binary mask
        probs = 1 / (1 + np.exp(-output))
        binary_mask = (probs[0, 0] > 0.5).astype(np.uint8)
        
        return binary_mask

    def predict_sliding_window(self, image: np.ndarray, window_size: int = 256, stride: int = 128, batch_size: int = 32) -> np.ndarray:
        """
        Performs batched sliding-window inference over a large image.
        Expects a standard RGB NumPy image array (H, W, 3).
        """
        h, w, _ = image.shape
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        patches = []
        coords = []
        
        # 1. Normalize and extract patches
        norm_img = (image.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        norm_img = norm_img.transpose(2, 0, 1)  # HWC to CHW
        
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                patch = norm_img[:, y:y+window_size, x:x+window_size]
                patches.append(patch)
                coords.append((y, x))
                
        if not patches:
            return prob_map

        # 2. Batch Inference Loop
        patches = np.array(patches, dtype=np.float32)
        num_patches = len(patches)
        outputs = []
        
        for i in range(0, num_patches, batch_size):
            batch_input = patches[i:i+batch_size]
            batch_output = self.session.run(None, {self.input_name: batch_input})[0]
            # Sigmoid activation
            batch_output = 1 / (1 + np.exp(-batch_output)) 
            outputs.append(batch_output)
            
        outputs = np.concatenate(outputs, axis=0)
        
        # 3. Reconstruct full scale image
        for idx, (y, x) in enumerate(coords):
            patch_out = outputs[idx, 0]
            prob_map[y:y+window_size, x:x+window_size] += patch_out
            count_map[y:y+window_size, x:x+window_size] += 1.0
            
        return np.divide(prob_map, count_map, out=np.zeros_like(prob_map), where=count_map != 0)