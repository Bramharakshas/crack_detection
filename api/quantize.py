import onnx
from onnxruntime.transformers.optimizer import optimize_model

input_model = "./model/deeplabv3_crack.onnx"
output_model = "./model/deeplabv3_crack_fp16.onnx"

print("Quantizing ONNX graph layout to FP16...")
# Optimize the graph configuration and cast weights to float16
optimized_model = optimize_model(input_model, model_type='generic')
optimized_model.convert_float_to_float16()
optimized_model.save_model_to_file(output_model)

print(f"Success! Permanently saved optimized model to: {output_model}")