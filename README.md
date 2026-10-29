# DeepLabV3 Crack Detection & Segmentation Pipeline

A high-performance, real-time computer vision system designed for precise crack detection and semantic segmentation on high-resolution images and video streams. Built with a **FastAPI** backend leveraging **ONNX Runtime (GPU/CUDA acceleration)** and a client-side video processing pipeline with tiled sliding-window inference and temporal prediction caching.

---

## 🚀 Key Features

* **Tiled Sliding-Window Inference**: Breaks down high-resolution images/videos into manageable patches (256x256 input with a 128 stride) to detect fine, hairline cracks without losing spatial context.
* **GPU-Accelerated Backend**: Powered by ONNX Runtime with `CUDAExecutionProvider` integration for ultra-fast model inference.
* **Smart Video Processing Client**: Includes automated frame-skipping and zero-order hold prediction lingering (caching) to achieve real-time video processing speeds without visual stutter or bounding box blinking.
* **RESTful FastAPI Service**: Clean API endpoints supporting single-frame inference (`/predict`) and high-resolution tiled image/video processing (`/predict-tiled`).

---

## 🛠️ Tech Stack

* **Deep Learning / Inference**: PyTorch (trained model), ONNX, ONNX Runtime (GPU)
* **API Framework**: FastAPI, Uvicorn
* **Image & Video Processing**: OpenCV, NumPy, Pillow (PIL)
* **Environment**: Anaconda, Python, Windows/Linux CUDA runtime support

---

## 📁 Project Structure

```text
crack_detection/
│
├── api/
│   ├── app.py              # FastAPI application & Windows DLL injection setup
│   ├── inference.py        # CrackDetector class (ONNX session & tiling logic)
│   └── test_video.py       # Client script for video processing & frame skipping
│
├── models/
│   └── best_model.onnx     # Exported DeepLabV3 ONNX model weights
│
├── vid/
│   └── input_video.mp4     # Sample input videos for testing
│
└── README.md

```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Bramharakshas/crack_detection.git
cd crack_detection

```

### 2. Create and Activate Conda Environment

```bash
conda create -n deploy_env python=3.10
conda activate deploy_env

```

### 3. Install Dependencies

Install PyTorch (if needed for preprocessing utils), FastAPI, OpenCV, and the GPU-accelerated ONNX runtime along with its matching NVIDIA libraries:

```bash
pip install fastapi uvicorn opencv-python numpy pillow requests
pip install onnxruntime-gpu nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12

```

---

## 🏃 Usage Instructions

### Step 1: Start the FastAPI Backend Server

Navigate to your API directory and launch the server using Uvicorn:

```bash
cd api
uvicorn app:app --reload

```

*Look for the startup log indicating successful initialization:*
`CrackDetector initialized using providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']`

### Step 2: Run the Video Inference Client

In a separate terminal window (with your conda environment activated), update your input video path inside `test_video.py` and run the client script:

```bash
python test_video.py

```

This will process the video utilizing frame-skipping and prediction lingering, outputting a smoothly annotated result video (`output_4.mp4`).

---

## 🔌 API Endpoints

* `POST /predict`: Accepts a single image file upload and returns detected bounding boxes and base64-encoded segmentation masks.
* `POST /predict-tiled`: Executes tiled sliding-window inference across high-resolution image inputs for precise macro/micro crack localization.
