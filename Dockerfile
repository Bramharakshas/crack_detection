FROM nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip python3-dev ffmpeg libsm6 libxext6
RUN pip3 install --no-cache-dir fastapi uvicorn onnxruntime-gpu numpy opencv-python-headless

COPY . /app
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]