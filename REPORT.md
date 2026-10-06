Mammogram Inference Service – Report
1. Overview
I built a Django REST API for mammogram inference. The API accepts a DICOM mammogram and clinical text and returns the prediction and probabilities from the MMBCD model.
2. Implementation
The pipeline is:
DICOM
  ↓
DICOM Preprocessing
  ↓
Detection Proposals
  ↓
NMS + Top-K Crops
  ↓
MMBCD Model
  ↓
Prediction

The project is divided into separate services for DICOM processing, detection, cropping, model inference, and API handling.
3. API
Implemented two endpoints:
- GET /health/ – checks whether the service is running.
- POST /predict/ – accepts a DICOM file and clinical text and returns the prediction and probabilities.
4. Docker
The application was containerized using Docker and Docker Compose.
I successfully tested the Dockerized API with a real DICOM mammogram and received an HTTP 200 response with the MMBCD prediction.
5. Challenges
The provided FocalNet-DINO model uses a custom CUDA operation. My local machine does not have NVIDIA CUDA, and the same CUDA extension also failed to compile in the available Colab PyTorch/CUDA environment due to compatibility issues.
Therefore, I used the provided detection proposals to validate the downstream MMBCD pipeline.
6. Result
The Django API, DICOM preprocessing, MMBCD inference, and Docker deployment were successfully tested. The FocalNet stage was validated using the provided detection proposals due to CUDA compatibility limitations.