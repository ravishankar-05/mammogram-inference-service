# Mammogram Inference Service

A Django REST API for serving mammogram deep-learning inference models.

## Overview

This project provides a REST API that accepts a mammogram DICOM file and clinical text, preprocesses the DICOM image, processes detection proposals, generates image crops, and performs MMBCD inference.

The service is containerized using Docker for reproducible deployment.

## Architecture

```text
DICOM Mammogram
       |
       v
DICOM Preprocessing
       |
       v
Detection Proposals
       |
       v
Non-Maximum Suppression
       |
       v
Top-K Region Crops
       |
       v
MMBCD Model
       |
       +---- Clinical Text
       |
       v
Prediction + Probabilities
```

## Project Structure

```text
inference_service/
|
├── inference/
│   ├── views.py
│   ├── urls.py
│   ├── config.py
│   ├── dicom_utils.py
│   ├── model_manager.py
│   ├── mmbcd_service.py
│   ├── focalnet_service.py
│   ├── detection_service.py
│   ├── crop_service.py
│   └── pipeline_service.py
|
├── inference_service/
│   ├── settings.py
│   └── urls.py
|
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── manage.py
└── README.md
```

## Technologies

- Python 3.10
- Django
- Django REST Framework
- PyTorch
- Torchvision
- Transformers
- OpenCV
- Pillow
- pydicom
- Docker
- Docker Compose

## API Endpoints

### Health Check

**GET /health/**

Example:

```bash
curl http://127.0.0.1:8000/health/
```

Response:

```json
{
    "status": "ok",
    "message": "Inference service is running"
}
```

### Prediction

**POST /predict/**

The endpoint accepts:

- `file`: DICOM mammogram file
- `clinical_text`: clinical information

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" 
  -F "file=@sample.dcm" 
  -F "clinical_text=bl mastalgia 2 months"
```

Response:

```json
{
    "status": "success",
    "filename": "sample.dcm",
    "image_height": 4616,
    "image_width": 3016,
    "model": "MMBCD",
    "prediction": 0,
    "probabilities": [
        0.9863116145133972,
        0.013688408769667149
    ]
}
```

## DICOM Preprocessing

The uploaded DICOM file is:

1. Read using pydicom.
2. Pixel data is extracted.
3. VOI LUT is applied.
4. Photometric interpretation is handled.
5. Pixel values are normalized.
6. The image is converted to an 8-bit image.

The resulting image is passed to the inference pipeline.

## Detection Processing

The detection proposals use the following format:

```text
cx cy width height confidence
```

The proposals are processed using:

1. Non-Maximum Suppression (NMS)
2. IoU thresholding
3. Top-K selection

The selected bounding boxes are converted from normalized coordinates into pixel coordinates.

## Image Cropping

The selected regions are cropped from the mammogram.

Each crop is:

- Resized to 224 x 224
- Converted to a tensor
- Normalized using ImageNet mean and standard deviation

The processed crops are then passed to MMBCD.

## MMBCD Inference

MMBCD uses the image regions together with the provided clinical text.

The model produces classification logits which are converted into probabilities using softmax.

The API returns:

- Prediction class
- Probability for each class

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Django:

```bash
python manage.py runserver
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## Running with Docker

Build the Docker image:

```bash
docker compose build
```

Start the service:

```bash
docker compose up
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Stop the service:

```bash
docker compose down
```

## Docker Validation

The Dockerized service was tested using a real DICOM mammogram.

The following were successfully validated:

- DICOM file upload
- DICOM reading
- DICOM preprocessing
- Detection proposal processing
- Non-Maximum Suppression
- Region crop generation
- MMBCD model loading
- MMBCD inference
- Django REST API
- Docker image build
- Docker container startup
- Health endpoint
- Prediction endpoint

## Example Docker Test

### Health Check

```bash
curl http://127.0.0.1:8000/health/
```

### Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict/" 
  -F "file=@sample.dcm" 
  -F "clinical_text=bl mastalgia 2 months"
```

## Model Notes

The provided FocalNet-DINO model requires a CUDA/NVIDIA environment for inference.

The local development machine does not have an NVIDIA CUDA environment. Therefore, the downstream MMBCD pipeline was validated using the provided detection proposal output.

The Django API, MMBCD inference pipeline, and Dockerized service were successfully tested locally.

## Challenges and Resolutions

### CUDA Environment

The supplied FocalNet-DINO implementation requires CUDA for inference. The local development machine does not have an NVIDIA CUDA environment.

The FocalNet-DINO inference pipeline was inspected to understand its expected inputs and outputs. The provided detection output was used to validate the downstream proposal processing, crop generation, and MMBCD inference stages.

### Docker Environment

Docker initially could not start because of a WSL issue on the development machine.

WSL was updated and Docker Desktop was then successfully started.

The Docker image was subsequently built successfully and the API was tested inside the container.

### Dependency Compatibility

During the Docker build, the initially specified OpenCV package version was not available for the Python environment.

The dependency was updated to:

```text
opencv-python==5.0.0.93
```

After this change, the Docker image built successfully.

## Validation Result

A real DICOM mammogram was successfully sent to the Dockerized API.

Example result:

```json
{
    "status": "success",
    "filename": "9a3af421-bb04-461c-9445-8f26f9ab34e7.dcm",
    "image_height": 4616,
    "image_width": 3016,
    "model": "MMBCD",
    "prediction": 0,
    "probabilities": [
        0.9863116145133972,
        0.013688408769667149
    ]
}
```