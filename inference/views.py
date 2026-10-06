from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status

from .dicom_utils import load_dicom_image
from .pipeline_service import InferencePipeline

import tempfile
import os
import cv2


# Create the pipeline once when Django starts.
pipeline = InferencePipeline()


@api_view(["GET"])
def health_check(request):
    """
    Health check endpoint.
    """

    return Response({
        "status": "ok",
        "message": "Inference service is running"
    })


@api_view(["POST"])
@parser_classes([MultiPartParser])
def predict(request):
    """
    Receive a DICOM file and run inference.
    """

    # Get the uploaded DICOM file.
    dicom_file = request.FILES.get("file")

    if not dicom_file:
        return Response(
            {
                "error": (
                    "No DICOM file provided. "
                    "Use the field name 'file'."
                )
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # Get clinical information.
    clinical_text = request.data.get(
        "clinical_text",
        ""
    )

    temp_dicom_path = None

    try:

        # -----------------------------------------
        # 1. Save uploaded DICOM temporarily
        # -----------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".dcm"
        ) as temp_file:

            for chunk in dicom_file.chunks():
                temp_file.write(chunk)

            temp_dicom_path = temp_file.name

        # -----------------------------------------
        # 2. Preprocess DICOM
        # -----------------------------------------

        image = load_dicom_image(
            temp_dicom_path
        )

        # -----------------------------------------
        # 3. Run inference pipeline
        # -----------------------------------------

        result = pipeline.predict(
            image,
            clinical_text
        )

        # -----------------------------------------
        # 4. Return response
        # -----------------------------------------

        return Response({
            "status": "success",
            "filename": dicom_file.name,
            "image_height": image.shape[0],
            "image_width": image.shape[1],
            "model": "MMBCD",
            "prediction": result["prediction"],
            "probabilities": result["probabilities"]
        })

    except Exception as e:

        return Response(
            {
                "error": "Inference request failed",
                "details": str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    finally:

        # Remove temporary DICOM file.
        if (
            temp_dicom_path is not None
            and os.path.exists(temp_dicom_path)
        ):
            os.remove(temp_dicom_path)