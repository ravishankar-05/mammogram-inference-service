from .config import (
    DETECTION_FILE,
    MMBCD_TOP_K,
    NMS_IOU_THRESHOLD
)

from .detection_service import DetectionService
from .crop_service import CropService
from .mmbcd_service import MMBCDService

import cv2
import tempfile
import os


class InferencePipeline:

    def __init__(self):
        """
        Initialize all inference components.
        """

        # Detection proposal service
        self.detection_service = DetectionService(
            DETECTION_FILE
        )

        # Crop generation service
        self.crop_service = CropService(
            topk=MMBCD_TOP_K,
            iou_threshold=NMS_IOU_THRESHOLD
        )

        # MMBCD inference service
        self.mmbcd_service = MMBCDService()

    def predict(self, image, clinical_text):
        """
        Run the complete detection -> crop -> MMBCD pipeline.
        """

        # -----------------------------------------
        # Step 1: Load detection proposals
        # -----------------------------------------

        proposals = (
            self.detection_service.load_detections()
        )

        # -----------------------------------------
        # Step 2: Generate image crops
        # -----------------------------------------

        crops = self.crop_service.create_crops(
            image,
            proposals
        )

        if not crops:
            raise ValueError(
                "No valid crops were generated."
            )

        # -----------------------------------------
        # Step 3: Save crops temporarily
        # -----------------------------------------

        crop_paths = []

        try:

            for crop in crops:

                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False
                )

                temp_path = temp_file.name
                temp_file.close()

                cv2.imwrite(
                    temp_path,
                    crop
                )

                crop_paths.append(temp_path)

            # -----------------------------------------
            # Step 4: Run MMBCD inference
            # -----------------------------------------

            result = self.mmbcd_service.predict(
                crop_paths,
                clinical_text
            )

            return result

        finally:

            # -----------------------------------------
            # Step 5: Delete temporary crop files
            # -----------------------------------------

            for path in crop_paths:

                if os.path.exists(path):
                    os.remove(path)

    def unload(self):
        """
        Release MMBCD resources.
        """

        self.mmbcd_service.unload_model()