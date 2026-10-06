import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def load_dicom_image(dicom_path):
    """
    Read a mammogram DICOM file and convert its pixel data
    into an 8-bit image.
    """

    # Read the DICOM file
    dataset = pydicom.dcmread(dicom_path)

    # Extract the image pixels
    image = dataset.pixel_array

    # Apply VOI LUT if available
    image = apply_voi_lut(image, dataset)

    # Handle the two common monochrome interpretations
    if dataset.PhotometricInterpretation == "MONOCHROME2":
        image = image - np.min(image)
    else:
        image = np.max(image) - image

    # Normalize pixel values to 0-255
    image = image / np.max(image)

    # Convert to 8-bit image
    image = (image * 255).astype(np.uint8)

    return image