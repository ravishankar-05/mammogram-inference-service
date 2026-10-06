import os


# Project root
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# MMBCD root directory
MMBCD_ROOT = os.getenv(
    "MMBCD_ROOT",
    os.path.join(
        BASE_DIR,
        "models",
        "MMBCD-main"
    )
)


# MMBCD trained checkpoint
MMBCD_CHECKPOINT = os.getenv(
    "MMBCD_CHECKPOINT",
    os.path.join(
        MMBCD_ROOT,
        "models",
        "mmbcd",
        "mmbcd_best.pt"
    )
)


# Detection proposals
DETECTION_FILE = os.getenv(
    "DETECTION_FILE",
    os.path.join(
        MMBCD_ROOT,
        "sample_data",
        "detection_output.txt"
    )
)


# Model configuration
MMBCD_IMAGE_SIZE = 224
MMBCD_TOP_K = 5
NMS_IOU_THRESHOLD = 0.1