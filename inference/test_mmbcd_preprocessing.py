import os
import sys
import copy
import random
import numpy as np

from PIL import Image
from torchvision import transforms


# Paths
IMAGE_PATH = r"C:\Users\ravis\Desktop\mammogram_data\mammogram_test.png"
DETECTION_PATH = r"C:\Users\ravis\Desktop\MMBCD-main\sample_data\detection_output.txt"
OUTPUT_DIR = r"C:\Users\ravis\Desktop\mammogram_data\mmbcd_crops"

IOU_THRESHOLD = 0.1
TOP_K = 5
IMG_SIZE = 224


def calculate_iou(box1, box2):
    x1, y1, w1, h1, c1 = box1
    x2, y2, w2, h2, c2 = box2

    x1 = x1 - w1 / 2
    y1 = y1 - h1 / 2

    x2 = x2 - w2 / 2
    y2 = y2 - h2 / 2

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)

    w_intersection = max(
        0,
        min(x1 + w1, x2 + w2) - x_intersection
    )

    h_intersection = max(
        0,
        min(y1 + h1, y2 + h2) - y_intersection
    )

    intersection_area = w_intersection * h_intersection

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area


def non_max_suppression(boxes, iou_threshold):
    boxes_copy = copy.deepcopy(boxes)

    while True:
        selected_indices = []
        removed_indices = []

        for i in range(len(boxes_copy)):

            if i in selected_indices or i in removed_indices:
                continue

            current_box = boxes_copy[i]
            selected_indices.append(i)

            for j in range(i + 1, len(boxes_copy)):

                if j in selected_indices or j in removed_indices:
                    continue

                if calculate_iou(
                    current_box,
                    boxes_copy[j]
                ) > iou_threshold:

                    removed_indices.append(j)

        selected_indices = sorted(selected_indices)

        if len(selected_indices) == len(boxes_copy):
            break

        boxes_copy = boxes_copy[selected_indices]

    return boxes_copy


def convert_yolo_to_pixel(box, image):
    width, height = image.size

    cx, cy, w, h = box

    x1 = int((cx - w / 2) * width)
    x2 = int((cx + w / 2) * width)

    y1 = int((cy - h / 2) * height)
    y2 = int((cy + h / 2) * height)

    return [x1, y1, x2, y2]


def main():

    print("Loading detection proposals...")

    boxes = np.loadtxt(
        DETECTION_PATH,
        dtype=np.float32
    )

    print("Original proposals:", len(boxes))

    print("Applying NMS...")

    proposals = non_max_suppression(
        boxes,
        IOU_THRESHOLD
    )

    print("Proposals after NMS:", len(proposals))

    if len(proposals) == 0:
        raise RuntimeError("No proposals remained after NMS.")

    # Match MMBCD behavior
    if len(proposals) < TOP_K:
        additional = random.choices(
            list(proposals),
            k=TOP_K - len(proposals)
        )

        proposals = np.concatenate(
            [proposals, additional]
        )

    proposals = proposals[:TOP_K]

    print("Proposals selected:", len(proposals))

    image = Image.open(IMAGE_PATH).convert("RGB")

    print("Image size:", image.size)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, box in enumerate(proposals):

        pixel_box = convert_yolo_to_pixel(
            box[:4],
            image
        )

        print(
            f"Crop {i + 1}: "
            f"normalized={box[:4]} "
            f"pixel={pixel_box}"
        )

        crop = image.crop(pixel_box)

        # Save the original crop for visual inspection
        crop.save(
            os.path.join(
                OUTPUT_DIR,
                f"crop_{i + 1}.png"
            )
        )

        # Convert exactly as MMBCD does
        tensor = transform(crop)

        print(
            f"Crop {i + 1} tensor shape:",
            tuple(tensor.shape)
        )

    print()
    print("MMBCD preprocessing test completed.")
    print("Crops saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()