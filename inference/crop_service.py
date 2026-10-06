import cv2
import numpy as np


class CropService:

    def __init__(self, topk=5, iou_threshold=0.1):
        self.topk = topk
        self.iou_threshold = iou_threshold

    def create_crops(self, image, proposals):
        """
        Create mammogram crops from normalized
        YOLO-format proposals.

        Proposal format:
        [cx, cy, width, height, confidence]
        """

        # Apply NMS first
        proposals = self.non_max_suppression(
            proposals,
            self.iou_threshold
        )

        # Keep only top K proposals
        proposals = proposals[:self.topk]

        crops = []

        height, width = image.shape[:2]

        for proposal in proposals:

            cx, cy, box_w, box_h, confidence = proposal

            # Convert normalized coordinates
            # to pixel coordinates.
            x1 = int((cx - box_w / 2) * width)
            y1 = int((cy - box_h / 2) * height)

            x2 = int((cx + box_w / 2) * width)
            y2 = int((cy + box_h / 2) * height)

            # Keep coordinates inside image boundaries.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Skip invalid boxes.
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            # MMBCD expects 224x224 crops.
            crop = cv2.resize(
                crop,
                (224, 224)
            )

            crops.append(crop)

        # MMBCD expects exactly top-k crops.
        # If fewer are available, duplicate the
        # last valid crop.
        if crops:

            while len(crops) < self.topk:
                crops.append(crops[-1].copy())

        return crops

    def non_max_suppression(
        self,
        boxes,
        iou_threshold
    ):
        """
        Remove highly overlapping bounding boxes.
        """

        boxes = np.asarray(
            boxes,
            dtype=np.float32
        )

        if len(boxes) == 0:
            return []

        selected = []

        # Work on a copy.
        remaining = boxes.copy()

        while len(remaining) > 0:

            current = remaining[0]

            selected.append(current)

            if len(remaining) == 1:
                break

            current_iou = []

            for box in remaining[1:]:

                iou = self.calculate_iou(
                    current[:4],
                    box[:4]
                )

                current_iou.append(iou)

            current_iou = np.array(
                current_iou
            )

            # Keep boxes whose overlap is
            # below the threshold.
            remaining = remaining[1:][
                current_iou < iou_threshold
            ]

        return selected

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU for normalized
        [cx, cy, width, height] boxes.
        """

        cx1, cy1, w1, h1 = box1
        cx2, cy2, w2, h2 = box2

        x1_min = cx1 - w1 / 2
        y1_min = cy1 - h1 / 2
        x1_max = cx1 + w1 / 2
        y1_max = cy1 + h1 / 2

        x2_min = cx2 - w2 / 2
        y2_min = cy2 - h2 / 2
        x2_max = cx2 + w2 / 2
        y2_max = cy2 + h2 / 2

        intersection_x1 = max(
            x1_min,
            x2_min
        )

        intersection_y1 = max(
            y1_min,
            y2_min
        )

        intersection_x2 = min(
            x1_max,
            x2_max
        )

        intersection_y2 = min(
            y1_max,
            y2_max
        )

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_width *
            intersection_height
        )

        area1 = w1 * h1
        area2 = w2 * h2

        union_area = (
            area1 +
            area2 -
            intersection_area
        )

        if union_area == 0:
            return 0.0

        return intersection_area / union_area