"""
ADAS Object Detection & Confidence Evaluator Module
Evaluates forward-facing vehicle and pedestrian detections under sensor degradation.
Measures:
- Detection Count (detected vs missed objects)
- Mean Detection Confidence (proxy metric)
- Visual Bounding Boxes with status tagging
"""

import numpy as np
import cv2

class ADASObjectDetector:
    def __init__(self):
        # Compatible with OpenCV 4.x and 5.x
        pass

    def get_ground_truth_objects(self, img_w: int, img_h: int):
        """
        Standard automotive ADAS forward scene objects (ground truth):
        - Lead vehicle (sedan directly ahead in ego lane)
        - Preceding vehicle in adjacent left lane (SUV)
        - Crossing pedestrian / cyclist near right curb
        - Distant vehicle approaching in opposing lane
        """
        objects = [
            {
                "id": "car_lead",
                "label": "Lead Vehicle (Sedan)",
                "bbox": [int(img_w * 0.44), int(img_h * 0.52), int(img_w * 0.16), int(img_h * 0.22)],
                "base_confidence": 0.94,
                "criticality": "HIGH (Collision Risk)"
            },
            {
                "id": "car_left",
                "label": "Adjacent SUV (Left Lane)",
                "bbox": [int(img_w * 0.18), int(img_h * 0.54), int(img_w * 0.18), int(img_h * 0.24)],
                "base_confidence": 0.89,
                "criticality": "MEDIUM (Lane Keeping)"
            },
            {
                "id": "pedestrian_right",
                "label": "Pedestrian (Crosswalk/Shoulder)",
                "bbox": [int(img_w * 0.78), int(img_h * 0.56), int(img_w * 0.06), int(img_h * 0.20)],
                "base_confidence": 0.86,
                "criticality": "CRITICAL (VRU Protection)"
            },
            {
                "id": "truck_distant",
                "label": "Distant Truck (Far Field)",
                "bbox": [int(img_w * 0.58), int(img_h * 0.48), int(img_w * 0.09), int(img_h * 0.12)],
                "base_confidence": 0.78,
                "criticality": "LOW (Far Path Planning)"
            }
        ]
        return objects

    def evaluate_detections(self, image: np.ndarray, ground_truth: list, detection_thresh: float = 0.45) -> dict:
        """
        Evaluates detection performance on the degraded frame.
        Physical rationale: Deep learning detectors (YOLO, Faster-RCNN) rely heavily on:
        1. High-frequency boundary gradients (edges of cars/pedestrians)
        2. Local contrast (SNR)
        3. Dynamic range without clipping (not 0 or 255)
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate global degradation factors
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        over_ratio = float(np.sum(gray >= 250) / gray.size)
        under_ratio = float(np.sum(gray <= 10) / gray.size)
        
        # Blur factor: sharp reference is ~400
        blur_factor = min(1.0, np.sqrt(max(0.01, laplacian_var / 350.0)))
        # Contrast factor
        std_intensity = float(np.std(gray))
        contrast_factor = min(1.0, std_intensity / 55.0)
        # Exposure penalty
        exp_penalty = min(0.9, 1.8 * over_ratio + 1.2 * under_ratio)
        
        detections = []
        confidences = []
        missed_count = 0
        detected_count = 0

        vis_image = image.copy()

        for obj in ground_truth:
            x, y, bw, bh = obj["bbox"]
            roi = gray[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
            
            if roi.size == 0:
                continue

            # ROI specific edge clarity
            roi_lap = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            roi_blur_factor = min(1.0, np.sqrt(max(0.01, roi_lap / 280.0)))
            
            # Local saturation in ROI
            roi_over = np.sum(roi >= 250) / roi.size
            roi_under = np.sum(roi <= 10) / roi.size
            roi_penalty = min(0.95, 2.2 * roi_over + 1.5 * roi_under)

            # Realistic degraded confidence calculation
            degraded_conf = obj["base_confidence"] * (
                0.40 * blur_factor +
                0.30 * roi_blur_factor +
                0.30 * contrast_factor
            ) * (1.0 - roi_penalty) * (1.0 - 0.5 * exp_penalty)
            
            degraded_conf = float(np.clip(degraded_conf, 0.05, 0.99))
            is_detected = degraded_conf >= detection_thresh
            
            if is_detected:
                detected_count += 1
                confidences.append(degraded_conf)
                # Green/Yellow box
                color = (0, 255, 0) if degraded_conf >= 0.70 else (0, 220, 255)
                status_text = f"{obj['label']}: {degraded_conf:.2f}"
            else:
                missed_count += 1
                confidences.append(degraded_conf)
                color = (0, 0, 255)  # Red box = MISSED / DROPPED
                status_text = f"MISSED: {obj['label']} ({degraded_conf:.2f})"

            detections.append({
                "id": obj["id"],
                "label": obj["label"],
                "bbox": [x, y, bw, bh],
                "confidence": round(degraded_conf, 3),
                "is_detected": is_detected,
                "criticality": obj["criticality"]
            })

            # Draw on visualization image
            cv2.rectangle(vis_image, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                vis_image,
                status_text,
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

        mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "detected_count": detected_count,
            "missed_count": missed_count,
            "total_objects": len(ground_truth),
            "mean_confidence": round(mean_confidence, 3),
            "detections": detections,
            "vis_image": vis_image
        }
