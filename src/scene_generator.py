"""
ADAS Test Scene Generator Module
Generates a realistic automotive front-camera road scene (1280x720) with:
- Highway/Urban road with perspective lane lines
- Lead vehicle (sedan), adjacent vehicle (SUV), crossing pedestrian, distant truck
- Realistic lighting, asphalt texture, and roadside structures
"""

import numpy as np
import cv2

def generate_adas_road_scene(width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Creates a photorealistic synthetic ADAS forward camera frame.
    """
    # 1. Background Sky and Horizon
    horizon_y = int(height * 0.44)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (sky blue to light horizon)
    for y in range(horizon_y):
        ratio = y / horizon_y
        b = int(240 - ratio * 40)
        g = int(210 - ratio * 30)
        r = int(170 - ratio * 20)
        image[y, :] = (b, g, r)

    # Distant mountains / cityscape on horizon
    for x in range(0, width, 40):
        m_height = int(np.sin(x * 0.015) * 25 + 30)
        cv2.ellipse(image, (x, horizon_y), (45, m_height), 0, 180, 360, (140, 150, 130), -1)

    # 2. Road Asphalt and Grass Shoulders
    # Grass terrain
    for y in range(horizon_y, height):
        ratio = (y - horizon_y) / (height - horizon_y)
        # Grass color gradient
        image[y, :] = (int(35 + 20 * ratio), int(95 + 35 * ratio), int(45 + 15 * ratio))

    # Road polygon (perspective trapezoid)
    road_top_left = (int(width * 0.45), horizon_y)
    road_top_right = (int(width * 0.55), horizon_y)
    road_bottom_right = (int(width * 0.95), height)
    road_bottom_left = (int(width * 0.05), height)

    road_pts = np.array([road_top_left, road_top_right, road_bottom_right, road_bottom_left], np.int32)
    cv2.fillPoly(image, [road_pts], (75, 75, 80))  # Dark asphalt

    # Add subtle asphalt texture noise
    noise = np.random.normal(0, 4, (height - horizon_y, width, 3)).astype(np.int16)
    road_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(road_mask, [road_pts], 255)
    road_region = image[horizon_y:, :].astype(np.int16) + noise
    image[horizon_y:, :] = np.where(
        road_mask[horizon_y:, :, np.newaxis] == 255,
        np.clip(road_region, 0, 255).astype(np.uint8),
        image[horizon_y:, :]
    )

    # 3. Lane Markings
    # Left solid yellow line
    cv2.line(image, (int(width * 0.46), horizon_y), (int(width * 0.08), height), (20, 200, 240), 4)
    # Right solid white line
    cv2.line(image, (int(width * 0.54), horizon_y), (int(width * 0.92), height), (240, 240, 240), 4)

    # Center dashed white lines
    num_dashes = 9
    for i in range(num_dashes):
        t1 = (i / num_dashes) ** 1.8
        t2 = ((i + 0.55) / num_dashes) ** 1.8
        y1 = int(horizon_y + t1 * (height - horizon_y))
        y2 = int(horizon_y + t2 * (height - horizon_y))
        x1 = int(width * 0.50 + (t1 * (width * 0.50 - width * 0.50)))
        x2 = int(width * 0.50 + (t2 * (width * 0.50 - width * 0.50)))
        # Left lane divider
        xl1 = int((1 - t1) * (width * 0.48) + t1 * (width * 0.36))
        xl2 = int((1 - t2) * (width * 0.48) + t2 * (width * 0.36))
        xr1 = int((1 - t1) * (width * 0.52) + t1 * (width * 0.64))
        xr2 = int((1 - t2) * (width * 0.52) + t2 * (width * 0.64))
        thickness = max(2, int(t2 * 6))
        cv2.line(image, (xl1, y1), (xl2, y2), (230, 230, 230), thickness)
        cv2.line(image, (xr1, y1), (xr2, y2), (230, 230, 230), thickness)

    # 4. Lead Vehicle (Sedan in center ego lane)
    # Bounding box roughly: [563, 374, 204, 158] (44% w, 52% h)
    car_x = int(width * 0.44)
    car_y = int(height * 0.52)
    car_w = int(width * 0.16)
    car_h = int(height * 0.22)

    # Car body (silver-blue metallic)
    cv2.rectangle(image, (car_x, car_y + int(car_h * 0.35)), (car_x + car_w, car_y + car_h), (90, 70, 60), -1)
    # Cabin / Roof
    cabin_pts = np.array([
        (car_x + int(car_w * 0.15), car_y + int(car_h * 0.35)),
        (car_x + int(car_w * 0.25), car_y),
        (car_x + int(car_w * 0.75), car_y),
        (car_x + int(car_w * 0.85), car_y + int(car_h * 0.35))
    ], np.int32)
    cv2.fillPoly(image, [cabin_pts], (80, 60, 50))
    # Rear windshield (tinted dark glass)
    windshield_pts = np.array([
        (car_x + int(car_w * 0.20), car_y + int(car_h * 0.33)),
        (car_x + int(car_w * 0.28), car_y + int(car_h * 0.06)),
        (car_x + int(car_w * 0.72), car_y + int(car_h * 0.06)),
        (car_x + int(car_w * 0.80), car_y + int(car_h * 0.33))
    ], np.int32)
    cv2.fillPoly(image, [windshield_pts], (40, 35, 30))

    # Taillights (bright red LEDs)
    tail_h = int(car_h * 0.12)
    tail_w = int(car_w * 0.22)
    cv2.rectangle(image, (car_x + 6, car_y + int(car_h * 0.45)), (car_x + 6 + tail_w, car_y + int(car_h * 0.45) + tail_h), (25, 30, 240), -1)
    cv2.rectangle(image, (car_x + car_w - 6 - tail_w, car_y + int(car_h * 0.45)), (car_x + car_w - 6, car_y + int(car_h * 0.45) + tail_h), (25, 30, 240), -1)

    # License plate
    lp_w = int(car_w * 0.32)
    lp_h = int(car_h * 0.14)
    lp_x = car_x + (car_w - lp_w) // 2
    lp_y = car_y + int(car_h * 0.68)
    cv2.rectangle(image, (lp_x, lp_y), (lp_x + lp_w, lp_y + lp_h), (220, 230, 240), -1)
    cv2.putText(image, "ADAS-01", (lp_x + 4, lp_y + lp_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

    # Tires & bumper shadow
    cv2.rectangle(image, (car_x + int(car_w * 0.08), car_y + car_h - 8), (car_x + int(car_w * 0.28), car_y + car_h + 10), (20, 20, 20), -1)
    cv2.rectangle(image, (car_x + int(car_w * 0.72), car_y + car_h - 8), (car_x + int(car_w * 0.92), car_y + car_h + 10), (20, 20, 20), -1)
    cv2.ellipse(image, (car_x + car_w // 2, car_y + car_h + 8), (int(car_w * 0.52), 9), 0, 0, 360, (25, 25, 25), -1)

    # 5. Adjacent Left Lane Vehicle (Dark SUV)
    suv_x = int(width * 0.18)
    suv_y = int(height * 0.54)
    suv_w = int(width * 0.18)
    suv_h = int(height * 0.24)
    cv2.rectangle(image, (suv_x, suv_y + int(suv_h * 0.32)), (suv_x + suv_w, suv_y + suv_h), (45, 45, 45), -1)
    cv2.rectangle(image, (suv_x + int(suv_w * 0.12), suv_y), (suv_x + int(suv_w * 0.88), suv_y + int(suv_h * 0.32)), (35, 35, 35), -1)
    cv2.rectangle(image, (suv_x + 8, suv_y + int(suv_h * 0.42)), (suv_x + 8 + int(suv_w * 0.24), suv_y + int(suv_h * 0.54)), (20, 20, 220), -1)
    cv2.rectangle(image, (suv_x + suv_w - 8 - int(suv_w * 0.24), suv_y + int(suv_h * 0.42)), (suv_x + suv_w - 8, suv_y + int(suv_h * 0.54)), (20, 20, 220), -1)
    cv2.ellipse(image, (suv_x + suv_w // 2, suv_y + suv_h + 6), (int(suv_w * 0.5), 8), 0, 0, 360, (20, 20, 20), -1)

    # 6. Crossing Pedestrian (Right shoulder / Crosswalk)
    ped_x = int(width * 0.78)
    ped_y = int(height * 0.56)
    ped_w = int(width * 0.06)
    ped_h = int(height * 0.20)
    # Head
    head_r = int(ped_w * 0.32)
    cv2.circle(image, (ped_x + ped_w // 2, ped_y + head_r), head_r, (170, 190, 210), -1)
    # Torso (Jacket)
    cv2.rectangle(image, (ped_x + int(ped_w * 0.18), ped_y + head_r * 2), (ped_x + int(ped_w * 0.82), ped_y + int(ped_h * 0.65)), (180, 50, 40), -1)
    # Legs
    cv2.line(image, (ped_x + int(ped_w * 0.35), ped_y + int(ped_h * 0.65)), (ped_x + int(ped_w * 0.25), ped_y + ped_h), (40, 40, 50), 4)
    cv2.line(image, (ped_x + int(ped_w * 0.65), ped_y + int(ped_h * 0.65)), (ped_x + int(ped_w * 0.75), ped_y + ped_h), (40, 40, 50), 4)

    # 7. Distant Truck (Far Field ahead)
    trk_x = int(width * 0.58)
    trk_y = int(height * 0.48)
    trk_w = int(width * 0.09)
    trk_h = int(height * 0.12)
    cv2.rectangle(image, (trk_x, trk_y), (trk_x + trk_w, trk_y + trk_h), (110, 120, 130), -1)
    cv2.rectangle(image, (trk_x + 4, trk_y + int(trk_h * 0.7)), (trk_x + 14, trk_y + int(trk_h * 0.88)), (20, 20, 200), -1)
    cv2.rectangle(image, (trk_x + trk_w - 14, trk_y + int(trk_h * 0.7)), (trk_x + trk_w - 4, trk_y + int(trk_h * 0.88)), (20, 20, 200), -1)

    return image
