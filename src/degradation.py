"""
Camera Degradation Module for ADAS Automotive Vision
Simulates realistic physical sensor and environmental degradation modes:
1. Defocus Blur / Lens Aberration
2. Vehicle Motion Blur (Pitch & Shake)
3. Sun Glare / High-beam Overexposure (CMOS blooming)
4. Night / Tunnel Underexposure (Low dynamic range)
5. Heavy Rain & High-gain Sensor Noise (Shot noise)
6. Lens Soiling / Mud Splatter (Front windshield / lens dirt)
"""

import numpy as np
import cv2

def apply_defocus_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Simulates optical defocus or condensation/mist on windshield.
    severity: 1 (mild) to 5 (severe)
    """
    ksize = 2 * severity * 2 + 1  # 5, 9, 13, 17, 21
    sigma = severity * 1.5
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return blurred

def apply_motion_blur(image: np.ndarray, severity: int = 1, angle: float = 15.0) -> np.ndarray:
    """
    Simulates vehicle pitch/vibration motion blur under low shutter speed.
    """
    size = int(severity * 7 + 3)
    kernel = np.zeros((size, size), dtype=np.float32)
    # Generate linear motion blur kernel
    alpha = np.deg2rad(angle)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    center = size // 2
    for i in range(size):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0
    return cv2.filter2D(image, -1, kernel)

def apply_glare_overexposure(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Simulates direct sunrise/sunset glare or oncoming high-beam headlights.
    Causes pixel saturation (overexposure blooming) in the field of view.
    """
    h, w = image.shape[:2]
    # Place glare center near horizon / upper center of ADAS camera
    center_x = int(w * 0.5)
    center_y = int(h * 0.35)
    
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    radius = (w * 0.25) * (0.6 + severity * 0.4)
    
    # Gaussian glare mask
    glare_mask = np.exp(-dist_sq / (2 * (radius ** 2)))
    glare_mask = np.clip(glare_mask * (0.6 + severity * 0.35), 0.0, 1.0)
    
    # Overexposure intensity boost
    glare_3ch = np.repeat(glare_mask[:, :, np.newaxis], 3, axis=2)
    result = image.astype(np.float32) * (1.0 + 0.2 * severity) + glare_3ch * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_night_underexposure(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Simulates unlit rural road / tunnel entry (extreme underexposure).
    Reduces brightness and applies gamma crush.
    """
    factor = max(0.08, 0.55 - (severity * 0.10))  # 0.45 down to 0.08
    gamma = 1.4 + severity * 0.3
    
    normalized = image.astype(np.float32) / 255.0
    dimmed = normalized * factor
    crushed = np.power(dimmed, gamma) * 255.0
    
    # Add subtle sensor read noise in dark regions
    noise = np.random.normal(0, 3 * severity, image.shape)
    result = np.clip(crushed + noise, 0, 255).astype(np.uint8)
    return result

def apply_rain_and_sensor_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Simulates heavy rain streaks combined with CMOS sensor gain noise (ISO shot noise).
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # 1. Rain streaks
    num_drops = int(severity * 400)
    streak_len = int(severity * 8 + 10)
    angle = np.deg2rad(75)  # slightly tilted rain
    dx = int(streak_len * np.cos(angle))
    dy = int(streak_len * np.sin(angle))
    
    rain_layer = np.zeros((h, w, 3), dtype=np.float32)
    for _ in range(num_drops):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        x2 = min(w - 1, x1 + dx)
        y2 = min(h - 1, y1 + dy)
        intensity = np.random.uniform(140, 240)
        cv2.line(rain_layer, (x1, y1), (x2, y2), (intensity, intensity, intensity), 1)
        
    # Rain blur
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    result = cv2.addWeighted(result, 1.0, rain_layer, 0.45, 0)
    
    # 2. Additive Gaussian sensor noise (gain noise)
    noise_std = severity * 6.5
    noise = np.random.normal(0, noise_std, image.shape)
    result = np.clip(result + noise, 0, 255).astype(np.uint8)
    return result

def apply_lens_soiling(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Simulates dirt/mud splash or salt residue on the automotive camera lens / bumper.
    """
    h, w = image.shape[:2]
    result = image.copy()
    num_splatters = severity * 3 + 2
    
    for _ in range(num_splatters):
        cx = np.random.randint(int(w * 0.1), int(w * 0.9))
        cy = np.random.randint(int(h * 0.2), int(h * 0.85))
        axes = (np.random.randint(20, 60 * severity), np.random.randint(15, 45 * severity))
        angle = np.random.randint(0, 180)
        
        # Mud color (brownish/gray opaque)
        color = (
            int(np.random.randint(30, 60)),
            int(np.random.randint(45, 75)),
            int(np.random.randint(55, 95))
        )
        overlay = result.copy()
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, color, -1)
        alpha = 0.75 + 0.05 * severity
        result = cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0)
        
    return cv2.GaussianBlur(result, (3, 3), 0)
