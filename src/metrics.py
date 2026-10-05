"""
Camera Image Quality Assessment (IQA) & Health Score Module for ADAS
Metrics:
- Laplacian Variance (Blur / Defocus metric)
- Exposure Saturation Ratio (Over-exposure / Under-exposure)
- Shannon Information Entropy
- ADAS Camera Degradation Health Score (0 - 100%)
"""

import numpy as np
import cv2

def compute_laplacian_blur_score(image: np.ndarray) -> float:
    """
    Computes the variance of the Laplacian of the image.
    High value = Sharp, in-focus, edge-rich image.
    Low value = Blurred, defocused, or soiled lens.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance

def compute_exposure_metrics(image: np.ndarray, over_thresh: int = 250, under_thresh: int = 10) -> dict:
    """
    Computes overexposure and underexposure pixel ratios.
    Overexposure indicates glare/headlight blinding or bright sun reflection.
    Underexposure indicates night/tunnel or extreme shadow.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    total_pixels = gray.size
    over_count = np.sum(gray >= over_thresh)
    under_count = np.sum(gray <= under_thresh)

    over_ratio = float(over_count / total_pixels)
    under_ratio = float(under_count / total_pixels)
    mean_intensity = float(np.mean(gray))

    return {
        "overexposure_ratio": over_ratio,
        "underexposure_ratio": under_ratio,
        "mean_intensity": mean_intensity
    }

def compute_shannon_entropy(image: np.ndarray) -> float:
    """
    Computes Shannon Information Entropy H(X) = -sum(p * log2(p)).
    Represents the information capacity and visual texture complexity.
    Blur, washed-out glare, or black night images show low entropy.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()
    # Filter out zero probabilities to avoid log2(0)
    p_nonzero = hist_norm[hist_norm > 0]
    entropy = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
    return entropy

def compute_robust_blur_and_noise(image: np.ndarray) -> dict:
    """
    Computes both raw Laplacian and robust pre-filtered Laplacian.
    Also calculates the High-Frequency Noise Index (HFNI) to detect sensor shot noise and rain streaks.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Raw Laplacian (sensitive to noise)
    raw_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Denoised Laplacian (applies 3x3 Gaussian smoothing first)
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0.8)
    denoised_var = float(cv2.Laplacian(smoothed, cv2.CV_64F).var())

    # 3. High-Frequency Noise Index: if raw is much larger than denoised, it's sensor noise/rain!
    noise_ratio = raw_var / max(1.0, denoised_var)

    return {
        "raw_blur_score": raw_var,
        "denoised_blur_score": denoised_var,
        "noise_ratio": float(noise_ratio)
    }

def compute_camera_health_score(
    blur_score: float,
    over_ratio: float,
    under_ratio: float,
    entropy: float,
    det_confidence: float = 1.0,
    noise_ratio: float = 1.0,
    base_noise_ratio: float = 1.0,
    ref_blur: float = 350.0,
    ref_entropy: float = 7.2,
    use_robust: bool = True
) -> dict:
    """
    Calculates Camera Degradation Health Score (0 - 100%).
    Supports both:
    - Naive mode (demonstrating metric failure case under rain/noise)
    - Robust mode (applying sensor noise penalty and exposure gating)
    """
    if not use_robust:
        # Naive calculation: vulnerable to high-frequency shot noise & night saturation
        s_blur = min(1.0, max(0.0, blur_score / ref_blur))
        penalty = 2.5 * over_ratio + 2.0 * under_ratio
        s_exposure = max(0.0, min(1.0, 1.0 - penalty))
        s_entropy = min(1.0, max(0.0, entropy / ref_entropy))
        s_det = min(1.0, max(0.0, det_confidence))

        composite = (0.35 * s_blur + 0.35 * s_exposure + 0.15 * s_entropy + 0.15 * s_det) * 100.0
        health_score = round(max(0.0, min(100.0, composite)), 2)
    else:
        # Robust Engineering Formulation:
        # 1. S_blur with relative noise penalty compared to baseline optical texture
        effective_blur = min(ref_blur, blur_score)
        s_blur = max(0.0, effective_blur / ref_blur)
        
        rel_noise = noise_ratio / max(0.1, base_noise_ratio)
        if rel_noise > 1.25:
            # Excessive high-frequency noise from rain streaks or CMOS sensor gain
            noise_penalty = min(0.85, (rel_noise - 1.0) * 0.75)
            s_blur = max(0.05, s_blur * (1.0 - noise_penalty))

        # 2. S_exposure with severe non-linear gating
        # If > 50% underexposed (night) or > 30% overexposed (glare), vision is compromised
        exp_damage = 3.0 * over_ratio + 2.5 * under_ratio
        s_exposure = max(0.0, min(1.0, 1.0 - exp_damage))

        # 3. S_entropy
        s_entropy = min(1.0, max(0.0, entropy / ref_entropy))

        # 4. S_detector
        s_det = min(1.0, max(0.0, det_confidence))

        composite = (0.35 * s_blur + 0.35 * s_exposure + 0.15 * s_entropy + 0.15 * s_det) * 100.0
        health_score = round(max(0.0, min(100.0, composite)), 2)

        # Critical Gating: If overexposure > 40% or underexposure > 60%, hard-cap health to < 35%
        if over_ratio > 0.40 or under_ratio > 0.60:
            health_score = min(health_score, 25.0)

    # Health status classification
    if health_score >= 75.0:
        status = "HEALTHY"
        action = "Full ADAS Operational. Nominal Vision Weight (100%)."
        color_code = "GREEN"
    elif health_score >= 45.0:
        status = "DEGRADED"
        action = "Vision Degraded. Reduce Camera Fusion Weight; Rely on Radar/LiDAR."
        color_code = "YELLOW"
    else:
        status = "CRITICAL"
        action = "Vision Sensor Blinded. Disengage Autosteer; Request Driver Takeover."
        color_code = "RED"

    return {
        "health_score": health_score,
        "status": status,
        "color_code": color_code,
        "action": action,
        "sub_scores": {
            "blur_component": round(s_blur * 100, 1),
            "exposure_component": round(s_exposure * 100, 1),
            "entropy_component": round(s_entropy * 100, 1),
            "detector_component": round(s_det * 100, 1)
        }
    }
