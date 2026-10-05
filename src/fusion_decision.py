"""
ADAS Sensor Fusion & Engineering Decision Module
Implements adaptive sensor weighting and fail-safe decision logic based on Camera Health Score.
"""

def evaluate_fusion_decision(health_score: float, detected_ratio: float) -> dict:
    """
    Computes adaptive sensor fusion weights and system fail-safe response.
    
    Parameters:
    - health_score: Composite camera health score (0 - 100%)
    - detected_ratio: Ratio of detected objects (detected / total)
    """
    if health_score >= 75.0 and detected_ratio >= 0.75:
        state = "NOMINAL_HEALTHY"
        level = "GREEN"
        weights = {"camera": 0.65, "radar": 0.25, "lidar": 0.10}
        adas_mode = "Full L2+ ADAS Active (ACC + Lane Centering)"
        speed_limit_factor = 1.0  # 100% allowed speed
        headway_time_sec = 1.5
        driver_alert = "None (System Operating Normally)"
        fail_safe_action = "No intervention needed."
    elif health_score >= 45.0 or detected_ratio >= 0.50:
        state = "DEGRADED_PERCEPTION"
        level = "YELLOW"
        # Shift fusion trust from camera to mmWave Radar and LiDAR
        weights = {"camera": 0.20, "radar": 0.55, "lidar": 0.25}
        adas_mode = "Degraded ADAS (Radar-Dominant Longitudinal Control)"
        speed_limit_factor = 0.80  # Speed restricted by 20%
        headway_time_sec = 2.5     # Increased safe following distance
        driver_alert = "WARNING: Vision Sensor Degraded. Keep hands on wheel."
        fail_safe_action = "Inhibit automated lane changes; widen safety margins."
    else:
        state = "CRITICAL_BLINDED"
        level = "RED"
        # Zero out camera weight to prevent ghost or missed braking commands
        weights = {"camera": 0.00, "radar": 0.70, "lidar": 0.30}
        adas_mode = "Fail-Safe Override (Vision Offline)"
        speed_limit_factor = 0.40
        headway_time_sec = 3.5
        driver_alert = "CRITICAL ALARM: Vision Blinded! TAKE OVER IMMEDIATELY!"
        fail_safe_action = (
            "Initiate Minimum Risk Maneuver (MRM): Hazard lights ON, "
            "gradual emergency braking at 1.8 m/s^2 to safe stop."
        )

    return {
        "state": state,
        "level": level,
        "fusion_weights": weights,
        "adas_mode": adas_mode,
        "speed_limit_factor": speed_limit_factor,
        "headway_time_sec": headway_time_sec,
        "driver_alert": driver_alert,
        "fail_safe_action": fail_safe_action
    }
