"""
Camera Degradation Health Score Benchmark Runner
Topic T1: Camera degradation health score (blur, exposure, entropy, detector confidence)
Target: ADAS Computer Vision & Sensor Fusion Fail-Safe
"""

import os
import sys
import json
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure standard UTF-8 stream output on Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.scene_generator import generate_adas_road_scene
from src.degradation import (
    apply_defocus_blur,
    apply_motion_blur,
    apply_glare_overexposure,
    apply_night_underexposure,
    apply_rain_and_sensor_noise,
    apply_lens_soiling
)
from src.metrics import (
    compute_laplacian_blur_score,
    compute_exposure_metrics,
    compute_shannon_entropy,
    compute_robust_blur_and_noise,
    compute_camera_health_score
)
from src.detector import ADASObjectDetector
from src.fusion_decision import evaluate_fusion_decision

def setup_directories():
    dirs = [
        "data",
        "outputs/images",
        "outputs/plots",
        "outputs/results"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_benchmark():
    setup_directories()
    print("================================================================================")
    print("[START] ADAS CAMERA DEGRADATION HEALTH SCORE BENCHMARK")
    print("================================================================================")

    # 1. Generate or load base ADAS camera frame
    base_img_path = "data/adas_baseline_frame.png"
    base_image = generate_adas_road_scene(width=1280, height=720)
    cv2.imwrite(base_img_path, base_image)
    print(f"[OK] Baseline ADAS frame generated and saved to: {base_img_path}")

    # Initialize Object Detector
    detector = ADASObjectDetector()
    ground_truth = detector.get_ground_truth_objects(img_w=1280, img_h=720)

    # 2. Define experimental degradation conditions
    conditions = [
        {
            "id": "cond_0_baseline",
            "name": "Baseline (Clear Daytime)",
            "category": "Nominal",
            "apply": lambda img: img.copy(),
            "param": "None (Nominal optical conditions)"
        },
        {
            "id": "cond_1_mild_blur",
            "name": "Mild Defocus Blur",
            "category": "Optical Defocus",
            "apply": lambda img: apply_defocus_blur(img, severity=2),
            "param": "Defocus Gaussian sigma=3.0, ksize=9"
        },
        {
            "id": "cond_2_motion_blur",
            "name": "Severe Motion Blur",
            "category": "Vehicle Dynamics",
            "apply": lambda img: apply_motion_blur(img, severity=4, angle=15.0),
            "param": "Linear shake kernel=31px, angle=15 deg"
        },
        {
            "id": "cond_3_sun_glare",
            "name": "Blinding Sun Glare",
            "category": "Overexposure",
            "apply": lambda img: apply_glare_overexposure(img, severity=4),
            "param": "Direct CMOS blooming, saturation intensity=4"
        },
        {
            "id": "cond_4_night_underexposure",
            "name": "Night / Unlit Road",
            "category": "Underexposure",
            "apply": lambda img: apply_night_underexposure(img, severity=4),
            "param": "Factor=0.15, gamma=2.6, read noise"
        },
        {
            "id": "cond_5_heavy_rain_noise",
            "name": "Heavy Rain & Shot Noise",
            "category": "Adverse Weather",
            "apply": lambda img: apply_rain_and_sensor_noise(img, severity=4),
            "param": "1600 rain streaks, Gaussian noise std=26.0"
        },
        {
            "id": "cond_6_lens_soiling",
            "name": "Lens Dirt / Mud Splatter",
            "category": "Lens Occlusion",
            "apply": lambda img: apply_lens_soiling(img, severity=3),
            "param": "11 mud splatter ellipses, alpha=0.90"
        }
    ]

    benchmark_records = []
    vis_images = {}

    # Calculate baseline reference values
    base_blur_info = compute_robust_blur_and_noise(base_image)
    base_blur = base_blur_info["denoised_blur_score"]
    base_raw_blur = base_blur_info["raw_blur_score"]
    base_noise_ratio = base_blur_info["noise_ratio"]
    base_entropy = compute_shannon_entropy(base_image)
    print(f"[METRIC] Reference Baseline: Denoised Blur={base_blur:.1f}, Raw={base_raw_blur:.1f}, Noise Ratio={base_noise_ratio:.1f}, Entropy={base_entropy:.2f}")

    # 3. Process each condition
    for cond in conditions:
        cond_id = cond["id"]
        cond_name = cond["name"]
        print(f"\n[EVAL] Evaluating: [{cond_name}] ...")

        # Apply degradation
        degraded_img = cond["apply"](base_image)
        raw_out_path = f"outputs/images/{cond_id}_raw.png"
        cv2.imwrite(raw_out_path, degraded_img)

        # Compute IQA metrics
        blur_info = compute_robust_blur_and_noise(degraded_img)
        raw_blur = blur_info["raw_blur_score"]
        denoised_blur = blur_info["denoised_blur_score"]
        noise_ratio = blur_info["noise_ratio"]

        exposure_metrics = compute_exposure_metrics(degraded_img)
        entropy_score = compute_shannon_entropy(degraded_img)

        # Run ADAS Object Detector
        det_result = detector.evaluate_detections(degraded_img, ground_truth)
        detected_count = det_result["detected_count"]
        missed_count = det_result["missed_count"]
        mean_conf = det_result["mean_confidence"]

        # Save annotated image
        annotated_out_path = f"outputs/images/{cond_id}_detected.png"
        cv2.imwrite(annotated_out_path, det_result["vis_image"])
        vis_images[cond_name] = det_result["vis_image"]

        # 1. Naive Health Score (baseline proxy without noise/exposure gating)
        naive_health = compute_camera_health_score(
            blur_score=raw_blur,
            over_ratio=exposure_metrics["overexposure_ratio"],
            under_ratio=exposure_metrics["underexposure_ratio"],
            entropy=entropy_score,
            det_confidence=mean_conf,
            noise_ratio=noise_ratio,
            base_noise_ratio=base_noise_ratio,
            ref_blur=base_raw_blur,
            ref_entropy=base_entropy,
            use_robust=False
        )

        # 2. Robust Proposed Health Score (with noise penalty & non-linear gating)
        robust_health = compute_camera_health_score(
            blur_score=denoised_blur,
            over_ratio=exposure_metrics["overexposure_ratio"],
            under_ratio=exposure_metrics["underexposure_ratio"],
            entropy=entropy_score,
            det_confidence=mean_conf,
            noise_ratio=noise_ratio,
            base_noise_ratio=base_noise_ratio,
            ref_blur=base_blur,
            ref_entropy=base_entropy,
            use_robust=True
        )

        # Evaluate Sensor Fusion & Safe ADAS Decision using Robust Score
        det_ratio = detected_count / len(ground_truth)
        fusion = evaluate_fusion_decision(robust_health["health_score"], det_ratio)

        record = {
            "Condition_ID": cond_id,
            "Condition_Name": cond_name,
            "Category": cond["category"],
            "Parameter": cond["param"],
            "Raw_Laplacian_Var": round(raw_blur, 2),
            "Denoised_Laplacian_Var": round(denoised_blur, 2),
            "Noise_Ratio_HFNI": round(noise_ratio, 2),
            "Overexposure_Ratio": round(exposure_metrics["overexposure_ratio"], 4),
            "Underexposure_Ratio": round(exposure_metrics["underexposure_ratio"], 4),
            "Shannon_Entropy": round(entropy_score, 3),
            "Objects_Detected": detected_count,
            "Objects_Missed": missed_count,
            "Detection_Ratio": round(det_ratio, 2),
            "Mean_Detector_Confidence": round(mean_conf, 3),
            "Naive_Health_Score_Pct": naive_health["health_score"],
            "Robust_Health_Score_Pct": robust_health["health_score"],
            "Health_Status": robust_health["status"],
            "Safety_Zone": robust_health["color_code"],
            "Weight_Camera": fusion["fusion_weights"]["camera"],
            "Weight_Radar": fusion["fusion_weights"]["radar"],
            "Weight_LiDAR": fusion["fusion_weights"]["lidar"],
            "ADAS_Operating_Mode": fusion["adas_mode"],
            "Speed_Limit_Factor": fusion["speed_limit_factor"],
            "Safe_Headway_Sec": fusion["headway_time_sec"],
            "Fail_Safe_Action": fusion["fail_safe_action"]
        }
        benchmark_records.append(record)

        print(f"   - Raw Blur: {raw_blur:.1f} | Denoised Blur: {denoised_blur:.1f} | Noise Ratio: {noise_ratio:.1f}")
        print(f"   - Exposure: Over={exposure_metrics['overexposure_ratio']:.1%}, Under={exposure_metrics['underexposure_ratio']:.1%}")
        print(f"   - Detections: {detected_count}/{len(ground_truth)} (Mean Conf: {mean_conf:.2f})")
        print(f"   - [HEALTH] Naive: {naive_health['health_score']:.1f}% -> Robust Health: {robust_health['health_score']:.1f}% [{robust_health['status']}]")
        print(f"   - [SAFETY] Safe Action: {fusion['adas_mode']}")

    # 4. Save results to CSV & JSON
    csv_path = "outputs/results/benchmark_results.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=benchmark_records[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_records)
    print(f"\n[SAVE] Benchmark records saved to CSV: {csv_path}")

    json_path = "outputs/results/metrics_summary.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Benchmark records saved to JSON: {json_path}")

    # 5. Generate Professional Charts & Plots
    print("\n[PLOTS] Generating scientific evaluation plots...")
    generate_plots(benchmark_records, vis_images)
    print("================================================================================")
    print("[SUCCESS] BENCHMARK RUN SUCCESSFULLY COMPLETED!")
    print("================================================================================")

def generate_plots(records: list, vis_images: dict):
    names = [r["Condition_Name"] for r in records]
    naive_scores = [r["Naive_Health_Score_Pct"] for r in records]
    robust_scores = [r["Robust_Health_Score_Pct"] for r in records]
    raw_blurs = [r["Raw_Laplacian_Var"] for r in records]
    denoised_blurs = [r["Denoised_Laplacian_Var"] for r in records]
    confidences = [r["Mean_Detector_Confidence"] for r in records]
    over_ratios = [r["Overexposure_Ratio"] * 100 for r in records]
    under_ratios = [r["Underexposure_Ratio"] * 100 for r in records]
    entropies = [r["Shannon_Entropy"] for r in records]
    cam_weights = [r["Weight_Camera"] * 100 for r in records]
    rad_weights = [r["Weight_Radar"] * 100 for r in records]
    lid_weights = [r["Weight_LiDAR"] * 100 for r in records]

    # PLOT 1: Naive vs Robust Health Score Comparison
    plt.figure(figsize=(12, 6.5))
    y = np.arange(len(names))
    h_bar = 0.38
    plt.barh(y - h_bar/2, naive_scores[::-1], h_bar, label="Naive Proxy Health Score (Metric Failure Case)", color="#95a5a6", edgecolor="black")
    
    robust_colors = []
    for s in robust_scores[::-1]:
        if s >= 75:
            robust_colors.append("#2ecc71")
        elif s >= 45:
            robust_colors.append("#f39c12")
        else:
            robust_colors.append("#e74c3c")
    bars_rob = plt.barh(y + h_bar/2, robust_scores[::-1], h_bar, label="Robust Proposed Health Score (With Noise & Gating)", color=robust_colors, edgecolor="black")

    plt.axvline(75, color="#27ae60", linestyle="--", linewidth=1.8, label="Healthy Threshold (>= 75%)")
    plt.axvline(45, color="#c0392b", linestyle="--", linewidth=1.8, label="Critical Threshold (< 45%)")
    plt.yticks(y, names[::-1], fontsize=10)
    plt.title("ADAS Camera Degradation: Naive Metric vs Proposed Robust Health Score", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Health Score (%)", fontsize=11, labelpad=8)
    plt.xlim(0, 115)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.95, fontsize=9.5)

    for bar in bars_rob:
        w = bar.get_width()
        plt.text(w + 1.2, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    p1 = "outputs/plots/health_score_comparison.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  [PLOT] Plot saved: {p1}")

    # PLOT 2: Optical Sub-metrics (Blur, Exposure, Entropy)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Subplot 2.1: Blur Comparison (Raw vs Denoised)
    width_b = 0.35
    x_b = np.arange(len(names))
    axes[0].bar(x_b - width_b/2, [max(0.1, b) for b in raw_blurs], width_b, label="Raw Laplacian (Noise Sensitive)", color="#e67e22", edgecolor="black")
    axes[0].bar(x_b + width_b/2, [max(0.1, b) for b in denoised_blurs], width_b, label="Denoised Laplacian (Robust)", color="#3498db", edgecolor="black")
    axes[0].set_yscale("log")
    axes[0].set_title("Laplacian Blur Variance (Log Scale)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Variance of Laplacian (log)", fontsize=10)
    axes[0].set_xticks(x_b)
    axes[0].set_xticklabels([n.split("(")[0].strip() for n in names], rotation=45, ha="right", fontsize=9)
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 2.2: Exposure Saturation
    width = 0.35
    x = np.arange(len(names))
    axes[1].bar(x - width/2, over_ratios, width, label="Overexposure (Glare)", color="#f1c40f", edgecolor="black")
    axes[1].bar(x + width/2, under_ratios, width, label="Underexposure (Night)", color="#34495e", edgecolor="black")
    axes[1].set_title("Exposure Saturation Ratios (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Saturated Pixels (%)", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.split("(")[0].strip() for n in names], rotation=45, ha="right", fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 2.3: Shannon Entropy
    axes[2].plot(range(len(names)), entropies, marker="o", color="#9b59b6", linewidth=2.5, markersize=8)
    axes[2].set_title("Shannon Information Entropy H(X)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Bits / Pixel", fontsize=10)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels([n.split("(")[0].strip() for n in names], rotation=45, ha="right", fontsize=9)
    axes[2].set_ylim(4.0, 8.0)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p2 = "outputs/plots/optical_metrics_trends.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  [PLOT] Plot saved: {p2}")

    # PLOT 3: Downstream Detector Confidence & Object Miss Rate
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    rects1 = ax1.bar(x, confidences, color="#2980b9", width=0.5, edgecolor="black", label="Mean Detection Confidence")
    ax1.set_ylabel("Detector Confidence [0.0 - 1.0]", color="#2980b9", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.split("(")[0].strip() for n in names], rotation=35, ha="right", fontsize=9.5)
    ax1.axhline(0.45, color="red", linestyle="--", linewidth=1.5, label="Detection Decision Threshold (0.45)")

    ax2 = ax1.twinx()
    missed_counts = [r["Objects_Missed"] for r in records]
    ax2.plot(x, missed_counts, color="#c0392b", marker="s", linewidth=2.5, markersize=8, label="Missed Objects Count")
    ax2.set_ylabel("Missed Objects (False Negatives)", color="#c0392b", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.5, 4.5)

    plt.title("Downstream Impact: Object Detector Confidence and Missed Objects", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    p3 = "outputs/plots/detector_confidence_impact.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  [PLOT] Plot saved: {p3}")

    # PLOT 4: Multi-Sensor Fusion Weights Dynamic Re-allocation
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(names))
    p_cam = plt.bar(x, cam_weights, width=0.55, label="Camera Weight (Visual)", color="#16a085", edgecolor="black")
    p_rad = plt.bar(x, rad_weights, width=0.55, bottom=cam_weights, label="Radar Weight (mmWave)", color="#2980b9", edgecolor="black")
    bottom_lid = np.array(cam_weights) + np.array(rad_weights)
    p_lid = plt.bar(x, lid_weights, width=0.55, bottom=bottom_lid, label="LiDAR Weight (Range)", color="#8e44ad", edgecolor="black")

    plt.ylabel("Adaptive Sensor Fusion Weight (%)", fontsize=11, fontweight="bold")
    plt.title("Fail-Safe Adaptive Sensor Fusion Re-weighting Policy", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, [n.split("(")[0].strip() for n in names], rotation=35, ha="right", fontsize=9.5)
    plt.ylim(0, 105)
    plt.legend(loc="upper right", framealpha=0.95)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    p4 = "outputs/plots/sensor_fusion_weights_tradeoff.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"  [PLOT] Plot saved: {p4}")

    # PLOT 5: Visual Collage Comparison (Grid of detections)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, r in enumerate(records):
        cond_name = r["Condition_Name"]
        img = vis_images[cond_name]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb_img)
        title_str = f"{cond_name}\nHealth: {r['Robust_Health_Score_Pct']:.1f}% ({r['Safety_Zone']})"
        axes[i].set_title(title_str, fontsize=10, fontweight="bold", color="darkred" if r['Safety_Zone']=="RED" else "darkgreen")
        axes[i].axis("off")
    # Hide the 8th unused subplot
    axes[7].axis("off")
    axes[7].text(
        0.5, 0.5,
        "ADAS Camera Degradation\nHealth Score Benchmark\n\n• Green: Healthy >= 75%\n• Yellow: Degraded 45-74%\n• Red: Critical < 45%",
        ha="center", va="center", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=1", facecolor="#ecf0f1", edgecolor="#bdc3c7")
    )
    plt.suptitle("Visual Benchmark Collage: Automotive Camera Perception under Optical Degradation", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    p5 = "outputs/plots/visual_grid_comparison.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"  [PLOT] Plot saved: {p5}")

if __name__ == "__main__":
    run_benchmark()
