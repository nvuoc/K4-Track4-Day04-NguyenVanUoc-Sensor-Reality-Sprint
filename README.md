**Repository Chuẩn Đề Bài:** `K4-Track4-Day04-NguyenVanUoc-Sensor-Reality-Sprint`  
**Chủ đề T1:** Camera degradation health score (blur, exposure, entropy, detector confidence; Computer vision, ADAS camera)  
**Tác giả:** **Nguyễn Văn Ước** - MSSV: **2A202602445** (Solo Researcher & Engineer - [TEAMMATES.md](file:///e:/Antigravity/ai20k_lab/K4-Track4-Day4-Sensor-Reality-Sprint/TEAMMATES.md))

---

## 📌 Tổng Quan Dự Án
Dự án nghiên cứu và định lượng sự suy giảm chất lượng của cảm biến camera trước trên xe tự hành ADAS dưới các điều kiện môi trường bất lợi (*Sensor Reality*): mờ quang học, rung lắc, chói sáng mặt trời, đêm tối, mưa lớn kết hợp nhiễu hạt CMOS, và bùn đất bám kính.

Dự án phát triển một hệ thống đo lường sức khỏe cảm biến theo thời gian thực (**Camera Degradation Health Score**), phát hiện lỗi sai lệch của metric truyền thống trong mưa/nhiễu, và đưa ra quyết định kỹ thuật tự động điều chỉnh trọng số trong hệ thống đa cảm biến (*Sensor Fusion*) cũng như kích hoạt cơ chế dừng xe an toàn (*Fail-Safe Minimum Risk Maneuver*).

---

## 📁 Cấu Trúc Repository
```
K4-Track4-Day4-Sensor-Reality-Sprint/
├── TEAMMATES.md                 # Thông tin thành viên & phân công nhiệm vụ
├── REPORT.md                    # Báo cáo kỹ thuật chi tiết chuẩn 5 mục theo rubric
├── SLIDES.md                    # Slide thuyết trình súc tích cho bài pitch 3-5 phút
├── README.md                    # Hướng dẫn sử dụng và tái hiện thực nghiệm
├── benchmark.py                 # Pipeline benchmark hoàn chỉnh xuất log, CSV và 5 biểu đồ
├── run_benchmark.bat            # Script chạy 1-chạm trên Windows
├── camera_health_benchmark_colab.ipynb # Notebook Jupyter chạy trực tiếp trên Google Colab
├── guide.txt.txt                # Hướng dẫn và tiêu chí đánh giá đề tài sprint
├── data/
│   └── adas_baseline_frame.png  # Khung hình chuẩn xe ADAS trên xa lộ (1280x720)
├── src/
│   ├── scene_generator.py       # Module sinh cảnh quan ADAS phối cảnh thực tế
│   ├── degradation.py           # Module mô phỏng 6 chế độ lỗi vật lý/quang học
│   ├── metrics.py               # Module tính toán IQA, HFNI và Health Score (Naive & Robust)
│   ├── detector.py              # Module đánh giá Object Detector & Confidence
│   └── fusion_decision.py       # Module thích ứng trọng số Sensor Fusion & State Machine
└── outputs/
    ├── images/                  # Ảnh trước/sau lỗi và bounding box trực quan
    ├── plots/                   # 5 biểu đồ khoa học định lượng (300 DPI)
    │   ├── health_score_comparison.png
    │   ├── optical_metrics_trends.png
    │   ├── detector_confidence_impact.png
    │   ├── sensor_fusion_weights_tradeoff.png
    │   └── visual_grid_comparison.png
    └── results/
        ├── benchmark_results.csv# Bảng số liệu chi tiết 7 kịch bản thử nghiệm
        └── metrics_summary.json # Kết quả định dạng JSON phục vụ tích hợp
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Lại Thực Nghiệm (Quick Start)

### 1. Yêu Cầu Môi Trường
- Python 3.9+ (Đã thử nghiệm hoàn hảo trên Python 3.12 và 3.14).
- Các thư viện thị giác máy tính và khoa học cơ bản:
  ```bash
  pip install numpy opencv-python matplotlib scipy pillow
  ```

### 2. Chạy Bằng Lệnh Batch (Một Chạm Trên Windows)
Nhấp đúp chuột vào file:
```cmd
run_benchmark.bat
```

### 3. Hoặc Chạy Trực Tiếp Qua Terminal
```bash
python benchmark.py
```

### 4. Chạy Trên Google Colab (Trình Duyệt Trực Tuyến & GPU)
Nếu muốn chạy trực tiếp trên Google Colab để gửi link cho giảng viên chấm bài:
1. Mở [Google Colab](https://colab.research.google.com/).
2. Chọn thẻ **Upload** và tải file `camera_health_benchmark_colab.ipynb` lên.
3. Chọn menu **Runtime > Change runtime type** và chọn **T4 GPU** (hoặc CPU).
4. Bấm **Runtime > Run all** — Toàn bộ kết quả, bảng so sánh và các biểu đồ phân tích sẽ hiển thị ngay trong giao diện web.

Thời gian chạy toàn bộ 7 kịch bản chỉ mất **khoảng 5 đến 8 giây** (rất nhẹ, chạy mượt mà trên cả CPU và GPU).

---

## 📊 Tóm Tắt Kết Quả Định Lượng Chính

| Kịch Bản | Raw Blur | Exposure Lỗi | Số Vật Thể | Mean Conf | Naive Health | **Robust Health** | Trạng Thái An Toàn | Trọng Số Sensor (Cam/Rad/LiDAR) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 425.3 | 0.0% | 4 / 4 | 0.82 | 97.3% | **97.3%** | 🟢 **HEALTHY** | 65% / 25% / 10% |
| **Defocus Blur** | 1.6 | 0.0% | 0 / 4 | 0.33 | 55.1% | **56.5%** | 🟡 **DEGRADED** | 20% / 55% / 25% |
| **Motion Blur** | 7.6 | 0.0% | 0 / 4 | 0.37 | 56.2% | **57.2%** | 🟡 **DEGRADED** | 20% / 55% / 25% |
| **Sun Glare** | 0.0 | 100.0% | 0 / 4 | 0.05 | 0.8% | **0.8%** | 🔴 **CRITICAL** | 0% / 70% / 30% |
| **Night / Dark** | 437.0 | 87.4% | 0 / 4 | 0.05 | 45.7% | **12.9%** | 🔴 **CRITICAL** | 0% / 70% / 30% |
| **Rain & Noise** | 6404.1 | 0.5% | 4 / 4 | 0.85 | **97.4% ⚠️** | **68.7%** | 🟡 **DEGRADED** | 20% / 55% / 25% |
| **Mud Splatter** | 33.6 | 0.0% | 2 / 4 | 0.47 | 59.9% | **73.5%** | 🟡 **DEGRADED** | 20% / 55% / 25% |

---

## 🎯 Điểm Nhấn Nghiên Cứu (Key Contributions)
1. **Phát hiện Failure Case kinh điển:** Phương sai vi phân Laplace ($\text{Var}(\nabla^2 I)$) bị nổ điểm trong môi trường mưa lớn và nhiễu gain hạt CMOS do xem nhiễu hạt ngẫu nhiên giữa các pixel lân cận là cạnh sắc nét (nhảy vọt từ 425 lên 6404).
2. **Đề xuất giải pháp lọc nhiễu hai pha:** Tích hợp tỷ số nhiễu tần số cao (*HFNI*) và cơ chế *Non-linear Exposure Gating* đưa điểm sức khỏe phản ánh đúng thực tế (từ 97.4% ảo xuống 68.7% chính xác).
3. **Bộ chuyển đổi trạng thái kỹ thuật (Engineering Decision):** Cung cấp luật tự động hạ trọng số camera từ $65\% \rightarrow 20\% \rightarrow 0\%$, tăng khoảng cách bám đuôi ACC từ $1.5\text{s} \rightarrow 2.5\text{s}$, và kích hoạt phanh tấp lề khẩn cấp (MRM).

---
*Mọi thắc mắc kỹ thuật vui lòng tham khảo chi tiết tại [REPORT.md](file:///e:/Antigravity/ai20k_lab/K4-Track4-Day4-Sensor-Reality-Sprint/REPORT.md) và [SLIDES.md](file:///e:/Antigravity/ai20k_lab/K4-Track4-Day4-Sensor-Reality-Sprint/SLIDES.md).*
