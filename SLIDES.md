# SLIDES THUYẾT TRÌNH PITCH (3 - 5 PHÚT)
## Đề Tài T1: Camera Degradation Health Score Trong Hệ Thống ADAS
**Người thực hiện:** **Nguyễn Văn Ước** - MSSV: **2A202602445** (Solo Researcher)  
**Lớp / Sprint:** K4 - Track 4 - Day 4 (Sensor Reality Sprint)  
**Tệp đính kèm:** [REPORT.md](file:///e:/Antigravity/ai20k_lab/K4-Track4-Day4-Sensor-Reality-Sprint/REPORT.md) | [benchmark_results.csv](file:///e:/Antigravity/ai20k_lab/K4-Track4-Day4-Sensor-Reality-Sprint/outputs/results/benchmark_results.csv)

---

### SLIDE 1: Problem & Motivation (Bài Toán Thực Tế Trong Tự Nhiên)
- **Bối cảnh:** Xe tự hành (L2+/L3 ADAS) dựa trên camera trước để thực hiện phanh khẩn cấp (AEB) và giữ làn (LKA).
- **Thực tế cảm biến (Sensor Reality):** Camera không hoàn hảo!
  * Mờ nhòe ống kính do hơi nước (*Defocus*), rung lắc xe (*Motion blur*).
  * Chói lóa mặt trời hoặc đèn pha ngược chiều (*Blinding Glare*).
  * Mưa bão lớn (*Heavy Rain*) và nhiễu hạt ISO cao (*CMOS Shot Noise*).
  * Thiếu sáng ban đêm (*Underexposure*) và bùn đất bám kính (*Soiling*).
- **Nguy cơ:** Khi ảnh suy giảm, detector bỏ sót xe trước / người đi bộ, dẫn đến quyết định phanh sai hoặc trễ!
- **Câu hỏi kỹ thuật:** Làm thế nào để xe **tự biết camera đang bị "ốm"** để hạ tải và chuyển sang radar/LiDAR trước khi tai nạn xảy ra?

---

### SLIDE 2: Methodology & Camera Health Score Formula (Phương Pháp)
- **Mục tiêu:** Tính toán chỉ số sức khỏe camera $H_{cam} \in [0, 100]\%$ siêu nhanh (< 3.5ms) không phụ thuộc model nặng:
  $$\mathbf{H_{cam} = 0.35 \cdot S_{blur} + 0.35 \cdot S_{exp} + 0.15 \cdot S_{ent} + 0.15 \cdot S_{det}}$$
- **Các thành phần IQA đo lường trực tiếp:**
  1. **$S_{blur}$ (Độ sắc nét):** Dựa trên phương sai toán tử Laplace $\text{Var}(\nabla^2 I)$.
  2. **$S_{exp}$ (Cân bằng phơi sáng):** Phạt tỷ lệ pixel cháy sáng ($I \ge 250$) và tối đen ($I \le 10$).
  3. **$S_{ent}$ (Entropy thông tin):** Shannon Entropy $H(X)$ đo độ phong phú của chi tiết điểm ảnh.
  4. **$S_{det}$ (Độ tin cậy nhận diện):** Mean Detection Confidence trên các vật thể ADAS trọng yếu.
- **3 Vùng an toàn:**
  * **GREEN ($\ge 75\%$):** Khỏe mạnh — Tin cậy camera 100%.
  * **YELLOW ($45 - 74\%$):** Suy giảm thị giác — Giảm trọng số camera, tăng khoảng cách an toàn.
  * **RED ($< 45\%$):** Tê liệt quang học — Kích hoạt Fail-Safe, bàn giao tài xế (Take-Over).

---

### SLIDE 3: Benchmark Results (Kết Quả Thực Nghiệm Định Lượng)
*Thử nghiệm có đối chứng trên 7 kịch bản camera ADAS chuẩn ($1280 \times 720$):*

| Kịch Bản Thử Nghiệm | Raw Blur | Exposure Lỗi | Số Vật Thể Phát Hiện | Mean Conf | **Robust Health** | Trạng Thái Hệ Thống |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Ngày trong)** | **425.3** | 0.0% | **4 / 4 (100%)** | **0.82** | **97.3%** | 🟢 **NOMINAL (GREEN)** |
| **Mild Defocus Blur** | 1.6 | 0.0% | 0 / 4 (0%) | 0.33 | **56.5%** | 🟡 **DEGRADED (YELLOW)** |
| **Severe Motion Blur** | 7.6 | 0.0% | 0 / 4 (0%) | 0.37 | **57.2%** | 🟡 **DEGRADED (YELLOW)** |
| **Blinding Sun Glare** | 0.0 | **100.0%** | 0 / 4 (0%) | 0.05 | **0.8%** | 🔴 **CRITICAL (RED)** |
| **Night / Unlit Road** | 437.0 | **87.4%** | 0 / 4 (0%) | 0.05 | **12.9%** | 🔴 **CRITICAL (RED)** |
| **Heavy Rain & Noise** | **6404.1** | 0.5% | 4 / 4 | 0.85 | **68.7%** | 🟡 **DEGRADED (YELLOW)** |
| **Lens Dirt / Mud** | 33.6 | 0.0% | 2 / 4 (50%) | 0.47 | **73.5%** | 🟡 **DEGRADED (YELLOW)** |

- **Minh chứng đồ thị:**
  * Biểu đồ Sức khỏe Cảm biến: [health_score_comparison.png](file:///e:/Antigravity/ai20k_lab/K4-Track4-Day4-Sensor-Reality-Sprint/outputs/plots/health_score_comparison.png)
  * Biểu đồ Bỏ sót vật thể: [detector_confidence_impact.png](file:///e:/Antigravity/ai20k_lab/K4-Track4-Day4-Sensor-Reality-Sprint/outputs/plots/detector_confidence_impact.png)
  * Toàn cảnh 7 khung hình: [visual_grid_comparison.png](file:///e:/Antigravity/ai20k_lab/K4-Track4-Day4-Sensor-Reality-Sprint/outputs/plots/visual_grid_comparison.png)

---

### SLIDE 4: Critical Failure Case & Proposed Solution (Phân Tích Lỗi Metric)
- **Nghịch lý tần số cao (High-Frequency Noise Paradox):**
  * Trong trời mưa to và nhiễu hạt gain, điểm Raw Laplacian nhảy vọt từ **425.3 lên 6404.1** (gấp 15 lần!).
  * **Hậu quả:** Naive Metric bị đánh lừa là ảnh "cực nét", cho điểm sức khỏe **97.4% (Ảo tưởng an toàn!)**.
- **Nguyên nhân cốt lõi:** Toán tử Laplace đạo hàm bậc hai xem nhiễu hạt ngẫu nhiên giữa các pixel lân cận là cạnh sắc nét!
- **Giải pháp cải tiến của nhóm:**
  1. Thêm chỉ số nhiễu hạt tần số cao: $\text{HFNI} = \frac{\text{Var}_{raw}}{\text{Var}_{denoised}}$. Khi mưa to, tỷ số này vọt lên 28.86 (gấp đôi mốc chuẩn 13.8).
  2. Kích hoạt hàm phạt suy hao $\text{Penalty}_{noise}$, đưa điểm sức khỏe chuẩn xác về **68.7% (DEGRADED)**.
  3. Thêm cơ chế **Non-linear Exposure Gating** để triệt tiêu việc nhiễu đen ban đêm đánh lừa hệ thống.

---

### SLIDE 5: Engineering Decision & Sensor Fusion Fallback (Quyết Định Kỹ Thuật)
- **Phân bổ trọng số Sensor Fusion động:**
  * Khi $H_{cam} \ge 75\%$: Camera giữ trọng số **65%**, Radar **25%**, LiDAR **10%** (Lái tự động mượt mà).
  * Khi $45\% \le H_{cam} < 75\%$: Hạ trọng số Camera còn **20%**, nhường quyền dẫn dắt khoảng cách cho Radar **55%** và LiDAR **25%**. Giới hạn tốc độ $-20\%$, tăng khoảng cách bám đuôi từ 1.5s lên 2.5s.
  * Khi $H_{cam} < 45\%$: Ngắt bỏ hoàn toàn dữ liệu Camera (**0%**). Phát chuông Take-Over khẩn cấp.
  * Nếu tài xế không phản ứng sau 4 giây: Kích hoạt **Minimum Risk Maneuver (MRM)** — Bật xi nhan khẩn cấp và phanh từ từ $1.8 \text{ m/s}^2$ tấp lề dừng an toàn.
- **Trade-off lớn nhất:**
  * Thuật toán IQA toán học chạy siêu nhanh (<3.5ms), hoàn toàn khả thi trên chip nhúng xe ô tô (Jetson/ECU) mà không tốn chi phí điện toán GPU của mạng nơ-ron sâu.

---
**Cảm ơn Thầy Cô và các bạn đã lắng nghe!**
