# Danh Sách Thành Viên Nhóm (TEAMMATES)

**Dự án:** Camera Degradation Health Score Benchmark (ADAS Autonomous Driving)  
**Mã lớp / Sprint:** K4 - Track 4 - Day 4 (Sensor Reality Sprint)  
**Chủ đề:** T1 - Camera Degradation Health Score (Blur, Exposure, Entropy, Detector Confidence)

---

## Thông Tin Thành Viên

> **Ghi chú:** Đề tài được thực hiện độc lập (Solo Member) hoàn thành trọn vẹn toàn bộ các vai trò từ nghiên cứu lý thuyết, xây dựng benchmark, đo lường định lượng, thiết kế kiến trúc kỹ thuật đến viết báo cáo.

| STT | Họ và Tên | MSSV | Vai Trò Chính | Đóng Góp |
|:---:|:---|:---:|:---|:---:|
| 1 | **Nguyễn Văn Ước** | **2A202602445** | Lead Researcher & Engineer (Solo) | 100% |

---

### Phân Công Nhiệm Vụ (Theo Rubric Đề Bài)
- **Problem Formulation & Claim:** Định nghĩa bài toán suy giảm chất lượng cảm biến camera trước trên xe ADAS (optical blur, overexposure/glare, underexposure, sensor noise).
- **Literature & Method:** Nghiên cứu các phương pháp đánh giá chất lượng ảnh (IQA - Image Quality Assessment, Laplacian Variance, Shannon Entropy, Saturation Ratios).
- **Benchmark Implementation:** Viết pipeline Python tự động tạo suy giảm sensor, tính toán metrics thời gian thực và đo detector confidence.
- **Data Analysis & Visualization:** Xuất log số liệu thực nghiệm, vẽ biểu đồ tương quan giữa độ suy giảm và sức khỏe cảm biến (Camera Health Score).
- **Engineering Decision & Safety Fallback:** Thiết kế bộ quy tắc chuyển đổi trạng thái an toàn (Healthy -> Degraded -> Critical) và cơ chế giảm tải trọng số camera trong Sensor Fusion (LiDAR/Radar).
