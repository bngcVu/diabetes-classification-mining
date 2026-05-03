# Mô tả Dataset: Diabetes Binary Health Indicators BRFSS 2015

## Tổng quan

Dataset **Diabetes Binary Health Indicators BRFSS 2015** là tập dữ liệu phục vụ bài toán phân loại nhị phân về tình trạng tiểu đường. Dữ liệu được trích từ khảo sát **Behavioral Risk Factor Surveillance System (BRFSS) 2015** của CDC.

- **File dữ liệu gốc:** `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`
- **Số dòng:** 253,680
- **Số cột:** 22 cột, gồm 1 biến mục tiêu và 21 biến đầu vào
- **Bài toán:** Binary Classification
- **Định dạng giá trị:** Các biến được mã hóa dạng số; nhiều cột nhị phân được lưu trong CSV ở dạng `0.0` và `1.0`

## Biến mục tiêu

| Tên biến | Kiểu | Miền giá trị | Mô tả |
|---|---|---:|---|
| `Diabetes_binary` | Binary | `0`, `1` | Tình trạng tiểu đường: `0` = không bị tiểu đường, `1` = bị tiểu đường |

### Phân bố lớp

| Giá trị | Số dòng | Tỷ lệ |
|---:|---:|---:|
| `0` | 218,334 | 86.07% |
| `1` | 35,346 | 13.93% |

Dataset có mất cân bằng lớp: nhóm `0` chiếm đa số, nhóm `1` chiếm khoảng 13.93%.

## Biến đầu vào

### Nhóm chỉ số sức khỏe lâm sàng

| Tên biến | Kiểu | Miền giá trị | Mô tả |
|---|---|---:|---|
| `HighBP` | Binary | `0`, `1` | Huyết áp cao: `0` = không, `1` = có |
| `HighChol` | Binary | `0`, `1` | Cholesterol cao: `0` = không, `1` = có |
| `CholCheck` | Binary | `0`, `1` | Đã kiểm tra cholesterol trong 5 năm qua |
| `BMI` | Numeric | `12`-`98` | Chỉ số khối cơ thể, tính theo cân nặng / chiều cao bình phương |
| `Stroke` | Binary | `0`, `1` | Tiền sử đột quỵ: `0` = không, `1` = có |
| `HeartDiseaseorAttack` | Binary | `0`, `1` | Bệnh tim hoặc từng bị nhồi máu cơ tim: `0` = không, `1` = có |

### Nhóm hành vi và lối sống

| Tên biến | Kiểu | Miền giá trị | Mô tả |
|---|---|---:|---|
| `Smoker` | Binary | `0`, `1` | Đã hút ít nhất 100 điếu thuốc trong đời: `0` = không, `1` = có |
| `PhysActivity` | Binary | `0`, `1` | Có hoạt động thể chất trong 30 ngày qua, không tính công việc: `0` = không, `1` = có |
| `Fruits` | Binary | `0`, `1` | Ăn trái cây từ 1 lần/ngày trở lên: `0` = không, `1` = có |
| `Veggies` | Binary | `0`, `1` | Ăn rau củ từ 1 lần/ngày trở lên: `0` = không, `1` = có |
| `HvyAlcoholConsump` | Binary | `0`, `1` | Uống rượu nhiều: nam trên 14 ly/tuần, nữ trên 7 ly/tuần |

### Nhóm tiếp cận y tế

| Tên biến | Kiểu | Miền giá trị | Mô tả |
|---|---|---:|---|
| `AnyHealthcare` | Binary | `0`, `1` | Có bảo hiểm y tế hoặc một hình thức tiếp cận chăm sóc y tế: `0` = không, `1` = có |
| `NoDocbcCost` | Binary | `0`, `1` | Không thể đi khám bác sĩ vì chi phí trong 12 tháng qua: `0` = không, `1` = có |
| `DiffWalk` | Binary | `0`, `1` | Gặp khó khăn khi đi lại hoặc leo cầu thang: `0` = không, `1` = có |

### Nhóm tự đánh giá sức khỏe

| Tên biến | Kiểu | Miền giá trị | Mô tả |
|---|---|---:|---|
| `GenHlth` | Ordinal | `1`-`5` | Sức khỏe tổng quát tự đánh giá: `1` = tuyệt vời, `2` = rất tốt, `3` = tốt, `4` = trung bình, `5` = kém |
| `MentHlth` | Numeric | `0`-`30` | Số ngày sức khỏe tâm thần không tốt trong 30 ngày qua |
| `PhysHlth` | Numeric | `0`-`30` | Số ngày sức khỏe thể chất không tốt trong 30 ngày qua |

### Nhóm nhân khẩu học

| Tên biến | Kiểu | Miền giá trị | Mô tả |
|---|---|---:|---|
| `Sex` | Binary | `0`, `1` | Giới tính: `0` = nữ, `1` = nam |
| `Age` | Ordinal | `1`-`13` | Nhóm tuổi: `1` = 18-24, `2` = 25-29, ..., `13` = 80+ |
| `Education` | Ordinal | `1`-`6` | Trình độ học vấn: `1` = chưa từng đi học/chỉ mẫu giáo, ..., `6` = cao đẳng/đại học trở lên |
| `Income` | Ordinal | `1`-`8` | Thu nhập hàng năm: `1` = dưới $10,000, ..., `8` = $75,000 trở lên |

## Ghi chú về kiểu dữ liệu

- Các cột binary có giá trị hợp lệ là `0` và `1`, nhưng trong file CSV được lưu dưới dạng số thực như `0.0` và `1.0`.
- Các cột ordinal là biến có thứ tự, nên cần cân nhắc khi mã hóa hoặc chuẩn hóa trong quá trình huấn luyện mô hình.
- `BMI` trong file thực tế có miền giá trị từ `12` đến `98`, không phải `10` đến `80`.
- Không nên xem các biến tự báo cáo như `Smoker`, `Fruits`, `Veggies`, `MentHlth`, `PhysHlth` là đo lường lâm sàng trực tiếp; đây là thông tin từ khảo sát.

## Ứng dụng trong dự án

Dataset này có thể được dùng để:

- Huấn luyện mô hình dự đoán khả năng một người thuộc nhóm có tiểu đường.
- Phân tích các yếu tố liên quan đến tiểu đường như huyết áp cao, cholesterol cao, BMI, tuổi, thu nhập và hoạt động thể chất.
- So sánh hiệu năng các thuật toán phân loại như Logistic Regression, Random Forest, XGBoost hoặc các mô hình ML khác.

## Nguồn dữ liệu

- **Tổ chức:** CDC, Centers for Disease Control and Prevention
- **Khảo sát:** BRFSS 2015, Behavioral Risk Factor Surveillance System
- **Phiên bản dataset:** Diabetes Binary Health Indicators BRFSS 2015

---

**Cập nhật lần cuối:** 2026-05-03
