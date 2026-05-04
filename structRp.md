# Cấu trúc báo cáo đề tài

---

## Phần chia công việc

| Thành viên | Công việc |
|---|---|
| ... | ... |

---

## Chương I: Giới thiệu

### 1.1. Data Mining là gì?

Data Mining (Khai phá dữ liệu) là quá trình trích xuất các thông tin có giá trị, patterns (mẫu ẩn) và tri thức từ các tập dữ liệu lớn. Data Mining kết hợp các kỹ thuật từ thống kê, học máy (Machine Learning) và cơ sở dữ liệu để phân tích dữ liệu từ nhiều góc độ khác nhau.

**Các bước chính trong Data Mining:**

1. **Thu thập dữ liệu (Data Collection):** Tập hợp dữ liệu từ các nguồn khác nhau.
2. **Làm sạch dữ liệu (Data Cleaning):** Xử lý dữ liệu thiếu, nhiễu, trùng lặp và không nhất quán.
3. **Khám phá dữ liệu (Data Exploration / EDA):** Phân tích thống kê mô tả, trực quan hóa dữ liệu.
4. **Chuyển đổi dữ liệu (Data Transformation):** Chuẩn hóa, mã hóa, tạo thuộc tính mới.
5. **Xây dựng mô hình (Model Building):** Áp dụng thuật toán học máy để dự đoán hoặc phân loại.
6. **Đánh giá mô hình (Model Evaluation):** Kiểm tra độ chính xác bằng Accuracy, Precision, Recall, F1, ROC-AUC.
7. **Triển khai (Deployment):** Đưa mô hình vào ứng dụng thực tế.

### 1.2. Khám phá dữ liệu (EDA)

> Thực hiện trong `notebooks/01_eda.ipynb`. Kết quả lưu tại `reports/figures/`.

#### 1.2.1. Thông tin tổng quan

- **Dataset:** Diabetes Binary Health Indicators BRFSS 2015 (CDC, Mỹ).
- **Kích thước:** 253,680 dòng × 22 cột.
- **Cấu trúc:** 21 thuộc tính đầu vào + 1 biến mục tiêu `Diabetes_binary`.

#### 1.2.2. Thống kê mô tả

```python
# File: notebooks/01_eda.ipynb
df.describe().T
```

Phân tích các giá trị đặc trưng:
- **BMI:** min, max, mean ≈ 28, phân phối gần chuẩn, lệch phải nhẹ.
- **MentHlth, PhysHlth:** Phân phối lệch phải mạnh, đa số giá trị = 0 (không có ngày nào không tốt).
- **Age:** Phân phối đều 13 nhóm tuổi (mã hóa ordinal).

#### 1.2.3. Phân phối biến mục tiêu

```python
# File: notebooks/01_eda.ipynb
target_counts = df["Diabetes_binary"].value_counts().sort_index()
# 0: 218,334 (86.07%) — Non-diabetic
# 1: 35,346 (13.93%) — Diabetic
```

- **Hiện tượng mất cân bằng lớp nghiêm trọng:** Class 0 chiếm ~86%, class 1 chiếm ~14% (tỷ lệ 6.18:1).
- **Ảnh hưởng:** Accuracy không phản ánh đúng chất lượng model. Cần dùng ROC-AUC, F1, Recall.

#### 1.2.4. Ma trận tương quan (Correlation Heatmap)

```python
# File: notebooks/01_eda.ipynb
corr = df.corr(numeric_only=True)
```

Các biến có tương quan mạnh nhất với `Diabetes_binary`:

| Thứ tự | Biến | Ý nghĩa |
|---:|---|---|
| 1 | `GenHlth` | Sức khỏe tổng quát |
| 2 | `HighBP` | Huyết áp cao |
| 3 | `BMI` | Chỉ số khối cơ thể |
| 4 | `DiffWalk` | Khó khăn đi bộ |
| 5 | `HighChol` | Cholesterol cao |
| 6 | `Age` | Nhóm tuổi |
| 7 | `HeartDiseaseorAttack` | Bệnh tim |

#### 1.2.5. Histogram các biến liên tục

```python
# File: notebooks/01_eda.ipynb
df[continuous_cols].hist(figsize=(12, 8), bins=30)
```

- **BMI:** Gần phân phối chuẩn, trung bình ~28.
- **MentHlth, PhysHlth:** Lệch phải mạnh, đa số giá trị = 0.
- **Age:** Phân phối đều các nhóm tuổi 1–13.

#### 1.2.6. Boxplot so sánh theo nhóm tiểu đường

```python
# File: notebooks/01_eda.ipynb
sns.boxplot(data=df, x="Diabetes_binary", y=col, ax=ax)
```

- Nhóm diabetic (label=1) có trung bình BMI, Age, GenHlth, PhysHlth **cao hơn**.
- Một số biến có outlier nhưng **không nên loại bỏ** vì có thể là ca thật có ý nghĩa y khoa.

#### 1.2.7. Tần suất các biến binary/ordinal

```python
# File: notebooks/01_eda.ipynb
for col in df.columns:
    print(df[col].value_counts().sort_index())
```

#### 1.2.8. Các biểu đồ EDA

| Biểu đồ | File | Mô tả |
|---|---|---|
| Phân phối target | `target_distribution.png` | Bar chart class 0 vs class 1 |
| Ma trận tương quan | `correlation_heatmap.png` | Heatmap tương quan 22×22 |
| Histogram liên tục | `continuous_histograms.png` | Histogram BMI, Age, MentHlth, PhysHlth |
| Boxplot | `boxplots.png` | Boxplot theo nhóm Diabetes_binary |

#### 1.2.9. Kết luận EDA

```
KẾT LUẬN EDA:
1. Dữ liệu sạch: 253,680 dòng, 22 cột, không có missing values.
2. Target mất cân bằng: ~86% non-diabetic, ~14% diabetic.
3. Features quan trọng: GenHlth, HighBP, BMI, DiffWalk, HighChol, Age.
4. BMI phân phối chuẩn, các biến khác chủ yếu là binary/ordinal.
5. Outlier không cần loại bỏ vì có ý nghĩa y khoa.
```

### 1.3. Mô tả dữ liệu

#### 1.3.1. Nguồn dữ liệu

- **Dataset:** Diabetes Binary Health Indicators BRFSS 2015.
- **Nguồn:** CDC — Behavioral Risk Factor Surveillance System (BRFSS).
  - Chương trình khảo sát sức khỏe lớn nhất qua điện thoại tại Mỹ.
  - Thu thập dữ liệu về yếu tố nguy cơ hành vi và tình trạng sức khỏe.
- **Link:** https://www.cdc.gov/brfss/
- **Đường dẫn:** `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

```python
# File: src/data/load_data.py
import pandas as pd
from src.utils.constants import RAW_DATA_PATH

def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)

# Sử dụng:
df = load_raw_data()
print(f"Số dòng: {len(df)}")        # 253,680
print(f"Số cột: {len(df.columns)}")  # 22
```

#### 1.3.2. Mô tả tổng quan

| Chỉ tiêu | Giá trị |
|---|---:|
| Số dòng | 253,680 |
| Số cột | 22 |
| Số thuộc tính đầu vào | 21 |
| Số biến mục tiêu | 1 |
| Giá trị thiếu | 0 |

#### 1.3.3. Mô tả các thuộc tính đầu vào

Danh sách 21 thuộc tính đầu vào (theo `src/utils/constants.py`):

```python
RAW_FEATURE_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",
]
```

| Tên cột | Mô tả | Kiểu | Giá trị |
|---|---|---|---|
| `HighBP` | Huyết áp cao | Binary | 0, 1 |
| `HighChol` | Cholesterol cao | Binary | 0, 1 |
| `CholCheck` | Kiểm tra cholesterol trong 5 năm | Binary | 0, 1 |
| `BMI` | Chỉ số khối cơ thể | Continuous | Số thực |
| `Smoker` | Hút thuốc >= 100 điếu | Binary | 0, 1 |
| `Stroke` | Từng bị đột quỵ | Binary | 0, 1 |
| `HeartDiseaseorAttack` | Bệnh tim / nhồi máu cơ tim | Binary | 0, 1 |
| `PhysActivity` | Tập thể dục trong 30 ngày | Binary | 0, 1 |
| `Fruits` | Ăn trái cây >= 1 lần/ngày | Binary | 0, 1 |
| `Veggies` | Ăn rau >= 1 lần/ngày | Binary | 0, 1 |
| `HvyAlcoholConsump` | Uống rượu nặng | Binary | 0, 1 |
| `AnyHealthcare` | Có bảo hiểm y tế | Binary | 0, 1 |
| `NoDocbcCost` | Không đi khám do chi phí | Binary | 0, 1 |
| `GenHlth` | Sức khỏe tổng quát | Ordinal | 1–5 |
| `MentHlth` | Số ngày sức khỏe tinh thần không tốt (30 ngày) | Continuous | 0–30 |
| `PhysHlth` | Số ngày sức khỏe thể chất không tốt (30 ngày) | Continuous | 0–30 |
| `DiffWalk` | Khó khăn khi đi bộ / leo cầu thang | Binary | 0, 1 |
| `Sex` | Giới tính | Binary | 0 (nữ), 1 (nam) |
| `Age` | Nhóm tuổi | Ordinal | 1–13 |
| `Education` | Trình độ học vấn | Ordinal | 1–6 |
| `Income` | Thu nhập hộ gia đình | Ordinal | 1–8 |

**Biến mục tiêu:**

| Tên cột | Mô tả | Kiểu | Giá trị |
|---|---|---|---|
| `Diabetes_binary` | Tình trạng tiểu đường | Binary | 0 (không), 1 (có) |

#### 1.3.4. Phân phối nhãn

| Giá trị | Ý nghĩa | Số dòng | Tỷ lệ |
|---:|---|---:|---:|
| 0 | Không bị tiểu đường | 218,334 | 86.07% |
| 1 | Bị tiểu đường | 35,346 | 13.93% |

**Nhận xét:** Dataset mất cân bằng lớp nghiêm trọng. Tỷ lệ 6.18:1. Cần dùng class_weight/scale_pos_weight và các metric phù hợp (ROC-AUC, F1, Recall) thay vì Accuracy.

---

## Chương II: Bài toán phân loại

### 2.1. Giới thiệu chung

#### 2.1.1. Định nghĩa bài toán

- **Loại bài toán:** Phân loại nhị phân (Binary Classification).
- **Mục tiêu:** Dự đoán nguy cơ mắc tiểu đường (0: không, 1: có) dựa trên 21 thuộc tính sức khỏe và nhân khẩu học.
- **Đầu vào:** 21 thuộc tính về huyết áp, cholesterol, BMI, tiền sử bệnh lý, thói quen sinh hoạt, nhân khẩu học.
- **Đầu ra:** Xác suất dự đoán (0–1) và nhãn phân loại.

#### 2.1.2. Mô hình bài toán

```
Đầu vào (21 features)
       ↓
Mô hình học máy
(Logistic Regression / Random Forest / XGBoost)
       ↓
Xác suất P(y=1|X) + Nhãn dự đoán (0 hoặc 1)
```

### 2.2. Lý do sử dụng Data Mining

#### 2.2.1. Thực trạng bệnh tiểu đường

Theo WHO và IDF:
- **537 triệu người** mắc tiểu đường năm 2021.
- Dự kiến **783 triệu người** vào năm 2045.
- **6.7 triệu ca tử vong** năm 2021.

#### 2.2.2. Tầm quan trọng của dự đoán sớm

- Phát hiện sớm giúp ngăn ngừa biến chứng: bệnh tim, suy thận, mù lòa.
- Giảm chi phí điều trị so với phát hiện muộn.
- Người dân chủ động thay đổi lối sống phòng bệnh.

#### 2.2.3. Ứng dụng Data Mining trong y tế

- Đánh giá nguy cơ tiểu đường nhanh chóng, chính xác.
- Xây dựng công cụ hỗ trợ quyết định lâm sàng (Clinical Decision Support System).
- Tự động hóa sàng lọc bệnh nhân có nguy cơ cao.

### 2.3. Thuật toán sử dụng

> Chi tiết từng thuật toán xem trong `ALGORITHMS.md`.

Project sử dụng **3 thuật toán** để so sánh và chọn model tốt nhất:

| # | Thuật toán | Phân loại | Mục đích |
|---|---|---|---|
| 1 | **Logistic Regression** | Tuyến tính | Baseline, dễ giải thích |
| 2 | **Random Forest** | Ensemble (Bagging) | Bắt quan hệ phi tuyến tính |
| 3 | **XGBoost** | Ensemble (Boosting) | **Model chính triển khai** |

#### 2.3.1. Logistic Regression

- **Loại:** Linear Classifier.
- **Hàm sigmoid:** P(y=1|X) = 1 / (1 + exp(-(w·X + b)))
- **Hàm mất mát:** Binary Cross-Entropy.
- **Ưu điểm:** Dễ giải thích, nhanh, có xác suất calibrated.
- **Nhược điểm:** Không bắt được quan hệ phi tuyến tính phức tạp.
- **Cấu hình:**

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
)
```

#### 2.3.2. Random Forest

- **Loại:** Ensemble — Bagging với Decision Trees.
- **Nguyên lý:** Nhiều Decision Trees huấn luyện trên tập con bootstrap, tổng hợp bằng majority voting.
- **Ưu điểm:** Bắt quan hệ phi tuyến, có Feature Importance tự nhiên.
- **Nhược điểm:** Huấn luyện chậm, khó giải thích chi tiết.
- **Cấu hình:**

```python
RandomForestClassifier(
    class_weight="balanced",
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
```

#### 2.3.3. XGBoost (Gradient Boosting)

- **Loại:** Ensemble — Gradient Boosting.
- **Nguyên lý:** Xây dựng tuần tự các Decision Trees, mỗi tree tiếp theo học từ sai số của các trees trước.
- **Cải tiến:** Regularization L1/L2, xử lý missing values tự động.
- **Ưu điểm:** ROC-AUC cao nhất, Feature Importance + SHAP, regularized.
- **Nhược điểm:** Huấn luyện chậm hơn, cần tuning.
- **Cấu hình:**

```python
scale_pos_weight = len(y_train) / (2 * np.bincount(y_train))[1]
# → ~6.1770

XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
)
```

---

## Chương III: Tiền xử lý dữ liệu

### 3.1. Feature Engineering

> Tạo các thuộc tính mới nhằm tăng khả năng dự đoán của mô hình.

#### 3.1.1. Tạo thuộc tính `BMI_category`

- **Phương pháp:** Rời rạc hóa (Binning) bằng `pd.cut()`, chia BMI thành 4 nhóm theo chuẩn WHO.
- **Công dụng:** Biểu diễn BMI theo nhóm thể trạng thay vì giá trị liên tục, giúp model dễ học.

| Khoảng BMI | Nhãn |
|---|---|
| (0, 18.5] | "thiếu cân" |
| (18.5, 25] | "bình thường" |
| (25, 30] | "thừa cân" |
| (30, +∞) | "béo phì" |

```python
# File: src/data/preprocess.py
data["BMI_category"] = pd.cut(
    data["BMI"],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=["thiếu cân", "bình thường", "thừa cân", "béo phì"],
    include_lowest=True,
)
```

#### 3.1.2. Tạo thuộc tính `comorbidity_score`

- **Phương pháp:** Tổng hợp số lượng bệnh lý đồng mắc (comorbidity count).
- **Công dụng:** Tóm tắt mức độ bệnh nền của một người — yếu tố dự báo quan trọng cho tiểu đường.

| Cột thành phần | Ý nghĩa |
|---|---|
| `HighBP` | Huyết áp cao |
| `HighChol` | Cholesterol cao |
| `Stroke` | Đột quỵ |
| `HeartDiseaseorAttack` | Bệnh tim |
| `DiffWalk` | Khó khăn đi bộ |

- **Kết quả:** Cột mới có giá trị **0–5** (tổng số bệnh đồng mắc).

```python
# File: src/data/preprocess.py
comorbidity_cols = [
    "HighBP", "HighChol", "Stroke",
    "HeartDiseaseorAttack", "DiffWalk",
]
data["comorbidity_score"] = data[comorbidity_cols].sum(axis=1)
```

#### 3.1.3. Tạo thuộc tính `healthy_lifestyle`

- **Phương pháp:** Tổng hợp 5 yếu tố lối sống lành mạnh.
- **Công dụng:** Tóm tắt mức độ lối sống tốt của một người.

| Cột thành phần | Ý nghĩa | Hệ số |
|---|---|---|
| `PhysActivity` | Tập thể dục | +1 nếu có |
| `Fruits` | Ăn trái cây | +1 nếu có |
| `Veggies` | Ăn rau | +1 nếu có |
| `Smoker` | Hút thuốc | +1 nếu KHÔNG hút |
| `HvyAlcoholConsump` | Uống rượu nặng | +1 nếu KHÔNG uống nặng |

- **Kết quả:** Cột mới có giá trị **0–5** (số thói quen lành mạnh).

```python
# File: src/data/preprocess.py
data["healthy_lifestyle"] = (
    data["PhysActivity"]
    + data["Fruits"]
    + data["Veggies"]
    + (1 - data["Smoker"])
    + (1 - data["HvyAlcoholConsump"])
)
```

### 3.2. Feature Transformation

> Chuẩn hóa và mã hóa các thuộc tính để phù hợp với thuật toán học máy.

#### 3.2.1. Kiểm tra chất lượng dữ liệu (Data Quality Check)

Trước khi biến đổi, cần kiểm tra và đánh giá chất lượng dữ liệu:

```python
# File: src/data/preprocess.py
def validate_data_quality(df):
    """Kiểm tra toàn diện chất lượng dữ liệu."""
    warnings = {}
    errors = []
    
    # 1. Kiểm tra Missing Values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        warnings['missing_values'] = missing_cols.to_dict()
    
    # 2. Kiểm tra giá trị không hợp lệ (binary columns)
    invalid_binary = {}
    for col in BINARY_COLS:
        if col in df.columns:
            values = set(df[col].dropna().unique())
            invalid = sorted(values - {0, 1, 0.0, 1.0})
            if invalid:
                invalid_binary[col] = invalid
    if invalid_binary:
        warnings['invalid_binary_values'] = invalid_binary
    
    # 3. Kiểm tra giá trị không hợp lệ (ordinal columns)
    invalid_ordinal = {}
    for col, (min_val, max_val) in ORDINAL_RANGES.items():
        if col in df.columns:
            invalid_count = (~df[col].between(min_val, max_val)).sum()
            if invalid_count > 0:
                invalid_ordinal[col] = int(invalid_count)
    if invalid_ordinal:
        warnings['invalid_ordinal_values'] = invalid_ordinal
    
    # 4. Kiểm tra Outliers (IQR method)
    outlier_info = {}
    for col in ['BMI', 'MentHlth', 'PhysHlth', 'Age']:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                outlier_info[col] = {
                    'count': int(outliers),
                    'percentage': float(f'{outliers / len(df) * 100:.2f}'),
                    'lower': float(lower),
                    'upper': float(upper)
                }
    if outlier_info:
        warnings['outliers'] = outlier_info
    
    if errors:
        raise DataValidationError("; ".join(errors))
    
    return warnings
```

**Kết quả kiểm tra trên dữ liệu gốc:**

| Kiểm tra | Kết quả |
|---|---|
| Missing Values | Không có (0 cột) |
| Giá trị không hợp lệ (binary) | Một số cột chứa giá trị ngoài {0, 1} |
| Giá trị không hợp lệ (ordinal) | Một số cột ngoài khoảng cho phép |
| Outliers | Có trên BMI, MentHlth, PhysHlth, Age |

#### 3.2.2. Làm sạch dữ liệu (Data Cleaning)

Thực hiện làm sạch dữ liệu toàn diện trong một pipeline duy nhất:

```python
# File: src/data/clean.py
def clean_data(df: pd.DataFrame, remove_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    stats = {
        "original_rows": len(df),
        "duplicates_removed": 0,
        "nulls_filled": {},
        "invalid_binary_filled": {},
        "ordinal_clipped": {},
        "outliers_clipped": {},
    }
    df = df.copy()
    
    # Bước 1: Xóa dòng trùng lặp
    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        stats["duplicates_removed"] = before - len(df)
    
    # Bước 2: Xử lý giá trị không hợp lệ trên cột binary
    for col in BINARY_COLS:
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
        invalid_mask = ~df[col].isin([0, 1, 0.0, 1.0])
        count = invalid_mask.sum()
        if count > 0:
            df.loc[invalid_mask, col] = mode_val
            stats["invalid_binary_filled"][col] = int(count)
    
    # Bước 3: Điền giá trị thiếu (missing values)
    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].isnull().any():
            if col in BINARY_COLS or col in ORDINAL_RANGES:
                fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            else:
                fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            stats["nulls_filled"][col] = int(df[col].isnull().sum())
    
    # Bước 4: Clip cột ordinal vào khoảng hợp lệ
    for col, (min_val, max_val) in ORDINAL_RANGES.items():
        if col not in df.columns:
            continue
        below = (df[col] < min_val).sum()
        above = (df[col] > max_val).sum()
        if below > 0 or above > 0:
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            stats["ordinal_clipped"][col] = {"below": int(below), "above": int(above)}
    
    # Bước 5: Clip outlier cho cột liên tục
    outlier_ranges = {
        "BMI": (10, 80),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
    }
    for col, (min_val, max_val) in outlier_ranges.items():
        if col not in df.columns:
            continue
        below = (df[col] < min_val).sum()
        above = (df[col] > max_val).sum()
        if below > 0 or above > 0:
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            stats["outliers_clipped"][col] = {"below": int(below), "above": int(above)}
    
    stats["final_rows"] = len(df)
    return df, stats
```

#### 3.2.3. Kiểm tra và xử lý dữ liệu trùng lặp

- **Kiểm tra:** Đếm số dòng trùng lặp hoàn toàn.
- **Xử lý:** Xóa các dòng trùng, chỉ giữ dòng đầu tiên.

```python
# File: src/data/clean.py
before = len(df)
df = df.drop_duplicates()
stats["duplicates_removed"] = before - len(df)
```

#### 3.2.4. Kiểm tra và xử lý dữ liệu thiếu (Missing Values)

Dữ liệu gốc không có missing values, nhưng pipeline vẫn xử lý phòng thủ khi upload dữ liệu mới:

| Loại cột | Phương pháp | Code |
|---|---|---|
| Binary | Mode (most frequent) | `df[col].mode().iloc[0]` |
| Ordinal | Mode | `df[col].mode().iloc[0]` |
| Continuous | Median | `df[col].median()` |

```python
# File: src/data/preprocess.py
def handle_missing_values(df):
    data = df.copy()
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if col in BINARY_COLS or col in ORDINAL_RANGES:
                data[col].fillna(data[col].mode()[0], inplace=True)
            elif col in CONTINUOUS_COLS:
                data[col].fillna(data[col].median(), inplace=True)
    return data
```

#### 3.2.5. Kiểm tra và xử lý giá trị không hợp lệ

- **Cột binary:** Thay giá trị ngoài {0, 1} bằng mode.
- **Cột ordinal:** Clip vào khoảng [min, max] cho phép.

| Cột | Khoảng hợp lệ |
|---|---|
| `GenHlth` | 1–5 |
| `Age` | 1–13 |
| `Education` | 1–6 |
| `Income` | 1–8 |

```python
# File: src/data/preprocess.py
def handle_invalid_values(df):
    data = df.copy()
    
    # Binary: thay giá trị không hợp lệ bằng NaN, sau đó fill bằng mode
    for col in BINARY_COLS:
        if col in data.columns:
            invalid_mask = ~data[col].isin([0, 1, 0.0, 1.0])
            if invalid_mask.sum() > 0:
                data.loc[invalid_mask, col] = np.nan
                data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Ordinal: clip vào khoảng hợp lệ
    for col, (min_val, max_val) in ORDINAL_RANGES.items():
        if col in data.columns:
            data[col] = data[col].clip(lower=min_val, upper=max_val)
    
    return data
```

#### 3.2.6. Kiểm tra và xử lý outlier

- **Phương pháp:** IQR (Interquartile Range) — giá trị ngoài [Q1 - 1.5×IQR, Q3 + 1.5×IQR].
- **Chiến lược:** Cap (gắn vào boundary) thay vì xóa row.

| Cột | Chiến lược | Chi tiết |
|---|---|---|
| `MentHlth` | Cap | Clip về [0, 30] |
| `PhysHlth` | Cap | Clip về [0, 30] |
| `BMI` | **Không xử lý** | Giá trị cao (VD: 98) vẫn hợp lệ về y khoa |

- **Lý do giữ nguyên BMI:** Trong dữ liệu y khoa, BMI > 40 vẫn là giá trị hợp lệ và có ý nghĩa dự báo. Loại bỏ outlier BMI có thể làm mất thông tin quan trọng về các trường hợp béo phì nặng — nhóm có nguy cơ tiểu đường cao nhất.

```python
# File: src/data/preprocess.py
def handle_outliers(df, strategy='cap'):
    data = df.copy()
    outlier_cols = ['MentHlth', 'PhysHlth']  # BMI removed - values like 98 are valid
    removed_indices = []
    
    for col in outlier_cols:
        if col in data.columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            outlier_mask = (data[col] < lower) | (data[col] > upper)
            
            if strategy == 'cap':
                data.loc[data[col] < lower, col] = lower
                data.loc[data[col] > upper, col] = upper
            elif strategy == 'remove':
                removed_indices.extend(data[outlier_mask].index.tolist())
    
    if strategy == 'remove' and removed_indices:
        data.drop(index=removed_indices, inplace=True)
    
    return data
```

#### 3.2.7. Chuẩn hóa dữ liệu bằng StandardScaler

- **Mục đích:** Đưa các biến liên tục về cùng scale (mean=0, std=1).
- **Các biến được scale:**

| Nhóm | Các cột | Lý do |
|---|---|---|
| Continuous gốc | `BMI`, `MentHlth`, `PhysHlth` | Scale khác nhau (BMI ~10-60, MentHlth ~0-30) |
| Ordinal | `Age` | Scale 1–13, cần normalize |
| Engineered numeric | `comorbidity_score`, `healthy_lifestyle` | Scale 0–5, cần normalize |

```python
# File: src/data/preprocess.py
from sklearn.preprocessing import StandardScaler

CONTINUOUS_COLS = [
    "BMI", "MentHlth", "PhysHlth", "Age",
    "comorbidity_score", "healthy_lifestyle",
]
```

#### 3.2.8. Mã hóa dữ liệu phân loại bằng OneHotEncoder

- **Biến được encode:** `BMI_category` (4 giá trị sau rời rạc hóa).
- **Kết quả:** Tạo 4 cột binary mới.

```python
# File: src/data/preprocess.py
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = ["BMI_category"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# BMI_category → BMI_category_thiếu cân, BMI_category_bình thường,
#               BMI_category_thừa cân, BMI_category_béo phì
```

#### 3.2.9. Giữ nguyên các thuộc tính binary và ordinal phù hợp

- **Binary (14 cột):** `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`.
- **Ordinal (3 cột):** `GenHlth` (1–5), `Education` (1–6), `Income` (1–8) — đã có thứ tự tự nhiên.

#### 3.2.10. Pipeline biến đổi đầy đủ (ColumnTransformer)

```python
# File: src/data/preprocess.py
from sklearn.compose import ColumnTransformer

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("continuous_scaler", StandardScaler(), CONTINUOUS_COLS),
            ("category_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="passthrough",  # Giữ nguyên các cột còn lại (binary + ordinal)
    )
```

**Tóm tắt Feature Transformation:**

| Nhóm | Các cột | Xử lý | Số cột sau xử lý |
|---|---|---|---|
| Continuous (6 cột) | BMI, MentHlth, PhysHlth, Age, comorbidity_score, healthy_lifestyle | StandardScaler | 6 |
| Categorical (1 cột) | BMI_category | OneHotEncoder | 4 |
| Binary (14 cột) | HighBP, HighChol, ..., Sex | Giữ nguyên | 14 |
| Ordinal (3 cột) | GenHlth, Education, Income | Giữ nguyên | 3 |
| **Tổng cộng** | **24 cột đầu vào** | — | **27 features** |

#### 3.2.11. Kiểm tra trùng lặp với dữ liệu hiện có

Khi upload dữ liệu mới trong hệ thống, cần kiểm tra trùng lặp với dữ liệu đã có:

```python
# File: src/data/clean.py
def check_duplicates_with_existing(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    key_columns: list = None
) -> dict:
    if key_columns is None:
        key_columns = RAW_FEATURE_COLUMNS.copy()
    
    available_keys = [c for c in key_columns if c in new_df.columns and c in existing_df.columns]
    if not available_keys:
        available_keys = RAW_FEATURE_COLUMNS.copy()
    
    # Hash-based comparison
    new_hashes = new_df[available_keys].apply(lambda x: hash(tuple(x)), axis=1)
    existing_hashes = existing_df[available_keys].apply(lambda x: hash(tuple(x)), axis=1)
    
    duplicates = new_hashes.isin(existing_hashes)
    
    return {
        "total_new_rows": len(new_df),
        "duplicate_rows": int(duplicates.sum()),
        "unique_new_rows": int(len(new_df) - duplicates.sum()),
        "duplicate_percent": round(duplicates.sum() / len(new_df) * 100, 2),
        "key_columns_used": available_keys,
    }
```

#### 3.2.12. Gộp dữ liệu (Merge Datasets)

```python
# File: src/data/clean.py
def merge_datasets(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    strategy: str = "keep_new"
) -> pd.DataFrame:
    """
    Gộp dữ liệu hiện có với dữ liệu mới.
    strategy: "keep_new" (chỉ giữ dòng mới), "keep_existing", "keep_all"
    """
    # Hash-based duplicate detection
    key_columns = RAW_FEATURE_COLUMNS.copy()
    available_keys = [c for c in key_columns if c in existing_df.columns and c in new_df.columns]
    
    existing_keys = existing_df[available_keys].apply(lambda x: hash(tuple(x)), axis=1)
    new_keys = new_df[available_keys].apply(lambda x: hash(tuple(x)), axis=1)
    is_duplicate = new_keys.isin(existing_keys)
    
    if strategy == "keep_new":
        unique_new = new_df[~is_duplicate]
        return pd.concat([existing_df, unique_new], ignore_index=True)
    elif strategy == "keep_existing":
        unique_new = new_df[~is_duplicate]
        return pd.concat([existing_df, unique_new], ignore_index=True)
    elif strategy == "keep_all":
        return pd.concat([existing_df, new_df], ignore_index=True)
```

#### 3.2.13. Kiểm tra và xử lý mất cân bằng lớp (Class Imbalance)

- **Tỷ lệ:** ~86% class 0, ~14% class 1 (ratio 6.18:1).
- **Phương pháp xử lý:**

| Model | Tham số | Giá trị |
|---|---|---|
| Logistic Regression | `class_weight` | `"balanced"` |
| Random Forest | `class_weight` | `"balanced"` |
| XGBoost | `scale_pos_weight` | `len(y) / (2 × count(class=1))` ≈ 6.18 |

```python
# File: notebooks/02_preprocessing_modeling.ipynb
# Class weight cho LR và RF
LogisticRegression(class_weight="balanced", ...)
RandomForestClassifier(class_weight="balanced", ...)

# scale_pos_weight cho XGBoost
scale_pos_weight = len(y_train) / (2 * np.bincount(y_train))[1]
# → 6.1770

XGBClassifier(scale_pos_weight=scale_pos_weight, ...)
```

- **SMOTE minh họa:** Có minh họa SMOTE trong notebook nhưng không sử dụng trong pipeline chính vì class_weight đã đủ hiệu quả.

#### 3.2.14. Tách features và target

```python
# File: src/data/preprocess.py
def split_features_target(df, clean=True, handle_outliers_strategy='cap'):
    if clean:
        cleaned_df, validation_warnings = clean_data(df, handle_outliers_strategy)
        engineered = add_engineered_features(cleaned_df)
    else:
        engineered = add_engineered_features(df)
        validation_warnings = {}
    
    X = engineered.drop(columns=[TARGET_COL])
    y = engineered[TARGET_COL].astype(int)
    
    return X, y, validation_warnings
```

#### 3.2.15. Feature Selection và giảm chiều (Notebook)

Thực hiện trong `notebooks/02_preprocessing_modeling.ipynb` để phân tích, không ảnh hưởng pipeline chính.

**ANOVA F-score:**

```python
# File: notebooks/02_preprocessing_modeling.ipynb
from sklearn.feature_selection import SelectKBest, f_classif

selector_f = SelectKBest(score_func=f_classif, k='all')
selector_f.fit(X_train_scaled, y_train)

feature_scores = pd.DataFrame({
    "feature": temp_feature_names,
    "f_score": selector_f.scores_,
    "p_value": selector_f.pvalues_
}).sort_values("f_score", ascending=False)
```

**Mutual Information:**

```python
# File: notebooks/02_preprocessing_modeling.ipynb
from sklearn.feature_selection import mutual_info_classif

mi_score_func = lambda X, y: mutual_info_classif(X, y, random_state=42)
selector_mi = SelectKBest(score_func=mi_score_func, k='all')
```

**PCA (Principal Component Analysis):**

```python
# File: notebooks/02_preprocessing_modeling.ipynb
from sklearn.decomposition import PCA

pca_95 = PCA(n_components=0.95)
X_train_pca = pca_95.fit_transform(X_train_scaled)
# → Từ 27 features → 14 principal components (giữ lại 95.55% phương sai)
```

### 3.3. Train/Test Split

#### 3.3.1. Phương pháp Stratified Sampling

- **Tỷ lệ:** 80% train / 20% test.
- **Random state:** 42 (đảm bảo tái lập kết quả).
- **Stratify:** Giữ nguyên tỷ lệ lớp trong cả train và test.

```python
# File: notebooks/02_preprocessing_modeling.ipynb
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

**Kết quả:**

| Tập | Số dòng | % Class 1 |
|---|---:|---:|
| Train | 202,944 | 13.93% |
| Test | 50,736 | 13.93% |

→ Stratified Sampling giữ nguyên tỷ lệ lớp 1 ở cả train và test.

#### 3.3.2. Lưu dữ liệu sau tiền xử lý

Dữ liệu sau tiền xử lý được lưu thành **4 file CSV** tại `data/processed/`:

| File | Nội dung |
|---|---|
| `X_train.csv` | 24 cột features của tập train |
| `X_test.csv` | 24 cột features của tập test |
| `y_train.csv` | Nhãn `Diabetes_binary` của tập train |
| `y_test.csv` | Nhãn `Diabetes_binary` của tập test |

#### 3.3.3. So sánh kích thước trước và sau tiền xử lý

| Giai đoạn | Số dòng | Số cột | Thay đổi |
|---|---:|---:|---|
| Dataset gốc | 253,680 | 22 | — |
| Sau Feature Engineering | 253,680 | 25 | +3 cột mới |
| X_train | 202,944 | 24 | Tách bỏ target |
| X_test | 50,736 | 24 | Tách bỏ target |
| y_train | 202,944 | 1 | Chỉ cột target |
| y_test | 50,736 | 1 | Chỉ cột target |

---

## Chương IV: Mô hình và Thực nghiệm

### 4.1. Xây dựng thuật toán

#### 4.1.1. Cấu trúc Pipeline huấn luyện

Mỗi model được đóng gói trong `Pipeline` gồm 2 bước:

```python
# File: notebooks/02_preprocessing_modeling.ipynb
Pipeline(steps=[
    ("preprocessor", build_preprocessor()),  # ColumnTransformer
    ("classifier", <model>)
])
```

**Ưu điểm:**
- Đảm bảo dữ liệu train và test được xử lý **hoàn toàn giống nhau**.
- Preprocessor chỉ fit trên train data, transform trên cả train và test → tránh data leakage.
- Khi predict ở production, toàn bộ preprocessing được tự động áp dụng.

#### 4.1.2. Pipeline đầy đủ cho từng model

```python
# File: notebooks/02_preprocessing_modeling.ipynb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Logistic Regression
("classifier", LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
))

# 2. Random Forest
("classifier", RandomForestClassifier(
    class_weight="balanced",
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
))

# 3. XGBoost
scale_pos_weight = len(y_train) / (2 * np.bincount(y_train))[1]
("classifier", XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
))
```

#### 4.1.3. Cross-Validation cho XGBoost

```python
# File: notebooks/02_preprocessing_modeling.ipynb
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_model = XGBClassifier(...)

scores = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring='roc_auc')
# → Trung bình 5-fold CV ROC-AUC
```

#### 4.1.4. Cách lưu model và feature columns

| File | Mô tả |
|---|---|
| `models/xgboost.joblib` | Full pipeline (preprocessor + XGBoost) |
| `models/feature_columns.joblib` | Danh sách 24 cột features |

```python
# File: backend/app.py
import joblib

MODEL_PATHS = {"xgboost": MODELS_DIR / "xgboost.joblib"}
feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")

# Dự đoán
X_input = prepare_prediction_frame(input_data).reindex(columns=feature_columns)
model = joblib.load(MODEL_PATHS["xgboost"])
probability = model.predict_proba(X_input)[:, 1][0]
```

### 4.2. Đánh giá mô hình

#### 4.2.1. Các chỉ số đánh giá

Do class imbalance, **Accuracy không phải metric phù hợp**. Project sử dụng:

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Không dùng làm metric chính |
| **Precision** | TP / (TP + FP) | Tỷ lệ dự đoán đúng trong số dự đoán là tiểu đường |
| **Recall** | TP / (TP + FN) | Tỷ lệ phát hiện đúng các ca tiểu đường |
| **F1-score** | 2 × (P × R) / (P + R) | Trung bình điều hòa Precision và Recall |
| **ROC-AUC** | Diện tích dưới ROC Curve | **Metric chính** — khả năng phân biệt |
| **Confusion Matrix** | TN, FP, FN, TP | Chi tiết các loại dự đoán đúng/sai |

#### 4.2.2. Confusion Matrix

|  | Dự đoán 0 | Dự đoán 1 |
|---|---|---|
| **Thực tế 0** | TN | FP |
| **Thực tế 1** | FN | TP |

**Ý nghĩa trong bài toán y tế:**
- **FN (False Negative):** Bỏ sót ca tiểu đường — **nguy hiểm nhất**, cần ưu tiên giảm thiểu.
- **FP (False Positive):** Cảnh báo nhầm — gây lo lắng không cần thiết nhưng ít nghiêm trọng hơn FN.

#### 4.2.3. ROC Curve

```python
# File: notebooks/03_model_explainability.ipynb
from sklearn.metrics import roc_curve, auc

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} AUC = {roc_auc:.4f}")
```

- Đường chéo (AUC=0.5) là random classifier.
- Đường cong càng góc trên trái → model càng tốt.

#### 4.2.4. Precision-Recall Curve

```python
# File: notebooks/03_model_explainability.ipynb
from sklearn.metrics import precision_recall_curve, average_precision_score

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
```

- Phù hợp hơn ROC cho class imbalance nghiêm trọng.
- Khi recall tăng → precision giảm (đặc trưng của class imbalance).

#### 4.2.5. Feature Importance

```python
# File: notebooks/03_model_explainability.ipynb
rf_features = get_transformed_feature_names(models["random_forest"])
rf_importance = models["random_forest"].named_steps["classifier"].feature_importances_

xgb_features = get_transformed_feature_names(models["xgboost"])
xgb_importance = models["xgboost"].named_steps["classifier"].feature_importances_
```

#### 4.2.6. SHAP Explainability cho XGBoost

```python
# File: notebooks/03_model_explainability.ipynb
import shap

explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_names)
```

- Giải thích từng dự đoán cụ thể.
- Hiểu hướng ảnh hưởng (tăng/giảm nguy cơ) của mỗi feature.

#### 4.2.7. Threshold Tuning

```python
# File: notebooks/03_model_explainability.ipynb
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for thresh in thresholds:
    y_pred = (y_proba_xgb >= thresh).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
```

- Threshold thấp → Recall cao, Precision thấp (phát hiện nhiều ca).
- Threshold cao → Precision cao, Recall thấp (ít cảnh báo nhầm).
- **Default threshold 0.5** là điểm cân bằng.

---

## Chương V: Kết quả

### 5.1. Kết quả đánh giá

#### 5.1.1. Bảng so sánh các mô hình

Kết quả đánh giá trên tập test (50,736 mẫu):

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7323 | 0.3124 | **0.7673** | 0.4441 | 0.8208 |
| Random Forest | 0.7849 | **0.3509** | 0.6396 | **0.4531** | 0.8182 |
| **XGBoost** | 0.7210 | 0.3061 | 0.7913 | 0.4414 | **0.8266** |

**Nhận xét:**
- **XGBoost** đạt **ROC-AUC cao nhất (0.8266)** → được chọn làm model triển khai.
- **Random Forest** đạt F1 và Precision cao nhất.
- **Logistic Regression** đạt Recall cao nhất.
- Accuracy không phản ánh đúng chất lượng model do class imbalance.

#### 5.1.2. Confusion Matrix từng mô hình

**XGBoost (model triển khai):**

|  | Dự đoán 0 | Dự đoán 1 |
|---|---|---|
| **Thực tế 0** | TN = 30,985 | FP = 12,682 |
| **Thực tế 1** | FN = 1,475 | TP = 5,594 |

**Random Forest:**

|  | Dự đoán 0 | Dự đoán 1 |
|---|---|---|
| **Thực tế 0** | TN = 35,303 | FP = 8,364 |
| **Thực tế 1** | FN = 2,548 | TP = 4,521 |

**So sánh:**
- XGBoost có **FN thấp hơn** (1,475 vs 2,548) → phát hiện nhiều ca tiểu đường hơn.
- Random Forest có **FP thấp hơn** (8,364 vs 12,682) → ít cảnh báo nhầm hơn.

#### 5.1.3. Phân tích Feature Importance

**XGBoost — Top 10 features quan trọng:**

| Thứ tự | Feature | Importance |
|---:|---|---:|
| 1 | `comorbidity_score` | 0.3152 |
| 2 | `HighBP` | 0.2732 |
| 3 | `BMI_category_obese` | 0.0992 |
| 4 | `GenHlth` | 0.0906 |
| 5 | `BMI_category_normal` | 0.0351 |
| 6 | `Age` | 0.0277 |
| 7 | `BMI` | 0.0270 |
| 8 | `HvyAlcoholConsump` | 0.0205 |
| 9 | `BMI_category_overweight` | 0.0191 |
| 10 | `CholCheck` | 0.0172 |

**Random Forest — Top 10 features quan trọng:**

| Thứ tự | Feature | Importance |
|---:|---|---:|
| 1 | `comorbidity_score` | 0.1539 |
| 2 | `GenHlth` | 0.1438 |
| 3 | `BMI` | 0.1010 |
| 4 | `Age` | 0.0957 |
| 5 | `HighBP` | 0.0799 |
| 6 | `Income` | 0.0492 |
| 7 | `PhysHlth` | 0.0434 |
| 8 | `HighChol` | 0.0431 |
| 9 | `MentHlth` | 0.0345 |
| 10 | `Education` | 0.0310 |

**Nhận xét:**
- `comorbidity_score` (feature do project tạo) là **feature quan trọng nhất** ở cả 2 model → Feature Engineering thành công.
- `BMI_category_obese` đứng thứ 3 trong XGBoost → việc rời rạc hóa BMI có giá trị.
- Top features đều thuộc nhóm chỉ số chuyển hóa (huyết áp, BMI) và sức khỏe tổng quát.

#### 5.1.4. Kết quả SHAP (XGBoost)

- **SHAP Summary Plot:** Hiển thị phân bố SHAP values cho từng feature.
- **SHAP Bar Plot:** Feature importance dựa trên SHAP values.
- Giúp giải thích **hướng ảnh hưởng** (tăng/giảm nguy cơ) của mỗi feature cho từng prediction.

#### 5.1.5. Kết quả Threshold Tuning

| Threshold | Precision | Recall | F1 | Nhận xét |
|---:|---:|---:|---:|---|
| 0.3 | Thấp | Cao | Trung bình | Phát hiện nhiều ca, nhiều cảnh báo nhầm |
| 0.4 | Trung bình | Trung bình-cao | Tốt | Cân bằng |
| **0.5** | **Trung bình** | **Cao** | **Tốt nhất** | **Default** |
| 0.6 | Khá | Trung bình | Trung bình | Ít cảnh báo nhầm |
| 0.7 | Cao | Thấp | Thấp | Chỉ khi rất chắc chắn |

→ Threshold 0.4–0.5 là phù hợp cho bài toán y tế.

### 5.2. Kịch bản báo cáo

#### 5.2.1. Hệ thống demo web

**Backend Flask (port 5000):**

| Endpoint | Mô tả |
|---|---|
| `GET /health` | Health check |
| `POST /predict` | Dự đoán 1 mẫu |
| `POST /predict-csv` | Dự đoán batch từ CSV |
| `POST /upload-new-data` | Upload CSV (không retrain) |
| `POST /upload-train` | Upload CSV + retrain |
| `GET /retrain-status` | Kiểm tra tiến trình retrain |
| `GET /retrain-details` | Chi tiết kết quả retrain |
| `POST /retrain` | Retrain từ new_data.csv |
| `POST /apply-new-model` | Admin phê duyệt model mới |
| `GET /retrain-history` | Lịch sử retrain |
| `GET /data-stats` | Thống kê dữ liệu |
| `GET /new-data-stats` | Thống kê dữ liệu mới |
| `GET /model-stats` | Metrics & curves |

**Frontend (port 8000):**

| Trang | File | Mô tả |
|---|---|---|
| Dự đoán | `index.html` | Nhập 21 features, xem kết quả |
| Biểu đồ | `charts.html` | 5 tab: Metrics, ROC, PR, Importance, Confusion |
| Admin | `admin.html` | Thống kê, upload, retrain, so sánh model |
| Upload | `upload.html` | Upload CSV + retrain |

#### 5.2.2. Luồng dự đoán

```
1. User nhập 21 thuộc tính trên frontend
       ↓
2. Frontend gửi POST /predict đến backend
       ↓
3. Backend kiểm tra đủ 21 features
       ↓
4. Backend sinh thêm 3 features: BMI_category, comorbidity_score, healthy_lifestyle
       ↓
5. Backend reindex đúng thứ tự 24 cột theo feature_columns.joblib
       ↓
6. Model pipeline (preprocessor + XGBoost) dự đoán xác suất
       ↓
7. Backend trả về: probability, label, label_text
       ↓
8. Frontend hiển thị kết quả với thanh tiến trình
```

#### 5.2.3. Luồng Retrain

```
1. Admin upload CSV có nhãn qua /upload-train
       ↓
2. Backend validate schema → clean data → check duplicates
       ↓
3. Backend gộp dữ liệu mới với data gốc (keep_new strategy)
       ↓
4. Backend lưu backup dataset version
       ↓
5. Backend lưu backup model versions
       ↓
6. Backend train model mới trên dữ liệu gộp
       ↓
7. Backend đánh giá model mới trên cùng test set
       ↓
8. Backend so sánh: old_metrics vs new_metrics
       ↓
9. Backend khuyến nghị: can_deploy = True/False
       ↓
10. Admin quyết định: cập nhật model mới hoặc giữ model cũ
```

---

## Chương VI: Kết luận

### 6.1. Tóm tắt kết quả đạt được

1. **Mô hình dự đoán:** XGBoost đạt ROC-AUC = 0.8266, khả năng phân biệt tốt giữa người có và không có nguy cơ tiểu đường.

2. **Feature Engineering hiệu quả:** `comorbidity_score` và `BMI_category` trở thành top features quan trọng nhất, chứng minh giá trị của việc tạo features mới.

3. **Hệ thống demo web đầy đủ:** 4 trang HTML, backend Flask REST API, cho phép dự đoán, xem biểu đồ, và quản trị.

4. **Cơ chế retrain hoàn chỉnh:** Upload dữ liệu mới, so sánh model cũ/mới, quyết định triển khai bởi admin.

5. **Module hóa code:** Logic tiền xử lý trong `src/data/` được dùng chung cho notebooks, training script và backend.

### 6.2. Hạn chế của đề tài

- **Dữ liệu là dữ liệu khảo sát, không phải dữ liệu lâm sàng trực tiếp:** Không thay thế được chẩn đoán y khoa; có thể có sai số do ký ức.
- **Dữ liệu mất cân bằng lớp:** 86%/14% ảnh hưởng đến precision của model.
- **Một số thuộc tính là thông tin tự báo cáo:** Có thể có sai lệch giữa thực tế và câu trả lời.
- **Mô hình chưa thay thế được chẩn đoán y khoa:** Chỉ mang tính hỗ trợ sàng lọc.

### 6.3. Hướng phát triển

- **Xử lý class imbalance nâng cao:** SMOTE, ADASYN, threshold tuning tự động.
- **Tối ưu hyperparameters:** GridSearchCV, RandomizedSearchCV, Bayesian Optimization (Optuna).
- **Giải thích mô hình nâng cao:** SHAP Interaction Values, LIME, phân tích subgroup.
- **Cải thiện giao diện:** Responsive design, dashboard admin, biểu đồ tương tác.
- **Giám sát model sau triển khai:** Model drift detection, A/B testing, logging.

---

## Tài liệu tham khảo

1. CDC. Behavioral Risk Factor Surveillance System (BRFSS) 2015. https://www.cdc.gov/brfss/
2. scikit-learn: Machine Learning in Python. https://scikit-learn.org/
3. XGBoost: Scalable and Flexible Gradient Boosting. https://xgboost.readthedocs.io/
4. SHAP (SHapley Additive exPlanations). https://shap.readthedocs.io/
5. Lundberg, S.M. and Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions.
6. Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
7. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
8. World Health Organization (WHO). Diabetes Fact Sheet, 2021.
9. International Diabetes Federation (IDF). IDF Diabetes Atlas, 10th Edition, 2021.
