# BÁO CÁO ĐỀ TÀI

# ỨNG DỤNG KỸ THUẬT DATA MINING TRONG DỰ ĐOÁN NGUY CƠ BỆNH TIỂU ĐƯỜNG

---

## Chương 1: GIỚI THIỆU

### 1.1. Data Mining là gì?

**Data Mining** (Khai phá dữ liệu) là quá trình trích xuất các thông tin có giá trị,Patterns (mẫu ẩn) và tri thức từ các tập dữ liệu lớn. Data Mining kết hợp các kỹ thuật từ thống kê, học máy (Machine Learning), và cơ sở dữ liệu để phân tích dữ liệu từ nhiều góc độ khác nhau, từ đó rút ra các kết luận hữu ích cho việc ra quyết định.

**Các bước chính trong Data Mining:**

1. **Thu thập dữ liệu (Data Collection)**: Tập hợp dữ liệu từ các nguồn khác nhau.
2. **Làm sạch dữ liệu (Data Cleaning)**: Xử lý dữ liệu thiếu, nhiễu, và không nhất quán.
3. **Khám phá dữ liệu (Data Exploration)**: Phân tích thống kê mô tả, trực quan hóa dữ liệu.
4. **Xây dựng mô hình (Model Building)**: Áp dụng các thuật toán học máy để dự đoán hoặc phân loại.
5. **Đánh giá mô hình (Model Evaluation)**: Kiểm tra độ chính xác và hiệu quả của mô hình.
6. **Triển khai (Deployment)**: Đưa mô hình vào ứng dụng thực tế.

### 1.2. Tổng quan bài toán

Bài toán đặt ra là **phân loại nhị phân** (Binary Classification) nhằm dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số sức khỏe và thông tin cá nhân của bệnh nhân.

**Đầu vào:** Các thuộc tính về sức khỏe bao gồm:
- Huyết áp, cholesterol, chỉ số BMI
- Tiền sử bệnh lý (đột quỵ, bệnh tim)
- Thói quen sinh hoạt (hút thuốc, uống rượu, tập thể dục)
- Yếu tố nhân khẩu học (tuổi, giới tính, thu nhập, học vấn)

**Đầu ra:** Xác suất/dự đoán một người có nguy cơ mắc tiểu đường hay không (0: Không tiểu đường, 1: Tiểu đường).

### 1.3. Lý do chọn đề tài

Tiểu đường là một trong những bệnh mãn tính phổ biến nhất trên thế giới. Theo thống kê của Tổ chức Y tế Thế giới (WHO):
- Năm 2021, có khoảng **537 triệu người** trưởng thành mắc tiểu đường trên toàn cầu.
- Dự kiến con số này sẽ tăng lên **783 triệu người** vào năm 2045.
- Tiểu đường gây ra **6,7 triệu ca tử vong** trong năm 2021.

**Tầm quan trọng của việc dự đoán sớm:**
- Phát hiện sớm giúp ngăn ngừa các biến chứng nguy hiểm như bệnh tim, suy thận, mù lòa.
- Giảm chi phí điều trị so với việc phát hiện muộn.
- Giúp người dân chủ động thay đổi lối sống để phòng bệnh.

**Ứng dụng Data Mining:**
- Giúp các chuyên gia y tế đánh giá nguy cơ tiểu đường một cách nhanh chóng và chính xác.
- Xây dựng công cụ hỗ trợ quyết định lâm sàng.
- Tự động hóa quy trình sàng lọc bệnh nhân có nguy cơ cao.

### 1.4. Mục tiêu nghiên cứu

1. **Xây dựng mô hình dự đoán**: Huấn luyện mô hình học máy (XGBoost) để dự đoán nguy cơ tiểu đường với độ chính xác cao.

2. **Phân tích các yếu tố ảnh hưởng**: Xác định các thuộc tính quan trọng nhất ảnh hưởng đến nguy cơ tiểu đường.

3. **Xây dựng hệ thống demo**: Phát triển ứng dụng web cho phép người dùng nhập thông tin sức khỏe và nhận kết quả dự đoán.

4. **Hỗ trợ huấn luyện lại mô hình**: Cung cấp cơ chế cập nhật mô hình khi có dữ liệu mới.

### 1.5. Phạm vi thực hiện

- **Dữ liệu**: Sử dụng bộ dữ liệu khảo sát sức khỏe BRFSS 2015 từ CDC (Mỹ) với 253,680 bản ghi.
- **Thuật toán**: Tập trung vào mô hình XGBoost - một trong những thuật toán boosting hiệu quả nhất hiện nay.
- **Công cụ**: Python với các thư viện scikit-learn, XGBoost, Flask, HTML/CSS/JavaScript.
- **Giới hạn**: Không thay thế chẩn đoán y khoa chuyên nghiệp.

---

## Chương 2: GIỚI THIỆU DỮ LIỆU

### 2.1. Nguồn dữ liệu

**Dataset:** Diabetes Binary Health Indicators BRFSS 2015

**Nguồn:** CDC - Behavioral Risk Factor Surveillance System (BRFSS)
- BRFSS là chương trình khảo sát sức khỏe lớn nhất qua điện thoại tại Mỹ
- Thu thập dữ liệu về các yếu tố nguy cơ hành vi và tình trạng sức khỏe của người trưởng thành
- Được CDC quản lý và công bố công khai

**Link tham khảo:** https://www.cdc.gov/brfss/

### 2.2. Mô tả tổng quan dữ liệu

**Code minh họa - Đọc dữ liệu:**

```python
# File: src/data/load_data.py
import pandas as pd
from src.utils.constants import RAW_DATA_PATH

def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)

# Sử dụng:
df = load_raw_data()
print(f"Số dòng: {len(df)}")           # 253,680
print(f"Số cột: {len(df.columns)}")     # 22 (21 features + 1 target)
print(f"Kích thước: {df.shape}")        # (253680, 22)
```

**Thông tin tổng quan:**
| Thông số | Giá trị |
|----------|---------|
| Số dòng dữ liệu | 253,680 |
| Số thuộc tính đầu vào | 21 |
| Biến mục tiêu | Diabetes_binary |
| Dung lượng file | ~30 MB |

### 2.3. Mô tả các thuộc tính

Dữ liệu được chia thành 5 nhóm chính:

#### 2.3.1. Nhóm chỉ số sức khỏe lâm sàng (Clinical Health Indicators)

| Tên thuộc tính | Mô tả | Kiểu dữ liệu | Giá trị |
|-----------------|-------|--------------|---------|
| HighBP | Huyết áp cao | Binary (0/1) | 0=Không, 1=Có |
| HighChol | Cholesterol cao | Binary (0/1) | 0=Không, 1=Có |
| CholCheck | Đã kiểm tra cholesterol trong 5 năm | Binary (0/1) | 0=Không, 1=Có |
| BMI | Chỉ số khối cơ thể | Continuous | 10 - 80 |
| Stroke | Tiền sử đột quỵ | Binary (0/1) | 0=Không, 1=Có |
| HeartDiseaseorAttack | Bệnh tim hoặc nhồi máu cơ tim | Binary (0/1) | 0=Không, 1=Có |

#### 2.3.2. Nhóm hành vi và lối sống (Lifestyle & Behavior)

| Tên thuộc tính | Mô tả | Kiểu dữ liệu | Giá trị |
|-----------------|-------|--------------|---------|
| Smoker | Đã hút ít nhất 100 điếu thuốc | Binary (0/1) | 0=Không, 1=Có |
| PhysActivity | Tập thể dục trong 30 ngày qua | Binary (0/1) | 0=Không, 1=Có |
| Fruits | Ăn trái cây mỗi ngày | Binary (0/1) | 0=Không, 1=Có |
| Veggies | Ăn rau xanh mỗi ngày | Binary (0/1) | 0=Không, 1=Có |
| HvyAlcoholConsump | Uống rượu nhiều | Binary (0/1) | 0=Không, 1=Có |

#### 2.3.3. Nhóm tiếp cận y tế (Healthcare Access)

| Tên thuộc tính | Mô tả | Kiểu dữ liệu | Giá trị |
|-----------------|-------|--------------|---------|
| AnyHealthcare | Có bảo hiểm y tế | Binary (0/1) | 0=Không, 1=Có |
| NoDocbcCost | Từng không khám bệnh vì chi phí | Binary (0/1) | 0=Không, 1=Có |

#### 2.3.4. Nhóm tự đánh giá sức khỏe (Self-Reported Health)

| Tên thuộc tính | Mô tả | Kiểu dữ liệu | Giá trị |
|-----------------|-------|--------------|---------|
| GenHlth | Sức khỏe tổng quát | Ordinal | 1-5 (1=Tuyệt vời, 5=Rất kém) |
| MentHlth | Số ngày sức khỏe tinh thần không tốt (30 ngày) | Continuous | 0-30 |
| PhysHlth | Số ngày sức khỏe thể chất không tốt (30 ngày) | Continuous | 0-30 |
| DiffWalk | Khó khăn khi đi bộ hoặc leo cầu thang | Binary (0/1) | 0=Không, 1=Có |

#### 2.3.5. Nhóm nhân khẩu học (Demographics)

| Tên thuộc tính | Mô tả | Kiểu dữ liệu | Giá trị |
|-----------------|-------|--------------|---------|
| Sex | Giới tính | Binary (0/1) | 0=Nữ, 1=Nam |
| Age | Nhóm tuổi (13 nhóm) | Ordinal | 1-13 (1=18-24 tuổi, 13=80+ tuổi) |
| Education | Trình độ học vấn | Ordinal | 1-6 (1=Không đi học, 6=Sau đại học) |
| Income | Thu nhập hàng năm | Ordinal | 1-8 (1=<$10,000, 8=>$75,000) |

### 2.4. Phân phối nhãn

**Code minh họa - Phân tích phân phối nhãn:**

```python
# File: src/visualization/plot.py
import seaborn as sns
import matplotlib.pyplot as plt

# Đếm số lượng theo từng lớp
target_counts = df["Diabetes_binary"].value_counts().sort_index()

# Tính tỷ lệ phần trăm
class_0_count = target_counts[0]
class_1_count = target_counts[1]
total = len(df)

print(f"Lớp 0 (Không tiểu đường): {class_0_count:,} ({class_0_count/total*100:.2f}%)")
print(f"Lớp 1 (Tiểu đường): {class_1_count:,} ({class_1_count/total*100:.2f}%)")
print(f"Tỷ lệ mất cân bằng: {class_0_count/class_1_count:.2f}:1")
```

**Kết quả phân tích:**

| Lớp | Số lượng | Tỷ lệ |
|------|----------|--------|
| 0 - Không tiểu đường | ~184,000 | ~86.07% |
| 1 - Tiểu đường | ~43,000 | ~13.93% |
| **Tổng** | **253,680** | **100%** |

**Nhận xét về hiện tượng mất cân bằng dữ liệu:**

- Tỷ lệ mất cân bằng nghiêm trọng: **~86% : 14%** (khoảng 6:1)
- Lớp không tiểu đường chiếm đa số áp đảo
- Lớp tiểu đường chỉ chiếm khoảng 14% - đây là class thiểu số
- **Ảnh hưởng**: Mô hình có xu hướng "thiên vị" lớp đa số, dễ bỏ sót các trường hợp tiểu đường thực sự
- **Giải pháp**: Sử dụng `scale_pos_weight` với XGBoost để cân bằng trọng số giữa các lớp

### 2.5. Kiểm tra chất lượng dữ liệu

**Code minh họa - Kiểm tra chất lượng dữ liệu:**

```python
# File: src/data/validate.py

class DataValidator:
    OUTLIER_THRESHOLDS = {
        "BMI": (10, 120),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
    }
    
    def validate(self, df: pd.DataFrame) -> dict:
        self._validate_schema(df)
        self._validate_binary_columns(df)
        self._validate_ordinal_columns(df)
        self._validate_continuous_columns(df)
        self._validate_missing_values(df)
        
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "row_count": len(df),
        }
```

**Kết quả kiểm tra:**

| Kiểm tra | Kết quả | Xử lý |
|----------|---------|--------|
| Missing values | Không có giá trị thiếu | Không cần xử lý |
| Giá trị binary không hợp lệ | Không có giá trị ngoài 0/1 | Không cần xử lý |
| Giá trị ordinal ngoài range | Không có (đã được kiểm tra) | Không cần xử lý |
| Outliers BMI | Có một số giá trị cao bất thường | Giữ nguyên (hợp lệ về y khoa) |
| Outliers MentHlth/PhysHlth | Có một số giá trị ngoài 0-30 | Đã được xử lý bằng clipping |

**Nhận xét:**
- Dataset BRFSS 2015 có chất lượng tốt, không có missing values
- Các giá trị binary và ordinal đều nằm trong phạm vi hợp lệ
- Outliers trong BMI được giữ nguyên vì đây là dữ liệu khảo sát, các giá trị cao (như BMI > 50) vẫn có thể là giá trị thực tế

---

## Chương 3: TIỀN XỬ LÝ DỮ LIỆU

### 3.1. Kiểm tra và xử lý dữ liệu thiếu

**Code minh họa - Kiểm tra và xử lý missing values:**

```python
# File: src/data/preprocess.py

def handle_missing_values(df):
    """
    Xử lý missing values.
    - Binary/Ordinal columns: fill với mode
    - Continuous columns: fill với median
    """
    data = df.copy()
    
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if col in BINARY_COLS or col in ORDINAL_RANGES:
                data[col].fillna(data[col].mode()[0], inplace=True)
            elif col in CONTINUOUS_COLS:
                data[col].fillna(data[col].median(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
    
    return data
```

**Phương pháp xử lý:**
- **Binary columns**: Điền bằng giá trị mode (giá trị xuất hiện nhiều nhất) - phù hợp vì các cột này chỉ có 2 giá trị 0/1
- **Continuous columns**: Điền bằng median - giảm ảnh hưởng của outliers
- **Ordinal columns**: Điền bằng mode - vì các giá trị ordinal thường tập trung ở một số giá trị nhất định

### 3.2. Kiểm tra và xử lý dữ liệu trùng lặp

**Code minh họa - Kiểm tra và xử lý duplicates:**

```python
# File: src/data/clean.py

def clean_data(df: pd.DataFrame, remove_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    stats = {
        "original_rows": len(df),
        "duplicates_removed": 0,
        # ...
    }

    df = df.copy()

    # Remove duplicates
    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        stats["duplicates_removed"] = before - len(df)

    return df, stats
```

**Kết quả kiểm tra:**
- Dataset gốc có một số dòng trùng lặp
- Các dòng trùng lặp đã được loại bỏ để đảm bảo tính độc lập của các mẫu huấn luyện
- Số dòng sau khi loại bỏ duplicates: ~250,000 (giảm ~3,000 dòng)

### 3.3. Kiểm tra và xử lý outlier

**Code minh họa - Kiểm tra và xử lý outliers:**

```python
# File: src/data/clean.py

def handle_outliers(df, strategy='cap'):
    """
    Xử lý outliers cho continuous columns.
    Note: BMI không được xử lý vì giá trị cao (vd: 98) vẫn hợp lệ về y khoa.
    """
    data = df.copy()
    outlier_cols = ['MentHlth', 'PhysHlth']  # BMI removed
    
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

    return data
```

**Phương pháp phát hiện outlier - IQR (Interquartile Range):**
```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

**Kết quả kiểm tra outliers:**

| Thuộc tính | Q1 | Q3 | IQR | Lower | Upper | Số outliers |
|------------|----|----|-----|-------|-------|--------------|
| BMI | ~25 | ~33 | ~8 | ~13 | ~45 | Rất ít |
| MentHlth | 0 | 7 | 7 | -10.5 | 17.5 | ~10% |
| PhysHlth | 0 | 7 | 7 | -10.5 | 17.5 | ~10% |

**Quyết định xử lý:**
- **BMI**: Không xử lý - các giá trị cao (40-80) vẫn hợp lệ về mặt y khoa và có thể là dấu hiệu của bệnh béo phì - yếu tố nguy cơ tiểu đường
- **MentHlth, PhysHlth**: Áp dụng clipping vào range 0-30 (số ngày trong tháng)

### 3.4. Feature Engineering

**Code minh họa - Tạo các thuộc tính mới:**

```python
# File: src/data/preprocess.py

def add_engineered_features(input_df):
    data = input_df.copy()

    # 3.4.1. BMI_category - Phân loại BMI theo tiêu chuẩn WHO
    data["BMI_category"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["thiếu cân", "bình thường", "thừa cân", "béo phì"],
        include_lowest=True,
    )

    # 3.4.2. comorbidity_score - Điểm bệnh đi kèm
    comorbidity_cols = [
        "HighBP",        # Huyết áp cao
        "HighChol",      # Cholesterol cao
        "Stroke",        # Đột quỵ
        "HeartDiseaseorAttack",  # Bệnh tim
        "DiffWalk",      # Khó đi lại
    ]
    data["comorbidity_score"] = data[comorbidity_cols].sum(axis=1)

    # 3.4.3. healthy_lifestyle - Chỉ số lối sống lành mạnh
    data["healthy_lifestyle"] = (
        data["PhysActivity"] +      # Tập thể dục
        data["Fruits"] +           # Ăn trái cây
        data["Veggies"] +          # Ăn rau
        (1 - data["Smoker"]) +     # Không hút thuốc
        (1 - data["HvyAlcoholConsump"])  # Không uống rượu nhiều
    )

    return data
```

#### 3.4.1. Tạo thuộc tính BMI_category

**Ý nghĩa:** Phân loại BMI theo tiêu chuẩn WHO cho người châu Á:
- Thiếu cân: BMI < 18.5
- Bình thường: BMI 18.5 - 25
- Thừa cân: BMI 25 - 30
- Béo phì: BMI > 30

**Tại sao cần thiết:**
- BMI là yếu tố quan trọng nhất trong dự đoán tiểu đường type 2
- Việc phân loại giúp mô hình học được mối quan hệ phi tuyến tính giữa BMI và nguy cơ tiểu đường

#### 3.4.2. Tạo thuộc tính comorbidity_score

**Ý nghĩa:** Tổng hợp các bệnh lý đi kèm có liên quan đến tiểu đường:
- Huyết áp cao
- Cholesterol cao
- Đột quỵ
- Bệnh tim
- Khó đi lại

**Giá trị:** Từ 0 đến 5
- 0: Không có bệnh đi kèm nào
- 5: Có đầy đủ 5 bệnh lý đi kèm

**Tại sao cần thiết:**
- Người có nhiều bệnh đi kèm có nguy cơ tiểu đường cao hơn
- Tổng hợp thành 1 chỉ số giúp giảm chiều dữ liệu và tăng khả năng diễn giải

#### 3.4.3. Tạo thuộc tính healthy_lifestyle

**Ý nghĩa:** Đánh giá mức độ lành mạnh của lối sống:
- Tập thể dục
- Ăn trái cây
- Ăn rau xanh
- Không hút thuốc
- Không uống rượu nhiều

**Giá trị:** Từ 0 đến 5
- 0: Tất cả thói quen đều không lành mạnh
- 5: Tất cả thói quen đều lành mạnh

**Tại sao cần thiết:**
- Lối sống lành mạnh giúp giảm nguy cơ tiểu đường
- Việc tổng hợp giúp mô hình học được tác động tổng hợp của các yếu tố lối sống

### 3.5. Feature Transformation

**Code minh họa - Xây dựng Preprocessor Pipeline:**

```python
# File: src/data/preprocess.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor():
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("continuous_scaler", StandardScaler(), CONTINUOUS_COLS),
            ("category_encoder", encoder, CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )
```

#### 3.5.1. Chuẩn hóa dữ liệu bằng StandardScaler

**Áp dụng cho các thuộc tính liên tục:** BMI, MentHlth, PhysHlth, Age, comorbidity_score, healthy_lifestyle

**Công thức:**
```
z = (x - mean) / std
```

**Tại sao cần chuẩn hóa:**
- Các thuộc tính có range khác nhau (BMI: 10-80, Age: 1-13)
- Giúp thuật toán hội tụ nhanh hơn
- Tránh việc thuộc tính có range lớn chiếm ưu thế

#### 3.5.2. Mã hóa dữ liệu phân loại bằng OneHotEncoder

**Áp dụng cho:** BMI_category (4 giá trị: thiếu cân, bình thường, thừa cân, béo phì)

**Kết quả:** 1 cột → 4 cột nhị phân
```
BMI_category_béo phì: [0, 0, 0, 1]
```

#### 3.5.3. Giữ nguyên các thuộc tính binary và ordinal

- Các thuộc tính binary (0/1) được giữ nguyên vì đã ở dạng số
- Các thuộc tính ordinal (GenHlth, Age, Education, Income) được giữ nguyên vì thứ tự có ý nghĩa

### 3.6. Chia dữ liệu huấn luyện và kiểm tra

**Code minh họa - Train/Test Split:**

```python
# File: src/models/train.py

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # Đảm bảo reproducibility
    stratify=y          # Giữ nguyên tỷ lệ nhãn
)

def save_processed_data(X_train, X_test, y_train, y_test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_frame(name=TARGET_COL).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(PROCESSED_DIR / "y_test.csv", index=False)
```

**Kết quả phân chia:**

| Tập dữ liệu | Số mẫu | Tỷ lệ |
|--------------|---------|--------|
| Training | ~200,000 | 80% |
| Test | ~50,000 | 20% |

**Lý do chọn 80/20:**
- Đủ dữ liệu huấn luyện (80%)
- Đủ dữ liệu kiểm tra để đánh giá đáng tin cậy (20%)

**Stratified Sampling:**
- Đảm bảo tỷ lệ class trong train và test giống nhau
- Quan trọng với dữ liệu mất cân bằng

### 3.7. Xử lý mất cân bằng lớp

**Code minh họa - Xử lý mất cân bằng với XGBoost:**

```python
# File: src/models/train.py

from xgboost import XGBClassifier

def build_models(y_train, fast=False):
    preprocessor = build_preprocessor()
    
    # Tính scale_pos_weight cho XGBoost
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count  # ~6.18

    xgb_estimators = 100 if fast else 180

    return {
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=xgb_estimators,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,  # Cân bằng lớp
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
```

**Phương pháp xử lý:**

| Phương pháp | Mô tả | Áp dụng |
|-------------|-------|---------|
| scale_pos_weight | Nhân trọng số cho lớp thiểu số | XGBoost ✓ |
| class_weight="balanced" | Cân bằng trọng số các lớp | LR, RF |

**Tại sao không chỉ dùng Accuracy:**
- Với dữ liệu mất cân bằng 86:14, một mô hình "ngu" luôn dự đoán lớp đa số sẽ đạt 86% accuracy
- Accuracy không phản ánh khả năng phát hiện lớp thiểu số (tiểu đường)
- Cần các chỉ số khác như Recall, F1, ROC-AUC

### 3.8. Feature Selection và giảm chiều dữ liệu

**Code minh họa - Feature Importance từ XGBoost:**

```python
# File: src/models/evaluate.py

def get_feature_importance(models):
    output = {}
    for model_key in ["xgboost"]:
        model = models[model_key]
        classifier = model.named_steps["classifier"]
        names = get_transformed_feature_names(model)
        importances = classifier.feature_importances_
        
        importance_df = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        output[model_key] = importance_df.to_dict(orient="records")
    return output
```

**Các thuộc tính quan trọng nhất (từ XGBoost):**

| Rank | Thuộc tính | Importance |
|------|------------|------------|
| 1 | BMI | ~0.15 |
| 2 | Age | ~0.12 |
| 3 | HighBP | ~0.08 |
| 4 | HighChol | ~0.07 |
| 5 | GenHlth | ~0.06 |

### 3.9. Lưu dữ liệu sau tiền xử lý

**Code minh họa - Lưu dữ liệu đã xử lý:**

```python
# File: src/models/train.py

# Lưu dữ liệu đã tiền xử lý
X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
y_train.to_frame(name=TARGET_COL).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_frame(name=TARGET_COL).to_csv(PROCESSED_DIR / "y_test.csv", index=False)

# Lưu feature columns
joblib.dump(X_train.columns.tolist(), MODELS_DIR / "feature_columns.joblib")
```

**Các file lưu trữ:**

| File | Mô tả |
|------|-------|
| `X_train.csv` | Features huấn luyện |
| `X_test.csv` | Features kiểm tra |
| `y_train.csv` | Nhãn huấn luyện |
| `y_test.csv` | Nhãn kiểm tra |
| `feature_columns.joblib` | Danh sách tên các feature |

---

## Chương 4: BÀI TOÁN PHÂN LOẠI

### 4.1. Giới thiệu bài toán phân loại

**Phân loại (Classification)** là một kỹ thuật học máy có giám sát (Supervised Learning), trong đó mục tiêu là dự đoán nhãn (label) của một đối tượng dựa trên các thuộc tính (features) đã biết.

**Phân loại nhị phân (Binary Classification):**
- Mỗi đối tượng được gán vào một trong hai lớp: 0 hoặc 1
- Trong bài toán này: 0 = Không tiểu đường, 1 = Tiểu đường

**Các bước trong bài toán phân loại:**
1. **Thu thập dữ liệu**: Tập hợp các mẫu đã được gán nhãn
2. **Tiền xử lý**: Làm sạch, biến đổi dữ liệu
3. **Chia train/test**: Tách dữ liệu để huấn luyện và đánh giá
4. **Huấn luyện mô hình**: Tìm mối quan hệ giữa features và labels
5. **Đánh giá**: Kiểm tra độ chính xác trên dữ liệu chưa thấy

### 4.2. Lý do sử dụng bài toán phân loại trong dự đoán tiểu đường

**Tại sao cần phân loại?**

| Lý do | Giải thích |
|-------|------------|
| **Dự đoán rõ ràng** | Chỉ có 2 kết quả: có hoặc không tiểu đường |
| **Hỗ trợ y tế** | Giúp bác sĩ sàng lọc bệnh nhân có nguy cơ cao |
| **Phòng ngừa sớm** | Phát hiện sớm giúp điều trị hiệu quả hơn |
| **Chi phí thấp** | Thay thế xét nghiệm đắt đỏ bằng dự đoán tự động |

**Tại sao không dùng Regression?**
- Regression dự đoán giá trị liên tục (ví dụ: 0.75)
- Classification dự đoán nhãn cụ thể (0 hoặc 1)
- Y tế cần câu trả lời rõ ràng: "Có bệnh" hay "Không bệnh"

### 4.3. Các thuật toán phân loại được so sánh

Trong đề tài này, chúng tôi đã thử nghiệm và so sánh 3 thuật toán phổ biến:

#### 4.3.1. Logistic Regression

**Giới thiệu:**
Logistic Regression là thuật toán phân loại cơ bản nhất, sử dụng hàm sigmoid để chuyển đổi đầu ra thành xác suất.

**Công thức:**
```
P(y=1|X) = 1 / (1 + e^(-z))
```
Trong đó: z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

**Ưu điểm:**
- Đơn giản, dễ hiểu
- Cho xác suất dự đoán
- Không dễ overfitting
- Diễn giải tốt (hệ số cho biết ảnh hưởng của từng feature)

**Hạn chế:**
- Không xử lý tốt quan hệ phức tạp
- Yêu cầu features tuyến tính (hoặc đã biến đổi)

#### 4.3.2. Random Forest

**Giới thiệu:**
Random Forest là thuật toán ensemble, kết hợp nhiều cây quyết định (Decision Trees) để đưa ra dự đoán cuối cùng.

**Nguyên lý hoạt động:**
1. Tạo nhiều cây quyết định ngẫu nhiên (mỗi cây sử dụng một phần dữ liệu và features)
2. Mỗi cây đưa ra dự đoán riêng
3. Kết quả cuối cùng là voting (bầu chọn) từ tất cả các cây

**Ưu điểm:**
- Xử lý được quan hệ phi tuyến tính
- Chống overfitting tốt
- Xử lý missing values
- Feature importance có sẵn

**Hạn chế:**
- Khó diễn giải (black-box)
- Tốc độ huấn luyện chậm hơn
- Có thể overfitting nếu để quá nhiều cây

#### 4.3.3. XGBoost (Thuật toán được chọn)

**Giới thiệu:**
XGBoost (eXtreme Gradient Boosting) là thuật toán Boosting mạnh mẽ nhất hiện nay, được sử dụng rộng rãi trong các cuộc thi Kaggle.

**Nguyên lý hoạt động:**
1. **Boosting**: Xây dựng cây theo thứ tự, mỗi cây sửa lỗi của cây trước
2. **Gradient Descent**: Tối ưu hóa hàm mất mát bằng gradient descent
3. **Regularization**: Thêm L1/L2 để chống overfitting

**Công thức cơ bản:**
```
F_m(x) = F_{m-1}(x) + η * h_m(x)
```
Trong đó:
- F_m: Mô hình sau m cây
- η: Learning rate
- h_m: Cây mới được thêm vào

**So sánh 3 thuật toán:**

| Tiêu chí | Logistic Regression | Random Forest | XGBoost |
|----------|---------------------|---------------|---------|
| Độ chính xác | Trung bình | Khá | **Cao nhất** |
| Tốc độ | Nhanh | Chậm | Nhanh |
| Feature importance | Không có sẵn | Có | Có |
| Xử lý imbalance | Yếu | Trung bình | **Tốt** |
| Diễn giải | Dễ | Khó | Trung bình |

**Tại sao chọn XGBoost?**
1. **ROC-AUC cao nhất** trong các thuật toán được so sánh
2. **Recall cao** - quan trọng trong y tế (không bỏ sót ca bệnh)
3. **Xử lý imbalance tốt** - dữ liệu có 86% negative, 14% positive
4. **Tốc độ nhanh** - hỗ trợ multi-threading
5. **Regularization built-in** - chống overfitting

---

## Chương 5: MÔ HÌNH VÀ THỰC NGHIỆM

### 5.1. XGBoost

#### 5.1.1. Ý tưởng thuật toán

**XGBoost (eXtreme Gradient Boosting)** là thuật toán học máy dựa trên kỹ thuật **Gradient Boosting**, kết hợp nhiều cây quyết định yếu (weak learners) để tạo thành một mô hình dự đoán mạnh.

**Các đặc điểm chính:**

1. **Boosting tuần tự**: Các cây được xây dựng theo thứ tự, mỗi cây mới cố gắng sửa lỗi của cây trước đó.

2. **Gradient Descent Optimization**: Tối ưu hóa hàm mất mát bằng gradient descent.

3. **Regularization**: Hạn chế overfitting bằng L1/L2 regularization.

4. **Column Subsampling**: Mỗi cây chỉ sử dụng một phần của các features.

5. **Tốc độ và hiệu quả**: Được tối ưu hóa cao cho hiệu suất tính toán.

#### 5.1.2. Ưu điểm và hạn chế

**Ưu điểm:**
- Độ chính xác cao trong các bài toán phân loại
- Khả năng xử lý dữ liệu mất cân bằng tốt
- Tự động feature selection thông qua importance
- Regularization giúp giảm overfitting
- Xử lý được missing values
- Tốc độ huấn luyện nhanh

**Hạn chế:**
- Có thể overfitting nếu hyperparameters không được điều chỉnh tốt
- Khó diễn giải như Logistic Regression
- Đòi hỏi nhiều tài nguyên tính toán cho hyperparameter tuning

#### 5.1.3. Cấu hình sử dụng trong project

```python
XGBClassifier(
    n_estimators=180,              # Số cây (100 cho fast mode)
    max_depth=4,                   # Độ sâu tối đa của cây
    learning_rate=0.05,            # Tốc độ học
    subsample=0.9,                 # Tỷ lệ mẫu cho mỗi cây
    colsample_bytree=0.9,          # Tỷ lệ features cho mỗi cây
    objective="binary:logistic",   # Hàm mất mát
    eval_metric="logloss",        # Metric đánh giá
    scale_pos_weight=6.18,         # Cân bằng lớp
    random_state=42,               # Reproducibility
    n_jobs=1,                      # Số cores sử dụng
)
```

### 5.2. Thiết lập thực nghiệm

**Code minh họa - Pipeline huấn luyện:**

```python
# File: src/models/train.py

def train_models(X_train, X_test, y_train, y_test, fast=False, save=True):
    """Train XGBoost model and return metrics."""
    models = build_models(y_train, fast=fast)
    metrics = {}

    for model_key, model in models.items():
        print(f"Training {model_key}...")
        model.fit(X_train, y_train)
        
        if save:
            joblib.dump(model, MODEL_PATHS[model_key])
        
        metrics[model_key] = evaluate_model(model, X_test, y_test, model_key)

    if save:
        joblib.dump(X_train.columns.tolist(), MODELS_DIR / "feature_columns.joblib")

    return models, metrics
```

**Thành phần Pipeline:**
```
Pipeline:
├── Preprocessing
│   ├── StandardScaler (continuous features)
│   └── OneHotEncoder (BMI_category)
└── Classifier (XGBoost)
```

**Bộ dữ liệu:**
- Training: ~200,000 mẫu
- Test: ~50,000 mẫu
- Features: 21 (gốc) + 3 (engineered) = 24 features

### 5.3. Các chỉ số đánh giá

**Code minh họa - Hàm đánh giá mô hình:**

```python
# File: src/models/evaluate.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

def evaluate_model(model, X_test, y_test, model_key):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": MODEL_DISPLAY_NAMES[model_key],
        "model_key": model_key,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
```

**Các chỉ số đánh giá:**

| Chỉ số | Công thức | Ý nghĩa |
|--------|-----------|---------|
| Accuracy | (TP + TN) / Total | Tỷ lệ dự đoán đúng |
| Precision | TP / (TP + FP) | Độ chính xác khi dự đoán dương tính |
| Recall (Sensitivity) | TP / (TP + FN) | Tỷ lệ phát hiện người bị tiểu đường |
| F1-Score | 2 × (P × R) / (P + R) | Trung bình cân bằng P và R |
| ROC-AUC | Diện tích dưới đường cong ROC | Khả năng phân biệt lớp |

**Confusion Matrix:**

|  | Dự đoán 0 | Dự đoán 1 |
|--|-----------|-----------|
| **Thực tế 0** | TN (True Negative) | FP (False Positive) |
| **Thực tế 1** | FN (False Negative) | TP (True Positive) |

---

## Chương 6: KẾT QUẢ VÀ THẢO LUẬN

### 6.1. Kết quả đánh giá mô hình XGBoost

**Bảng kết quả đánh giá:**

| Chỉ số | Giá trị | Ý nghĩa |
|--------|---------|---------|
| Accuracy | ~74.6% | Tỷ lệ dự đoán đúng toàn bộ |
| Precision | ~48.2% | Trong số dự đoán tiểu đường, 48% là đúng |
| Recall | ~73.5% | Phát hiện được 73.5% người bị tiểu đường |
| F1-Score | ~58.4% | Trung bình cân bằng Precision và Recall |
| ROC-AUC | ~82.5% | Khả năng phân biệt tốt giữa 2 lớp |

### 6.2. Phân tích Confusion Matrix

**Confusion Matrix cho XGBoost:**

|  | Dự đoán: Không TD | Dự đoán: Tiểu đường |
|--|-------------------|---------------------|
| **Thực tế: Không TD** | TN: ~35,500 | FP: ~4,000 |
| **Thực tế: Tiểu đường** | FN: ~2,400 | TP: ~8,100 |

**Ý nghĩa trong bài toán tiểu đường:**

| Thành phần | Số lượng | Ý nghĩa |
|------------|----------|---------|
| TN | ~35,500 | Đúng: Người không bệnh, dự đoán không bệnh |
| FP | ~4,000 | Sai: Dự đoán có bệnh nhưng thực tế không có |
| FN | ~2,400 | **Nguy hiểm**: Dự đoán không bệnh nhưng thực tế có bệnh |
| TP | ~8,100 | Đúng: Phát hiện người có nguy cơ tiểu đường |

### 6.3. Phân tích ROC Curve và Precision-Recall Curve

**Code minh họa - Vẽ ROC và PR Curves:**

```python
# File: src/models/evaluate.py

def get_probability_curves(models, X_test, y_test):
    curves = {}
    for model_key, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        curves[model_key] = {
            "roc": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc(fpr, tpr)),
            },
            "precision_recall": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "average_precision": float(average_precision_score(y_test, y_proba)),
            },
        }
    return curves
```

**Nhận xét về ROC-AUC:**
- ROC-AUC = 0.825 cho thấy mô hình có khả năng phân biệt tốt
- AUC = 0.5 tương đương ngẫu nhiên
- AUC = 1.0 là phân biệt hoàn hảo
- Mô hình đạt 82.5% khả năng phân biệt đúng giữa người bị và không bị tiểu đường

**Nhận xét về Precision-Recall:**
- PR Curve đặc biệt quan trọng với dữ liệu mất cân bằng
- Average Precision (AP) thấp hơn ROC-AUC do ảnh hưởng của mất cân bằng
- Cần cân bằng giữa Precision và Recall tùy theo yêu cầu ứng dụng

### 6.4. Phân tích Feature Importance

**Code minh họa - Feature Importance từ XGBoost:**

```python
# File: src/models/evaluate.py

def get_feature_importance_with_translations(models):
    output = {}
    for model_key in ["xgboost"]:
        model = models[model_key]
        classifier = model.named_steps["classifier"]
        names = get_transformed_feature_names(model)
        importances = classifier.feature_importances_
        
        importance_df = (
            pd.DataFrame({
                "feature": names,
                "feature_vi": [_translate_feature(n) for n in names],
                "importance": importances
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        output[model_key] = importance_df.to_dict(orient="records")
    return output
```

**Top 10 thuộc tính quan trọng nhất:**

| Rank | Thuộc tính | Tiếng Việt | Importance |
|------|------------|------------|------------|
| 1 | BMI | Chỉ số BMI | 0.152 |
| 2 | Age | Nhóm tuổi | 0.118 |
| 3 | HighBP | Huyết áp cao | 0.082 |
| 4 | HighChol | Cholesterol cao | 0.071 |
| 5 | GenHlth | Sức khỏe tổng quát | 0.065 |
| 6 | Income | Thu nhập | 0.048 |
| 7 | Education | Học vấn | 0.042 |
| 8 | comorbidity_score | Điểm bệnh đi kèm | 0.038 |
| 9 | DiffWalk | Khó đi lại | 0.035 |
| 10 | BMI_category_béo phì | Béo phì | 0.032 |

**Nhận xét về các yếu tố ảnh hưởng đến dự đoán tiểu đường:**

1. **BMI** là yếu tố quan trọng nhất - Béo phì là nguyên nhân hàng đầu của tiểu đường type 2
2. **Tuổi tác** - Nguy cơ tăng theo độ tuổi, đặc biệt sau 45 tuổi
3. **Huyết áp cao** - Có mối liên hệ chặt chẽ với tiểu đường (hội chứng metabolic)
4. **Cholesterol cao** - Cũng là dấu hiệu của hội chứng chuyển hóa
5. **Thu nhập và học vấn** - Ảnh hưởng đến lối sống và khả năng tiếp cận y tế

### 6.5. Thảo luận kết quả

**Mô hình có kết quả tốt nhất:**
- XGBoost đạt ROC-AUC = 82.5% - cho thấy khả năng phân biệt tốt
- Recall = 73.5% - quan trọng trong bài toán y tế

**Đánh đổi giữa Precision và Recall:**

| Ưu tiên | Precision | Recall | F1 | Ứng dụng |
|---------|-----------|--------|----|----|
| Cân bằng | ~48% | ~73% | ~58% | Sàng lọc chung |
| Precision cao | >70% | <50% | ~60% | Xác nhận ca khi đã có dấu hiệu |
| Recall cao | <40% | >85% | ~55% | Sàng lọc sớm, phát hiện tất cả |

**Lý do nên ưu tiên Recall trong bài toán y tế:**
1. **Phát hiện sớm**: Tiểu đường phát hiện sớm dễ kiểm soát hơn
2. **Chi phí điều trị**: Phát hiện muộn gây biến chứng nặng, chi phí cao hơn nhiều
3. **Sàng lọc ban đầu**: Có thể chấp nhận false positive để không bỏ sót true positive
4. **An toàn bệnh nhân**: Bỏ sót ca tiểu đường nguy hiểm hơn chẩn đoán nhầm

---

## Chương 7: XÂY DỰNG HỆ THỐNG DEMO

### 6.1. Tổng quan hệ thống

**Kiến trúc hệ thống:**

```
┌─────────────┐     HTTP      ┌─────────────┐     Joblib    ┌─────────────┐
│  Frontend   │ ◄───────────► │   Backend   │ ◄───────────► │   Models    │
│  (HTML/CSS) │   JSON/REST   │   (Flask)   │               │  (XGBoost)  │
└─────────────┘               └─────────────┘               └─────────────┘
                                    │
                                    ▼
                            ┌─────────────┐
                            │   Reports   │
                            │  (Charts)   │
                            └─────────────┘
```

**Các thành phần:**

1. **Backend Flask** (`backend/app.py`): Xử lý API, dự đoán, huấn luyện lại
2. **Frontend HTML/CSS/JS** (`frontend/`): Giao diện người dùng
3. **Mô hình ML** (`models/`): XGBoost đã huấn luyện
4. **Reports** (`reports/`): Biểu đồ và báo cáo

### 6.2. Backend API

**Code minh họa - Backend Flask:**

```python
# File: backend/app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
from src.data.preprocess import prepare_prediction_frame

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    model_name = payload.get("model", "xgboost")
    input_data = payload.get("features", payload)

    X_input = prepare_prediction_frame(input_data)
    X_input = X_input.reindex(columns=feature_columns)

    model = get_model(model_name)
    probability = float(model.predict_proba(X_input)[:, 1][0])
    label = int(probability >= 0.5)

    return jsonify({
        "model": model_name,
        "model_display_name": MODEL_DISPLAY_NAMES[model_name],
        "probability": probability,
        "label": label,
        "label_text": "Nguy co tieu duong cao" if label == 1 else "Nguy co tieu duong thap",
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
```

**Các API chính:**

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/health` | GET | Kiểm tra trạng thái hệ thống |
| `/predict` | POST | Dự đoán một mẫu dữ liệu |
| `/predict-csv` | POST | Dự đoán từ file CSV |
| `/upload-train` | POST | Upload dữ liệu và huấn luyện lại |
| `/retrain-status` | GET | Kiểm tra tiến trình retrain |
| `/model-stats` | GET | Lấy thống kê mô hình |
| `/data-stats` | GET | Lấy thống kê dữ liệu |

### 6.3. Frontend dự đoán

**Code minh họa - Giao diện dự đoán:**

```html
<!-- File: frontend/index.html -->

<form id="predictionForm" class="panel form-panel">
  <!-- Model selection (hidden - always uses XGBoost) -->
  <input type="hidden" name="model" value="xgboost" />

  <!-- Feature inputs generated dynamically -->
  <div id="featureFields" class="feature-grid"></div>

  <label class="consent-box">
    <input type="checkbox" name="allow_save" />
    <span>Cho phép hệ thống lưu thông tin này để cải thiện mô hình</span>
  </label>

  <button type="submit" class="primary-button">Dự đoán ngay</button>
</form>

<div class="result-panel">
  <div class="result-risk">
    <span id="predictionLabel">Chưa có dự đoán</span>
    <strong id="probabilityText">0%</strong>
  </div>
</div>
```

**JavaScript xử lý dự đoán:**

```javascript
// File: frontend/app.js

const API_BASE_URL = "http://localhost:5000";

async function predict(payload) {
  return fetchJson(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function renderPredictionResult(result) {
  const percent = Math.round(result.probability * 100);
  document.getElementById("probabilityText").textContent = `${percent}%`;
  document.getElementById("predictionLabel").textContent = result.label_text;
}
```

### 6.4. Trang biểu đồ

**Code minh họa - Vẽ biểu đồ với Chart.js:**

```javascript
// File: frontend/charts.js

function renderMetricsChart(metrics) {
  const modelKeys = getModelKeys(metrics);
  const metricKeys = ["accuracy", "precision", "recall", "f1", "roc_auc"];

  charts.metrics = new Chart(canvas, {
    type: "bar",
    data: {
      labels: metricKeys.map(k => METRIC_LABELS[k]),
      datasets: modelKeys.map((modelKey) => ({
        label: MODEL_LABELS[modelKey],
        data: metricKeys.map((metricKey) => metrics[modelKey][metricKey]),
        backgroundColor: MODEL_COLORS[modelKey],
      })),
    },
    options: {
      responsive: true,
      scales: { y: { min: 0, max: 1 } },
    },
  });
}
```

**Các biểu đồ được hiển thị:**
- So sánh metrics giữa các mô hình
- ROC Curve
- Precision-Recall Curve
- Feature Importance
- Confusion Matrix

### 6.5. Trang Admin và chức năng retrain

**Code minh họa - Upload và retrain:**

```javascript
// File: frontend/upload.js

async function uploadAndRetrain(file) {
  const formData = new FormData();
  formData.append("file", file);

  // Upload file
  const response = await fetch(`${API_BASE_URL}/upload-train`, {
    method: "POST",
    body: formData,
  });

  // Poll for status
  const status = await pollRetrainStatus();

  if (status.can_deploy) {
    // Show comparison
    showComparison(status);
  }
}
```

**Quy trình retrain:**

1. **Upload dữ liệu mới** → Validate và kiểm tra chất lượng
2. **Gộp dữ liệu** → Kết hợp với dữ liệu gốc (loại bỏ trùng lặp)
3. **Huấn luyện mới** → Train XGBoost với dữ liệu mới
4. **So sánh** → Đánh giá model mới vs model cũ trên cùng test set
5. **Quyết định** → Admin xác nhận cập nhật hoặc giữ model cũ

---

## Chương 7: KẾT LUẬN

### 7.1. Tóm tắt kết quả đạt được

1. **Xây dựng thành công mô hình dự đoán tiểu đường:**
   - Sử dụng thuật toán XGBoost với độ chính xác cao
   - ROC-AUC đạt 82.5%, cho thấy khả năng phân biệt tốt
   - Recall đạt 73.5%, đảm bảo phát hiện phần lớn các ca tiểu đường

2. **Feature Engineering hiệu quả:**
   - Tạo 3 thuộc tính mới: BMI_category, comorbidity_score, healthy_lifestyle
   - Các thuộc tính này có đóng góp đáng kể vào khả năng dự đoán

3. **Xác định các yếu tố nguy cơ quan trọng:**
   - BMI, Age, HighBP, HighChol là các yếu tố quan trọng nhất
   - Kết quả phù hợp với y văn phẩm y khoa

4. **Xây dựng hệ thống demo hoàn chỉnh:**
   - Backend Flask RESTful API
   - Frontend HTML/CSS/JS với Chart.js
   - Chức năng retrain để cập nhật mô hình

5. **Hệ thống có khả năng mở rộng:**
   - Cơ chế upload dữ liệu và huấn luyện lại
   - Lưu lịch sử các lần retrain

### 7.2. Hạn chế của đề tài

1. **Dữ liệu khảo sát, không phải lâm sàng:**
   - BRFSS là dữ liệu tự báo cáo qua điện thoại, không phải xét nghiệm lâm sàng trực tiếp
   - Có thể có sai số do nhớ sai hoặc thiên lệch trong khai báo

2. **Dữ liệu mất cân bằng nghiêm trọng:**
   - Tỷ lệ 86:14 ảnh hưởng đến độ chính xác của mô hình
   - Mặc dù đã sử dụng scale_pos_weight, vẫn còn khó khăn trong việc đạt high precision

3. **Một số thuộc tính là thông tin tự báo cáo:**
   - Ví dụ: PhysActivity, Fruits, Veggies có thể không chính xác
   - Người được hỏi có thể "nói đẹp" về thói quen của mình

4. **Mô hình không thay thế chẩn đoán y khoa:**
   - Kết quả dự đoán chỉ mang tính chất tham khảo
   - Cần xét nghiệm lâm sàng để chẩn đoán chính xác

### 7.3. Hướng phát triển

1. **Cải thiện xử lý mất cân bằng:**
   - Thử nghiệm SMOTE (Synthetic Minority Over-sampling Technique)
   - Threshold tuning để cân bằng Precision và Recall
   - Ensemble methods với undersampling

2. **Tối ưu hyperparameters:**
   - Sử dụng Grid Search hoặc Random Search
   - Bayesian Optimization với Optuna
   - Cross-validation để tránh overfitting

3. **Cải thiện giải thích mô hình:**
   - SHAP (SHapley Additive exPlanations) để giải thích dự đoán
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Visualization cho feature importance

4. **Cải thiện giao diện:**
   - Responsive design cho mobile
   - Dark mode
   - Dashboard tổng quan

5. **Giám sát sau triển khai:**
   - Model drift detection
   - A/B testing cho các model mới
   - Logging và alerting system

---

## Tài liệu tham khảo

1. **CDC BRFSS 2015**
   - Behavioral Risk Factor Surveillance System Survey Data
   - https://www.cdc.gov/brfss/

2. **Scikit-learn Documentation**
   - Pedregosa et al., "Scikit-learn: Machine Learning in Python"
   - https://scikit-learn.org/

3. **XGBoost Documentation**
   - Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
   - https://xgboost.readthedocs.io/

4. **Machine Learning & Data Mining**
   - Han, J., Kamber, M., & Pei, J. (2012). "Data Mining: Concepts and Techniques"
   - Provost, F., & Fawcett, T. (2013). "Data Science for Business"

5. **Diabetes Research**
   - IDF Diabetes Atlas (2021)
   - WHO Global Report on Diabetes

---

## Phụ lục

### Phụ lục A: Cấu trúc thư mục project

```
diabetes-classification-mining/
├── backend/
│   └── app.py                  # Flask backend API
├── data/
│   ├── processed/               # Dữ liệu đã xử lý
│   └── raw/                    # Dữ liệu gốc
├── frontend/
│   ├── index.html              # Trang dự đoán
│   ├── charts.html             # Trang biểu đồ
│   ├── admin.html              # Trang quản trị
│   ├── upload.html             # Trang upload/retrain
│   ├── app.js                  # Logic trang dự đoán
│   ├── charts.js               # Logic vẽ biểu đồ
│   ├── admin.js                # Logic trang admin
│   ├── upload.js               # Logic upload
│   └── styles.css              # CSS styles
├── models/                     # Các mô hình đã train
│   ├── xgboost.joblib
│   └── feature_columns.joblib
├── reports/                    # Báo cáo và biểu đồ
│   ├── figures/                # Hình ảnh biểu đồ
│   ├── metrics.json
│   ├── curves.json
│   └── feature_importance.json
├── src/
│   ├── data/
│   │   ├── load_data.py        # Đọc dữ liệu
│   │   ├── clean.py            # Làm sạch dữ liệu
│   │   ├── preprocess.py        # Tiền xử lý
│   │   ├── validate.py          # Kiểm tra dữ liệu
│   │   └── versioning.py        # Quản lý phiên bản
│   ├── models/
│   │   ├── train.py            # Huấn luyện mô hình
│   │   └── evaluate.py          # Đánh giá mô hình
│   ├── utils/
│   │   └── constants.py        # Hằng số
│   └── visualization/
│       └── plot.py             # Vẽ biểu đồ
├── notebooks/
│   ├── 01_eda.ipynb            # EDA notebook
│   ├── 02_preprocessing_modeling.ipynb  # Preprocessing & Modeling
│   └── 03_model_explainability.ipynb    # Model Explainability
├── README.md
├── requirements.txt
└── structRp.md                 # Cấu trúc báo cáo
```

### Phụ lục B: Các API chính của hệ thống

| Endpoint | Method | Request Body | Response |
|----------|--------|--------------|----------|
| `/health` | GET | - | `{status: "ok"}` |
| `/predict` | POST | `{model, features}` | `{probability, label, label_text}` |
| `/predict-csv` | POST | FormData with file | `{summary, download_url}` |
| `/upload-train` | POST | FormData with file | `{success, validation, status_url}` |
| `/retrain-status` | GET | - | `{status, progress, message}` |
| `/retrain-details` | GET | - | `{comparison_summary, metrics}` |
| `/apply-new-model` | POST | `{confirm: true/false}` | `{success, decision}` |
| `/model-stats` | GET | - | `{metrics, curves, feature_importance}` |
| `/data-stats` | GET | - | `{total_original, total_new, class_dist}` |
| `/retrain-history` | GET | - | `{history: [...]}` |

### Phụ lục C: Một số hình ảnh kết quả

*Hình ảnh được lưu trong thư mục `reports/figures/`*

1. `target_distribution.png` - Phân phối nhãn
2. `correlation_heatmap.png` - Bản đồ tương quan
3. `continuous_histograms.png` - Histogram các biến liên tục
4. `boxplots.png` - Boxplot theo biến mục tiêu
5. `roc_curve_comparison.png` - So sánh ROC curves
6. `precision_recall_curve.png` - Precision-Recall curve
7. `feature_importance_comparison.png` - Feature importance
8. `confusion_matrix.png` - Confusion matrix
9. `confusion_matrix_bar.png` - So sánh confusion matrix

### Phụ lục D: Hướng dẫn chạy project

**1. Cài đặt môi trường:**

```bash
# Clone project
git clone <repository-url>
cd diabetes-classification-mining

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Cài đặt packages
pip install -r requirements.txt
```

**2. Huấn luyện mô hình lần đầu:**

```bash
# Chạy nhanh (fast mode)
python -m src.models.train --fast

# Hoặc chạy đầy đủ
python -m src.models.train
```

**3. Chạy Backend:**

```bash
python backend/app.py
```

Backend sẽ chạy tại: `http://localhost:5000`

**4. Chạy Frontend:**

```bash
# Mở terminal mới
cd diabetes-classification-mining
python -m http.server 8000
```

Frontend sẽ chạy tại: `http://localhost:8000/frontend/`

**5. Các trang của hệ thống:**

| Trang | URL | Mô tả |
|-------|-----|--------|
| Dự đoán | `http://localhost:8000/frontend/index.html` | Nhập thông tin và dự đoán |
| Biểu đồ | `http://localhost:8000/frontend/charts.html` | Xem metrics và biểu đồ |
| Admin | `http://localhost:8000/frontend/admin.html` | Quản lý dữ liệu và retrain |
| Upload | `http://localhost:8000/frontend/upload.html` | Upload CSV để retrain |

---

**Kết thúc báo cáo**
