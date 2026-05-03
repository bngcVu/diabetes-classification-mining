# Kỹ Thuật Tiền Xử Lý Dữ Liệu - Diabetes Classification

## Mục lục
1. [Tổng quan Dataset](#1-tổng-quan-dataset)
2. [Các kỹ thuật tiền xử lý đã sử dụng](#2-các-kỹ-thuật-tiền-xử-lý-đã-sử-dụng)

---

## 1. Tổng quan Dataset

### 1.1 Nguồn dữ liệu
- **File**: `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`
- **Số dòng**: 253,680
- **Số cột**: 22

### 1.2 Cấu trúc dữ liệu

| Loại cột | Các trường | Mô tả |
|----------|-------------|--------|
| **Target** | `Diabetes_binary` | Biến mục tiêu (0: Không tiểu đường, 1: Tiểu đường) |
| **Binary (14 cột)** | `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` | Giá trị 0/1 |
| **Ordinal (4 cột)** | `GenHlth` (1-5), `Age` (1-13), `Education` (1-6), `Income` (1-8) | Thứ tự có ý nghĩa |
| **Continuous (3 cột)** | `BMI`, `MentHlth`, `PhysHlth` | Giá trị liên tục |

---

## 2. Các kỹ thuật tiền xử lý đã sử dụng

### 2.1 Data Integration (Tích hợp dữ liệu)

**Mục đích**: Load dữ liệu từ file CSV vào môi trường làm việc.

**File nguồn**: `src/data/load_data.py`

```python
# D:\diabetes-classification-mining\src\data\load_data.py (dòng 6-7)
def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)
```

**Sử dụng trong notebook**: Cell [2]

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb
from src.data.load_data import load_raw_data

df = load_raw_data()
```

---

### 2.2 Data Quality Validation (Kiểm tra chất lượng dữ liệu)

**Mục đích**: Phát hiện missing values, giá trị không hợp lệ, và outliers.

**File nguồn**: `src/data/validate.py` và `src/data/preprocess.py`

#### 2.2.1 Kiểm tra Missing Values

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 21-37)
def validate_data_quality(df):
    """Kiểm tra toàn diện chất lượng dữ liệu."""
    # 1. Kiểm tra Missing Values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        warnings['missing_values'] = missing_cols.to_dict()
```

#### 2.2.2 Kiểm tra Giá trị không hợp lệ (Binary Columns)

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 39-49)
# 2. Kiểm tra giá trị không hợp lệ (binary columns)
invalid_binary = {}
for col in BINARY_COLS:
    if col in df.columns:
        values = set(df[col].dropna().unique())
        invalid = sorted(values - {0, 1, 0.0, 1.0})
        if invalid:
            invalid_binary[col] = invalid
```

#### 2.2.3 Kiểm tra Giá trị không hợp lệ (Ordinal Columns)

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 51-59)
# 3. Kiểm tra giá trị không hợp lệ (ordinal columns)
invalid_ordinal = {}
for col, (min_val, max_val) in ORDINAL_RANGES.items():
    if col in df.columns:
        invalid_count = (~df[col].between(min_val, max_val)).sum()
        if invalid_count > 0:
            invalid_ordinal[col] = int(invalid_count)
```

#### 2.2.4 Kiểm tra Outliers (IQR Method)

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 61-79)
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
```

---

### 2.3 Data Cleaning (Làm sạch dữ liệu)

**Mục đích**: Xử lý missing values, giá trị không hợp lệ, và outliers.

#### 2.3.1 Xử lý Missing Values

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 88-105)
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

#### 2.3.2 Xử lý Giá trị không hợp lệ

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 108-128)
def handle_invalid_values(df):
    """
    Xử lý giá trị không hợp lệ.
    - Binary columns: replace invalid với NaN, sau đó fill
    - Ordinal columns: clip vào range hợp lệ
    """
    data = df.copy()
    
    for col in BINARY_COLS:
        if col in data.columns:
            invalid_mask = ~data[col].isin([0, 1, 0.0, 1.0])
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                data.loc[invalid_mask, col] = np.nan
                data[col].fillna(data[col].mode()[0], inplace=True)
    
    for col, (min_val, max_val) in ORDINAL_RANGES.items():
        if col in data.columns:
            data[col] = data[col].clip(lower=min_val, upper=max_val)
    
    return data
```

#### 2.3.3 Xử lý Outliers (Capping Strategy)

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 131-161)
def handle_outliers(df, strategy='cap'):
    """
    Xử lý outliers cho continuous columns.
    
    Args:
        strategy: 'cap' (gắn vào boundary) hoặc 'remove' (xóa row)
    """
    data = df.copy()
    outlier_cols = ['BMI', 'MentHlth', 'PhysHlth']
    
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
                data.drop(index=data[outlier_mask].index, inplace=True)
    
    return data
```

---

### 2.4 Feature Engineering (Tạo thuộc tính mới)

**Mục đích**: Tạo các features mới từ các features hiện có để cải thiện mô hình.

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 196-223)
def add_engineered_features(input_df):
    data = input_df.copy()

    # 1. BMI_category - Rời rạc hóa BMI thành các nhóm
    data["BMI_category"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["thiếu cân", "bình thường", "thừa cân", "béo phì"],
        include_lowest=True,
    )

    # 2. comorbidity_score - Tổng hợp các bệnh đi kèm
    comorbidity_cols = [
        "HighBP",
        "HighChol",
        "Stroke",
        "HeartDiseaseorAttack",
        "DiffWalk",
    ]
    data["comorbidity_score"] = data[comorbidity_cols].sum(axis=1)

    # 3. healthy_lifestyle - Chỉ số lối sống lành mạnh
    data["healthy_lifestyle"] = (
        data["PhysActivity"]
        + data["Fruits"]
        + data["Veggies"]
        + (1 - data["Smoker"])
        + (1 - data["HvyAlcoholConsump"])
    )

    return data
```

**Sử dụng trong notebook**: Cell về Feature Engineering

---

### 2.5 Sampling (Lấy mẫu)

**Mục đích**: Chia dữ liệu thành tập train/test với stratified sampling.

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 1624-1636)

# Simple Random Sampling (không stratify)
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    df.drop(columns=[target_col]),
    df[target_col],
    test_size=0.2,
    random_state=42
)

# Stratified Sampling (có stratify) - ĐƯỢC SỬ DỤNG
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=[target_col]),
    df[target_col],
    test_size=0.2,
    random_state=42,
    stratify=df[target_col]
)
```

---

### 2.6 Attribute Transformation (Chuyển đổi thuộc tính)

**Mục đích**: Chuẩn hóa và mã hóa dữ liệu trước khi đưa vào mô hình.

#### 2.6.1 StandardScaler (Chuẩn hóa)

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 2157-2161)
# Tạo ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("continuous_scaler", StandardScaler(), continuous_cols_to_scale),
        ("category_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder="passthrough"
)

# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 2421-2422)
# Fit preprocessor
X_train_scaled = preprocessor.fit_transform(X_train_fe)
X_test_scaled = preprocessor.transform(X_test_fe)
```

**Continuous columns được chuẩn hóa**:
- `BMI`
- `MentHlth`
- `PhysHlth`
- `Age`
- `comorbidity_score`
- `healthy_lifestyle`

#### 2.6.2 OneHotEncoder (Mã hóa One-Hot)

```python
# D:\diabetes-classification-mining\src\data\preprocess.py (dòng 248-260)
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

**Categorical columns được mã hóa**:
- `BMI_category` (4 giá trị: "thiếu cân", "bình thường", "thừa cân", "béo phì")

---

### 2.7 Feature Selection (Lựa chọn thuộc tính)

**Mục đích**: Chọn các features quan trọng nhất cho mô hình.

#### 2.7.1 SelectKBest với ANOVA F-test

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 2431-2433)
# SelectKBest với ANOVA F-test
selector_f = SelectKBest(score_func=f_classif, k='all')
selector_f.fit(X_train_scaled, y_train)
```

#### 2.7.2 Mutual Information

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 2535-2538)
# Mutual Information
mi_score_func = lambda X, y: mutual_info_classif(X, y, random_state=42)
selector_mi = SelectKBest(score_func=mi_score_func, k='all')
selector_mi.fit(X_train_scaled, y_train)
```

---

### 2.8 Dimensionality Reduction (Giảm số chiều)

**Mục đích**: Giảm số lượng features mà vẫn giữ lại phần lớn thông tin.

#### 2.8.1 PCA (Principal Component Analysis)

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 2743-2745)
# Fit PCA để xem phân bố phương sai
pca_full = PCA()
pca_full.fit(X_train_scaled)

# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 2861-2864)
# PCA với 95% variance
pca_95 = PCA(n_components=0.95)
X_train_pca = pca_95.fit_transform(X_train_scaled)
X_test_pca = pca_95.transform(X_test_scaled)
```

**Kết quả**:
- Số components: 14
- Giải thích được: 95.55% variance
- Giảm từ 27 features xuống 14 features

---

### 2.9 Imbalanced Data Handling (Xử lý mất cân bằng)

**Mục đích**: Xử lý class imbalance (13.9% positive vs 86.1% negative).

#### 2.9.1 Class Weighting

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 2907-2911)
# Class Weighting được sử dụng trong các mô hình:
# - Logistic Regression: class_weight='balanced'
# - Random Forest: class_weight='balanced'
# - XGBoost: scale_pos_weight
```

#### 2.9.2 SMOTE (Minh họa)

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 2913-2914)
# SMOTE (Synthetic Minority Over-sampling Technique)
# - Tạo mẫu synthetic cho lớp thiểu số
# - Được minh họa nhưng không sử dụng trong production
```

---

### 2.10 Cross-Validation

**Mục đích**: Đánh giá mô hình một cách robust bằng cách chia dữ liệu thành nhiều folds.

```python
# D:\diabetes-classification-mining\notebooks\02_preprocessing_modeling.ipynb (dòng 100)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
```

---

## Tổng kết

| Kỹ thuật | File nguồn | Mô tả |
|----------|------------|-------|
| Data Integration | `src/data/load_data.py` | Load dữ liệu từ CSV |
| Data Validation | `src/data/validate.py` | Kiểm tra schema và chất lượng |
| Missing Values | `src/data/preprocess.py` | Fill bằng mode/median |
| Invalid Values | `src/data/preprocess.py` | Xử lý giá trị ngoài range |
| Outliers | `src/data/preprocess.py` | Capping bằng IQR |
| Feature Engineering | `src/data/preprocess.py` | BMI_category, comorbidity_score, healthy_lifestyle |
| Sampling | `notebooks/02_preprocessing_modeling.ipynb` | Stratified train/test split |
| StandardScaler | `notebooks/02_preprocessing_modeling.ipynb` | Chuẩn hóa continuous features |
| OneHotEncoder | `notebooks/02_preprocessing_modeling.ipynb` | Mã hóa categorical features |
| Feature Selection | `notebooks/02_preprocessing_modeling.ipynb` | SelectKBest (F-test, MI) |
| PCA | `notebooks/02_preprocessing_modeling.ipynb` | Giảm chiều 27 → 14 |
| Class Weighting | `notebooks/02_preprocessing_modeling.ipynb` | Xử lý imbalanced data |
| Cross-Validation | `notebooks/02_preprocessing_modeling.ipynb` | StratifiedKFold |

---

## Files chính

- **Data Loading**: `src/data/load_data.py`
- **Data Validation**: `src/data/validate.py`
- **Data Preprocessing**: `src/data/preprocess.py`
- **Notebook chính**: `notebooks/02_preprocessing_modeling.ipynb`
- **Constants**: `src/utils/constants.py`
