# Báo cáo bước tiền xử lý dữ liệu

Báo cáo này trình bày quá trình tiền xử lý dữ liệu trong dự án phân loại tình trạng tiểu đường. Nội dung được tổng hợp từ notebook:

```text
notebooks/02_preprocessing_modeling.ipynb
```

và module xử lý dữ liệu:

```text
src/data/preprocess.py
```

## 1. Thông tin dữ liệu gốc

Bộ dữ liệu gốc được sử dụng là **Diabetes Binary Health Indicators BRFSS 2015**, lưu tại:

```text
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
```

Dữ liệu ban đầu có:

| Nội dung | Giá trị |
|---|---:|
| Số dòng | 253,680 |
| Số cột | 22 |
| Số thuộc tính đầu vào | 21 |
| Số biến mục tiêu | 1 |

Biến mục tiêu là `Diabetes_binary`:

| Giá trị | Ý nghĩa | Số dòng | Tỷ lệ |
|---:|---|---:|---:|
| 0 | Không bị tiểu đường | 218,334 | 86.07% |
| 1 | Bị tiểu đường | 35,346 | 13.93% |

Dữ liệu gốc không có giá trị thiếu. Tuy nhiên, dữ liệu bị mất cân bằng lớp vì lớp `0` chiếm phần lớn, còn lớp `1` chỉ chiếm 13.93%.

Code đọc dữ liệu trong notebook:

```python
from src.data.load_data import load_raw_data

df = load_raw_data()
print(f"Shape: {df.shape}")
print(f"So dong: {df.shape[0]:,}")
print(f"So cot: {df.shape[1]}")
df.head()
```

Code kiểm tra phân phối biến mục tiêu:

```python
target_col = "Diabetes_binary"

print("Phan phoi target trong du lieu goc:")
target_dist = df[target_col].value_counts().sort_index()
print(target_dist)
print(f"\nTi le: {target_dist[0]/len(df)*100:.1f}% non-diabetic, {target_dist[1]/len(df)*100:.1f}% diabetic")
```

## 2. Các kỹ thuật tiền xử lý dữ liệu

### 2.1. Kiểm tra chất lượng dữ liệu

Trước khi huấn luyện mô hình, dữ liệu được kiểm tra giá trị thiếu, kiểu dữ liệu, phân phối nhãn và các giá trị bất thường. Trong module `src/data/preprocess.py`, hàm `get_quality_report()` được dùng để tạo báo cáo chất lượng dữ liệu.

Code trong dự án:

```python
def get_quality_report(df):
    report = {
        "shape": list(df.shape),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df[TARGET_COL].value_counts().sort_index().to_dict(),
        "invalid_binary_values": {},
        "invalid_ordinal_counts": {},
        "outlier_counts": {},
    }
```

Ý nghĩa:

- `shape`: kiểm tra số dòng và số cột.
- `missing_values`: kiểm tra giá trị thiếu.
- `target_distribution`: kiểm tra phân phối biến mục tiêu.
- `invalid_binary_values`: kiểm tra giá trị không hợp lệ ở các biến nhị phân.
- `invalid_ordinal_counts`: kiểm tra giá trị ngoài miền hợp lệ ở các biến thứ bậc.
- `outlier_counts`: thống kê số lượng ngoại lệ ở một số biến quan trọng.

### 2.2. Chia dữ liệu bằng Stratified Sampling

Dữ liệu được chia thành tập huấn luyện và tập kiểm tra theo tỷ lệ 80:20. Do biến mục tiêu bị mất cân bằng lớp, notebook sử dụng **Stratified Sampling** để giữ tỷ lệ hai lớp trong train/test tương tự dữ liệu gốc.

Code trong notebook:

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=[target_col]),
    df[target_col],
    test_size=0.2,
    random_state=42,
    stratify=df[target_col]
)

print("Stratified Sampling (co stratify=y):")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"\nTrain target dist: {y_train.value_counts(normalize=True).to_dict()}")
print(f"Test target dist: {y_test.value_counts(normalize=True).to_dict()}")
```

Việc sử dụng:

```python
stratify=df[target_col]
```

giúp tập train và test giữ ổn định tỷ lệ lớp `0` và `1`.

### 2.3. Tạo thuộc tính mới

Notebook tạo thêm ba thuộc tính mới nhằm tăng khả năng biểu diễn thông tin của dữ liệu.

| Thuộc tính mới | Nguồn tạo | Ý nghĩa |
|---|---|---|
| `BMI_category` | Từ `BMI` | Chia BMI thành các nhóm thể trạng |
| `comorbidity_score` | Từ các biến bệnh lý | Tổng số tình trạng bệnh đi kèm |
| `healthy_lifestyle` | Từ các biến lối sống | Điểm lối sống lành mạnh |

Code tạo `BMI_category`:

```python
df["BMI_category"] = pd.cut(
    df["BMI"],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=["underweight", "normal", "overweight", "obese"],
    include_lowest=True
)
```

Code tạo `comorbidity_score`:

```python
comorbidity_cols = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk"]
df["comorbidity_score"] = df[comorbidity_cols].sum(axis=1)
```

Code tạo `healthy_lifestyle`:

```python
df["healthy_lifestyle"] = (
    df["PhysActivity"]
    + df["Fruits"]
    + df["Veggies"]
    + (1 - df["Smoker"])
    + (1 - df["HvyAlcoholConsump"])
)
```

Trong module `src/data/preprocess.py`, các bước này cũng được đóng gói trong hàm:

```python
def add_engineered_features(input_df):
    data = input_df.copy()

    data["BMI_category"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["underweight", "normal", "overweight", "obese"],
        include_lowest=True,
    )

    comorbidity_cols = [
        "HighBP",
        "HighChol",
        "Stroke",
        "HeartDiseaseorAttack",
        "DiffWalk",
    ]
    data["comorbidity_score"] = data[comorbidity_cols].sum(axis=1)

    data["healthy_lifestyle"] = (
        data["PhysActivity"]
        + data["Fruits"]
        + data["Veggies"]
        + (1 - data["Smoker"])
        + (1 - data["HvyAlcoholConsump"])
    )

    return data
```

### 2.4. Tách đặc trưng đầu vào và biến mục tiêu

Sau khi tạo thêm thuộc tính mới, dữ liệu được tách thành:

- `X`: các thuộc tính đầu vào.
- `y`: biến mục tiêu `Diabetes_binary`.

Code trong notebook:

```python
X = df.drop(columns=[target_col, "BMI_category_encoded"])
y = df[target_col].astype(int)
feature_columns = X.columns.tolist()

X_train_fe = X.loc[X_train.index].copy()
X_test_fe = X.loc[X_test.index].copy()
```

Code tương ứng trong `src/data/preprocess.py`:

```python
def split_features_target(df):
    engineered = add_engineered_features(df)
    X = engineered.drop(columns=[TARGET_COL])
    y = engineered[TARGET_COL].astype(int)
    return X, y
```

### 2.5. Chuẩn hóa và mã hóa thuộc tính

Notebook sử dụng `ColumnTransformer` để áp dụng các kỹ thuật khác nhau cho từng nhóm biến:

- `StandardScaler`: chuẩn hóa các biến liên tục.
- `OneHotEncoder`: mã hóa one-hot biến phân loại `BMI_category`.
- Các biến còn lại được giữ nguyên bằng `remainder="passthrough"`.

Các biến được chuẩn hóa:

```text
BMI, MentHlth, PhysHlth, Age, comorbidity_score, healthy_lifestyle
```

Biến được mã hóa one-hot:

```text
BMI_category
```

Code trong notebook:

```python
continuous_cols_to_scale = ["BMI", "MentHlth", "PhysHlth", "Age", "comorbidity_score", "healthy_lifestyle"]
categorical_cols = ["BMI_category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("continuous_scaler", StandardScaler(), continuous_cols_to_scale),
        ("category_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder="passthrough"
)
```

Code tương ứng trong `src/data/preprocess.py`:

```python
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

### 2.6. Lựa chọn thuộc tính và giảm số chiều

Notebook có phân tích thêm các kỹ thuật:

- `SelectKBest` với ANOVA F-test.
- `SelectKBest` với Mutual Information.
- PCA để phân tích giảm số chiều.

Code ANOVA F-score:

```python
selector_f = SelectKBest(score_func=f_classif, k='all')
selector_f.fit(X_train_scaled, y_train)

feature_scores = pd.DataFrame({
    "feature": temp_feature_names,
    "f_score": selector_f.scores_,
    "p_value": selector_f.pvalues_
}).sort_values("f_score", ascending=False)
```

Code Mutual Information:

```python
mi_score_func = lambda X, y: mutual_info_classif(X, y, random_state=42)
selector_mi = SelectKBest(score_func=mi_score_func, k='all')
selector_mi.fit(X_train_scaled, y_train)
```

Code PCA:

```python
pca_95 = PCA(n_components=0.95)
X_train_pca = pca_95.fit_transform(X_train_scaled)
X_test_pca = pca_95.transform(X_test_scaled)
```

Với dữ liệu hiện tại, sau khi scale/encode, số đặc trưng là 27. PCA với ngưỡng 95% phương sai giảm còn 14 thành phần chính và giải thích khoảng 95.55% phương sai.

## 3. So sánh dữ liệu ban đầu và dữ liệu sau tiền xử lý

Bảng so sánh kích thước dữ liệu:

| Giai đoạn | Số dòng | Số cột |
|---|---:|---:|
| Dataset gốc | 253,680 | 22 |
| Sau feature engineering | 253,680 | 25 |
| `X_train` sau tiền xử lý | 202,944 | 24 |
| `X_test` sau tiền xử lý | 50,736 | 24 |
| `y_train` | 202,944 | 1 |
| `y_test` | 50,736 | 1 |

Số cột tăng từ 22 lên 25 vì dữ liệu được bổ sung ba thuộc tính mới:

```text
BMI_category
comorbidity_score
healthy_lifestyle
```

Sau đó, cột mục tiêu `Diabetes_binary` được tách riêng thành `y_train` và `y_test`, nên `X_train` và `X_test` còn 24 cột đầu vào.

Kích thước train/test:

| Tập dữ liệu | Số dòng | Tỷ lệ |
|---|---:|---:|
| Train | 202,944 | 80% |
| Test | 50,736 | 20% |
| Tổng | 253,680 | 100% |

Phân phối nhãn sau khi chia:

| Tập dữ liệu | Lớp 0 | Tỷ lệ lớp 0 | Lớp 1 | Tỷ lệ lớp 1 |
|---|---:|---:|---:|---:|
| `y_train` | 174,667 | 86.07% | 28,277 | 13.93% |
| `y_test` | 43,667 | 86.07% | 7,069 | 13.93% |

Code so sánh kích thước trước và sau tiền xử lý trong notebook:

```python
# So sanh kich thuoc dataset goc va dataset sau tien xu ly
raw_data_path = ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
raw_shape = pd.read_csv(raw_data_path).shape

engineered_shape = df.drop(columns=["BMI_category_encoded"]).shape

comparison_df = pd.DataFrame({
    "Giai doan": [
        "Dataset goc",
        "Sau feature engineering",
        "X_train sau tien xu ly",
        "X_test sau tien xu ly",
        "y_train",
        "y_test"
    ],
    "So dong": [
        raw_shape[0],
        engineered_shape[0],
        X_train_fe.shape[0],
        X_test_fe.shape[0],
        y_train.shape[0],
        y_test.shape[0]
    ],
    "So cot": [
        raw_shape[1],
        engineered_shape[1],
        X_train_fe.shape[1],
        X_test_fe.shape[1],
        1,
        1
    ]
})

display(comparison_df)

print("Tong so dong sau khi chia train/test:", X_train_fe.shape[0] + X_test_fe.shape[0])
print(f"Ty le train: {X_train_fe.shape[0] / raw_shape[0] * 100:.2f}%")
print(f"Ty le test: {X_test_fe.shape[0] / raw_shape[0] * 100:.2f}%")
```

Sau tiền xử lý, các tập dữ liệu chính được lưu trong thư mục:

```text
data/processed/
```

gồm:

```text
X_train.csv
X_test.csv
y_train.csv
y_test.csv
```

Tóm lại, quá trình tiền xử lý đã giúp dữ liệu sạch hơn về mặt tổ chức, bổ sung thêm các thuộc tính có ý nghĩa, chia dữ liệu theo tỷ lệ hợp lý và giữ ổn định phân phối lớp giữa tập huấn luyện và tập kiểm tra.
