# Báo cáo tiền xử lý dữ liệu

Báo cáo này trình bày bước tiền xử lý dữ liệu trong dự án phân loại tình trạng tiểu đường. Nội dung được tổng hợp từ notebook `notebooks/02_preprocessing_modeling.ipynb` và code backend trong các file `src/data/preprocess.py`, `src/models/train.py`, `backend/app.py`.

Các vị trí code chính:

| Nội dung | Vị trí trong project |
|---|---|
| Đọc dữ liệu gốc | `notebooks/02_preprocessing_modeling.ipynb`, mục **2. Load du lieu (Ky thuat Tich hop)** |
| Kiểm tra chất lượng dữ liệu | `notebooks/02_preprocessing_modeling.ipynb`, mục **3. Kiem tra chat luong du lieu** và `src/data/preprocess.py`, hàm `get_quality_report()` |
| Chia train/test | `notebooks/02_preprocessing_modeling.ipynb`, mục **4.4. Minh hoa Stratified Sampling** và `src/models/train.py`, hàm `main()` / `retrain_pipeline()` |
| Tạo feature mới | `notebooks/02_preprocessing_modeling.ipynb`, mục **5. Ky thuat Tao moi thuoc tinh dac trung** và `src/data/preprocess.py`, hàm `add_engineered_features()` |
| Tách X/y | `notebooks/02_preprocessing_modeling.ipynb`, mục **6.3. Tao Preprocessor** và `src/data/preprocess.py`, hàm `split_features_target()` |
| Scale/encode | `notebooks/02_preprocessing_modeling.ipynb`, mục **6.3. Tao Preprocessor** và `src/data/preprocess.py`, hàm `build_preprocessor()` |
| So sánh trước/sau xử lý | `notebooks/02_preprocessing_modeling.ipynb`, mục **6.4. So sanh kich thuoc truoc va sau tien xu ly** |
| Lưu file CSV sau xử lý | `src/models/train.py`, hàm `save_processed_data()` |
| Dùng feature mới khi predict | `backend/app.py`, endpoint `/predict` và `src/data/preprocess.py`, hàm `prepare_prediction_frame()` |

## 1. Tổng quan dữ liệu gốc

Dữ liệu gốc của dự án là bộ **Diabetes Binary Health Indicators BRFSS 2015**, được lưu tại:

```text
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
```

Bộ dữ liệu ban đầu có kích thước:

| Nội dung | Giá trị |
|---|---:|
| Số dòng | 253,680 |
| Số cột | 22 |
| Số thuộc tính đầu vào | 21 |
| Số biến mục tiêu | 1 |

Biến mục tiêu là `Diabetes_binary`, biểu diễn tình trạng tiểu đường:

| Giá trị | Ý nghĩa | Số dòng | Tỷ lệ |
|---:|---|---:|---:|
| 0 | Không bị tiểu đường | 218,334 | 86.07% |
| 1 | Bị tiểu đường | 35,346 | 13.93% |

Dữ liệu gốc không có giá trị thiếu, nhưng có hiện tượng mất cân bằng lớp vì lớp `0` chiếm 86.07%, còn lớp `1` chỉ chiếm 13.93%. Vì vậy, khi chia dữ liệu cần giữ ổn định tỷ lệ lớp và khi đánh giá mô hình không nên chỉ dựa vào Accuracy.

Code đọc dữ liệu trong notebook:

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **2. Load du lieu (Ky thuat Tich hop)**.

```python
from src.data.load_data import load_raw_data

df = load_raw_data()
print(f"Shape: {df.shape}")
print(f"So dong: {df.shape[0]:,}")
print(f"So cot: {df.shape[1]}")
df.head()
```

Code kiểm tra phân phối biến mục tiêu:

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **4.2. Xem phan phoi target truoc khi lay mau**.

```python
target_col = "Diabetes_binary"

print("Phan phoi target trong du lieu goc:")
target_dist = df[target_col].value_counts().sort_index()
print(target_dist)
print(f"\nTi le: {target_dist[0]/len(df)*100:.1f}% non-diabetic, {target_dist[1]/len(df)*100:.1f}% diabetic")
```

Trong backend, đường dẫn dữ liệu gốc được khai báo tại `src/utils/constants.py`.

Vị trí: `src/utils/constants.py`, phần khai báo hằng số đường dẫn và tên biến mục tiêu.

```python
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET_COL = "Diabetes_binary"
```

## 2. Các kỹ thuật tiền xử lý dữ liệu trong project

### 2.1. Kiểm tra chất lượng dữ liệu

Project có hàm `get_quality_report()` trong `src/data/preprocess.py` để kiểm tra chất lượng dữ liệu. Hàm này thống kê kích thước dữ liệu, giá trị thiếu, phân phối target, giá trị không hợp lệ và outlier.

Code thực tế:

Vị trí: `src/data/preprocess.py`, hàm `get_quality_report()`.

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

    for col in BINARY_COLS:
        values = set(df[col].dropna().unique())
        invalid = sorted(values - {0, 1, 0.0, 1.0})
        report["invalid_binary_values"][col] = invalid

    for col, (min_value, max_value) in ORDINAL_RANGES.items():
        report["invalid_ordinal_counts"][col] = int((~df[col].between(min_value, max_value)).sum())

    for col in ["BMI", "MentHlth", "PhysHlth", "Age"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        report["outlier_counts"][col] = int(((df[col] < lower) | (df[col] > upper)).sum())

    return report
```

Trong quá trình train chính, báo cáo chất lượng dữ liệu được tạo tại `src/models/train.py`.

Vị trí: `src/models/train.py`, hàm `main()`.

```python
df = load_raw_data()
quality_report = get_quality_report(df)
save_json(quality_report, REPORTS_DIR / "data_quality_report.json")
```

### 2.2. Chia dữ liệu bằng Stratified Sampling

Do dữ liệu bị mất cân bằng lớp, notebook và backend đều chia dữ liệu bằng **Stratified Sampling**. Cách chia này giúp tỷ lệ lớp `0` và `1` trong tập train/test tương tự dữ liệu gốc.

Code trong notebook:

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **4.4. Minh hoa Stratified Sampling**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=[target_col]),
    df[target_col],
    test_size=0.2,
    random_state=42,
    stratify=df[target_col]
)
```

Code thực tế trong backend training pipeline.

Vị trí: `src/models/train.py`, hàm `main()`; logic tương tự cũng xuất hiện trong hàm `retrain_pipeline()`.

```python
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

Ý nghĩa:

- `test_size=0.2`: 20% dữ liệu dùng làm tập test.
- `random_state=42`: giúp kết quả chia dữ liệu có thể tái lập.
- `stratify=y`: giữ tỷ lệ lớp của biến mục tiêu trong train/test.

### 2.3. Tạo thuộc tính mới

Project sinh thêm ba thuộc tính mới từ dữ liệu gốc:

| Thuộc tính mới | Cách tạo | Tác dụng |
|---|---|---|
| `BMI_category` | Rời rạc hóa `BMI` thành nhóm | Biểu diễn BMI theo nhóm thể trạng |
| `comorbidity_score` | Cộng các bệnh lý đi kèm | Tóm tắt mức độ bệnh nền |
| `healthy_lifestyle` | Tổng hợp hành vi sống lành mạnh | Tóm tắt mức độ lối sống tốt |

Code thực tế trong `src/data/preprocess.py`:

Vị trí: `src/data/preprocess.py`, hàm `add_engineered_features()`.

```python
def add_engineered_features(input_df):
    data = input_df.copy()

    data["BMI_category"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["thiếu cân", "bình thường", "thừa cân", "béo phì"],
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

Ba thuộc tính này không yêu cầu người dùng nhập trực tiếp trên frontend. Frontend gửi dữ liệu gốc lên backend, backend tự sinh thêm các thuộc tính mới để đảm bảo logic tiền xử lý thống nhất giữa train và predict.

### 2.4. Tách đặc trưng đầu vào và biến mục tiêu

Sau khi tạo feature mới, dữ liệu được tách thành:

- `X`: tập thuộc tính đầu vào.
- `y`: nhãn mục tiêu `Diabetes_binary`.

Code thực tế trong `src/data/preprocess.py`:

Vị trí: `src/data/preprocess.py`, hàm `split_features_target()`.

```python
def split_features_target(df):
    engineered = add_engineered_features(df)
    X = engineered.drop(columns=[TARGET_COL])
    y = engineered[TARGET_COL].astype(int)
    return X, y
```

Trong notebook, logic tương ứng là:

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **6.3. Tao Preprocessor**.

```python
X = df.drop(columns=[target_col, "BMI_category_encoded"])
y = df[target_col].astype(int)
feature_columns = X.columns.tolist()

X_train_fe = X.loc[X_train.index].copy()
X_test_fe = X.loc[X_test.index].copy()
```

Sau bước này, `X_train` và `X_test` chứa các thuộc tính đầu vào, còn `y_train` và `y_test` chỉ chứa nhãn `Diabetes_binary`.

### 2.5. Chuẩn hóa và mã hóa thuộc tính

Project sử dụng `ColumnTransformer` để áp dụng các phép biến đổi khác nhau cho từng nhóm biến:

- `StandardScaler`: chuẩn hóa biến liên tục.
- `OneHotEncoder`: mã hóa biến phân loại `BMI_category`.
- `remainder="passthrough"`: giữ nguyên các biến còn lại.

Các biến được chuẩn hóa:

```text
BMI, MentHlth, PhysHlth, Age, comorbidity_score, healthy_lifestyle
```

Biến được mã hóa one-hot:

```text
BMI_category
```

Code trong notebook:

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **6.2. Xac dinh cac nhom bien** và **6.3. Tao Preprocessor**.

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

Code thực tế trong `src/data/preprocess.py`:

Vị trí: `src/data/preprocess.py`, hàm `build_preprocessor()`.

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

Code này được dùng trong mô hình ở `src/models/train.py`.

Vị trí: `src/models/train.py`, hàm `build_models()`.

```python
def build_models(y_train, fast=False):
    preprocessor = build_preprocessor()
    ...
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(...)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", RandomForestClassifier(...)),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", XGBClassifier(...)),
            ]
        ),
    }
```

Như vậy, quá trình scale/encode được đặt trực tiếp trong `Pipeline`, giúp đảm bảo dữ liệu train và dữ liệu predict được xử lý cùng một cách.

### 2.6. Dùng feature mới trong backend dự đoán

Trong API `/predict`, frontend chỉ gửi 21 thuộc tính gốc. Backend kiểm tra đủ cột gốc, sau đó gọi `prepare_prediction_frame()` để sinh thêm feature mới trước khi đưa vào model.

Code trong `backend/app.py`:

Vị trí: `backend/app.py`, endpoint `/predict`, hàm `predict()`.

```python
missing = [col for col in RAW_FEATURE_COLUMNS if col not in input_data]
if missing:
    return jsonify({"error": "Thieu features", "missing_features": missing}), 400

X_input = prepare_prediction_frame(input_data).reindex(columns=feature_columns)
model = get_model(model_name)
probability = float(model.predict_proba(X_input)[:, 1][0])
label = int(probability >= 0.5)
```

Code trong `src/data/preprocess.py`:

Vị trí: `src/data/preprocess.py`, hàm `prepare_prediction_frame()`.

```python
def prepare_prediction_frame(payload):
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    raw = pd.DataFrame([{col: payload[col] for col in RAW_FEATURE_COLUMNS}])
    return add_engineered_features(raw)
```

Điều này cho thấy backend có sử dụng các cột mới trong dự đoán, nhưng người dùng không cần nhập chúng thủ công.

### 2.7. Lựa chọn thuộc tính và giảm số chiều trong notebook

Notebook có thêm phần phân tích feature selection và PCA để đánh giá vai trò của các thuộc tính:

Code ANOVA F-score:

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **7.2. Chi so F (ANOVA F-score)**.

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

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **7.4. Mutual Information**.

```python
mi_score_func = lambda X, y: mutual_info_classif(X, y, random_state=42)
selector_mi = SelectKBest(score_func=mi_score_func, k='all')
selector_mi.fit(X_train_scaled, y_train)
```

Code PCA:

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **8.4. PCA voi so components toi uu**.

```python
pca_95 = PCA(n_components=0.95)
X_train_pca = pca_95.fit_transform(X_train_scaled)
X_test_pca = pca_95.transform(X_test_scaled)
```

Với dữ liệu hiện tại, sau scale/encode, số đặc trưng là 27. PCA với ngưỡng giữ lại 95% phương sai giảm còn 14 thành phần chính và giải thích khoảng 95.55% phương sai.

### 2.8. Xử lý dữ liệu mới khi retrain

Khi retrain, backend lấy dữ liệu mới, kiểm tra, làm sạch, gộp với dữ liệu hiện có, sau đó gọi lại pipeline train. Trong `src/models/train.py`, hàm `retrain_pipeline()` vẫn gọi `split_features_target(df)`, nên các feature mới tiếp tục được sinh ra trong quá trình retrain.

Code thực tế:

Vị trí: `src/models/train.py`, hàm `retrain_pipeline()`.

```python
training_columns = RAW_FEATURE_COLUMNS + [TARGET_COL]
df = df[training_columns].copy()

X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Như vậy, train lần đầu, predict và retrain đều dùng cùng logic tạo feature mới.

## 3. So sánh dữ liệu trước và sau tiền xử lý

Bảng so sánh kích thước dữ liệu:

| Giai đoạn | Số dòng | Số cột |
|---|---:|---:|
| Dataset gốc | 253,680 | 22 |
| Sau feature engineering | 253,680 | 25 |
| `X_train` sau tiền xử lý | 202,944 | 24 |
| `X_test` sau tiền xử lý | 50,736 | 24 |
| `y_train` | 202,944 | 1 |
| `y_test` | 50,736 | 1 |

Giải thích sự thay đổi:

- Dataset gốc có 22 cột, gồm 21 thuộc tính đầu vào và 1 biến mục tiêu.
- Sau feature engineering, dữ liệu có thêm 3 cột mới: `BMI_category`, `comorbidity_score`, `healthy_lifestyle`.
- Vì vậy, số cột tăng từ 22 lên 25.
- Khi tách dữ liệu thành `X` và `y`, cột mục tiêu `Diabetes_binary` được tách riêng ra khỏi tập đặc trưng.
- Do đó, `X_train` và `X_test` có 24 cột đầu vào, còn `y_train` và `y_test` chỉ có 1 cột target.

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

Kết quả trên cho thấy Stratified Sampling đã giữ nguyên tỷ lệ lớp giữa train và test.

Code so sánh kích thước trong notebook:

Vị trí: `notebooks/02_preprocessing_modeling.ipynb`, mục **6.4. So sanh kich thuoc truoc va sau tien xu ly**.

```python
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
```

### Vì sao dữ liệu sau xử lý được chia thành nhiều file CSV?

Trong project thực tế, dữ liệu sau xử lý được lưu thành **4 file CSV chính**, không phải 3 file:

```text
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
```

Code lưu file trong `src/models/train.py`:

Vị trí: `src/models/train.py`, hàm `save_processed_data()`.

```python
def save_processed_data(X_train, X_test, y_train, y_test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_frame(name=TARGET_COL).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(PROCESSED_DIR / "y_test.csv", index=False)
```

Ý nghĩa từng file:

| File | Nội dung | Mục đích |
|---|---|---|
| `X_train.csv` | Các thuộc tính đầu vào của tập train | Dùng để huấn luyện mô hình |
| `y_train.csv` | Nhãn `Diabetes_binary` của tập train | Là đáp án đúng cho mô hình học |
| `X_test.csv` | Các thuộc tính đầu vào của tập test | Dùng để kiểm tra mô hình trên dữ liệu chưa học |
| `y_test.csv` | Nhãn `Diabetes_binary` của tập test | Dùng để so sánh với dự đoán và tính metric |

Việc tách thành các file riêng có ba lợi ích chính:

1. Tách rõ dữ liệu đầu vào và nhãn mục tiêu.  
   `X` chứa đặc trưng, `y` chứa đáp án. Đây là cấu trúc chuẩn của bài toán supervised learning.

2. Tách rõ dữ liệu huấn luyện và dữ liệu kiểm tra.  
   Model chỉ học từ `X_train`, `y_train`; sau đó được đánh giá bằng `X_test`, `y_test`.

3. Dễ tái sử dụng trong backend.  
   Backend có thể đọc lại các file này khi retrain, thống kê dữ liệu hoặc phục hồi model candidate.

Ví dụ trong `backend/app.py`, khi retrain từ dữ liệu mới, backend đọc lại dữ liệu train gốc.

Vị trí: `backend/app.py`, endpoint `/retrain`, hàm `retrain_from_new_data()`.

```python
X_train_path = PROCESSED_DIR / "X_train.csv"
y_train_path = PROCESSED_DIR / "y_train.csv"

X_existing = pd.read_csv(X_train_path)
y_existing = pd.read_csv(y_train_path)
df_existing = pd.concat([X_existing, y_existing[TARGET_COL]], axis=1)
```

Tóm lại, dữ liệu được chia thành `X_train`, `X_test`, `y_train`, `y_test` để phục vụ đúng quy trình huấn luyện và đánh giá mô hình: học trên train, kiểm tra trên test, đồng thời giữ nhãn tách riêng với thuộc tính đầu vào.
