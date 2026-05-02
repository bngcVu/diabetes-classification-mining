# Kế hoạch xây dựng hệ thống Data Mining cho dự đoán tiểu đường

Tài liệu này mô tả chi tiết quy trình xây dựng hệ thống data mining sử dụng dataset:

`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

Dataset BRFSS 2015 có `253,680` mẫu. File CSV gồm `22` cột nếu tính cả biến mục tiêu, trong đó:

- `Diabetes_binary`: biến target, nhận giá trị `0` hoặc `1`.
- `21` cột còn lại là feature đầu vào, đều ở dạng số, chủ yếu là binary hoặc ordinal.
- Không có missing values theo mô tả dataset.

Mục tiêu hệ thống:

- Phân tích dữ liệu sức khỏe liên quan đến nguy cơ tiểu đường.
- Huấn luyện và so sánh 3 mô hình phân loại, gồm một baseline tuyến tính và hai mô hình phi tuyến.
- Đánh giá mô hình phù hợp với dữ liệu mất cân bằng.
- Đóng gói mô hình thành backend API.
- Xây dựng frontend nhập thông tin sức khỏe và hiển thị kết quả dự đoán.

---

## 1. Cấu trúc dự án đề xuất

```text
diabetes-classification-mining/
├── data/
│   ├── raw/
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_modeling.ipynb
│   └── 03_model_explainability.ipynb
├── reports/
│   ├── figures/
│   │   ├── target_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── continuous_histograms.png
│   │   ├── boxplots.png
│   │   ├── grouped_bar_metrics.png
│   │   ├── roc_curve_comparison.png
│   │   ├── precision_recall_curve.png
│   │   ├── feature_importance_comparison.png
│   │   ├── confusion_matrix_bar.png
│   │   └── confusion_matrix.png
│   └── metrics.json
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│       └── constants.py
├── backend/
│   ├── app.py
│   ├── schemas.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── requirements.txt
├── README.md
└── DATAMINING_SYSTEM_PLAN.md
```

Vai trò chính của từng phần:

- `data/raw`: lưu dữ liệu gốc, không chỉnh sửa trực tiếp.
- `data/processed`: lưu dữ liệu đã chia train/test hoặc đã xử lý.
- `notebooks`: nơi thực hiện EDA, thử nghiệm mô hình, trực quan hóa.
- `reports`: lưu biểu đồ và kết quả đánh giá.
- `models`: lưu mô hình, scaler và danh sách feature bằng `joblib`.
- `src`: mã nguồn dùng lại được cho tiền xử lý, train, evaluate, predict.
- `backend`: API phục vụ dự đoán.
- `frontend`: giao diện nhập thông tin và hiển thị kết quả.

---

## 2. Danh sách cột trong dataset

Target:

```text
Diabetes_binary
```

Features:

```text
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income
```

Các biến nên xem là liên tục hoặc gần liên tục để vẽ histogram, boxplot và scaling:

```text
BMI
MentHlth
PhysHlth
Age
```

Các biến ordinal:

```text
GenHlth
Age
Education
Income
```

Các biến binary:

```text
HighBP
HighChol
CholCheck
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
DiffWalk
Sex
```

---

## 3. Notebook 01 - EDA

File đề xuất:

```text
notebooks/01_eda.ipynb
```

Mục tiêu:

- Kiểm tra kích thước dữ liệu.
- Kiểm tra kiểu dữ liệu.
- Kiểm tra missing values.
- Thống kê mô tả.
- Xem phân phối target.
- Kiểm tra mất cân bằng lớp.
- Phân tích tương quan.
- Trực quan hóa các biến quan trọng.

### 3.1. Import thư viện

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
```

### 3.2. Load dữ liệu

```python
DATA_PATH = "../data/raw/diabetes_binary_health_indicators_BRFSS2015.csv"

df = pd.read_csv(DATA_PATH)

df.head()
```

### 3.3. Kiểm tra thông tin tổng quan

```python
df.shape
df.info()
df.dtypes
```

Kỳ vọng:

```text
Số dòng: 253,680
Số cột: 22 cột gồm 1 target và 21 feature
Missing values: 0
Kiểu dữ liệu: numeric
```

### 3.4. Kiểm tra missing values

```python
missing_count = df.isnull().sum()
missing_percent = df.isnull().mean() * 100

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent
})

missing_report
```

Nếu tất cả đều bằng `0`, có thể ghi nhận dataset sạch, không cần imputation.

### 3.5. Thống kê mô tả

```python
df.describe().T
```

Nên chú ý:

- `BMI`: kiểm tra min, max, mean, outlier.
- `MentHlth`: số ngày sức khỏe tinh thần không tốt trong 30 ngày.
- `PhysHlth`: số ngày sức khỏe thể chất không tốt trong 30 ngày.
- `Age`: nhóm tuổi dạng ordinal.
- `GenHlth`: mức sức khỏe tổng quát dạng ordinal.

### 3.6. Đếm tần suất từng biến

Với target:

```python
df["Diabetes_binary"].value_counts()
df["Diabetes_binary"].value_counts(normalize=True) * 100
```

Kỳ vọng phân phối xấp xỉ:

```text
0 - non-diabetic: khoảng 86%
1 - diabetic: khoảng 14%
```

Với toàn bộ biến categorical/binary/ordinal:

```python
for col in df.columns:
    print(f"\n{col}")
    print(df[col].value_counts().sort_index())
```

### 3.7. Vẽ phân phối biến target

```python
target_counts = df["Diabetes_binary"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title("Target Distribution")
plt.xlabel("Diabetes_binary")
plt.ylabel("Count")
plt.xticks([0, 1], ["Non-diabetic", "Diabetic"])
plt.tight_layout()
plt.savefig("../reports/figures/target_distribution.png", dpi=150)
plt.show()
```

Kết luận cần ghi trong notebook:

- Dataset mất cân bằng rõ rệt.
- Accuracy có thể gây hiểu nhầm nếu model chỉ dự đoán đa số là `0`.
- Cần ưu tiên `ROC-AUC`, `F1-score`, `Recall` cho lớp diabetic.

### 3.8. Correlation heatmap

```python
corr = df.corr(numeric_only=True)

plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../reports/figures/correlation_heatmap.png", dpi=150)
plt.show()
```

Xem feature liên quan mạnh với `Diabetes_binary`:

```python
target_corr = corr["Diabetes_binary"].sort_values(ascending=False)
target_corr
```

Các biến thường có liên quan đáng chú ý:

- `GenHlth`
- `HighBP`
- `BMI`
- `DiffWalk`
- `HighChol`
- `Age`
- `HeartDiseaseorAttack`
- `PhysHlth`

### 3.9. Histogram các biến liên tục

```python
continuous_cols = ["BMI", "Age", "MentHlth", "PhysHlth"]

df[continuous_cols].hist(figsize=(12, 8), bins=30)
plt.suptitle("Histograms of Continuous / Ordinal Features")
plt.tight_layout()
plt.savefig("../reports/figures/continuous_histograms.png", dpi=150)
plt.show()
```

### 3.10. Boxplot theo target

```python
continuous_cols = ["BMI", "Age", "MentHlth", "PhysHlth"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, col in zip(axes, continuous_cols):
    sns.boxplot(data=df, x="Diabetes_binary", y=col, ax=ax)
    ax.set_title(f"{col} by Diabetes_binary")
    ax.set_xlabel("Diabetes_binary")
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig("../reports/figures/boxplots.png", dpi=150)
plt.show()
```

Gợi ý nhận xét:

- Nhóm diabetic thường có `BMI`, `Age`, `GenHlth`, `PhysHlth` cao hơn.
- `MentHlth` có thể lệch phải vì nhiều người có giá trị `0`.
- Một số biến có outlier nhưng không nhất thiết phải loại bỏ vì mô hình tree-based xử lý khá tốt.

---

## 4. Notebook 02 - Tiền xử lý và huấn luyện mô hình

File đề xuất:

```text
notebooks/02_preprocessing_modeling.ipynb
```

Mục tiêu:

- Kiểm tra chất lượng dữ liệu: null, outlier, giá trị hợp lệ.
- Tạo feature mới: `BMI_category`, `comorbidity_score`, `healthy_lifestyle`.
- Encoding biến category mới nếu có.
- Tách `X`, `y`.
- Chia train/test theo stratified split.
- Scale các biến liên tục.
- Xử lý mất cân bằng lớp.
- Feature Selection tùy chọn.
- Huấn luyện 3 mô hình: Logistic Regression baseline, Random Forest và XGBoost.
- Lưu model bằng `joblib`.

Pipeline tổng quát:

```text
Raw CSV (21 features trong file hiện tại)
    ↓
1. Kiểm tra chất lượng dữ liệu (null, outlier, giá trị hợp lệ)
    ↓
2. Feature Engineering (BMI_category, comorbidity_score, healthy_lifestyle)
    ↓
3. Encoding (nếu có biến category mới)
    ↓
4. Train/Test split (stratify=y)
    ↓
5. Chuẩn hóa StandardScaler (chỉ fit trên train)
    ↓
6. Xử lý mất cân bằng SMOTE (chỉ trên train)
    ↓
7. Feature Selection (tùy chọn, chỉ fit trên train)
    ↓
Đưa vào 3 thuật toán: Logistic Regression, Random Forest, XGBoost
```

Lưu ý: dataset gốc trong repo có `21` feature đầu vào và `1` target. Nếu tài liệu khác ghi `20 features`, cần kiểm tra xem có đang loại bỏ một cột nào trước khi train hay không.

### 4.1. Import thư viện

```python
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
```

Nếu dùng SMOTE:

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
```

### 4.2. Kiểm tra chất lượng dữ liệu

```python
df.shape
df.info()
df.isnull().sum()
df.describe().T
```

Kiểm tra giá trị hợp lệ cho các nhóm biến:

```python
binary_cols = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex"
]

ordinal_ranges = {
    "GenHlth": (1, 5),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8)
}

for col in binary_cols:
    invalid_values = sorted(set(df[col].dropna().unique()) - {0, 1, 0.0, 1.0})
    print(col, invalid_values)

for col, (min_value, max_value) in ordinal_ranges.items():
    invalid_count = (~df[col].between(min_value, max_value)).sum()
    print(col, invalid_count)
```

Kiểm tra outlier cho các biến gần liên tục:

```python
continuous_cols = ["BMI", "MentHlth", "PhysHlth", "Age"]

for col in continuous_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
    print(col, outlier_count)
```

Không nên tự động xóa outlier ngay. Với dữ liệu sức khỏe, giá trị cực đoan có thể là ca thật và có ý nghĩa dự đoán.

### 4.3. Feature Engineering

Tạo thêm 3 feature:

- `BMI_category`: nhóm BMI dạng category.
- `comorbidity_score`: tổng số tình trạng bệnh hoặc dấu hiệu nguy cơ đi kèm.
- `healthy_lifestyle`: điểm lối sống lành mạnh.

```python
def add_engineered_features(input_df):
    data = input_df.copy()

    data["BMI_category"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["underweight", "normal", "overweight", "obese"],
        include_lowest=True
    )

    comorbidity_cols = [
        "HighBP",
        "HighChol",
        "Stroke",
        "HeartDiseaseorAttack",
        "DiffWalk"
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

df_fe = add_engineered_features(df)
```

Sau bước này, số feature tăng lên vì có thêm feature mới. Nếu `BMI_category` được one-hot encoding, số cột model nhận vào sẽ lớn hơn số feature gốc.

### 4.4. Tách feature và target

```python
target_col = "Diabetes_binary"

X = df_fe.drop(columns=[target_col])
y = df_fe[target_col].astype(int)

feature_columns = X.columns.tolist()
```

### 4.5. Train/test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Lý do dùng `stratify=y`:

- Giữ tỷ lệ `0` và `1` gần giống giữa train và test.
- Quan trọng vì dataset mất cân bằng.

### 4.6. Encoding và Feature scaling

Các biến cần scale:

```python
continuous_cols = [
    "BMI",
    "MentHlth",
    "PhysHlth",
    "Age",
    "comorbidity_score",
    "healthy_lifestyle"
]
categorical_cols = ["BMI_category"]
```

`BMI_category` là biến category mới nên cần encoding:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("continuous_scaler", StandardScaler(), continuous_cols),
        ("category_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)
```

Lưu ý:

- Random Forest và XGBoost không bắt buộc phải scaling.
- Logistic Regression bắt buộc nên dùng `StandardScaler` vì mô hình tuyến tính nhạy với scale của feature.
- Trong project này, dùng chung `ColumnTransformer` với `StandardScaler` cho `BMI`, `MentHlth`, `PhysHlth`, `Age` để baseline Logistic Regression được huấn luyện đúng.
- Nếu dùng pipeline, scaler chỉ fit trên train set, tránh data leakage.

### 4.7. Cách xử lý mất cân bằng lớp

Có 2 hướng chính.

#### Cách 1: Dùng class weight

Logistic Regression baseline:

```python
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])
```

Random Forest:

```python
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])
```

XGBoost:

```python
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / positive_count

xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    ))
])
```

#### Cách 2: Dùng SMOTE

```python
logistic_smote_model = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])

rf_smote_model = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])
```

Lưu ý quan trọng:

- Chỉ áp dụng SMOTE trên train set.
- Không áp dụng SMOTE trước khi train/test split.
- Không áp dụng SMOTE trên test set.

Khuyến nghị cho project này:

- Dùng Logistic Regression với `StandardScaler` và `class_weight="balanced"` làm baseline đầu tiên.
- Bắt đầu với `class_weight="balanced"` cho Random Forest.
- Dùng `scale_pos_weight` cho XGBoost.
- Chỉ thử SMOTE ở bước mở rộng nếu recall hoặc F1 cho lớp diabetic còn thấp.

### 4.8. Feature Selection tùy chọn

Feature Selection nên được fit trong pipeline để chỉ học từ train set. Có thể dùng `SelectKBest` sau preprocessing và trước classifier.

Ví dụ với Logistic Regression:

```python
logistic_selected_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_classif, k=20)),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])
```

Lưu ý:

- `k=20` chỉ là ví dụ, cần tuning bằng validation hoặc cross-validation.
- Không fit feature selection trên toàn bộ dataset trước khi split.
- Với Random Forest và XGBoost, feature selection có thể không cần thiết vì mô hình tree-based tự giảm ảnh hưởng feature yếu khá tốt.

### 4.9. Huấn luyện mô hình

```python
logistic_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
```

### 4.10. Hàm đánh giá dùng lại

```python
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print(model_name)
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", metrics["roc_auc"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    return metrics
```

```python
logistic_metrics = evaluate_model(logistic_model, X_test, y_test, "Logistic Regression")
rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
```

### 4.11. Lưu mô hình và artifacts

```python
joblib.dump(logistic_model, "../models/logistic_regression.joblib")
joblib.dump(rf_model, "../models/random_forest.joblib")
joblib.dump(xgb_model, "../models/xgboost.joblib")
joblib.dump(feature_columns, "../models/feature_columns.joblib")
```

Lưu metrics:

```python
import json

metrics = {
    "logistic_regression": logistic_metrics,
    "random_forest": rf_metrics,
    "xgboost": xgb_metrics
}

with open("../reports/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
```

---

## 5. Ba thuật toán đề xuất

### 5.1. Logistic Regression baseline

Logistic Regression là lựa chọn rất tốt để làm baseline so sánh trước khi dùng các mô hình phức tạp hơn.

Lý do chọn:

- Đơn giản, train nhanh và dễ tái lập.
- Dễ giải thích bằng hệ số của mô hình.
- Cho biết mức hiệu năng tối thiểu hợp lý của một mô hình tuyến tính.
- Có `class_weight="balanced"` để xử lý mất cân bằng lớp.
- Kết hợp tốt với `StandardScaler`, đặc biệt với các biến như `BMI`, `MentHlth`, `PhysHlth`, `Age`.

Điểm cần chú ý:

- Bắt buộc nên đặt trong `Pipeline` có `StandardScaler` để tránh scale feature làm lệch hệ số.
- Khó học quan hệ phi tuyến và tương tác phức tạp giữa các biến.
- Nếu Logistic Regression cho ROC-AUC hoặc F1 gần Random Forest/XGBoost, nên ưu tiên mô hình đơn giản hơn khi cần giải thích.

Code baseline:

```python
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])
```

### 5.2. Random Forest

Lý do chọn:

- Phù hợp dữ liệu tabular.
- Xử lý tốt quan hệ phi tuyến.
- Ít yêu cầu tiền xử lý phức tạp.
- Có thể giải thích bằng feature importance.
- Có tham số `class_weight="balanced"` để xử lý mất cân bằng lớp.

Điểm cần chú ý:

- Có thể tốn thời gian train nếu `n_estimators` lớn.
- Feature importance dạng impurity có thể thiên vị với feature nhiều mức giá trị.
- Cần đánh giá bằng ROC-AUC, F1, recall thay vì chỉ accuracy.

### 5.3. XGBoost

Lý do chọn:

- Thường cho hiệu năng cao trên dữ liệu tabular.
- Có khả năng học các tương tác phức tạp giữa feature.
- Hỗ trợ `scale_pos_weight` cho dữ liệu mất cân bằng.
- Có thể giải thích bằng SHAP values.

Điểm cần chú ý:

- Cần cài thêm thư viện `xgboost`.
- Có nhiều hyperparameter cần điều chỉnh.
- Nếu không tuning cẩn thận có thể overfit.

### 5.4. Cách so sánh hợp lý

Thứ tự nên chạy:

1. Logistic Regression để có baseline tuyến tính.
2. Random Forest để kiểm tra cải thiện từ mô hình phi tuyến dễ giải thích.
3. XGBoost để tối ưu hiệu năng trên dữ liệu tabular.

Kết luận mô hình không nên dựa vào accuracy thuần túy. Nếu Logistic Regression có F1 hoặc ROC-AUC thấp hơn rõ rệt, dùng Random Forest hoặc XGBoost. Nếu Logistic Regression gần tương đương, baseline này là lựa chọn tốt vì đơn giản và dễ giải thích.

---

## 6. Notebook 03 - Giải thích mô hình

File đề xuất:

```text
notebooks/03_model_explainability.ipynb
```

Mục tiêu:

- Xem feature importance của Random Forest.
- Dùng SHAP để giải thích XGBoost.
- Phân tích vì sao mô hình dự đoán một người có nguy cơ tiểu đường cao.

### 6.1. Random Forest feature importance

```python
rf_classifier = rf_model.named_steps["classifier"]

feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
feature_names = [name.replace("continuous_scaler__", "").replace("remainder__", "") for name in feature_names]

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_classifier.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.head(15)
```

```python
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x="importance", y="feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.tight_layout()
plt.show()
```

### 6.2. SHAP cho XGBoost

```python
import shap

xgb_classifier = xgb_model.named_steps["classifier"]
X_test_transformed = xgb_model.named_steps["preprocessor"].transform(X_test)

explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
```

Lưu ý:

- SHAP giúp giải thích từng dự đoán cụ thể.
- Rất phù hợp khi cần trình bày vì sao một người được dự đoán có nguy cơ cao.

---

## 7. Đánh giá mô hình

Vì dataset mất cân bằng, không nên dùng accuracy làm tiêu chí chính.

Các metric nên báo cáo:

- `Accuracy`: dùng để tham khảo tổng quan.
- `Precision`: trong số người dự đoán diabetic, bao nhiêu người thực sự diabetic.
- `Recall`: trong số người thực sự diabetic, mô hình phát hiện được bao nhiêu.
- `F1-score`: cân bằng giữa precision và recall.
- `ROC-AUC`: khả năng phân biệt giữa hai lớp trên nhiều threshold.
- `Confusion matrix`: xem lỗi false positive và false negative.

Metric ưu tiên:

```text
1. ROC-AUC
2. F1-score
3. Recall của lớp diabetic
4. Precision của lớp diabetic
5. Accuracy
```

Các biểu đồ nên có để so sánh 3 mô hình:

```text
Biểu đồ 1 - Grouped bar metrics
Biểu đồ 2 - ROC Curve
Biểu đồ 3 - Precision vs Recall
Biểu đồ 4 - Feature Importance
Biểu đồ 5 - Confusion Matrix dạng bar
```

### 7.1. Biểu đồ 1 - Grouped bar metrics

Mục tiêu:

- Nhìn tổng thể một lần để thấy mô hình nào dẫn đầu từng metric.
- Không dùng accuracy làm tiêu chí chính vì dataset mất cân bằng.
- Ưu tiên xem `roc_auc`, `f1`, `recall`.

```python
metrics_df = pd.DataFrame([
    logistic_metrics,
    rf_metrics,
    xgb_metrics
])

plot_metrics = metrics_df.melt(
    id_vars="model",
    value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
    var_name="metric",
    value_name="score"
)

plt.figure(figsize=(11, 6))
sns.barplot(data=plot_metrics, x="metric", y="score", hue="model")
plt.ylim(0, 1)
plt.title("Grouped Bar Metrics Comparison")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("../reports/figures/grouped_bar_metrics.png", dpi=150)
plt.show()
```

Nhận xét cần ghi:

- Accuracy chỉ để tham khảo.
- Nếu model có recall cao hơn, model đó phát hiện được nhiều ca diabetic hơn.
- Nếu model có precision cao hơn, model đó ít báo nhầm diabetic hơn.

### 7.2. Biểu đồ 2 - ROC Curve của 3 mô hình

ROC Curve rất quan trọng trong bài toán y tế vì cho thấy khả năng phân biệt giữa hai lớp trên nhiều threshold. Đường cong càng phình lên góc trên trái thì AUC càng cao. Đường chéo xám là random classifier, mô hình tốt phải vượt rõ đường này.

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

logistic_proba = logistic_model.predict_proba(X_test)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)

logistic_auc = auc(logistic_fpr, logistic_tpr)
rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

plt.figure(figsize=(8, 6))
plt.plot(logistic_fpr, logistic_tpr, label=f"Logistic Regression AUC = {logistic_auc:.4f}")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest AUC = {rf_auc:.4f}")
plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost AUC = {xgb_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/figures/roc_curve_comparison.png", dpi=150)
plt.show()
```

### 7.3. Biểu đồ 3 - Precision vs Recall

Mục tiêu:

- Thấy rõ sự đánh đổi giữa precision và recall.
- RF có thể có precision cao hơn, nghĩa là ít báo nhầm hơn.
- XGBoost có thể có recall cao hơn, nghĩa là bắt được nhiều ca bệnh hơn.
- Với bài toán y tế, thường nên ưu tiên recall để giảm số ca bệnh bị bỏ sót.

```python
logistic_precision, logistic_recall, _ = precision_recall_curve(y_test, logistic_proba)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_proba)
xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_proba)

logistic_ap = average_precision_score(y_test, logistic_proba)
rf_ap = average_precision_score(y_test, rf_proba)
xgb_ap = average_precision_score(y_test, xgb_proba)

plt.figure(figsize=(8, 6))
plt.plot(logistic_recall, logistic_precision, label=f"Logistic Regression AP = {logistic_ap:.4f}")
plt.plot(rf_recall, rf_precision, label=f"Random Forest AP = {rf_ap:.4f}")
plt.plot(xgb_recall, xgb_precision, label=f"XGBoost AP = {xgb_ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/figures/precision_recall_curve.png", dpi=150)
plt.show()
```

Nhận xét cần ghi:

- Nếu mục tiêu là sàng lọc nguy cơ, recall cao thường quan trọng hơn precision.
- Precision quá thấp sẽ tạo nhiều cảnh báo nhầm.
- Có thể tuning threshold để chọn điểm cân bằng phù hợp.

### 7.4. Biểu đồ 4 - Feature Importance

Mục tiêu:

- So sánh xem Random Forest và XGBoost có đồng thuận về feature quan trọng không.
- `HighBP` và `BMI` thường là các feature quan trọng.
- Frontend có thể làm toggle giữa RF và XGBoost để xem từng mô hình.

Lấy tên feature sau preprocessing:

```python
def get_transformed_feature_names(model):
    preprocessor = model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    return [
        name.replace("continuous_scaler__", "")
            .replace("category_encoder__", "")
            .replace("remainder__", "")
        for name in names
    ]

feature_names = get_transformed_feature_names(rf_model)
```

Tạo importance dataframe:

```python
rf_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.named_steps["classifier"].feature_importances_,
    "model": "Random Forest"
})

xgb_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": xgb_model.named_steps["classifier"].feature_importances_,
    "model": "XGBoost"
})

importance_compare = pd.concat([rf_importance, xgb_importance], ignore_index=True)

top_features = (
    importance_compare
    .groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .index
)

plt.figure(figsize=(11, 7))
sns.barplot(
    data=importance_compare[importance_compare["feature"].isin(top_features)],
    x="importance",
    y="feature",
    hue="model"
)
plt.title("Feature Importance Comparison: Random Forest vs XGBoost")
plt.tight_layout()
plt.savefig("../reports/figures/feature_importance_comparison.png", dpi=150)
plt.show()
```

Gợi ý cho frontend:

- Tab `Feature Importance`.
- Toggle hoặc segmented control gồm `Random Forest` và `XGBoost`.
- Khi chọn model, chart chỉ hiển thị top feature của model đó.

### 7.5. Biểu đồ 5 - Confusion Matrix dạng bar

Confusion matrix dạng bar giúp so sánh trực tiếp số lượng:

- `TP`: bắt đúng ca diabetic.
- `FN`: bỏ sót ca diabetic.
- `FP`: báo nhầm diabetic.
- `TN`: bắt đúng non-diabetic.

Với bài toán y tế, `FN` là chỉ số rất quan trọng vì đây là số ca bệnh bị bỏ sót.

```python
def confusion_items(metrics):
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

confusion_bar_df = pd.DataFrame([
    {"model": "Logistic Regression", **confusion_items(logistic_metrics)},
    {"model": "Random Forest", **confusion_items(rf_metrics)},
    {"model": "XGBoost", **confusion_items(xgb_metrics)}
])

confusion_bar_long = confusion_bar_df.melt(
    id_vars="model",
    value_vars=["TP", "FN", "FP", "TN"],
    var_name="confusion_type",
    value_name="count"
)

plt.figure(figsize=(11, 6))
sns.barplot(
    data=confusion_bar_long,
    x="confusion_type",
    y="count",
    hue="model"
)
plt.title("Confusion Matrix Bar Comparison")
plt.xlabel("Confusion Matrix Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../reports/figures/confusion_matrix_bar.png", dpi=150)
plt.show()
```

Nhận xét cần ghi:

- Model có `TP` cao hơn bắt được nhiều ca diabetic hơn.
- Model có `FN` thấp hơn bỏ sót ít ca bệnh hơn.
- Trong y tế, giảm `FN` thường quan trọng hơn tăng accuracy tổng thể.

### 7.6. Confusion matrix dạng ma trận

```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ConfusionMatrixDisplay.from_estimator(
    logistic_model,
    X_test,
    y_test,
    ax=axes[0],
    cmap="Purples"
)
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_estimator(
    rf_model,
    X_test,
    y_test,
    ax=axes[1],
    cmap="Blues"
)
axes[1].set_title("Random Forest")

ConfusionMatrixDisplay.from_estimator(
    xgb_model,
    X_test,
    y_test,
    ax=axes[2],
    cmap="Greens"
)
axes[2].set_title("XGBoost")

plt.tight_layout()
plt.savefig("../reports/figures/confusion_matrix.png", dpi=150)
plt.show()
```

### 7.7. Threshold tuning

Mặc định threshold là `0.5`. Với bài toán sức khỏe, có thể muốn tăng recall cho lớp diabetic bằng cách giảm threshold.

```python
threshold = 0.35
y_pred_custom = (xgb_proba >= threshold).astype(int)

print(classification_report(y_test, y_pred_custom))
print(confusion_matrix(y_test, y_pred_custom))
```

Ý nghĩa:

- Threshold thấp hơn giúp phát hiện nhiều ca nguy cơ hơn.
- Nhưng false positive cũng tăng.
- Cần chọn threshold theo mục tiêu nghiệp vụ.

---

## 8. Backend API

Có thể dùng Flask hoặc FastAPI. Với project này, Flask là lựa chọn đơn giản và dễ triển khai.

File đề xuất:

```text
backend/app.py
```

### 8.1. API endpoints

#### `POST /predict`

Nhận JSON input gồm 21 feature:

```json
{
  "HighBP": 1,
  "HighChol": 1,
  "CholCheck": 1,
  "BMI": 30,
  "Smoker": 0,
  "Stroke": 0,
  "HeartDiseaseorAttack": 0,
  "PhysActivity": 1,
  "Fruits": 1,
  "Veggies": 1,
  "HvyAlcoholConsump": 0,
  "AnyHealthcare": 1,
  "NoDocbcCost": 0,
  "GenHlth": 3,
  "MentHlth": 2,
  "PhysHlth": 5,
  "DiffWalk": 0,
  "Sex": 1,
  "Age": 9,
  "Education": 5,
  "Income": 6
}
```

Response:

```json
{
  "model": "xgboost",
  "probability": 0.72,
  "label": 1,
  "label_text": "High diabetes risk"
}
```

#### `GET /model-stats`

Response:

```json
{
  "logistic_regression": {
    "accuracy": 0.80,
    "precision": 0.36,
    "recall": 0.72,
    "f1": 0.48,
    "roc_auc": 0.80,
    "confusion_matrix": [[36000, 7500], [2600, 6600]]
  },
  "random_forest": {
    "accuracy": 0.84,
    "precision": 0.48,
    "recall": 0.62,
    "f1": 0.54,
    "roc_auc": 0.82,
    "confusion_matrix": [[41000, 2500], [4200, 5000]]
  },
  "xgboost": {
    "accuracy": 0.86,
    "precision": 0.52,
    "recall": 0.65,
    "f1": 0.58,
    "roc_auc": 0.84,
    "confusion_matrix": [[41500, 2000], [3900, 5300]]
  }
}
```

Các số ở trên chỉ là ví dụ. Khi chạy training thật, cần ghi bằng metrics thực tế từ notebook.

### 8.2. Flask skeleton

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import json

app = Flask(__name__)
CORS(app)

MODEL_PATHS = {
    "logistic_regression": "../models/logistic_regression.joblib",
    "random_forest": "../models/random_forest.joblib",
    "xgboost": "../models/xgboost.joblib"
}

models = {
    name: joblib.load(path)
    for name, path in MODEL_PATHS.items()
}

feature_columns = joblib.load("../models/feature_columns.joblib")
raw_feature_columns = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]

def add_engineered_features(input_df):
    data = input_df.copy()
    data["BMI_category"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=["underweight", "normal", "overweight", "obese"],
        include_lowest=True
    )
    data["comorbidity_score"] = data[
        ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk"]
    ].sum(axis=1)
    data["healthy_lifestyle"] = (
        data["PhysActivity"]
        + data["Fruits"]
        + data["Veggies"]
        + (1 - data["Smoker"])
        + (1 - data["HvyAlcoholConsump"])
    )
    return data

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    model_name = payload.get("model", "xgboost")
    input_data = payload.get("features", payload)

    if model_name not in models:
        return jsonify({"error": "Unsupported model"}), 400

    missing_features = [col for col in raw_feature_columns if col not in input_data]
    if missing_features:
        return jsonify({
            "error": "Missing features",
            "missing_features": missing_features
        }), 400

    X_raw = pd.DataFrame([input_data], columns=raw_feature_columns)
    X_input = add_engineered_features(X_raw)
    X_input = X_input.reindex(columns=feature_columns)
    model = models[model_name]

    probability = float(model.predict_proba(X_input)[:, 1][0])
    label = int(probability >= 0.5)

    return jsonify({
        "model": model_name,
        "probability": probability,
        "label": label,
        "label_text": "High diabetes risk" if label == 1 else "Low diabetes risk"
    })

@app.route("/model-stats", methods=["GET"])
def model_stats():
    with open("../reports/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return jsonify(metrics)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

Nếu có Feature Engineering trong notebook, backend phải áp dụng đúng cùng logic trước khi gọi model. Frontend vẫn chỉ cần gửi 21 feature gốc; `BMI_category`, `comorbidity_score`, `healthy_lifestyle` nên được backend tạo lại để tránh người dùng gửi sai feature dẫn xuất.

### 8.3. Backend requirements

File:

```text
backend/requirements.txt
```

Nội dung:

```text
flask
flask-cors
pandas
scikit-learn
xgboost
joblib
```

---

## 9. Frontend HTML + Chart.js

Phần này là hướng dẫn frontend chi tiết trong tài liệu tổng thể. Frontend dùng HTML, CSS, JavaScript thuần và Chart.js để nhập 21 chỉ số sức khỏe, gọi backend Flask, hiển thị xác suất dự đoán và so sánh 3 mô hình:

- Logistic Regression baseline
- Random Forest
- XGBoost

Backend giả định chạy tại:

```text
http://localhost:5000
```

Các endpoint cần dùng:

```text
POST /predict
GET /model-stats
```

---

### 9.1. Mục tiêu frontend

Frontend cần có các chức năng chính:

- Form nhập đầy đủ 21 feature của dataset.
- Select chọn mô hình: `logistic_regression`, `random_forest`, `xgboost`.
- Gửi dữ liệu tới `POST /predict`.
- Hiển thị xác suất nguy cơ tiểu đường.
- Hiển thị nhãn dự đoán: nguy cơ thấp hoặc nguy cơ cao.
- Gọi `GET /model-stats` để lấy metrics.
- Hiển thị bảng so sánh 3 mô hình.
- Vẽ confusion matrix hoặc bảng confusion matrix.
- Vẽ ROC curve bằng Chart.js nếu backend hoặc file JSON có dữ liệu ROC.

Lưu ý: frontend chỉ hiển thị kết quả tham khảo nguy cơ, không thay thế chẩn đoán y tế.

---

### 9.2. Cấu trúc thư mục frontend

```text
frontend/
├── index.html
├── styles.css
└── app.js
```

Vai trò từng file:

- `index.html`: khung giao diện, form nhập liệu, vùng hiển thị kết quả, canvas Chart.js.
- `styles.css`: bố cục, màu sắc, responsive layout.
- `app.js`: xử lý form, gọi API, render kết quả, render metrics và chart.

Nếu chạy frontend bằng static server:

```bash
python -m http.server 8000
```

Truy cập:

```text
http://localhost:8000/frontend/
```

---

### 9.3. Danh sách feature cần nhập

Frontend phải gửi đúng 21 feature, đúng tên cột như khi train model.

```text
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income
```

Target `Diabetes_binary` không được đưa vào form vì đây là nhãn cần dự đoán.

---

### 9.4. Kiểu input đề xuất

#### 9.4.1. Binary select

Các biến nhận `0` hoặc `1`:

```text
HighBP
HighChol
CholCheck
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
DiffWalk
Sex
```

Gợi ý label:

- `0`: Không
- `1`: Có

Riêng `Sex` có thể ghi:

- `0`: Nữ
- `1`: Nam

#### 9.4.2. Number input

Các biến nên dùng number input:

```text
BMI
MentHlth
PhysHlth
```

Range gợi ý:

- `BMI`: 10 đến 80
- `MentHlth`: 0 đến 30
- `PhysHlth`: 0 đến 30

#### 9.4.3. Ordinal select

Các biến ordinal:

```text
GenHlth: 1 đến 5
Age: 1 đến 13
Education: 1 đến 6
Income: 1 đến 8
```

Gợi ý mô tả:

- `GenHlth`: 1 là rất tốt, 5 là kém.
- `Age`: nhóm tuổi mã hóa từ 1 đến 13 theo dataset BRFSS.
- `Education`: mức học vấn mã hóa từ 1 đến 6.
- `Income`: nhóm thu nhập mã hóa từ 1 đến 8.

---

### 9.5. Thiết kế màn hình

Trang chính nên chia thành 3 vùng:

1. Khu vực form nhập chỉ số sức khỏe.
2. Khu vực kết quả dự đoán.
3. Khu vực so sánh mô hình.

Bố cục gợi ý:

```text
┌──────────────────────────────────────────────┐
│ Header: Diabetes Risk Prediction              │
├───────────────────────┬──────────────────────┤
│ Form nhập 21 feature  │ Kết quả dự đoán       │
│ Chọn model            │ Probability bar       │
│ Nút Predict           │ Label + selected model│
├───────────────────────┴──────────────────────┤
│ Tabs: Metrics | Confusion Matrix | ROC Curve  │
└──────────────────────────────────────────────┘
```

Không nên làm landing page dài. Màn hình đầu tiên nên là công cụ nhập liệu và dự đoán.

---

### 9.6. `index.html` skeleton

```html
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Diabetes Risk Prediction</title>
    <link rel="stylesheet" href="styles.css" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  </head>
  <body>
    <main class="app-shell">
      <header class="app-header">
        <div>
          <h1>Dự đoán nguy cơ tiểu đường</h1>
          <p>Nhập các chỉ số sức khỏe để xem xác suất dự đoán từ mô hình.</p>
        </div>
      </header>

      <section class="workspace">
        <form id="predictionForm" class="panel form-panel">
          <div class="section-title">
            <h2>Thông tin đầu vào</h2>
          </div>

          <label>
            Mô hình
            <select name="model" id="modelSelect">
              <option value="xgboost">XGBoost</option>
              <option value="random_forest">Random Forest</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
          </label>

          <div id="featureFields" class="feature-grid"></div>

          <button type="submit" class="primary-button">Dự đoán</button>
          <p id="formError" class="error-message" hidden></p>
        </form>

        <section class="panel result-panel">
          <div class="section-title">
            <h2>Kết quả dự đoán</h2>
          </div>

          <div class="result-summary">
            <span id="predictionLabel">Chưa có dự đoán</span>
            <strong id="probabilityText">0%</strong>
          </div>

          <div class="probability-track">
            <div id="probabilityBar" class="probability-bar"></div>
          </div>

          <dl class="result-details">
            <div>
              <dt>Mô hình</dt>
              <dd id="selectedModelText">-</dd>
            </div>
            <div>
              <dt>Nhãn</dt>
              <dd id="labelValue">-</dd>
            </div>
          </dl>
        </section>
      </section>

      <section class="panel comparison-panel">
        <div class="tabs" role="tablist">
          <button type="button" class="tab-button active" data-tab="metrics">Metrics</button>
          <button type="button" class="tab-button" data-tab="confusion">Confusion Matrix</button>
          <button type="button" class="tab-button" data-tab="roc">ROC Curve</button>
        </div>

        <div id="metricsTab" class="tab-content active">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody id="metricsBody"></tbody>
          </table>
        </div>

        <div id="confusionTab" class="tab-content">
          <div id="confusionMatrices" class="matrix-grid"></div>
        </div>

        <div id="rocTab" class="tab-content">
          <canvas id="rocChart" height="120"></canvas>
        </div>
      </section>
    </main>

    <script src="app.js"></script>
  </body>
</html>
```

---

### 9.7. `app.js` skeleton

#### 9.7.1. Khai báo API và metadata

```javascript
const API_BASE_URL = "http://localhost:5000";

const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost"
};

const BINARY_FIELDS = [
  "HighBP",
  "HighChol",
  "CholCheck",
  "Smoker",
  "Stroke",
  "HeartDiseaseorAttack",
  "PhysActivity",
  "Fruits",
  "Veggies",
  "HvyAlcoholConsump",
  "AnyHealthcare",
  "NoDocbcCost",
  "DiffWalk",
  "Sex"
];

const NUMBER_FIELDS = {
  BMI: { min: 10, max: 80, value: 28 },
  MentHlth: { min: 0, max: 30, value: 0 },
  PhysHlth: { min: 0, max: 30, value: 0 }
};

const ORDINAL_FIELDS = {
  GenHlth: { min: 1, max: 5, value: 3 },
  Age: { min: 1, max: 13, value: 8 },
  Education: { min: 1, max: 6, value: 5 },
  Income: { min: 1, max: 8, value: 6 }
};

const FEATURE_ORDER = [
  "HighBP",
  "HighChol",
  "CholCheck",
  "BMI",
  "Smoker",
  "Stroke",
  "HeartDiseaseorAttack",
  "PhysActivity",
  "Fruits",
  "Veggies",
  "HvyAlcoholConsump",
  "AnyHealthcare",
  "NoDocbcCost",
  "GenHlth",
  "MentHlth",
  "PhysHlth",
  "DiffWalk",
  "Sex",
  "Age",
  "Education",
  "Income"
];
```

#### 9.7.2. Render form động

```javascript
function renderFeatureFields() {
  const container = document.querySelector("#featureFields");

  container.innerHTML = FEATURE_ORDER.map((field) => {
    if (BINARY_FIELDS.includes(field)) {
      const zeroLabel = field === "Sex" ? "Nữ" : "Không";
      const oneLabel = field === "Sex" ? "Nam" : "Có";

      return `
        <label>
          ${field}
          <select name="${field}">
            <option value="0">${zeroLabel}</option>
            <option value="1">${oneLabel}</option>
          </select>
        </label>
      `;
    }

    if (NUMBER_FIELDS[field]) {
      const config = NUMBER_FIELDS[field];

      return `
        <label>
          ${field}
          <input
            type="number"
            name="${field}"
            min="${config.min}"
            max="${config.max}"
            value="${config.value}"
            required
          />
        </label>
      `;
    }

    const config = ORDINAL_FIELDS[field];
    const options = Array.from(
      { length: config.max - config.min + 1 },
      (_, index) => config.min + index
    )
      .map((value) => (
        `<option value="${value}" ${value === config.value ? "selected" : ""}>${value}</option>`
      ))
      .join("");

    return `
      <label>
        ${field}
        <select name="${field}">
          ${options}
        </select>
      </label>
    `;
  }).join("");
}
```

#### 9.7.3. Tạo payload đúng định dạng backend

Backend trong kế hoạch nhận payload dạng:

```json
{
  "model": "xgboost",
  "features": {
    "HighBP": 1,
    "BMI": 30
  }
}
```

Hàm tạo payload:

```javascript
function buildPayload(form) {
  const formData = new FormData(form);
  const selectedModel = formData.get("model");
  const features = {};

  FEATURE_ORDER.forEach((field) => {
    features[field] = Number(formData.get(field));
  });

  return {
    model: selectedModel,
    features
  };
}
```

#### 9.7.4. Gọi API dự đoán

```javascript
async function predict(payload) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Prediction request failed");
  }

  return data;
}
```

#### 9.7.5. Render kết quả dự đoán

```javascript
function renderPredictionResult(result) {
  const percent = Math.round(result.probability * 100);

  document.querySelector("#probabilityText").textContent = `${percent}%`;
  document.querySelector("#probabilityBar").style.width = `${percent}%`;
  document.querySelector("#predictionLabel").textContent = result.label_text;
  document.querySelector("#selectedModelText").textContent = MODEL_LABELS[result.model] || result.model;
  document.querySelector("#labelValue").textContent = String(result.label);
}
```

#### 9.7.6. Submit form

```javascript
async function handleSubmit(event) {
  event.preventDefault();

  const form = event.currentTarget;
  const errorElement = document.querySelector("#formError");

  try {
    errorElement.hidden = true;
    const payload = buildPayload(form);
    const result = await predict(payload);
    renderPredictionResult(result);
  } catch (error) {
    errorElement.textContent = error.message;
    errorElement.hidden = false;
  }
}
```

---

### 9.8. Render model stats

Backend `GET /model-stats` trả về metrics đã lưu từ notebook, ví dụ:

```json
{
  "logistic_regression": {
    "accuracy": 0.80,
    "precision": 0.36,
    "recall": 0.72,
    "f1": 0.48,
    "roc_auc": 0.80,
    "confusion_matrix": [[36000, 7500], [2600, 6600]]
  },
  "random_forest": {
    "accuracy": 0.84,
    "precision": 0.48,
    "recall": 0.62,
    "f1": 0.54,
    "roc_auc": 0.82,
    "confusion_matrix": [[41000, 2500], [4200, 5000]]
  },
  "xgboost": {
    "accuracy": 0.86,
    "precision": 0.52,
    "recall": 0.65,
    "f1": 0.58,
    "roc_auc": 0.84,
    "confusion_matrix": [[41500, 2000], [3900, 5300]]
  }
}
```

#### 9.8.1. Fetch stats

```javascript
async function fetchModelStats() {
  const response = await fetch(`${API_BASE_URL}/model-stats`);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Cannot load model stats");
  }

  return data;
}
```

#### 9.8.2. Render metrics table

```javascript
function formatMetric(value) {
  if (typeof value !== "number") {
    return "-";
  }

  return value.toFixed(3);
}

function renderMetricsTable(stats) {
  const rows = Object.entries(stats).map(([modelName, metrics]) => `
    <tr>
      <td>${MODEL_LABELS[modelName] || modelName}</td>
      <td>${formatMetric(metrics.accuracy)}</td>
      <td>${formatMetric(metrics.precision)}</td>
      <td>${formatMetric(metrics.recall)}</td>
      <td>${formatMetric(metrics.f1)}</td>
      <td>${formatMetric(metrics.roc_auc)}</td>
    </tr>
  `);

  document.querySelector("#metricsBody").innerHTML = rows.join("");
}
```

#### 9.8.3. Render confusion matrix

Quy ước confusion matrix:

```text
[[TN, FP],
 [FN, TP]]
```

Code render:

```javascript
function renderConfusionMatrices(stats) {
  const html = Object.entries(stats).map(([modelName, metrics]) => {
    const matrix = metrics.confusion_matrix || [[0, 0], [0, 0]];
    const [[tn, fp], [fn, tp]] = matrix;

    return `
      <article class="matrix-card">
        <h3>${MODEL_LABELS[modelName] || modelName}</h3>
        <div class="confusion-grid">
          <div><span>TN</span><strong>${tn}</strong></div>
          <div><span>FP</span><strong>${fp}</strong></div>
          <div><span>FN</span><strong>${fn}</strong></div>
          <div><span>TP</span><strong>${tp}</strong></div>
        </div>
      </article>
    `;
  });

  document.querySelector("#confusionMatrices").innerHTML = html.join("");
}
```

---

### 9.9. ROC curve bằng Chart.js

`/model-stats` trong kế hoạch cơ bản chỉ trả metrics và confusion matrix. Nếu muốn frontend vẽ ROC curve thật, backend nên trả thêm dữ liệu ROC theo dạng:

```json
{
  "logistic_regression": {
    "fpr": [0.0, 0.12, 0.25, 1.0],
    "tpr": [0.0, 0.48, 0.68, 1.0],
    "auc": 0.80
  },
  "random_forest": {
    "fpr": [0.0, 0.1, 0.2, 1.0],
    "tpr": [0.0, 0.5, 0.7, 1.0],
    "auc": 0.82
  },
  "xgboost": {
    "fpr": [0.0, 0.08, 0.18, 1.0],
    "tpr": [0.0, 0.55, 0.75, 1.0],
    "auc": 0.84
  }
}
```

Nếu chưa có dữ liệu ROC, frontend vẫn có thể hiển thị metrics và confusion matrix trước.

Code render ROC:

```javascript
let rocChart;

function toRocPoints(rocData) {
  return rocData.fpr.map((fpr, index) => ({
    x: fpr,
    y: rocData.tpr[index]
  }));
}

function renderRocChart(rocStats) {
  const canvas = document.querySelector("#rocChart");

  if (rocChart) {
    rocChart.destroy();
  }

  rocChart = new Chart(canvas, {
    type: "line",
    data: {
      datasets: [
        {
          label: `Logistic Regression AUC = ${rocStats.logistic_regression.auc.toFixed(3)}`,
          data: toRocPoints(rocStats.logistic_regression),
          borderColor: "#7c3aed",
          backgroundColor: "transparent",
          parsing: false
        },
        {
          label: `Random Forest AUC = ${rocStats.random_forest.auc.toFixed(3)}`,
          data: toRocPoints(rocStats.random_forest),
          borderColor: "#2563eb",
          backgroundColor: "transparent",
          parsing: false
        },
        {
          label: `XGBoost AUC = ${rocStats.xgboost.auc.toFixed(3)}`,
          data: toRocPoints(rocStats.xgboost),
          borderColor: "#16a34a",
          backgroundColor: "transparent",
          parsing: false
        }
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: "linear",
          min: 0,
          max: 1,
          title: {
            display: true,
            text: "False Positive Rate"
          }
        },
        y: {
          min: 0,
          max: 1,
          title: {
            display: true,
            text: "True Positive Rate"
          }
        }
      }
    }
  });
}
```

---

### 9.10. Tabs

```javascript
function setupTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  const contents = {
    metrics: document.querySelector("#metricsTab"),
    confusion: document.querySelector("#confusionTab"),
    roc: document.querySelector("#rocTab")
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;

      buttons.forEach((item) => item.classList.remove("active"));
      Object.values(contents).forEach((content) => content.classList.remove("active"));

      button.classList.add("active");
      contents[tab].classList.add("active");
    });
  });
}
```

---

### 9.11. Khởi tạo app

```javascript
async function init() {
  renderFeatureFields();
  setupTabs();

  document.querySelector("#predictionForm").addEventListener("submit", handleSubmit);

  try {
    const stats = await fetchModelStats();
    renderMetricsTable(stats);
    renderConfusionMatrices(stats);
  } catch (error) {
    console.error(error);
  }
}

init();
```

Nếu backend bổ sung endpoint hoặc field chứa ROC data, gọi thêm:

```javascript
renderRocChart(rocStats);
```

---

### 9.12. `styles.css` skeleton

```css
:root {
  color-scheme: light;
  --bg: #f6f7fb;
  --panel: #ffffff;
  --text: #172033;
  --muted: #667085;
  --border: #d9deea;
  --primary: #2563eb;
  --danger: #dc2626;
  --success: #16a34a;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
}

.app-shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 24px 0 40px;
}

.app-header {
  margin-bottom: 20px;
}

.app-header h1 {
  margin: 0 0 6px;
  font-size: 28px;
}

.app-header p {
  margin: 0;
  color: var(--muted);
}

.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.8fr);
  gap: 16px;
  align-items: start;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px;
}

.section-title h2 {
  margin: 0 0 14px;
  font-size: 18px;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

label {
  display: grid;
  gap: 6px;
  font-size: 13px;
  color: var(--muted);
}

input,
select,
button {
  min-height: 38px;
  border-radius: 6px;
  font: inherit;
}

input,
select {
  width: 100%;
  border: 1px solid var(--border);
  padding: 0 10px;
  background: #ffffff;
  color: var(--text);
}

.primary-button {
  width: 100%;
  margin-top: 14px;
  border: 0;
  background: var(--primary);
  color: #ffffff;
  cursor: pointer;
}

.result-summary {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  margin-bottom: 14px;
}

.result-summary strong {
  font-size: 32px;
}

.probability-track {
  height: 14px;
  overflow: hidden;
  border-radius: 999px;
  background: #e8edf7;
}

.probability-bar {
  width: 0%;
  height: 100%;
  background: var(--success);
  transition: width 180ms ease;
}

.result-details {
  display: grid;
  gap: 10px;
  margin: 18px 0 0;
}

.result-details div {
  display: flex;
  justify-content: space-between;
  gap: 16px;
}

.result-details dt {
  color: var(--muted);
}

.result-details dd {
  margin: 0;
  font-weight: 700;
}

.comparison-panel {
  margin-top: 16px;
}

.tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 14px;
}

.tab-button {
  border: 1px solid var(--border);
  background: #ffffff;
  padding: 0 12px;
  cursor: pointer;
}

.tab-button.active {
  border-color: var(--primary);
  color: var(--primary);
}

.tab-content {
  display: none;
}

.tab-content.active {
  display: block;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  border-bottom: 1px solid var(--border);
  padding: 10px;
  text-align: left;
}

.matrix-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.matrix-card h3 {
  margin: 0 0 10px;
  font-size: 15px;
}

.confusion-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.confusion-grid div {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px;
}

.confusion-grid span {
  display: block;
  color: var(--muted);
  font-size: 12px;
}

.confusion-grid strong {
  font-size: 18px;
}

.error-message {
  color: var(--danger);
  margin: 10px 0 0;
}

@media (max-width: 900px) {
  .workspace,
  .matrix-grid {
    grid-template-columns: 1fr;
  }

  .feature-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .app-shell {
    width: min(100% - 20px, 1180px);
  }

  .feature-grid {
    grid-template-columns: 1fr;
  }

  .tabs {
    flex-wrap: wrap;
  }
}
```

---

### 9.13. Đồng bộ với backend

Frontend cần khớp backend ở các điểm sau:

- `model` gửi lên phải là một trong: `logistic_regression`, `random_forest`, `xgboost`.
- `features` phải có đủ 21 key và đúng tên cột.
- Tất cả giá trị gửi lên phải là number, không gửi string.
- Backend phải bật CORS nếu frontend chạy ở port khác.
- Backend phải load đủ các model:
  - `models/logistic_regression.joblib`
  - `models/random_forest.joblib`
  - `models/xgboost.joblib`
  - `models/feature_columns.joblib`

Ví dụ request hợp lệ:

```json
{
  "model": "xgboost",
  "features": {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 30,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 3,
    "MentHlth": 2,
    "PhysHlth": 5,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 9,
    "Education": 5,
    "Income": 6
  }
}
```

Ví dụ response hợp lệ:

```json
{
  "model": "xgboost",
  "probability": 0.72,
  "label": 1,
  "label_text": "High diabetes risk"
}
```

---

### 9.14. Checklist hoàn thành frontend

- [ ] Có file `frontend/index.html`.
- [ ] Có file `frontend/styles.css`.
- [ ] Có file `frontend/app.js`.
- [ ] Form nhập đủ 21 feature.
- [ ] Có select chọn `Logistic Regression`, `Random Forest`, `XGBoost`.
- [ ] Payload gửi đúng dạng `{ model, features }`.
- [ ] Tất cả feature được ép kiểu `Number`.
- [ ] Gọi được `POST /predict`.
- [ ] Hiển thị xác suất dự đoán.
- [ ] Hiển thị nhãn dự đoán.
- [ ] Gọi được `GET /model-stats`.
- [ ] Hiển thị bảng metrics cho 3 mô hình.
- [ ] Hiển thị confusion matrix cho 3 mô hình.
- [ ] Vẽ ROC curve bằng Chart.js nếu có ROC data.
- [ ] Giao diện responsive trên mobile.
- [ ] Có thông báo lỗi khi backend không chạy hoặc request sai.

---

### 9.15. Thứ tự triển khai đề xuất

1. Tạo `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`.
2. Render form động từ danh sách `FEATURE_ORDER`.
3. Build payload và log ra console để kiểm tra đủ 21 feature.
4. Chạy backend Flask tại `http://localhost:5000`.
5. Gọi thử `POST /predict`.
6. Render probability bar và label.
7. Gọi `GET /model-stats`.
8. Render metrics table và confusion matrix.
9. Bổ sung ROC data vào backend hoặc file JSON.
10. Render ROC curve bằng Chart.js.

## 10. Quy trình triển khai từng bước

### Bước 1: Chuẩn bị môi trường

Tạo virtual environment:

```bash
python -m venv .venv
```

Kích hoạt trên Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Cài thư viện:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap joblib flask flask-cors notebook
```

Xuất requirements:

```bash
pip freeze > requirements.txt
```

### Bước 2: Tạo thư mục dự án

```bash
mkdir notebooks reports reports/figures models src src/data src/models src/utils backend frontend data/processed
```

### Bước 3: Làm EDA

Thực hiện trong:

```text
notebooks/01_eda.ipynb
```

Output cần có:

- `target_distribution.png`
- `correlation_heatmap.png`
- `continuous_histograms.png`
- `boxplots.png`
- Nhận xét mất cân bằng lớp.
- Nhận xét feature liên quan với target.

### Bước 4: Tiền xử lý và train model

Thực hiện trong:

```text
notebooks/02_preprocessing_modeling.ipynb
```

Output cần có:

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/feature_columns.joblib`
- `reports/metrics.json`
- Feature Engineering gồm `BMI_category`, `comorbidity_score`, `healthy_lifestyle`.
- Encoding cho `BMI_category`.
- StandardScaler fit trên train thông qua pipeline.
- SMOTE hoặc class balancing chỉ áp dụng trên train.

### Bước 5: Đánh giá và giải thích

Thực hiện trong:

```text
notebooks/03_model_explainability.ipynb
```

Output cần có:

- Grouped bar metrics so sánh 3 mô hình.
- ROC curve so sánh 3 mô hình.
- Precision-Recall curve so sánh 3 mô hình.
- Feature Importance so sánh Random Forest và XGBoost.
- Confusion matrix dạng bar.
- Confusion matrix dạng ma trận.
- SHAP summary plot nếu dùng XGBoost.

### Bước 6: Xây backend

Thực hiện:

```bash
cd backend
python app.py
```

API chạy tại:

```text
http://localhost:5000
```

Kiểm thử:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"xgboost\",\"features\":{\"HighBP\":1,\"HighChol\":1,\"CholCheck\":1,\"BMI\":30,\"Smoker\":0,\"Stroke\":0,\"HeartDiseaseorAttack\":0,\"PhysActivity\":1,\"Fruits\":1,\"Veggies\":1,\"HvyAlcoholConsump\":0,\"AnyHealthcare\":1,\"NoDocbcCost\":0,\"GenHlth\":3,\"MentHlth\":2,\"PhysHlth\":5,\"DiffWalk\":0,\"Sex\":1,\"Age\":9,\"Education\":5,\"Income\":6}}"
```

### Bước 7: Xây frontend

Mở:

```text
frontend/index.html
```

Hoặc chạy server tĩnh:

```bash
python -m http.server 8000
```

Truy cập:

```text
http://localhost:8000/frontend/
```

---

## 11. Checklist hoàn thành project

### EDA

- [ ] Load đúng file CSV.
- [ ] Kiểm tra shape: `253,680` dòng, `22` cột gồm target.
- [ ] Kiểm tra không có missing values.
- [ ] Chạy `df.describe()`.
- [ ] Chạy `value_counts()` cho target và các biến phân loại.
- [ ] Vẽ phân phối target.
- [ ] Nhận xét mất cân bằng lớp.
- [ ] Vẽ correlation heatmap.
- [ ] Vẽ histogram cho `BMI`, `Age`, `MentHlth`, `PhysHlth`.
- [ ] Vẽ boxplot các biến liên tục theo target.

### Tiền xử lý

- [ ] Kiểm tra chất lượng dữ liệu: null, outlier, giá trị hợp lệ.
- [ ] Tạo `BMI_category`.
- [ ] Tạo `comorbidity_score`.
- [ ] Tạo `healthy_lifestyle`.
- [ ] Encoding `BMI_category` nếu dùng feature này.
- [ ] Tách `X`, `y`.
- [ ] Chia train/test theo tỷ lệ `80/20`.
- [ ] Dùng `stratify=y`.
- [ ] Scale `BMI`, `MentHlth`, `PhysHlth`, `Age` và các feature numeric mới nếu cần.
- [ ] Xử lý mất cân bằng bằng `class_weight`, `scale_pos_weight` hoặc SMOTE.
- [ ] Nếu dùng SMOTE, chỉ fit/resample trên train.
- [ ] Feature Selection tùy chọn và chỉ fit trên train.
- [ ] Không gây data leakage.

### Modeling

- [ ] Train Logistic Regression baseline với `StandardScaler`.
- [ ] Train Random Forest.
- [ ] Train XGBoost.
- [ ] So sánh metric.
- [ ] Lưu model bằng `joblib`.
- [ ] Lưu danh sách feature.
- [ ] Lưu metrics ra JSON.

### Đánh giá

- [ ] Báo cáo Accuracy.
- [ ] Báo cáo Precision.
- [ ] Báo cáo Recall.
- [ ] Báo cáo F1-score.
- [ ] Báo cáo ROC-AUC.
- [ ] Vẽ grouped bar metrics.
- [ ] Vẽ ROC curve của 3 mô hình trên cùng chart.
- [ ] Vẽ Precision-Recall curve.
- [ ] Vẽ Feature Importance của Random Forest và XGBoost.
- [ ] Vẽ confusion matrix dạng bar.
- [ ] Vẽ confusion matrix dạng ma trận.
- [ ] Nhận xét false positive và false negative.

### Backend

- [ ] Có endpoint `POST /predict`.
- [ ] Có endpoint `GET /model-stats`.
- [ ] Validate đủ 21 feature.
- [ ] Tạo lại `BMI_category`, `comorbidity_score`, `healthy_lifestyle` trong backend trước khi predict.
- [ ] Trả về probability.
- [ ] Trả về label.
- [ ] Load model từ `models/`.

### Frontend

- [ ] Có form nhập chỉ số sức khỏe.
- [ ] Có lựa chọn model.
- [ ] Gọi được API `/predict`.
- [ ] Hiển thị xác suất bằng thanh phần trăm.
- [ ] Hiển thị nhãn dự đoán.
- [ ] Có tab so sánh mô hình.
- [ ] Có grouped bar metrics bằng Chart.js.
- [ ] Có ROC curve bằng Chart.js.
- [ ] Có Precision-Recall curve bằng Chart.js.
- [ ] Có Feature Importance toggle giữa Random Forest và XGBoost.
- [ ] Có confusion matrix dạng bar hoặc bảng metric.

---

## 12. Gợi ý nội dung báo cáo cuối

Báo cáo nên có các phần:

1. Giới thiệu bài toán
2. Mô tả dataset
3. EDA
4. Tiền xử lý
5. Mô hình đề xuất
6. Kết quả đánh giá
7. So sánh Logistic Regression, Random Forest và XGBoost
8. Giải thích feature quan trọng
9. Thiết kế backend API
10. Thiết kế frontend
11. Hạn chế
12. Hướng phát triển

Các hạn chế nên nêu:

- Dataset mất cân bằng.
- Dữ liệu dạng khảo sát nên có thể có sai lệch tự báo cáo.
- `Diabetes_binary` chỉ là phân loại nhị phân, không phân biệt tiền tiểu đường hoặc loại tiểu đường.
- Model chỉ hỗ trợ tham khảo nguy cơ, không thay thế chẩn đoán y tế.

Hướng phát triển:

- Tuning hyperparameter bằng GridSearchCV hoặc RandomizedSearchCV.
- Tuning thêm Logistic Regression, Random Forest và XGBoost để tối ưu F1-score hoặc ROC-AUC.
- Thử SMOTE hoặc undersampling.
- Tối ưu threshold theo mục tiêu recall.
- Thêm SHAP explanation cho từng prediction trên frontend.
- Dockerize backend và frontend.
- Lưu lịch sử prediction vào database.

