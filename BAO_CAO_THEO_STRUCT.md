# CHƯƠNG I: GIỚI THIỆU

## 1.1 Data Mining là gì?
Data Mining là quá trình khai phá tri thức từ dữ liệu thông qua thống kê, học máy và trực quan hóa, nhằm phát hiện các mẫu (patterns) và quy luật ẩn. Trong y tế, Data Mining hỗ trợ dự báo nguy cơ bệnh, sàng lọc sớm và tối ưu hóa quyết định điều trị.

## 1.2 Tổng quan bài toán
Đề tài tập trung vào bài toán phân loại nhị phân nguy cơ tiểu đường dựa trên dữ liệu khảo sát sức khỏe BRFSS 2015 của CDC. Mục tiêu là dự đoán nhãn `Diabetes_binary` với hai lớp: không mắc (0) và mắc (1).

## 1.3 Mục tiêu nghiên cứu
- Xây dựng quy trình tiền xử lý dữ liệu y tế đáng tin cậy.
- Thử nghiệm và so sánh các mô hình phân loại phổ biến.
- Lựa chọn mô hình phù hợp với dữ liệu mất cân bằng.
- Tích hợp mô hình vào hệ thống hỗ trợ dự đoán và tái huấn luyện.

# CHƯƠNG II: GIỚI THIỆU DỮ LIỆU

## 2.1 Nguồn dữ liệu (CDC)
Dữ liệu sử dụng là **Diabetes Binary Health Indicators BRFSS 2015**, trích từ khảo sát **Behavioral Risk Factor Surveillance System (BRFSS) 2015** của CDC. Bộ dữ liệu có:
- 253,680 dòng.
- 22 cột gồm 1 nhãn và 21 đặc trưng.
- Định dạng số hóa (nhiều cột nhị phân được lưu dưới dạng 0.0 và 1.0).

## 2.2 Mô tả các thuộc tính
Các nhóm thuộc tính chính:
- **Chỉ số lâm sàng:** HighBP, HighChol, BMI, Stroke, HeartDiseaseorAttack, ...
- **Hành vi và lối sống:** Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, ...
- **Tiếp cận y tế:** AnyHealthcare, NoDocbcCost, DiffWalk, ...
- **Tự đánh giá sức khỏe:** GenHlth, MentHlth, PhysHlth.
- **Nhân khẩu học:** Sex, Age, Education, Income.

## 2.3 Phân phối nhãn (mất cân bằng ~14%)
Tỷ lệ nhãn dương khoảng **13.93%**, nhãn âm khoảng **86.07%**. Dữ liệu mất cân bằng, cần ưu tiên các chỉ số như Recall, F1-score và ROC-AUC để đánh giá mô hình.

## 2.4 Đường dẫn dữ liệu trong code
- File dữ liệu gốc: [data/raw/diabetes_binary_health_indicators_BRFSS2015.csv](data/raw/diabetes_binary_health_indicators_BRFSS2015.csv)
- Khai báo trong: [src/utils/constants.py](src/utils/constants.py)

```python
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET_COL = "Diabetes_binary"
RAW_FEATURE_COLUMNS = [
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
	"Income",
]
```

# CHƯƠNG III: TIỀN XỬ LÝ DỮ LIỆU

## 3.1 Kiểm tra chất lượng dữ liệu
### 3.1.1 Missing Values
Kiểm tra dữ liệu thiếu để xác định mức độ ảnh hưởng và chiến lược xử lý (loại bỏ hoặc điền giá trị phù hợp).

### 3.1.2 Duplicates
Phát hiện và loại bỏ bản ghi trùng lặp nhằm đảm bảo mỗi quan sát là độc lập.

### 3.1.3 Outliers
Phân tích ngoại lệ ở các biến liên tục (BMI, MentHlth, PhysHlth) để tránh làm sai lệch phân phối.

### 3.1.4 Code kiểm tra chất lượng dữ liệu (đường dẫn)
- Kiểm tra tổng quan trong pipeline training: [src/data/preprocess.py](src/data/preprocess.py)
- Validate CSV khi upload: [src/data/validate.py](src/data/validate.py)
- Làm sạch dữ liệu khi upload: [src/data/clean.py](src/data/clean.py)

```python
def validate_data_quality(df):
	missing = df.isnull().sum()
	# ... kiểm tra missing, binary, ordinal, outliers
```

```python
class DataValidator:
	def _validate_schema(self, df):
		required_cols = RAW_FEATURE_COLUMNS.copy()
		# ... kiểm tra thiếu/cột thừa
```

```python
def clean_data(df: pd.DataFrame, remove_duplicates: bool = True):
	df = df.drop_duplicates()
	# ... fill missing, clip ordinal, clip outliers
```

## 3.2 Feature Engineering
### 3.2.1 BMI_category (Binning)
Chuyển BMI từ liên tục thành nhóm (thiếu cân, bình thường, thừa cân, béo phì) để tăng khả năng diễn giải.

### 3.2.2 comorbidity_score (Aggregation)
Tổng hợp các biến bệnh nền thành một điểm rủi ro tổng hợp.

### 3.2.3 healthy_lifestyle (Aggregation)
Tổng hợp các hành vi lành mạnh thành một chỉ số lối sống.

### 3.2.4 Code feature engineering (đường dẫn)
Xem trong [src/data/preprocess.py](src/data/preprocess.py):

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

## 3.3 Feature Transformation
### 3.3.1 Scaling (continuous_cols)
Áp dụng chuẩn hóa (StandardScaler) cho các biến liên tục, đảm bảo mô hình học ổn định và không bị lệch thang đo. Việc chuẩn hóa chỉ fit trên tập train.

### 3.3.2 Encoding (categorical_cols)
Mã hóa các biến phân loại/ordinal để phù hợp với đầu vào mô hình (ví dụ: one-hot hoặc ordinal encoding).

### 3.3.3 Binary (giữ nguyên)
Các biến nhị phân được giữ nguyên để tránh mất thông tin.

### 3.3.4 Code scaling/encoding (đường dẫn)
Preprocessor được cấu hình trong [src/data/preprocess.py](src/data/preprocess.py) và danh sách cột trong [src/utils/constants.py](src/utils/constants.py):

```python
return ColumnTransformer(
	transformers=[
		("continuous_scaler", StandardScaler(), CONTINUOUS_COLS),
		("category_encoder", encoder, CATEGORICAL_COLS),
	],
	remainder="passthrough",
)
```

```python
CONTINUOUS_COLS = [
	"BMI",
	"MentHlth",
	"PhysHlth",
	"Age",
	"comorbidity_score",
	"healthy_lifestyle",
]

CATEGORICAL_COLS = ["BMI_category"]
```

## 3.4 Xử lý mất cân bằng lớp
### 3.4.1 SMOTE
Sinh mẫu giả cho lớp thiểu số để giảm mất cân bằng.

### 3.4.2 Class Weight
Gán trọng số cho lớp thiểu số trong hàm mất mát.

### 3.4.3 Code xử lý mất cân bằng (class weight)
Trong [src/models/train.py](src/models/train.py):

```python
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / positive_count

XGBClassifier(
	# ...
	scale_pos_weight=scale_pos_weight,
	random_state=42,
)
```

SMOTE được thực nghiệm trong notebook: [notebooks/02_preprocessing_modeling.ipynb](notebooks/02_preprocessing_modeling.ipynb)

## 3.5 Train/Test Split
### 3.5.1 Random Sampling
Chia dữ liệu theo tỷ lệ 80/20 để đánh giá tổng quát mô hình.

### 3.5.2 Stratified Sampling ✅
Dùng `stratify=y` nhằm giữ phân phối nhãn tương tự giữa train và test.

### 3.5.3 Code train/test split (stratify)
Trong [src/models/train.py](src/models/train.py):

```python
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)
```

# CHƯƠNG IV: BÀI TOÁN PHÂN LOẠI

## 4.1 Giới thiệu chung về bài toán phân loại
Phân loại là bài toán dự đoán nhãn rời rạc dựa trên đặc trưng đầu vào. Ở đây, nhãn là nguy cơ tiểu đường (0 hoặc 1).

## 4.2 Lý do sử dụng bài toán phân loại
Mục tiêu là phân loại người dùng vào hai nhóm nguy cơ, phù hợp với mô hình phân loại nhị phân.

## 4.3 Thuật toán sử dụng
### 4.3.1 Logistic Regression
Mô hình tuyến tính, dễ diễn giải, dùng làm baseline.

### 4.3.2 Random Forest
Mô hình tổ hợp nhiều cây quyết định, giảm overfitting và xử lý phi tuyến tốt.

### 4.3.3 XGBoost
Mô hình boosting mạnh cho dữ liệu tabular, tối ưu hiệu năng và thường đạt kết quả tốt.

#### Code mô hình trong dự án (đường dẫn)
- Pipeline XGBoost trong script training: [src/models/train.py](src/models/train.py)
- Baseline Logistic Regression và Random Forest: [notebooks/02_preprocessing_modeling.ipynb](notebooks/02_preprocessing_modeling.ipynb)

```python
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
					scale_pos_weight=scale_pos_weight,
					random_state=42,
					n_jobs=1,
				),
			),
		]
	),
}
```

# CHƯƠNG V: MÔ HÌNH VÀ THỰC NGHIỆM

## 5.1 Thiết lập thực nghiệm
### 5.1.1 Môi trường thực nghiệm
Mô hình được huấn luyện trong môi trường Python, sử dụng các thư viện tiêu chuẩn như pandas, scikit-learn, xgboost và joblib. Kết quả được lưu tại [reports](reports).

### 5.1.2 Tham số các mô hình
Các mô hình được huấn luyện với tham số mặc định hoặc tinh chỉnh cơ bản để đảm bảo tính công bằng khi so sánh.

### 5.1.3 Script và lệnh chạy thực tế
- Script training: [src/models/train.py](src/models/train.py)
- Lệnh chạy nhanh: `python -m src.models.train --fast`
- Lệnh chạy đầy đủ: `python -m src.models.train`

Output chính sau huấn luyện:
- [models/xgboost.joblib](models/xgboost.joblib)
- [models/feature_columns.joblib](models/feature_columns.joblib)
- [data/processed](data/processed)
- [reports/metrics.json](reports/metrics.json)
- [reports/curves.json](reports/curves.json)
- [reports/feature_importance.json](reports/feature_importance.json)
- [reports/feature_importance_vi.json](reports/feature_importance_vi.json)
- [reports/training_summary.json](reports/training_summary.json)

## 5.2 Các chỉ số đánh giá
### 5.2.1 Accuracy
Tỷ lệ dự đoán đúng trên toàn bộ dữ liệu.

### 5.2.2 Precision
Tỷ lệ dự đoán dương đúng trên tổng dự đoán dương.

### 5.2.3 Recall
Tỷ lệ phát hiện đúng dương tính trên tổng mẫu dương thực tế.

### 5.2.4 F1-score
Trung bình điều hòa giữa Precision và Recall, phù hợp cho dữ liệu mất cân bằng.

### 5.2.5 ROC-AUC
Đánh giá khả năng phân tách hai lớp ở nhiều ngưỡng.

### 5.2.6 Code tính metrics và lưu báo cáo
Xem trong [src/models/evaluate.py](src/models/evaluate.py):

```python
def evaluate_model(model, X_test, y_test, model_key):
	y_pred = model.predict(X_test)
	y_proba = model.predict_proba(X_test)[:, 1]

	return {
		"accuracy": float(accuracy_score(y_test, y_pred)),
		"precision": float(precision_score(y_test, y_pred, zero_division=0)),
		"recall": float(recall_score(y_test, y_pred, zero_division=0)),
		"f1": float(f1_score(y_test, y_pred, zero_division=0)),
		"roc_auc": float(roc_auc_score(y_test, y_proba)),
		"confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
	}
```

```python
def save_artifacts(models, metrics, curves, feature_importance):
	save_json(metrics, REPORTS_DIR / "metrics.json")
	save_json(curves, REPORTS_DIR / "curves.json")
	save_json(feature_importance, REPORTS_DIR / "feature_importance.json")
```

## 5.3 Kết quả so sánh 3 thuật toán
### 5.3.1 Bảng so sánh đầy đủ
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7323 | 0.3124 | 0.7673 | 0.4441 | 0.8208 |
| Random Forest | 0.7849 | 0.3509 | 0.6396 | 0.4531 | 0.8182 |
| XGBoost | 0.7210 | 0.3061 | 0.7913 | 0.4414 | 0.8266 |

Nguồn số liệu: [reports/metrics.json](reports/metrics.json)

### 5.3.2 Biểu đồ trực quan
Kết quả được minh họa bằng ROC curve và confusion matrix để so sánh trực quan hiệu suất của các mô hình.

Đường dẫn hình ảnh: [reports/figures](reports/figures)

## 5.4 Lý do chọn XGBoost
### 5.4.1 Dựa trên kết quả so sánh
XGBoost đạt ROC-AUC cao nhất và Recall tốt nhất, phù hợp bài toán y tế ưu tiên phát hiện sớm.

### 5.4.2 ROC-AUC cao nhất (0.8266)
Cho thấy khả năng phân tách hai lớp tốt nhất trong ba mô hình.

### 5.4.3 Recall tốt nhất (0.7913)
Giảm nguy cơ bỏ sót ca bệnh, quan trọng trong sàng lọc y tế.

### 5.4.4 Phù hợp bài toán y tế
Đánh đổi Precision để tăng Recall là hợp lý khi ưu tiên phát hiện bệnh sớm.

# CHƯƠNG VI: KẾT QUẢ VÀ THẢO LUẬN

## 6.1 Kết quả chi tiết XGBoost
### 6.1.1 Kết quả trên tập test
XGBoost đạt ROC-AUC 0.8266 và Recall 0.7913, thể hiện khả năng phát hiện lớp dương tốt.

### 6.1.2 Ma trận nhầm lẫn (Confusion Matrix)
Ma trận nhầm lẫn (theo thứ tự TN, FP, FN, TP):
| | Pred 0 | Pred 1 |
|---|---:|---:|
| Actual 0 | 30,985 | 12,682 |
| Actual 1 | 1,475 | 5,594 |

### 6.1.3 Đường cong ROC
ROC curve thể hiện mô hình có khả năng phân tách hai lớp tốt ở nhiều ngưỡng, ROC-AUC đạt 0.8266.

#### Đường dẫn hình ảnh và code vẽ
- Hình lưu tại: [reports/figures](reports/figures)
- Code vẽ: [src/models/evaluate.py](src/models/evaluate.py)

```python
def plot_roc_curves(curves):
	for model_key, data in curves.items():
		roc = data["roc"]
		# ... plot ROC

def plot_confusion_matrices(metrics):
	for model_key, values in metrics.items():
		matrix = values["confusion_matrix"]
		# ... plot confusion matrix
```

## 6.2 Hệ thống Admin & Retrain
### 6.2.1 Trang quản trị Admin
Trang quản trị hiển thị thống kê dữ liệu mới, phân phối lớp và trạng thái retrain. Các điều kiện retrain gồm đủ mẫu, có đủ 2 class và đủ số ngày kể từ lần retrain trước.

### 6.2.2 Luồng Upload CSV
- Kiểm tra định dạng CSV và tên cột bắt buộc.
- Kiểm tra kích thước file và số dòng tối đa.
- Xác thực schema, làm sạch dữ liệu và kiểm tra trùng lặp.
- Lưu lịch sử upload để theo dõi.

### 6.2.3 Luồng Retrain
- Kiểm tra điều kiện retrain và dữ liệu có nhãn.
- Gộp dữ liệu mới với dữ liệu hiện có.
- Tiền xử lý đồng bộ.
- Huấn luyện lại mô hình XGBoost.
- So sánh mô hình cũ và mới theo các chỉ số chính.
- Quyết định triển khai và lưu lịch sử retrain.

### 6.2.4 Kết quả Retrain thực tế
Kết quả retrain được ghi vào [reports/retrain_history.json](reports/retrain_history.json), lưu thời điểm, số dòng mới, và so sánh ROC-AUC giữa mô hình cũ và mới để phục vụ quyết định cập nhật.

### 6.2.5 Code backend/frontend và đường dẫn thực tế
- API backend: [backend/app.py](backend/app.py)
- Giao diện admin (retrain/metrics): [frontend/admin.js](frontend/admin.js)
- Giao diện upload/predict: [frontend/upload.js](frontend/upload.js)

```python
@app.route("/predict", methods=["POST"])
def predict():
	ensure_feature_columns_loaded()
	X_input = prepare_prediction_frame(input_data).reindex(columns=feature_columns)
	probability = float(model.predict_proba(X_input)[:, 1][0])
	# ... return label, probability

@app.route("/predict-csv", methods=["POST"])
def predict_csv():
	# ... read CSV, validate columns, predict batch

@app.route("/upload-train", methods=["POST"])
def upload_train():
	# ... validate CSV, clean data, retrain pipeline
```

```javascript
const response = await fetch(`${API_BASE_URL}/predict-csv`, {
  method: "POST",
  body: formData,
});
```

## 6.3 Phân tích sai số
Sai số chủ yếu xuất hiện ở các trường hợp ranh giới (ví dụ: BMI cao nhưng ít bệnh nền hoặc lối sống lành mạnh). Việc ưu tiên Recall làm tăng False Positive nhưng giảm False Negative, phù hợp mục tiêu sàng lọc.

## 6.4 Đề xuất cải tiến
- Tinh chỉnh siêu tham số sâu hơn (Grid/Random Search).
- Thử thêm LightGBM, CatBoost.
- Cân nhắc calibration xác suất và threshold theo mục tiêu y tế.
- Bổ sung dữ liệu hoặc đặc trưng y tế chuyên sâu.

# CHƯƠNG VII: KẾT LUẬN

## 7.1 Tổng kết kết quả đạt được
Đề tài đã xây dựng quy trình tiền xử lý, thử nghiệm mô hình và tích hợp hệ thống hỗ trợ dự đoán cũng như retrain. XGBoost được chọn nhờ ROC-AUC và Recall cao.

## 7.2 Hạn chế của nghiên cứu
- Chưa tối ưu toàn diện siêu tham số.
- Dữ liệu chỉ từ một nguồn và một thời điểm.
- Một số đặc trưng là tự báo cáo, có thể có sai lệch.

## 7.3 Hướng phát triển trong tương lai
- Mở rộng dữ liệu đa nguồn hoặc nhiều năm.
- Thêm giải thích mô hình (SHAP/LIME).
- Triển khai thử nghiệm thực tế trong môi trường y tế.