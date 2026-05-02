# Diabetes Classification Mining

Hệ thống data mining dự đoán nguy cơ tiểu đường từ dataset BRFSS 2015:

`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

Dataset có `253,680` dòng, `21` feature đầu vào và target `Diabetes_binary`.

## Thành phần chính

- EDA và kiểm tra chất lượng dữ liệu.
- Feature Engineering: `BMI_category`, `comorbidity_score`, `healthy_lifestyle`.
- Train/test split `80/20` với `stratify=y`.
- `StandardScaler` fit trong pipeline, chỉ fit trên train.
- Logistic Regression baseline, Random Forest, XGBoost.
- Metrics, ROC curve, Precision-Recall curve, Feature Importance, Confusion Matrix.
- Backend Flask: `POST /predict`, `GET /model-stats`.
- Frontend HTML/CSS/JS + Chart.js.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Train lại mô hình

Chạy nhanh:

```bash
python -m src.models.train --fast
```

Chạy nhiều estimators hơn:

```bash
python -m src.models.train
```

Output chính:

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`
- `models/feature_columns.joblib`
- `reports/metrics.json`
- `reports/curves.json`
- `reports/feature_importance.json`
- `reports/figures/*.png`

## Chạy backend

```bash
python backend/app.py
```

API chạy tại:

```text
http://localhost:5000
```

Kiểm tra:

```bash
curl http://localhost:5000/health
```

## Chạy frontend

```bash
python -m http.server 8000
```

Mở:

```text
http://localhost:8000/frontend/
```

## Notebook

- `notebooks/01_eda.ipynb`
- `notebooks/02_preprocessing_modeling.ipynb`
- `notebooks/03_model_explainability.ipynb`

Các notebook dùng lại module trong `src/` để tránh lệch logic giữa notebook, training script và backend.
