from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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

BINARY_COLS = [
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
    "Sex",
]

ORDINAL_RANGES = {
    "GenHlth": (1, 5),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

CONTINUOUS_COLS = [
    "BMI",
    "MentHlth",
    "PhysHlth",
    "Age",
    "comorbidity_score",
    "healthy_lifestyle",
]

CATEGORICAL_COLS = ["BMI_category"]

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "xgboost": MODELS_DIR / "xgboost.joblib",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Retrain & Deployment thresholds
RETRAIN_THRESHOLDS = {
    "min_improvement_accuracy": 0.01,     # F1 must improve by at least 1%
    "min_improvement_f1": 0.01,
    "min_improvement_roc_auc": 0.005,
    "prefer_recall_for_medical": True,  # In healthcare, recall is often more important
}

# Data upload limits
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_ROWS = 500000

# New data collection for incremental learning
NEW_DATA_PATH = PROCESSED_DIR / "new_data.csv"
