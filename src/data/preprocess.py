import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.constants import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    ORDINAL_RANGES,
    RAW_FEATURE_COLUMNS,
    TARGET_COL,
)


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


def split_features_target(df):
    engineered = add_engineered_features(df)
    X = engineered.drop(columns=[TARGET_COL])
    y = engineered[TARGET_COL].astype(int)
    return X, y


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


def prepare_prediction_frame(payload):
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    raw = pd.DataFrame([{col: payload[col] for col in RAW_FEATURE_COLUMNS}])
    return add_engineered_features(raw)
