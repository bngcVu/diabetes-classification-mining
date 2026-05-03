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


class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


def validate_data_quality(df):
    """
    Kiểm tra toàn diện chất lượng dữ liệu.
    
    Raises DataValidationError nếu có vấn đề nghiêm trọng.
    Returns dict với warnings nếu có vấn đề nhẹ.
    """
    warnings = {}
    errors = []
    
    # 1. Kiểm tra Missing Values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        warnings['missing_values'] = missing_cols.to_dict()
        if len(missing_cols) > len(df.columns) * 0.5:
            errors.append(f"Quá nhiều missing values: {missing_cols.to_dict()}")
    
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
        errors.append(f"Giá trị không hợp lệ trong binary columns: {invalid_binary}")
    
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
    
    # Raise error nếu có vấn đề nghiêm trọng
    if errors:
        raise DataValidationError("; ".join(errors))
    
    return warnings


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


def handle_outliers(df, strategy='cap'):
    """
    Xử lý outliers cho continuous columns.
    
    Args:
        strategy: 'cap' (gắn vào boundary) hoặc 'remove' (xóa row)
    """
    data = df.copy()
    outlier_cols = ['BMI', 'MentHlth', 'PhysHlth']
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


def clean_data(df, handle_outliers_strategy='cap'):
    """
    Hàm chính để làm sạch dữ liệu.
    
    Pipeline:
    1. Validate data quality
    2. Handle missing values
    3. Handle invalid values
    4. Handle outliers
    
    Args:
        df: DataFrame cần làm sạch
        handle_outliers_strategy: 'cap' hoặc 'remove'
    
    Returns:
        Cleaned DataFrame
    """
    # Validate trước
    validation_warnings = validate_data_quality(df)
    
    # 1. Handle missing values
    data = handle_missing_values(df)
    
    # 2. Handle invalid values
    data = handle_invalid_values(data)
    
    # 3. Handle outliers
    data = handle_outliers(data, strategy=handle_outliers_strategy)
    
    return data, validation_warnings


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


def split_features_target(df, clean=True, handle_outliers_strategy='cap'):
    """
    Tách features và target từ DataFrame.
    
    Args:
        df: DataFrame đầu vào
        clean: Nếu True, áp dụng full cleaning pipeline
        handle_outliers_strategy: 'cap' hoặc 'remove'
    """
    if clean:
        cleaned_df, validation_warnings = clean_data(df, handle_outliers_strategy)
        engineered = add_engineered_features(cleaned_df)
    else:
        engineered = add_engineered_features(df)
        validation_warnings = {}
    
    X = engineered.drop(columns=[TARGET_COL])
    y = engineered[TARGET_COL].astype(int)
    
    return X, y, validation_warnings


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
    """Prepare DataFrame từ payload cho prediction."""
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    raw = pd.DataFrame([{col: payload[col] for col in RAW_FEATURE_COLUMNS}])
    return add_engineered_features(raw)


# Legacy alias cho backward compatibility
get_quality_report = validate_data_quality
