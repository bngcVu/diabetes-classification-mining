"""Clean and prepare data for training."""

import pandas as pd
from src.utils.constants import (
    BINARY_COLS,
    ORDINAL_RANGES,
    RAW_FEATURE_COLUMNS,
    TARGET_COL,
)


def clean_data(df: pd.DataFrame, remove_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Clean incoming dataframe and return cleaned data with stats.

    Returns:
        tuple: (cleaned_df, cleaning_stats)
    """
    stats = {
        "original_rows": len(df),
        "duplicates_removed": 0,
        "nulls_filled": {},
        "invalid_binary_filled": {},
        "ordinal_clipped": {},
        "outliers_clipped": {},
    }

    df = df.copy()

    # 1. Remove duplicates
    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        stats["duplicates_removed"] = before - len(df)

    # 2. Clean binary columns
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
        invalid_mask = ~df[col].isin([0, 1, 0.0, 1.0])
        count = invalid_mask.sum()
        if count > 0:
            df.loc[invalid_mask, col] = mode_val
            stats["invalid_binary_filled"][col] = int(count)

    # 3. Fill missing values
    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].isnull().any():
            if col in BINARY_COLS or col in ORDINAL_RANGES:
                fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            else:
                fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            stats["nulls_filled"][col] = int(df[col].isnull().sum())

    # 4. Clip ordinal columns to valid range
    for col, (min_val, max_val) in ORDINAL_RANGES.items():
        if col not in df.columns:
            continue
        below = (df[col] < min_val).sum()
        above = (df[col] > max_val).sum()
        if below > 0 or above > 0:
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            stats["ordinal_clipped"][col] = {"below": int(below), "above": int(above)}

    # 5. Clip outliers for continuous columns
    outlier_ranges = {
        "BMI": (10, 80),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
    }
    for col, (min_val, max_val) in outlier_ranges.items():
        if col not in df.columns:
            continue
        below = (df[col] < min_val).sum()
        above = (df[col] > max_val).sum()
        if below > 0 or above > 0:
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            stats["outliers_clipped"][col] = {"below": int(below), "above": int(above)}

    stats["final_rows"] = len(df)
    return df, stats


def check_duplicates_with_existing(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    key_columns: list = None
) -> dict:
    """
    Check for duplicates between new data and existing data.

    Args:
        new_df: New data to check
        existing_df: Existing dataset
        key_columns: Columns to use for matching (default: all feature columns)

    Returns:
        dict with duplicate statistics
    """
    if key_columns is None:
        key_columns = RAW_FEATURE_COLUMNS.copy()

    available_keys = [c for c in key_columns if c in new_df.columns and c in existing_df.columns]
    if not available_keys:
        available_keys = RAW_FEATURE_COLUMNS.copy()

    # Hash-based comparison
    new_hashes = new_df[available_keys].apply(lambda x: hash(tuple(x)), axis=1)
    existing_hashes = existing_df[available_keys].apply(lambda x: hash(tuple(x)), axis=1)

    duplicates = new_hashes.isin(existing_hashes)
    duplicate_count = duplicates.sum()

    return {
        "total_new_rows": len(new_df),
        "duplicate_rows": int(duplicate_count),
        "unique_new_rows": int(len(new_df) - duplicate_count),
        "duplicate_percent": round(duplicate_count / len(new_df) * 100, 2) if len(new_df) > 0 else 0,
        "key_columns_used": available_keys,
    }


def merge_datasets(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    strategy: str = "keep_new"
) -> pd.DataFrame:
    """
    Merge existing and new datasets.

    Args:
        existing_df: Existing dataset
        new_df: New data to add
        strategy: How to handle duplicates
            - "keep_new": Keep new values when duplicate
            - "keep_existing": Keep existing values when duplicate
            - "keep_all": Keep all (no deduplication)

    Returns:
        Merged dataframe
    """
    if strategy == "keep_all":
        return pd.concat([existing_df, new_df], ignore_index=True)

    # Find duplicates
    key_columns = RAW_FEATURE_COLUMNS.copy()
    available_keys = [c for c in key_columns if c in existing_df.columns and c in new_df.columns]

    existing_keys = existing_df[available_keys].apply(lambda x: hash(tuple(x)), axis=1)
    new_keys = new_df[available_keys].apply(lambda x: hash(tuple(x)), axis=1)

    is_duplicate = new_keys.isin(existing_keys)

    if strategy == "keep_new":
        # Only keep new rows that are not duplicates
        unique_new = new_df[~is_duplicate]
        return pd.concat([existing_df, unique_new], ignore_index=True)

    elif strategy == "keep_existing":
        # Remove duplicates from new data
        unique_new = new_df[~is_duplicate]
        return pd.concat([existing_df, unique_new], ignore_index=True)

    return pd.concat([existing_df, new_df], ignore_index=True)
