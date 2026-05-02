"""Validate incoming CSV data against schema and data quality rules."""

import pandas as pd
from src.utils.constants import (
    BINARY_COLS,
    ORDINAL_RANGES,
    RAW_FEATURE_COLUMNS,
    TARGET_COL,
)


class ValidationError(Exception):
    pass


class DataValidator:
    """Validate CSV data for training/prediction."""

    # Outlier thresholds for continuous columns
    OUTLIER_THRESHOLDS = {
        "BMI": (10, 80),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
    }

    def __init__(self, require_target=False):
        self.require_target = require_target
        self.errors = []
        self.warnings = []

    def validate(self, df: pd.DataFrame) -> dict:
        """Run all validations and return report."""
        self.errors = []
        self.warnings = []

        self._validate_schema(df)
        if self.errors:
            raise ValidationError("; ".join(self.errors))

        self._validate_binary_columns(df)
        self._validate_ordinal_columns(df)
        self._validate_continuous_columns(df)
        self._validate_missing_values(df)

        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "row_count": len(df),
        }

    def _validate_schema(self, df: pd.DataFrame):
        required_cols = RAW_FEATURE_COLUMNS.copy()
        if self.require_target:
            required_cols.append(TARGET_COL)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.errors.append(f"Thieu cot: {missing}")

        extra = [c for c in df.columns if c not in required_cols]
        if extra:
            self.warnings.append(f"Cot them: {extra}")

    def _validate_binary_columns(self, df: pd.DataFrame):
        for col in BINARY_COLS:
            if col not in df.columns:
                continue
            invalid = df[col].dropna().apply(lambda x: x not in [0, 1, 0.0, 1.0])
            count = invalid.sum()
            if count > 0:
                self.warnings.append(f"{col}: {count} gia tri khong phai 0/1 (da thay = mode)")

    def _validate_ordinal_columns(self, df: pd.DataFrame):
        for col, (min_val, max_val) in ORDINAL_RANGES.items():
            if col not in df.columns:
                continue
            out_of_range = ~df[col].between(min_val, max_val)
            count = out_of_range.sum()
            if count > 0:
                self.warnings.append(
                    f"{col}: {count} gia tri ngoai khoang [{min_val}, {max_val}]"
                )

    def _validate_continuous_columns(self, df: pd.DataFrame):
        for col, (min_val, max_val) in self.OUTLIER_THRESHOLDS.items():
            if col not in df.columns:
                continue
            outliers = ~df[col].between(min_val, max_val)
            count = outliers.sum()
            if count > 0:
                self.warnings.append(
                    f"{col}: {count} outliers ngoai khoang [{min_val}, {max_val}]"
                )

    def _validate_missing_values(self, df: pd.DataFrame):
        missing = df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        if not cols_with_missing.empty:
            self.warnings.append(
                f"Cot co gia tri thieu: {cols_with_missing.to_dict()}"
            )


def validate_csv(df: pd.DataFrame, require_target: bool = False) -> dict:
    """Convenience function to validate a dataframe."""
    validator = DataValidator(require_target=require_target)
    return validator.validate(df)


def get_validation_report(df: pd.DataFrame, require_target: bool = False) -> dict:
    """Get detailed validation report."""
    validator = DataValidator(require_target=require_target)
    validator.validate(df)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_counts": df.isnull().sum().to_dict(),
        "duplicate_count": df.duplicated().sum(),
        "errors": validator.errors,
        "warnings": validator.warnings,
    }
