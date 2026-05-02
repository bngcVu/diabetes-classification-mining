"""Versioning for datasets and models."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.constants import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


def create_backup(data_dir: Path, label: str = None) -> Path:
    """
    Create a timestamped backup of the processed data directory.

    Args:
        data_dir: Directory to backup
        label: Optional label for the backup

    Returns:
        Path to the backup directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}"
    if label:
        backup_name = f"{label}_{backup_name}"
    
    backup_dir = data_dir.parent / f"{data_dir.name}_{backup_name}"
    shutil.copytree(data_dir, backup_dir)
    return backup_dir


def save_dataset_version(
    df: pd.DataFrame,
    name: str = "combined",
    include_timestamp: bool = True
) -> Path:
    """
    Save a dataset version with metadata.

    Args:
        df: DataFrame to save
        name: Dataset name
        include_timestamp: Include timestamp in filename

    Returns:
        Path to saved dataset
    """
    versions_dir = PROCESSED_DIR / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if include_timestamp:
        filename = f"{name}_{timestamp}.csv"
    else:
        filename = f"{name}.csv"

    filepath = versions_dir / filename
    df.to_csv(filepath, index=False)

    # Save metadata
    meta = {
        "filename": filename,
        "rows": len(df),
        "columns": list(df.columns),
        "saved_at": timestamp,
    }
    meta_path = versions_dir / f"{name}_{timestamp}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return filepath


def save_model_version(model_name: str) -> Path:
    """
    Save a backup of the current model as a versioned copy.

    Args:
        model_name: Name of the model (e.g., 'xgboost')

    Returns:
        Path to the saved model version
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    versions_dir = MODELS_DIR / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"{model_name}_{timestamp}.joblib"
    version_path = versions_dir / version_name

    shutil.copy2(model_path, version_path)
    return version_path


def get_latest_model_version(model_name: str) -> Path:
    """Get path to the latest version of a model."""
    versions_dir = MODELS_DIR / "versions"
    if not versions_dir.exists():
        return MODELS_DIR / f"{model_name}.joblib"

    pattern = f"{model_name}_*.joblib"
    versions = sorted(versions_dir.glob(pattern), key=lambda x: x.stat().st_mtime)
    
    if versions:
        return versions[-1]
    return MODELS_DIR / f"{model_name}.joblib"


def rollback_to_version(resource: str, version_id: str) -> bool:
    """
    Rollback to a previous version.

    Args:
        resource: Type of resource ('model', 'data')
        version_id: Version identifier (timestamp)

    Returns:
        True if successful
    """
    if resource == "model":
        # Rollback model
        model_file = MODELS_DIR / "xgboost.joblib"  # Default model
        versions_dir = MODELS_DIR / "versions"
        version_file = versions_dir / f"xgboost_{version_id}.joblib"
        
        if version_file.exists():
            shutil.copy2(version_file, model_file)
            return True
    elif resource == "data":
        # Rollback processed data
        backup_dir = PROCESSED_DIR.parent / f"processed_backup_{version_id}"
        if backup_dir.exists():
            shutil.rmtree(PROCESSED_DIR)
            shutil.copytree(backup_dir, PROCESSED_DIR)
            return True

    return False


def get_version_history(resource: str = "model") -> list:
    """Get list of available versions for a resource."""
    if resource == "model":
        versions_dir = MODELS_DIR / "versions"
        if not versions_dir.exists():
            return []
        
        versions = []
        for f in sorted(versions_dir.glob("*.joblib"), key=lambda x: x.stat().st_mtime, reverse=True):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                timestamp = "_".join(parts[1:])
                versions.append({
                    "filename": f.name,
                    "timestamp": timestamp,
                    "path": str(f),
                })
        return versions

    return []


def save_training_log(log_data: dict, log_type: str = "retrain") -> Path:
    """
    Save a training/retraining log.

    Args:
        log_data: Dictionary containing log information
        log_type: Type of log ('retrain', 'upload')

    Returns:
        Path to saved log
    """
    logs_dir = REPORTS_DIR / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{log_type}_{timestamp}.json"

    log_entry = {
        "timestamp": timestamp,
        "log_type": log_type,
        **log_data
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    return log_file
