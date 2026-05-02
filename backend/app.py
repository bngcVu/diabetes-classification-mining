import json
import logging
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocess import prepare_prediction_frame
from src.data.validate import validate_csv, get_validation_report
from src.data.clean import clean_data, check_duplicates_with_existing, merge_datasets
from src.data.versioning import save_dataset_version, save_model_version, save_training_log
from src.models.train import clear_candidate_models, deploy_candidate_models, retrain_pipeline
from src.utils.constants import (
    MAX_UPLOAD_SIZE_MB,
    MAX_UPLOAD_ROWS,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    MODELS_DIR,
    NEW_DATA_PATH,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    RAW_FEATURE_COLUMNS,
    REPORTS_DIR,
    TARGET_COL,
)

app = Flask(__name__)
CORS(app)


def load_models():
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


models = {}
feature_columns = None
retrain_status = {"status": "idle", "progress": 0, "message": "", "result": None}
RETRAIN_HISTORY_PATH = REPORTS_DIR / "retrain_history.json"


def ensure_feature_columns_loaded():
    global feature_columns
    if feature_columns is None:
        feature_path = MODELS_DIR / "feature_columns.joblib"
        if not feature_path.exists():
            raise FileNotFoundError("Missing models/feature_columns.joblib")
        feature_columns = joblib.load(feature_path)


def get_model(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name not in models:
        path = MODEL_PATHS[model_name]
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        models[model_name] = joblib.load(path)

    return models[model_name]


def save_prediction_data(input_data: dict, result: dict):
    """
    Lưu input và kết quả dự đoán vào new_data.csv.
    Mỗi dòng chứa: các features + prediction + probability + timestamp.
    """
    import os
    timestamp = datetime.now().isoformat()
    row = {**input_data, TARGET_COL: result["label"], "probability": result["probability"], "source": "prediction", "timestamp": timestamp}

    df_new = pd.DataFrame([row])
    if NEW_DATA_PATH.exists():
        df_existing = pd.read_csv(NEW_DATA_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(NEW_DATA_PATH, index=False)
    logger.info(f"[SAVE-PREDICTION] Saved prediction to {NEW_DATA_PATH}, total rows: {len(df_combined)}")


def read_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max(0, sum(1 for _ in f) - 1)


def get_retrain_history():
    return read_json(RETRAIN_HISTORY_PATH, [])


def append_retrain_history(decision: str, result: dict):
    history = get_retrain_history()
    comparison = result.get("comparison", {}).get("comparison", {})
    old_metrics = result.get("old_metrics_same_test", {})
    new_metrics = result.get("new_metrics", {})
    xgb_old = old_metrics.get("xgboost", {})
    xgb_new = new_metrics.get("xgboost", {})
    entry = {
        "id": len(history) + 1,
        "created_at": datetime.now().isoformat(),
        "decision": decision,
        "new_rows": result.get("data_stats", {}).get("new_rows", 0),
        "total_rows": result.get("data_stats", {}).get("total_rows", 0),
        "old_roc_auc": xgb_old.get("roc_auc"),
        "new_roc_auc": xgb_new.get("roc_auc"),
        "comparison": comparison,
    }
    history.append(entry)
    write_json(RETRAIN_HISTORY_PATH, history)
    return entry


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        ensure_feature_columns_loaded()
        payload = request.get_json(force=True) or {}
        model_name = payload.get("model", "xgboost")
        input_data = payload.get("features", payload)
        allow_save = bool(payload.get("allow_save", False))

        logger.info(f"[PREDICT] Model: {model_name}, Features: {input_data}")

        if model_name not in MODEL_PATHS:
            return jsonify({"error": "Mo hinh khong ho tro", "supported_models": list(MODEL_PATHS.keys())}), 400

        missing = [col for col in RAW_FEATURE_COLUMNS if col not in input_data]
        if missing:
            return jsonify({"error": "Thieu features", "missing_features": missing}), 400

        X_input = prepare_prediction_frame(input_data).reindex(columns=feature_columns)
        model = get_model(model_name)
        probability = float(model.predict_proba(X_input)[:, 1][0])
        label = int(probability >= 0.5)

        result_payload = {
            "model": model_name,
            "model_display_name": MODEL_DISPLAY_NAMES[model_name],
            "probability": probability,
            "label": label,
            "label_text": "Nguy co tieu duong cao" if label == 1 else "Nguy co tieu duong thap",
        }

        if allow_save:
            try:
                save_prediction_data(input_data, result_payload)
            except Exception as save_err:
                logger.warning(f"[PREDICT] Could not save prediction data: {save_err}")

        logger.info(f"[PREDICT] Result: prob={probability:.6f}, label={label}")

        return jsonify(result_payload)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "hint": "Chay: python -m src.models.train --fast"}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Du doan that bai", "details": str(exc)}), 500


@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    """Upload CSV file and get batch predictions (without retraining)"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Khong co file duoc upload"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Ten file rong"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Chi chap nhan file CSV"}), 400

        model_name = request.form.get("model", "xgboost")
        if model_name not in MODEL_PATHS:
            return jsonify({"error": "Mo hinh khong ho tro", "supported_models": list(MODEL_PATHS.keys())}), 400

        ensure_feature_columns_loaded()

        # Read CSV with various encoding support
        df = pd.read_csv(file, encoding='utf-8', encoding_errors='ignore')
        logger.info(f"[PREDICT-CSV] File columns: {list(df.columns)}")
        logger.info(f"[PREDICT-CSV] Required columns: {RAW_FEATURE_COLUMNS}")
        logger.info(f"[PREDICT-CSV] Rows: {len(df)}")

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Check required columns
        missing_cols = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            # Try case-insensitive matching
            col_mapping = {col.lower(): col for col in df.columns}
            for missing in missing_cols[:]:
                if missing.lower() in col_mapping:
                    df = df.rename(columns={col_mapping[missing.lower()]: missing})
                    missing_cols.remove(missing)

        missing_cols = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            return jsonify({
                "error": "Thieu cac cot trong file CSV",
                "missing_columns": missing_cols,
                "required_columns": RAW_FEATURE_COLUMNS,
                "found_columns": list(df.columns)
            }), 400

        # Prepare data
        X_input = prepare_prediction_frame(df[RAW_FEATURE_COLUMNS].to_dict(orient="records"))
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)

        # Predict
        model = get_model(model_name)
        probabilities = model.predict_proba(X_input)[:, 1]
        labels = (probabilities >= 0.5).astype(int)

        # Create results dataframe
        results_df = df.copy()
        results_df["probability"] = probabilities
        results_df["prediction"] = labels
        results_df["risk_level"] = results_df["prediction"].map({
            0: "Nguy co thap",
            1: "Nguy co cao"
        })

        # Save to CSV
        output_dir = ROOT_DIR / "predictions"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"predictions_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)

        # Lưu input + kết quả vào new_data.csv
        try:
            timestamp = datetime.now().isoformat()
            for idx, row in df.iterrows():
                input_data_row = row[RAW_FEATURE_COLUMNS].to_dict()
                result_row = {
                    "label": int(labels[idx]),
                    "probability": float(probabilities[idx]),
                    "source": "predict_csv",
                    "timestamp": timestamp
                }
                combined_row = {**input_data_row, TARGET_COL: result_row["label"], **result_row}
                df_single = pd.DataFrame([combined_row])
                
                if NEW_DATA_PATH.exists():
                    df_existing = pd.read_csv(NEW_DATA_PATH)
                    df_combined = pd.concat([df_existing, df_single], ignore_index=True)
                else:
                    df_combined = df_single
                
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                df_combined.to_csv(NEW_DATA_PATH, index=False)
            logger.info(f"[PREDICT-CSV] Saved {total} predictions to {NEW_DATA_PATH}")
        except Exception as save_err:
            logger.warning(f"[PREDICT-CSV] Could not save prediction data: {save_err}")

        # Summary stats
        total = len(results_df)
        high_risk = (labels == 1).sum()
        low_risk = (labels == 0).sum()

        logger.info(f"[PREDICT-CSV] Model: {model_name}, Rows: {total}, High risk: {high_risk}")

        return jsonify({
            "success": True,
            "model": model_name,
            "summary": {
                "total_records": int(total),
                "high_risk": int(high_risk),
                "low_risk": int(low_risk),
                "high_risk_percent": round(high_risk / total * 100, 2)
            },
            "download_path": str(output_path),
            "download_url": f"/predictions/{output_path.name}"
        })
    except Exception as exc:
        logger.error(f"[PREDICT-CSV] Error: {exc}")
        # Return more detailed error for debugging
        import traceback
        return jsonify({
            "error": "Xu ly file that bai",
            "details": str(exc),
            "hint": "Kiem tra file CSV: dam bao co day du cac cot HighBP, HighChol, BMI, Age,..."
        }), 500


@app.route("/predictions/<filename>")
def download_prediction(filename):
    """Download prediction results"""
    from flask import send_from_directory
    output_dir = ROOT_DIR / "predictions"
    return send_from_directory(output_dir, filename, as_attachment=True)


@app.route("/upload-train", methods=["POST"])
def upload_train():
    """
    Full data mining pipeline:
    1. Upload & validate CSV
    2. Clean data
    3. Check duplicates
    4. Merge with existing data
    5. Retrain model
    6. Compare & decide deploy
    """
    global retrain_status

    try:
        # Step 1: Check file
        if "file" not in request.files:
            return jsonify({"error": "Khong co file duoc upload"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Ten file rong"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Chi chap nhan file CSV"}), 400

        # Check file size
        file.seek(0, 2)
        size_mb = file.tell() / (1024 * 1024)
        file.seek(0)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            return jsonify({"error": f"File qua lon ({size_mb:.1f}MB). Max: {MAX_UPLOAD_SIZE_MB}MB"}), 400

        # Step 2: Read & validate CSV
        logger.info("[UPLOAD-TRAIN] Reading CSV...")
        df_new = pd.read_csv(file)

        if len(df_new) > MAX_UPLOAD_ROWS:
            return jsonify({"error": f"File qua nhieu dong ({len(df_new)}). Max: {MAX_UPLOAD_ROWS}"}), 400

        # Check for target column
        has_target = TARGET_COL in df_new.columns
        if not has_target:
            return jsonify({
                "error": f"File phai co cot '{TARGET_COL}' de huấn luyện lại model",
                "required_columns": RAW_FEATURE_COLUMNS + [TARGET_COL]
            }), 400

        # Validate schema
        logger.info("[UPLOAD-TRAIN] Validating schema...")
        validation = validate_csv(df_new, require_target=True)
        if not validation["valid"]:
            return jsonify({
                "error": "Validation failed",
                "details": validation["errors"]
            }), 400

        # Step 3: Clean data
        logger.info("[UPLOAD-TRAIN] Cleaning data...")
        df_cleaned, cleaning_stats = clean_data(df_new)

        # Step 4: Check duplicates with existing data
        logger.info("[UPLOAD-TRAIN] Checking duplicates...")
        existing_path = PROCESSED_DIR / "y_train.csv"
        duplicate_info = {"total_new_rows": len(df_cleaned), "duplicate_rows": 0, "unique_new_rows": len(df_cleaned)}

        if existing_path.exists():
            df_existing = pd.read_csv(existing_path)
            if TARGET_COL in df_existing.columns:
                # Load full existing data
                X_existing = pd.read_csv(PROCESSED_DIR / "X_train.csv")
                existing_full = pd.concat([X_existing, df_existing[TARGET_COL]], axis=1)
                duplicate_info = check_duplicates_with_existing(df_cleaned, existing_full)

        # Step 5: Merge datasets
        logger.info("[UPLOAD-TRAIN] Merging datasets...")
        if existing_path.exists():
            X_existing = pd.read_csv(PROCESSED_DIR / "X_train.csv")
            y_existing = pd.read_csv(PROCESSED_DIR / "y_train.csv")
            existing_full = pd.concat([X_existing, y_existing[TARGET_COL]], axis=1)
            df_combined = merge_datasets(existing_full, df_cleaned, strategy="keep_new")
        else:
            df_combined = df_cleaned

        # Save dataset version
        save_dataset_version(df_combined, name="combined")

        # Lưu dữ liệu vào new_data.csv để /retrain có thể sử dụng
        try:
            timestamp = datetime.now().isoformat()
            df_for_new_data = df_cleaned.copy()
            df_for_new_data["source"] = "upload_train"
            df_for_new_data["timestamp"] = timestamp
            
            if NEW_DATA_PATH.exists():
                df_existing = pd.read_csv(NEW_DATA_PATH)
                df_combined_new = pd.concat([df_existing, df_for_new_data], ignore_index=True)
            else:
                df_combined_new = df_for_new_data
            
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            df_combined_new.to_csv(NEW_DATA_PATH, index=False)
            logger.info(f"[UPLOAD-TRAIN] Saved {len(df_cleaned)} rows to {NEW_DATA_PATH}, total: {len(df_combined_new)}")
        except Exception as save_err:
            logger.warning(f"[UPLOAD-TRAIN] Could not save to new_data.csv: {save_err}")

        # Step 6: Retrain in background
        logger.info("[UPLOAD-TRAIN] Starting retrain...")

        # Backup current model
        for model_name in MODEL_PATHS.keys():
            try:
                save_model_version(model_name)
            except Exception as e:
                logger.warning(f"Could not backup model {model_name}: {e}")

        retrain_status = {"status": "running", "progress": 10, "message": "Bat dau huan luyen...", "result": None}

        # Run retrain in background thread
        def background_retrain():
            global retrain_status
            try:
                retrain_status["progress"] = 30
                retrain_status["message"] = "Huan luyen model..."

                result = retrain_pipeline(df_combined, fast=False)

                retrain_status["progress"] = 90
                retrain_status["message"] = "Hoan tat!"

                if result.get("can_deploy"):
                    retrain_status["status"] = "completed_candidate"
                    retrain_status["message"] = "Model moi tot hon. Dang cho admin xac nhan cap nhat."
                else:
                    retrain_status["status"] = "completed_candidate"
                    retrain_status["message"] = "Model moi da train xong nhung chua du dieu kien khuyen nghi cap nhat."

                retrain_status["result"] = result
                retrain_status["progress"] = 100

                # Save training log
                save_training_log({
                    "data_stats": result.get("data_stats", {}),
                    "cleaning_stats": cleaning_stats,
                    "duplicate_info": duplicate_info,
                    "comparison": result.get("comparison", {}),
                    "deployed": result.get("deployed", False)
                }, log_type="retrain")

            except Exception as e:
                logger.error(f"[RETRAIN] Error: {e}")
                retrain_status = {"status": "failed", "progress": 0, "message": str(e), "result": None}

        thread = threading.Thread(target=background_retrain)
        thread.daemon = True
        thread.start()

        return jsonify({
            "success": True,
            "message": "Bat dau qua trinh huan luyen",
            "validation": {
                "row_count": len(df_new),
                "warnings": validation.get("warnings", [])
            },
            "cleaning_stats": cleaning_stats,
            "duplicate_info": duplicate_info,
            "combined_rows": len(df_combined),
            "status_url": "/retrain-status"
        })

    except Exception as exc:
        logger.error(f"[UPLOAD-TRAIN] Error: {exc}")
        return jsonify({"error": "Upload that bai", "details": str(exc)}), 500


@app.route("/upload-new-data", methods=["POST"])
def upload_new_data():
    """Upload labeled CSV into new_data.csv without starting retrain."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Khong co file duoc upload"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Ten file rong"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Chi chap nhan file CSV"}), 400

        file.seek(0, 2)
        size_mb = file.tell() / (1024 * 1024)
        file.seek(0)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            return jsonify({"error": f"File qua lon ({size_mb:.1f}MB). Max: {MAX_UPLOAD_SIZE_MB}MB"}), 400

        df_new = pd.read_csv(file)
        if len(df_new) > MAX_UPLOAD_ROWS:
            return jsonify({"error": f"File qua nhieu dong ({len(df_new)}). Max: {MAX_UPLOAD_ROWS}"}), 400

        validation = validate_csv(df_new, require_target=True)
        if not validation["valid"]:
            return jsonify({"error": "Validation failed", "details": validation["errors"]}), 400

        df_cleaned, cleaning_stats = clean_data(df_new)
        timestamp = datetime.now().isoformat()
        df_cleaned["source"] = "admin_upload"
        df_cleaned["timestamp"] = timestamp

        if NEW_DATA_PATH.exists():
            df_existing = pd.read_csv(NEW_DATA_PATH)
            df_combined = pd.concat([df_existing, df_cleaned], ignore_index=True)
        else:
            df_combined = df_cleaned

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(NEW_DATA_PATH, index=False)

        return jsonify({
            "success": True,
            "message": "Da upload CSV vao du lieu moi",
            "rows_added": len(df_cleaned),
            "total_new": len(df_combined),
            "validation": {"row_count": len(df_new), "warnings": validation.get("warnings", [])},
            "cleaning_stats": cleaning_stats,
        })

    except Exception as exc:
        logger.error(f"[UPLOAD-NEW-DATA] Error: {exc}")
        return jsonify({"error": "Upload that bai", "details": str(exc)}), 500


@app.route("/retrain-status", methods=["GET"])
def get_retrain_status():
    """Get current retrain status"""
    global retrain_status
    return jsonify(retrain_status)


@app.route("/retrain-details", methods=["GET"])
def get_retrain_details():
    """Get detailed retrain results after completion"""
    global retrain_status

    if retrain_status["status"] == "idle":
        return jsonify({"error": "Khong co qua trinh retrain nao"}), 404

    if retrain_status["status"] == "running":
        return jsonify({
            "status": "running",
            "progress": retrain_status["progress"],
            "message": retrain_status["message"]
        })

    if retrain_status["status"] == "failed":
        return jsonify({
            "status": "failed",
            "error": retrain_status["message"]
        })

    # Completed - return full results
    result = retrain_status.get("result", {})

    comparison = result.get("comparison", {})
    comparison_summary = {}
    if comparison:
        comp_data = comparison.get("comparison", {})
        if comp_data and all(isinstance(value, dict) and "old" in value for value in comp_data.values()):
            for metric, data in comp_data.items():
                comparison_summary[metric] = {
                    "old": data.get("old", 0),
                    "new": data.get("new", 0),
                    "improved": data.get("improved", False),
                    "change_percent": data.get("percent_change", 0)
                }
        else:
            comparison_summary = comp_data

    return jsonify({
        "status": retrain_status["status"],
        "deployed": result.get("deployed", False),
        "can_deploy": result.get("can_deploy", False),
        "pending_admin_decision": result.get("pending_admin_decision", False),
        "recommendation": result.get("recommendation", "unknown"),
        "duration_seconds": result.get("training_duration_seconds", 0),
        "data_stats": result.get("data_stats", {}),
        "old_metrics": result.get("old_metrics_same_test", {}),
        "new_metrics": result.get("new_metrics", {}),
        "comparison_summary": comparison_summary,
        "full_comparison": result.get("comparison", {}),
        "message": retrain_status["message"]
    })


@app.route("/retrain", methods=["POST"])
def retrain_from_new_data():
    """
    Retrain model bằng cách gộp new_data.csv (dữ liệu từ predict) với dữ liệu gốc.
    Yêu cầu: new_data.csv phải có cột Diabetes_binary (nhãn thật).
    Nếu không có nhãn, trả lỗi và gợi ý dùng /upload-train.
    """
    global retrain_status, models, feature_columns

    try:
        # Bước 1: Kiểm tra new_data.csv
        if not NEW_DATA_PATH.exists():
            return jsonify({
                "error": "Chua co du lieu moi de retrain",
                "hint": "Su dung /predict nhieu lan hoac upload du lieu co nhan qua /upload-train"
            }), 400

        df_new = pd.read_csv(NEW_DATA_PATH)

        # Kiểm tra có cột target không
        if TARGET_COL not in df_new.columns:
            return jsonify({
                "error": f"Du lieu moi chua co cot nhan '{TARGET_COL}'",
                "hint": "Vui long them cot nhan vao new_data.csv hoac su dung /upload-train de upload du lieu co nhan"
            }), 400

        # Kiểm tra số lượng dữ liệu mới
        if len(df_new) < 10:
            return jsonify({
                "error": f"Chi co {len(df_new)} dong du lieu moi. Can it nhat 10 dong de retrain.",
                "current_rows": len(df_new)
            }), 400

        # Bước 2: Load dữ liệu gốc
        X_train_path = PROCESSED_DIR / "X_train.csv"
        y_train_path = PROCESSED_DIR / "y_train.csv"
        if not X_train_path.exists() or not y_train_path.exists():
            return jsonify({
                "error": "Khong tim thay du lieu huan luyen goc",
                "hint": "Chay: python -m src.models.train"
            }), 503

        X_existing = pd.read_csv(X_train_path)
        y_existing = pd.read_csv(y_train_path)

        # Bước 3: Merge dữ liệu
        df_existing = pd.concat([X_existing, y_existing[TARGET_COL]], axis=1)

        # Clean dữ liệu mới
        df_new_clean, cleaning_stats = clean_data(df_new)

        # Kiểm tra duplicates
        duplicate_info = check_duplicates_with_existing(df_new_clean, df_existing)

        # Merge
        df_combined = merge_datasets(df_existing, df_new_clean, strategy="keep_new")

        # Backup model cũ
        for model_name in MODEL_PATHS.keys():
            try:
                save_model_version(model_name)
            except Exception as e:
                logger.warning(f"Could not backup model {model_name}: {e}")

        # Bước 4: Cập nhật retrain status
        retrain_status = {"status": "running", "progress": 10, "message": "Bat dau gop du lieu...", "result": None}

        # Chạy retrain trong background thread
        def background_retrain():
            global retrain_status, models, feature_columns
            try:
                retrain_status["progress"] = 30
                retrain_status["message"] = "Huan luyen model..."

                result = retrain_pipeline(df_combined, fast=False)

                retrain_status["progress"] = 90
                retrain_status["message"] = "Hoan tat!"

                if result.get("can_deploy"):
                    retrain_status["status"] = "completed_candidate"
                    retrain_status["message"] = "Model moi tot hon. Dang cho admin xac nhan cap nhat."
                else:
                    retrain_status["status"] = "completed_candidate"
                    retrain_status["message"] = "Model moi da train xong nhung chua du dieu kien khuyen nghi cap nhat."

                retrain_status["result"] = result
                retrain_status["progress"] = 100

                # Clear cached models để load model mới
                models.clear()
                feature_columns = None

                save_training_log({
                    "data_stats": result.get("data_stats", {}),
                    "cleaning_stats": cleaning_stats,
                    "duplicate_info": duplicate_info,
                    "comparison": result.get("comparison", {}),
                    "deployed": result.get("deployed", False)
                }, log_type="retrain_from_new_data")

            except Exception as e:
                logger.error(f"[RETRAIN-NEW-DATA] Error: {e}")
                retrain_status = {"status": "failed", "progress": 0, "message": str(e), "result": None}

        thread = threading.Thread(target=background_retrain)
        thread.daemon = True
        thread.start()

        return jsonify({
            "success": True,
            "message": "Bat dau qua trinh retrain voi du lieu moi",
            "stats": {
                "new_data_rows": len(df_new),
                "existing_data_rows": len(df_existing),
                "combined_rows": len(df_combined),
                "new_unique_rows": duplicate_info.get("unique_new_rows", len(df_new_clean))
            },
            "status_url": "/retrain-status"
        })

    except Exception as exc:
        logger.error(f"[RETRAIN] Error: {exc}")
        return jsonify({"error": "Retrain that bai", "details": str(exc)}), 500


@app.route("/apply-new-model", methods=["POST"])
def apply_new_model():
    """Admin confirms whether the latest candidate models should replace active models."""
    global retrain_status, models, feature_columns

    payload = request.get_json(force=True) or {}
    confirm = bool(payload.get("confirm", False))
    result = retrain_status.get("result") or {}

    if retrain_status.get("status") not in {"completed_candidate", "completed", "completed_keep_old"}:
        return jsonify({"error": "Chua co candidate model nao de quyet dinh"}), 400

    try:
        if confirm:
            for model_name in MODEL_PATHS.keys():
                try:
                    save_model_version(model_name)
                except Exception as exc:
                    logger.warning(f"Could not backup model {model_name}: {exc}")

            deploy_candidate_models()
            clear_candidate_models()
            models.clear()
            feature_columns = None
            retrain_status["status"] = "completed"
            retrain_status["message"] = "Admin da cap nhat model moi."
            retrain_status["result"]["deployed"] = True
            retrain_status["result"]["pending_admin_decision"] = False
            history_entry = append_retrain_history("updated", result)
            save_training_log({"decision": "updated", "result": result}, log_type="admin_apply_model")
            return jsonify({"success": True, "decision": "updated", "history": history_entry})

        clear_candidate_models()
        retrain_status["status"] = "completed_keep_old"
        retrain_status["message"] = "Admin da giu model cu."
        if retrain_status.get("result"):
            retrain_status["result"]["deployed"] = False
            retrain_status["result"]["pending_admin_decision"] = False
        history_entry = append_retrain_history("kept_old", result)
        save_training_log({"decision": "kept_old", "result": result}, log_type="admin_apply_model")
        return jsonify({"success": True, "decision": "kept_old", "history": history_entry})

    except Exception as exc:
        logger.error(f"[APPLY-MODEL] Error: {exc}")
        return jsonify({"error": "Khong the ap dung quyet dinh model", "details": str(exc)}), 500


@app.route("/retrain-history", methods=["GET"])
def retrain_history():
    return jsonify({"history": list(reversed(get_retrain_history()))})


@app.route("/data-stats", methods=["GET"])
def data_stats():
    total_original = count_csv_rows(RAW_DATA_PATH)
    if total_original == 0:
        total_original = count_csv_rows(PROCESSED_DIR / "X_train.csv") + count_csv_rows(PROCESSED_DIR / "X_test.csv")

    daily_counts = []
    class_dist = {"0": 0, "1": 0}
    total_new = 0
    last_updated = None
    labeled_count = 0

    if NEW_DATA_PATH.exists():
        df = pd.read_csv(NEW_DATA_PATH)
        total_new = len(df)

        if "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
            counts = dates.dropna().value_counts().sort_index()
            daily_counts = [{"date": str(date), "count": int(count)} for date, count in counts.items()]
            valid_dates = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
            if not valid_dates.empty:
                last_updated = valid_dates.max().isoformat()

        if TARGET_COL in df.columns:
            labels = df[TARGET_COL].dropna().astype(int)
            labeled_count = int(len(labels))
            counts = labels.value_counts().to_dict()
            class_dist = {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}

    history = get_retrain_history()
    last_retrain = history[-1]["created_at"] if history else None

    class_total = class_dist["0"] + class_dist["1"]
    conditions = {
        "enough_samples": total_new >= 500,
        "current_samples": total_new,
        "has_two_classes": class_dist["0"] > 0 and class_dist["1"] > 0,
        "days_since_last_retrain_ok": True,
    }
    if last_retrain:
        days_since = (datetime.now() - datetime.fromisoformat(last_retrain)).days
        conditions["days_since_last_retrain"] = days_since
        conditions["days_since_last_retrain_ok"] = days_since >= 30

    return jsonify({
        "total_original": total_original,
        "total_new": total_new,
        "total_labeled": labeled_count,
        "total_combined": total_original + total_new,
        "last_updated": last_updated,
        "last_retrain": last_retrain,
        "daily_counts": daily_counts,
        "class_dist": class_dist,
        "class_percent": {
            "0": round(class_dist["0"] / class_total * 100, 2) if class_total else 0,
            "1": round(class_dist["1"] / class_total * 100, 2) if class_total else 0,
        },
        "conditions": conditions,
    })


@app.route("/new-data-stats", methods=["GET"])
def get_new_data_stats():
    """Trả về số liệu thống kê về dữ liệu mới đã thu thập"""
    if not NEW_DATA_PATH.exists():
        return jsonify({
            "exists": False,
            "row_count": 0,
            "labeled_count": 0,
            "unlabeled_count": 0,
            "message": "Chua co du lieu moi"
        })

    df = pd.read_csv(NEW_DATA_PATH)
    has_target = TARGET_COL in df.columns

    return jsonify({
        "exists": True,
        "row_count": len(df),
        "labeled_count": int(df[TARGET_COL].notna().sum()) if has_target else 0,
        "unlabeled_count": int(df[TARGET_COL].isna().sum()) if has_target else 0,
        "columns": list(df.columns),
        "first_row": df.iloc[0].to_dict() if len(df) > 0 else None,
        "last_updated": pd.read_csv(NEW_DATA_PATH, nrows=1).to_dict(orient="records")[-1].get("timestamp") if len(df) > 0 else None
    })


@app.route("/model-stats", methods=["GET"])
def model_stats():
    metrics = read_json(REPORTS_DIR / "metrics.json", {})
    curves = read_json(REPORTS_DIR / "curves.json", {})
    feature_importance = read_json(REPORTS_DIR / "feature_importance_vi.json",
                                   read_json(REPORTS_DIR / "feature_importance.json", {}))
    training_summary = read_json(REPORTS_DIR / "training_summary.json", {})
    quality_report = read_json(REPORTS_DIR / "data_quality_report.json", {})

    return jsonify(
        {
            "metrics": metrics,
            "curves": curves,
            "feature_importance": feature_importance,
            "training_summary": training_summary,
            "data_quality_report": quality_report,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
