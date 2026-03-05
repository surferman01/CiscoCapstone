# analysis.py
import os
import re
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder


# -----------------------------
# Loading
# -----------------------------
def _load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(path)
    # attempt csv fallback
    return pd.read_csv(path)


def _split_features(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col]
    X = df.drop(columns=[target_col], errors="ignore")
    return X, y


# -----------------------------
# Weak non-measurement detection
# -----------------------------
_ID_NAME_RE = re.compile(
    r"(serial|serno|s/n|sn\b|lot|wafer|unit|device|die|barcode|uuid|guid|hash|mac|imei|imsi|ip\b|hostname|name\b|id\b|identifier|index)",
    re.IGNORECASE,
)
_DATE_NAME_RE = re.compile(r"(date|time|timestamp|datetime)", re.IGNORECASE)


def _weak_non_measurement_candidates(df: pd.DataFrame) -> List[str]:
    """
    Conservative:
      - obvious ID/date columns by name
      - datetime dtype
      - mostly-unique string columns that are mostly non-numeric (typical IDs)
    Does NOT drop numeric columns unless name strongly suggests ID/date.
    """
    n = len(df)
    if n == 0:
        return []

    drops: set[str] = set()
    for c in df.columns:
        name = str(c)

        # Strong name-based signals
        if _ID_NAME_RE.search(name) or _DATE_NAME_RE.search(name):
            drops.add(c)
            continue

        s = df[c]

        # Datetime dtype
        if pd.api.types.is_datetime64_any_dtype(s):
            drops.add(c)
            continue

        # Mostly-unique string IDs
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.dropna()
            if len(non_null) < max(10, int(0.1 * n)):
                continue

            uniq_ratio = non_null.nunique(dropna=True) / max(1, len(non_null))
            num_parse = pd.to_numeric(non_null, errors="coerce")
            numericish_ratio = float(num_parse.notna().mean())

            if uniq_ratio > 0.95 and numericish_ratio < 0.20:
                drops.add(c)

    return sorted(drops)


def _apply_exclusions_for_training(
    df: pd.DataFrame,
    target_col: str,
    recommended_targets: Optional[List[str]] = None,
    explicit_excludes: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop safe-to-drop columns from FEATURES while preserving chosen target_col.
    Also excludes other recommended target columns (except chosen target).
    """
    recommended_targets = recommended_targets or []
    explicit_excludes = explicit_excludes or []

    weak_meta = _weak_non_measurement_candidates(df)

    # Exclude recommended targets except chosen target
    rec_excludes = [
        c for c in recommended_targets if c in df.columns and c != target_col
    ]

    drop_cols = set(weak_meta) | set(rec_excludes) | set(explicit_excludes)
    drop_cols.discard(target_col)

    df2 = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    meta = {
        "dropped_weak_non_measurement": weak_meta,
        "dropped_recommended_targets": rec_excludes,
        "dropped_explicit": [c for c in explicit_excludes if c in df.columns],
        "total_dropped": int(len([c for c in drop_cols if c in df.columns])),
    }
    return df2, meta


# -----------------------------
# XGBoost preprocessing
# -----------------------------
def _prep_for_xgboost(X: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost expects numeric features.
    One-hot encode categoricals and force float matrix.
    """
    X = X.copy()

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X.columns if c not in num_cols]

    if len(num_cols):
        X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))

    for c in cat_cols:
        X[c] = X[c].astype("object").where(~X[c].isna(), "NA").astype(str)

    X_oh = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Force numeric float matrix (prevents object/ufunc sqrt errors later)
    X_oh = X_oh.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_oh.astype(np.float32)


def _align_features_to_training(
    X_new: pd.DataFrame, training_features: List[str]
) -> pd.DataFrame:
    X_new = X_new.copy()
    for c in training_features:
        if c not in X_new.columns:
            X_new[c] = 0.0
    X_new = X_new[training_features]
    return X_new


# -----------------------------
# ROC computation
# -----------------------------
def _compute_roc(
    y_true_labels: np.ndarray, y_prob: Optional[np.ndarray], class_names: List[str]
) -> List[Dict[str, Any]]:
    if y_prob is None:
        return []
    y_true = np.asarray(y_true_labels).astype(str)
    out = []
    # one-vs-rest ROC
    for i, cls in enumerate(class_names):
        y_bin = (y_true == str(cls)).astype(int)
        # if class missing in y_true, skip
        if y_bin.sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            out.append(
                {"class": str(cls), "fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
            )
        except Exception:
            continue
    return out


# -----------------------------
# Per-class importance tables
# -----------------------------
def _build_per_class_importance_tables(
    X_num: pd.DataFrame,
    y_true_labels: pd.Series,
    contrib: np.ndarray,
    class_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Builds one table per class:
      rank, feature, importance, share_pct, direction, failure_avg, pass_avg, pass_std

    "failure_avg" = mean feature value within that class
    "pass_avg/std" = mean/std in baseline class if 'PASS' exists else in "not this class"
    """
    X_vals = X_num.to_numpy(dtype=np.float64, copy=False)
    feats = list(X_num.columns)
    y_series = pd.Series(y_true_labels).astype(str).reset_index(drop=True)

    # Choose baseline group:
    baseline = "PASS" if "PASS" in [str(c) for c in class_names] else None

    # Normalize contrib shape:
    contrib = np.asarray(contrib)
    # XGBoost multiclass can be (N, K*(F+1)) already reshaped by caller, or (N, K, F+1)
    if contrib.ndim == 2:
        # binary: (N, F+1) -> treat as one "class" table, duplicated per class for UI consistency
        # we'll just use same importances per class
        shap_by_class = {str(cls): contrib[:, :-1] for cls in class_names}
    elif contrib.ndim == 3:
        # (N, K, F+1)
        shap_by_class = {}
        for i, cls in enumerate(class_names):
            shap_by_class[str(cls)] = contrib[:, i, :-1]
    else:
        # unknown -> bail out
        return {}

    out: Dict[str, pd.DataFrame] = {}

    for cls in class_names:
        cls_str = str(cls)
        shap_vals = shap_by_class.get(cls_str)
        if shap_vals is None:
            continue

        # class mask
        mask_cls = (y_series == cls_str).to_numpy()
        if mask_cls.sum() == 0:
            continue

        # mean |shap| over samples of that class
        imp = np.nanmean(np.abs(shap_vals[mask_cls, :]), axis=0)
        if not np.isfinite(imp).any():
            continue

        # direction sign (mean shap, class subset)
        direction = np.nanmean(shap_vals[mask_cls, :], axis=0)

        # feature stats
        failure_avg = np.nanmean(X_vals[mask_cls, :], axis=0)

        if baseline is not None and baseline != cls_str:
            mask_base = (y_series == baseline).to_numpy()
            if mask_base.sum() > 0:
                pass_avg = np.nanmean(X_vals[mask_base, :], axis=0)
                pass_std = np.nanstd(X_vals[mask_base, :], axis=0)
            else:
                pass_avg = np.nanmean(X_vals[~mask_cls, :], axis=0)
                pass_std = np.nanstd(X_vals[~mask_cls, :], axis=0)
        else:
            # not this class
            pass_avg = np.nanmean(X_vals[~mask_cls, :], axis=0)
            pass_std = np.nanstd(X_vals[~mask_cls, :], axis=0)

        total = float(np.nansum(imp)) if np.nansum(imp) != 0 else 1.0
        share_pct = (imp / total) * 100.0

        df_imp = (
            pd.DataFrame(
                {
                    "feature": feats,
                    "importance": imp,
                    "share_pct": share_pct,
                    "direction": direction,
                    "failure_avg": failure_avg,
                    "pass_avg": pass_avg,
                    "pass_std": pass_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        df_imp.insert(0, "rank", np.arange(1, len(df_imp) + 1))
        out[cls_str] = df_imp

    return out


# -----------------------------
# Train: XGBoost
# -----------------------------
def _train_xgboost(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    import xgboost as xgb
    from xgboost import XGBClassifier

    y_series = pd.Series(y).astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    class_names = [str(c) for c in le.classes_]
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Target must have at least 2 unique values.")

    X_num = _prep_for_xgboost(X)
    training_features = list(X_num.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_num,
        y_enc,
        test_size=float(cfg.get("test_size", 0.2)),
        random_state=int(cfg.get("random_state", 67)),
        stratify=y_enc,
    )

    use_gpu_requested = bool(cfg.get("use_gpu", True))

    if num_classes == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
        num_class_param = None
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        num_class_param = num_classes

    base_params = dict(
        n_estimators=int(cfg.get("iterations", 500)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_depth=int(cfg.get("depth", 8)),
        subsample=float(cfg.get("subsample", 0.9)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
        objective=objective,
        eval_metric=eval_metric,
        random_state=int(cfg.get("random_state", 67)),
        verbosity=int(cfg.get("verbosity", 1)),
    )
    if num_class_param is not None:
        base_params["num_class"] = num_class_param

    # GPU fallback (prevents your gpu_hist invalid error)
    if use_gpu_requested:
        candidates = [{"tree_method": "gpu_hist"}, {"tree_method": "hist"}]
    else:
        candidates = [{"tree_method": "hist"}]

    model = None
    last_err = None
    for extra in candidates:
        try:
            model = XGBClassifier(**base_params, **extra)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=bool(cfg.get("verbose", False)),
            )
            break
        except Exception as e:
            last_err = e
            model = None

    if model is None:
        raise RuntimeError(f"XGBoost training failed. Last error: {last_err}")

    y_pred_enc = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test_labels, y_pred_labels)
    f1w = f1_score(y_test_labels, y_pred_labels, average="weighted")

    report_dict = classification_report(
        y_test_labels, y_pred_labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(y_test_labels, y_pred_labels, zero_division=0)

    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)

    counts = (
        pd.Series(y_test_labels)
        .value_counts()
        .reindex(class_names)
        .fillna(0)
        .astype(int)
    )
    bins_df = pd.DataFrame({"bin_name": class_names, "count": counts.values})

    roc_list = _compute_roc(y_test_labels, y_prob, class_names)

    # Contributions (pred_contribs)
    dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
    contrib = model.get_booster().predict(dtest, pred_contribs=True)
    contrib = np.asarray(contrib)

    # Handle flattened multiclass contrib: (N, K*(F+1))
    if (
        contrib.ndim == 2
        and num_classes > 2
        and contrib.shape[1] == num_classes * (X_test.shape[1] + 1)
    ):
        contrib = contrib.reshape(contrib.shape[0], num_classes, X_test.shape[1] + 1)

    shap_importance = _build_per_class_importance_tables(
        X_test.reset_index(drop=True),
        pd.Series(y_test_labels).reset_index(drop=True),
        contrib,
        class_names=class_names,
    )

    wavg = report_dict.get("weighted avg", {}) or {}
    return {
        "model_name": "XGBClassifier",
        "trained_model": model,
        "training_features": training_features,
        "label_classes": class_names,  # label encoder classes
        "class_names": class_names,
        "metrics": {
            "Accuracy": float(acc),
            "Precision_weighted": float(wavg.get("precision", 0.0)),
            "Recall_weighted": float(wavg.get("recall", 0.0)),
            "F1_weighted": float(f1w),
        },
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "roc": roc_list,
        "bins": bins_df,
        "shap_importance": shap_importance,
    }


# -----------------------------
# Train: CatBoost
# -----------------------------
def _train_catboost(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    from catboost import CatBoostClassifier, Pool

    y_series = pd.Series(y).astype(str)
    class_names = sorted(y_series.dropna().unique().tolist())
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Target must have at least 2 unique values.")

    # CatBoost can handle numeric and some categoricals, but we'll keep it simple:
    X_cb = X.copy()

    # Fill numeric NaNs (safe default)
    num_cols = X_cb.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        X_cb[num_cols] = X_cb[num_cols].fillna(X_cb[num_cols].median(numeric_only=True))

    # Convert remaining non-numeric to string so CatBoost can treat them as categorical if needed
    # (But if you want strict numeric-only training, we can enforce that.)
    for c in X_cb.columns:
        if c not in num_cols:
            X_cb[c] = X_cb[c].astype("object").where(~X_cb[c].isna(), "NA").astype(str)

    training_features = list(X_cb.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_cb,
        y_series,
        test_size=float(cfg.get("test_size", 0.2)),
        random_state=int(cfg.get("random_state", 67)),
        stratify=y_series,
    )

    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    use_gpu = bool(cfg.get("use_gpu", True))
    task_type = "GPU" if use_gpu else "CPU"

    model = CatBoostClassifier(
        iterations=int(cfg.get("iterations", 500)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        depth=int(cfg.get("depth", 8)),
        loss_function="MultiClass" if num_classes > 2 else "Logloss",
        eval_metric="Accuracy",
        random_seed=int(cfg.get("random_state", 67)),
        task_type=task_type,
        verbose=int(cfg.get("cat_verbose", 100)),
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred).reshape(-1).astype(str)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test.astype(str), y_pred)
    f1w = f1_score(y_test.astype(str), y_pred, average="weighted")

    report_dict = classification_report(
        y_test.astype(str), y_pred, output_dict=True, zero_division=0
    )
    report_text = classification_report(y_test.astype(str), y_pred, zero_division=0)

    cm = confusion_matrix(y_test.astype(str), y_pred, labels=class_names)

    counts = (
        pd.Series(y_test.astype(str))
        .value_counts()
        .reindex(class_names)
        .fillna(0)
        .astype(int)
    )
    bins_df = pd.DataFrame({"bin_name": class_names, "count": counts.values})

    roc_list = _compute_roc(y_test.astype(str), y_prob, class_names)

    # CatBoost SHAP values
    shap_importance = {}
    try:
        # For multiclass, CatBoost may return (N, classes, F+1)
        shap_vals = model.get_feature_importance(test_pool, type="ShapValues")
        shap_vals = np.asarray(shap_vals)
        # If binary: (N, F+1) -> use same for all classes
        # If multiclass: (N, K, F+1)
        shap_importance = _build_per_class_importance_tables(
            X_test.reset_index(drop=True)
            if isinstance(X_test, pd.DataFrame)
            else pd.DataFrame(X_test),
            pd.Series(y_test.astype(str)).reset_index(drop=True),
            shap_vals,
            class_names=class_names,
        )
    except Exception:
        shap_importance = {}

    wavg = report_dict.get("weighted avg", {}) or {}
    return {
        "model_name": "CatBoostClassifier",
        "trained_model": model,
        "training_features": training_features,
        "class_names": class_names,
        "metrics": {
            "Accuracy": float(acc),
            "Precision_weighted": float(wavg.get("precision", 0.0)),
            "Recall_weighted": float(wavg.get("recall", 0.0)),
            "F1_weighted": float(f1w),
        },
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "roc": roc_list,
        "bins": bins_df,
        "shap_importance": shap_importance,
    }


# -----------------------------
# Save / Load model artifacts
# -----------------------------
def save_model_artifact(artifact: dict, out_path: str) -> str:
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    return out_path


def load_model_artifact(path: str) -> dict:
    with open(path, "rb") as f:
        art = pickle.load(f)
    if not isinstance(art, dict) or "model_type" not in art or "model" not in art:
        raise ValueError("Invalid model artifact (missing keys).")
    return art


# -----------------------------
# Main API: training
# -----------------------------
def run_analysis(data_path: str, cfg: dict) -> dict:
    """
    Called by GUI to train + evaluate + SHAP/importance tables.
    """
    cfg = cfg or {}
    model_type = str(cfg.get("model_type", "XGBoost")).strip()
    use_gpu = bool(cfg.get("use_gpu", True))

    df = _load_data(data_path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Loaded dataset is empty or invalid.")

    target_col = cfg.get("target_column")
    if not target_col or target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    recommended_targets = cfg.get("recommended_targets", []) or []
    explicit_excludes = cfg.get("exclude_columns", []) or []

    df_filtered, drop_meta = _apply_exclusions_for_training(
        df=df,
        target_col=target_col,
        recommended_targets=recommended_targets,
        explicit_excludes=explicit_excludes,
    )

    X, y = _split_features(df_filtered, target_col)
    y_series = pd.Series(y).astype(str)
    classes = sorted(y_series.dropna().unique().tolist())
    if len(classes) < 2:
        raise ValueError(f"Target '{target_col}' must have at least 2 unique values.")

    if model_type.lower() == "catboost":
        results = _train_catboost(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "CatBoostClassifier"
    elif model_type.lower() == "xgboost":
        results = _train_xgboost(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "XGBClassifier"
    else:
        raise ValueError("model_type must be 'CatBoost' or 'XGBoost'")

    # attach meta + dataframe for UI table
    meta = results.get("meta", {}) or {}
    meta.update(
        {
            "mode": "train",
            "model": model_name,
            "model_type": model_type,
            "use_gpu_requested": use_gpu,
            "target_column": target_col,
            "num_classes": len(classes),
            "classes": classes,
            "data_path": data_path,
            "n_rows": int(df.shape[0]),
            "n_cols_original": int(df.shape[1]),
            "n_cols_after_filter": int(df_filtered.shape[1]),
            "dropped_columns_info": drop_meta,
        }
    )
    results["meta"] = meta
    results["dataframe"] = df_filtered.copy()

    # build artifact for saving
    artifact = {
        "model_type": model_type,
        "model": results.get("trained_model"),
        "target_column": target_col,
        "class_names": results.get("class_names") or classes,
        "label_classes": results.get("label_classes")
        or results.get("class_names")
        or classes,
        "training_features": results.get("training_features") or [],
        "recommended_targets": recommended_targets,
        "exclude_columns": explicit_excludes,
    }
    results["artifact"] = artifact

    return results


# -----------------------------
# Main API: inference (no retrain)
# -----------------------------
def run_analysis_with_artifact(
    data_path: str, artifact_path: str, cfg: Optional[dict] = None
) -> dict:
    """
    Load a saved model artifact and evaluate/analyze a new dataset.
    """
    cfg = cfg or {}
    art = load_model_artifact(artifact_path)

    df = _load_data(data_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    target_col = art.get("target_column")
    if not target_col or target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in new dataset.")

    df_filtered, drop_meta = _apply_exclusions_for_training(
        df=df,
        target_col=target_col,
        recommended_targets=art.get("recommended_targets", []) or [],
        explicit_excludes=art.get("exclude_columns", []) or [],
    )

    X_raw, y_raw = _split_features(df_filtered, target_col)
    y_true = pd.Series(y_raw).astype(str).values

    model_type = str(art.get("model_type", "")).strip()
    model = art.get("model")
    class_names = [
        str(c)
        for c in (art.get("class_names") or sorted(pd.Series(y_true).unique().tolist()))
    ]
    training_features = art.get("training_features") or []

    y_prob = None

    if model_type.lower() == "xgboost":
        import xgboost as xgb

        X_num = _prep_for_xgboost(X_raw)
        if training_features:
            X_num = _align_features_to_training(X_num, training_features)

        y_pred_enc = model.predict(X_num)
        y_prob = model.predict_proba(X_num)

        label_classes = art.get("label_classes") or class_names
        label_classes = [str(c) for c in label_classes]
        y_pred = (
            pd.Series(y_pred_enc)
            .map(lambda i: label_classes[int(i)])
            .astype(str)
            .values
        )

        cm = confusion_matrix(y_true, y_pred, labels=label_classes)

        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        report_text = classification_report(y_true, y_pred, zero_division=0)

        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted")

        counts = (
            pd.Series(y_true)
            .value_counts()
            .reindex(label_classes)
            .fillna(0)
            .astype(int)
        )
        bins_df = pd.DataFrame({"bin_name": label_classes, "count": counts.values})

        roc_list = _compute_roc(y_true, y_prob, label_classes)

        dmat = xgb.DMatrix(X_num, feature_names=list(X_num.columns))
        contrib = model.get_booster().predict(dmat, pred_contribs=True)
        contrib = np.asarray(contrib)

        # attempt reshape for multiclass flattened
        if (
            contrib.ndim == 2
            and len(label_classes) > 2
            and contrib.shape[1] == len(label_classes) * (X_num.shape[1] + 1)
        ):
            contrib = contrib.reshape(
                contrib.shape[0], len(label_classes), X_num.shape[1] + 1
            )

        shap_importance = _build_per_class_importance_tables(
            X_num.reset_index(drop=True),
            pd.Series(y_true).reset_index(drop=True),
            contrib,
            class_names=label_classes,
        )

        wavg = report_dict.get("weighted avg", {}) or {}
        return {
            "model_name": "XGBClassifier (loaded)",
            "metrics": {
                "Accuracy": float(acc),
                "Precision_weighted": float(wavg.get("precision", 0.0)),
                "Recall_weighted": float(wavg.get("recall", 0.0)),
                "F1_weighted": float(f1w),
            },
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "classification_report_text": report_text,
            "roc": roc_list,
            "bins": bins_df,
            "shap_importance": shap_importance,
            "class_names": label_classes,
            "dataframe": df_filtered.copy(),
            "meta": {
                "mode": "inference",
                "artifact_path": artifact_path,
                "model_type": model_type,
                "target_column": target_col,
                "num_classes": len(label_classes),
                "classes": label_classes,
                "dropped_columns_info": drop_meta,
            },
        }

    elif model_type.lower() == "catboost":
        from catboost import Pool

        X_cb = X_raw.copy()

        # fill numeric NaNs
        num_cols = X_cb.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            X_cb[num_cols] = X_cb[num_cols].fillna(
                X_cb[num_cols].median(numeric_only=True)
            )
        for c in X_cb.columns:
            if c not in num_cols:
                X_cb[c] = (
                    X_cb[c].astype("object").where(~X_cb[c].isna(), "NA").astype(str)
                )

        if training_features:
            for c in training_features:
                if c not in X_cb.columns:
                    X_cb[c] = 0.0
            X_cb = X_cb[training_features]

        y_pred = model.predict(X_cb)
        y_pred = np.asarray(y_pred).reshape(-1).astype(str)

        try:
            y_prob = model.predict_proba(X_cb)
        except Exception:
            y_prob = None

        cm = confusion_matrix(y_true, y_pred, labels=class_names)

        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        report_text = classification_report(y_true, y_pred, zero_division=0)

        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted")

        counts = (
            pd.Series(y_true).value_counts().reindex(class_names).fillna(0).astype(int)
        )
        bins_df = pd.DataFrame({"bin_name": class_names, "count": counts.values})

        roc_list = _compute_roc(y_true, y_prob, class_names)

        shap_importance = {}
        try:
            pool = Pool(X_cb, y_true)
            shap_vals = model.get_feature_importance(pool, type="ShapValues")
            shap_vals = np.asarray(shap_vals)
            shap_importance = _build_per_class_importance_tables(
                X_cb.reset_index(drop=True),
                pd.Series(y_true).reset_index(drop=True),
                shap_vals,
                class_names=class_names,
            )
        except Exception:
            shap_importance = {}

        wavg = report_dict.get("weighted avg", {}) or {}
        return {
            "model_name": "CatBoostClassifier (loaded)",
            "metrics": {
                "Accuracy": float(acc),
                "Precision_weighted": float(wavg.get("precision", 0.0)),
                "Recall_weighted": float(wavg.get("recall", 0.0)),
                "F1_weighted": float(f1w),
            },
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "classification_report_text": report_text,
            "roc": roc_list,
            "bins": bins_df,
            "shap_importance": shap_importance,
            "class_names": class_names,
            "dataframe": df_filtered.copy(),
            "meta": {
                "mode": "inference",
                "artifact_path": artifact_path,
                "model_type": model_type,
                "target_column": target_col,
                "num_classes": len(class_names),
                "classes": class_names,
                "dropped_columns_info": drop_meta,
            },
        }

    else:
        raise ValueError(f"Unknown model_type in artifact: {model_type}")
