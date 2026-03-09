from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from core.evaluation import (
    _build_per_class_importance_tables,
    _build_visual_plot_payload,
    _compute_roc,
)


def _train_catboost(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    from catboost import CatBoostClassifier, Pool

    y_series = pd.Series(y).astype(str)
    class_names = sorted(y_series.dropna().unique().tolist())
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Target must have at least 2 unique values.")

    X_cb = X.copy()
    num_cols = X_cb.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        X_cb[num_cols] = X_cb[num_cols].fillna(X_cb[num_cols].median(numeric_only=True))

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

    shap_importance = {}
    try:
        shap_vals = model.get_feature_importance(test_pool, type="ShapValues")
        shap_vals = np.asarray(shap_vals)
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
    visual_plots = _build_visual_plot_payload(
        X_test.reset_index(drop=True)
        if isinstance(X_test, pd.DataFrame)
        else pd.DataFrame(X_test),
        pd.Series(y_test.astype(str)).reset_index(drop=True),
        y_prob,
        class_names,
        shap_importance,
    )

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
        "visual_plots": visual_plots,
    }
