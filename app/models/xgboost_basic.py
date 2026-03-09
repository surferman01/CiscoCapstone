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
from sklearn.preprocessing import LabelEncoder

from core.evaluation import (
    _build_per_class_importance_tables,
    _build_visual_plot_payload,
    _compute_roc,
)
from core.preprocessing import _prep_for_xgboost


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

    dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
    contrib = model.get_booster().predict(dtest, pred_contribs=True)
    contrib = np.asarray(contrib)

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
    visual_plots = _build_visual_plot_payload(
        X_test.reset_index(drop=True),
        pd.Series(y_test_labels).reset_index(drop=True),
        y_prob,
        class_names,
        shap_importance,
    )

    wavg = report_dict.get("weighted avg", {}) or {}
    return {
        "model_name": "XGBClassifier",
        "trained_model": model,
        "training_features": training_features,
        "label_classes": class_names,
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
