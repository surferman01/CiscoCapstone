from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from core.data import _load_data, _split_features, load_model_artifact
from core.evaluation import (
    _build_per_class_importance_tables,
    _build_xgb_contrib_importance_tables,
    _build_visual_plot_payload,
    _compute_roc,
    _predict_with_thresholds,
)
from core.preprocessing import (
    _align_features_to_training,
    _apply_exclusions_for_training,
    _prep_for_xgboost,
)
from models.catboost_runner import _train_catboost
from models.xgboost_basic import _train_xgboost
from models.xgboost_mega import (
    _train_xgboost_mega_hierarchical,
    _train_xgboost_mega_multiclass,
    _train_xgboost_mega_ovr,
)


def _extract_training_config(cfg: dict) -> dict:
    return {
        "model_type": str(cfg.get("model_type", "")).strip(),
        "use_gpu": bool(cfg.get("use_gpu", True)),
        "hyperparameters": {
            k: v
            for k, v in cfg.items()
            if k
            not in {
                "model_type",
                "use_gpu",
                "target_column",
                "recommended_targets",
                "exclude_columns",
            }
        },
    }


def run_analysis(data_path: str, cfg: dict) -> dict:
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

    model_type_l = model_type.lower()
    if model_type_l == "catboost":
        results = _train_catboost(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "CatBoostClassifier"
    elif model_type_l == "xgboost":
        results = _train_xgboost(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "XGBClassifier"
    elif model_type_l == "mega multiclass xgboost":
        results = _train_xgboost_mega_multiclass(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "Mega Multiclass XGBoost"
    elif model_type_l == "mega ovr xgboost":
        results = _train_xgboost_mega_ovr(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "Mega OVR XGBoost"
    elif model_type_l == "mega hierarchical xgboost":
        results = _train_xgboost_mega_hierarchical(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "Mega Hierarchical XGBoost"
    else:
        raise ValueError(
            "model_type must be one of: 'CatBoost', 'XGBoost', "
            "'Mega Multiclass XGBoost', 'Mega OVR XGBoost', 'Mega Hierarchical XGBoost'"
        )

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
            "training_config": _extract_training_config(cfg),
        }
    )
    results["meta"] = meta
    results["dataframe"] = df_filtered.copy()

    artifact = {
        "model_type": model_type,
        "model": results.get("trained_model"),
        "target_column": target_col,
        "class_names": results.get("class_names") or classes,
        "label_classes": results.get("label_classes") or results.get("class_names") or classes,
        "training_features": results.get("training_features") or [],
        "recommended_targets": recommended_targets,
        "exclude_columns": explicit_excludes,
        "training_config": _extract_training_config(cfg),
    }
    results["artifact"] = artifact

    return results


def run_analysis_with_artifact(
    data_path: str, artifact_path: str, cfg: Optional[dict] = None
) -> dict:
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
            pd.Series(y_pred_enc).map(lambda i: label_classes[int(i)]).astype(str).values
        )

        cm = confusion_matrix(y_true, y_pred, labels=label_classes)
        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        report_text = classification_report(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted")

        counts = (
            pd.Series(y_true).value_counts().reindex(label_classes).fillna(0).astype(int)
        )
        bins_df = pd.DataFrame({"bin_name": label_classes, "count": counts.values})
        roc_list = _compute_roc(y_true, y_prob, label_classes)

        dmat = xgb.DMatrix(X_num, feature_names=list(X_num.columns))
        contrib = model.get_booster().predict(dmat, pred_contribs=True)
        contrib = np.asarray(contrib)

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
        visual_plots = _build_visual_plot_payload(
            X_num.reset_index(drop=True),
            pd.Series(y_true).reset_index(drop=True),
            y_prob,
            label_classes,
            shap_importance,
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
            "visual_plots": visual_plots,
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
                "training_config": art.get("training_config", {}),
            },
        }

    if model_type.lower() == "catboost":
        from catboost import Pool

        X_cb = X_raw.copy()

        num_cols = X_cb.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            X_cb[num_cols] = X_cb[num_cols].fillna(X_cb[num_cols].median(numeric_only=True))
        for c in X_cb.columns:
            if c not in num_cols:
                X_cb[c] = X_cb[c].astype("object").where(~X_cb[c].isna(), "NA").astype(str)

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

        counts = pd.Series(y_true).value_counts().reindex(class_names).fillna(0).astype(int)
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
        visual_plots = _build_visual_plot_payload(
            X_cb.reset_index(drop=True),
            pd.Series(y_true).reset_index(drop=True),
            y_prob,
            class_names,
            shap_importance,
        )

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
            "visual_plots": visual_plots,
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
                "training_config": art.get("training_config", {}),
            },
        }

    if model_type.lower() in {
        "mega multiclass xgboost",
        "mega ovr xgboost",
        "mega hierarchical xgboost",
    }:
        if not isinstance(model, dict):
            raise ValueError("Invalid mega pipeline artifact model payload.")
        pipeline_type = str(model.get("type", "")).strip().lower()
        label_classes = [str(c) for c in (model.get("label_classes") or class_names)]
        if not label_classes:
            raise ValueError("Artifact is missing class labels.")
        X_num = _prep_for_xgboost(X_raw)
        feat_order = model.get("training_features") or training_features or list(X_num.columns)
        X_num = _align_features_to_training(X_num, feat_order)
        imputer = model.get("imputer")
        X_eval = (
            imputer.transform(X_num.to_numpy(dtype=np.float32))
            if imputer is not None
            else X_num.to_numpy(dtype=np.float32)
        )

        y_pred_idx = None
        if pipeline_type == "mega_multiclass_xgb":
            multi_model = model.get("model")
            y_pred_idx = multi_model.predict(X_eval).astype(int)
            y_prob = multi_model.predict_proba(X_eval)
        elif pipeline_type == "mega_ovr_xgb":
            models = model.get("models") or []
            thresholds = np.asarray(model.get("thresholds"), dtype=float)
            if not models or thresholds.size != len(models):
                raise ValueError("Invalid OVR mega artifact internals.")
            y_prob = np.column_stack([m.predict_proba(X_eval)[:, 1] for m in models])
            y_pred_idx = _predict_with_thresholds(y_prob, thresholds).astype(int)
        elif pipeline_type == "mega_hierarchical_xgb":
            stage1 = model.get("stage1")
            stage2 = model.get("stage2")
            pass_idx = int(model.get("pass_idx", 0))
            fail_idxs = [int(i) for i in (model.get("fail_class_indices") or [])]
            threshold = float(model.get("threshold", 0.5))
            prob_fail = stage1.predict_proba(X_eval)[:, 1]
            if stage2 is None:
                fail_probs = np.ones((len(X_eval), max(len(fail_idxs), 1)), dtype=float)
            else:
                fail_probs = stage2.predict_proba(X_eval)

            y_prob = np.zeros((len(X_eval), len(label_classes)), dtype=float)
            y_prob[:, pass_idx] = 1.0 - prob_fail
            for i, cls_idx in enumerate(fail_idxs):
                y_prob[:, cls_idx] = prob_fail * fail_probs[:, i]

            fail_choice = np.argmax(fail_probs, axis=1)
            fail_pred = np.array([fail_idxs[i] for i in fail_choice], dtype=int)
            y_pred_idx = np.where(prob_fail >= threshold, fail_pred, pass_idx).astype(int)
        else:
            raise ValueError(f"Unknown mega pipeline artifact type: {pipeline_type}")

        y_pred = np.array([label_classes[int(i)] for i in y_pred_idx], dtype=str)
        cm = confusion_matrix(y_true, y_pred, labels=label_classes)
        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        report_text = classification_report(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted")
        counts = pd.Series(y_true).value_counts().reindex(label_classes).fillna(0).astype(int)
        bins_df = pd.DataFrame({"bin_name": label_classes, "count": counts.values})
        roc_list = _compute_roc(y_true, y_prob, label_classes)
        X_eval_df = pd.DataFrame(X_eval, columns=feat_order).reset_index(drop=True)
        y_true_series = pd.Series(y_true).reset_index(drop=True)
        shap_importance = {}
        if pipeline_type == "mega_multiclass_xgb":
            shap_importance = _build_xgb_contrib_importance_tables(
                multi_model,
                X_eval_df,
                y_true_series,
                label_classes,
            )
        visual_plots = _build_visual_plot_payload(
            X_eval_df,
            y_true_series,
            y_prob,
            label_classes,
            shap_importance,
        )
        wavg = report_dict.get("weighted avg", {}) or {}
        return {
            "model_name": f"{model_type} (loaded)",
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
                "training_config": art.get("training_config", {}),
                "feature_importance_source": (
                    "xgboost_pred_contribs" if shap_importance else "unavailable"
                ),
            },
        }

    raise ValueError(f"Unknown model_type in artifact: {model_type}")
