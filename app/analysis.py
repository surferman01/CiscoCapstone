import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
)

# --- Optional backends; we import inside helpers to make tooling friendlier ----


def _prepare_xy(df: pd.DataFrame):
    """
    Expect: id, test_bin, optional test_result, plus feature columns.
    Returns X, y after dropping id/test_result and filling NaNs numerically.
    """
    df = df.copy()
    if "test_result" in df.columns:
        df = df.drop(columns=["test_result"])

    y = df["test_bin"]
    X = df.drop(columns=[c for c in ["id", "test_bin"] if c in df.columns])

    # numeric-only median fill (prevents XGB/CatBoost from choking on NaNs)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def _train_catboost(X, y, cfg):
    from catboost import CatBoostClassifier, Pool

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 67),
        stratify=y,
    )

    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    params = dict(
        iterations=cfg.get("iterations", 500) # CHANGE TO ~10 FOR TESTING GUI CHANGES
        learning_rate=cfg.get("learning_rate", 0.05),
        depth=cfg.get("depth", 8),
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=cfg.get("random_state", 67),
        verbose=cfg.get("verbose", 100),
    )

    # Try GPU first if requested, fall back to CPU
    use_gpu = cfg.get("use_gpu", True)
    if use_gpu:
        params.update({"task_type": "GPU"})
        if "devices" in cfg:
            params["devices"] = cfg["devices"]

    try:
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    except Exception:
        params.pop("task_type", None)
        params.pop("devices", None)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # predictions
    y_pred = model.predict(X_test).flatten()
    y_prob = model.predict_proba(X_test)

    # metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # SHAP per-class importance with directional stats and normalized shares
    shap_importance = {}
    try:
        shap_values = model.get_feature_importance(train_pool, type="ShapValues")
        shap_arr = np.array(shap_values)
        # Expected shape: (n_samples, n_classes, n_features + 1)
        if shap_arr.ndim == 3:
            class_labels = getattr(model, "classes_", None)
            if class_labels is None:
                class_labels = np.unique(y)
            feature_values = shap_arr[:, :, :-1]
            y_train_str = y_train.astype(str)
            pass_mask = y_train_str.str.upper() == "PASS"
            pass_means = X_train[pass_mask].mean(numeric_only=True)
            pass_std = X_train[pass_mask].std(numeric_only=True)
            for idx, cls in enumerate(class_labels):
                if idx >= feature_values.shape[1]:
                    break
                cls_mask = y_train_str == str(cls)
                cls_means = X_train[cls_mask].mean(numeric_only=True)
                class_vals = feature_values[:, idx, :]
                mean_abs = np.abs(class_vals).mean(axis=0)
                mean_signed = class_vals.mean(axis=0)
                total = mean_abs.sum()
                df = pd.DataFrame({"feature": list(X.columns), "importance": mean_abs})
                df["share_pct"] = np.where(total > 0, (df["importance"] / total) * 100, 0)
                df["direction"] = mean_signed
                df["failure_avg"] = df["feature"].map(cls_means.to_dict())
                df["pass_avg"] = df["feature"].map(pass_means.to_dict())
                df["pass_std"] = df["feature"].map(pass_std.to_dict())
                df = df.sort_values("importance", ascending=False)
                df["rank"] = np.arange(1, len(df) + 1)
                shap_importance[str(cls)] = df
    except Exception:
        shap_importance = {}

    # ROC (one-vs-rest)
    roc = []
    classes = np.unique(y_test)
    for i, cls in enumerate(classes):
        try:
            y_true = (y_test == cls).astype(int)
            y_score = y_prob[:, i]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc.append(
                {"class": str(cls), "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
            )
        except Exception:
            pass

    return {
        "model_name": "CatBoostClassifier",
        "classification_report": report,
        "shap_importance": shap_importance,
        "roc": roc,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def _train_xgboost(X, y, cfg):
    """
    Multi-class XGBoost (softprob). Tries GPU first (gpu_hist/gpu_predictor), falls back to CPU.
    """
    from xgboost import XGBClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 67),
        stratify=y,
    )

    params = dict(
        n_estimators=cfg.get("n_estimators", 500),
        learning_rate=cfg.get("learning_rate", 0.05),
        max_depth=cfg.get("max_depth", 8),
        subsample=cfg.get("subsample", 1.0),
        colsample_bytree=cfg.get("colsample_bytree", 1.0),
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=cfg.get("random_state", 67),
        n_jobs=cfg.get("n_jobs", 0),  # 0 lets XGB choose
    )

    use_gpu = cfg.get("use_gpu", True)
    if use_gpu:
        params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    else:
        params.update({"tree_method": "hist"})

    try:
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=cfg.get("verbose", 100) > 0,
        )
    except Exception:
        # fallback to CPU if GPU isn’t available
        params.update({"tree_method": "hist"})
        params.pop("predictor", None)
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=cfg.get("verbose", 100) > 0,
        )

    # predictions
    y_prob = model.predict_proba(X_test)
    y_pred = y_prob.argmax(axis=1)

    # If target labels aren’t 0..K-1, map back
    classes_ = getattr(model, "classes_", None)
    if classes_ is not None and not np.array_equal(classes_, np.arange(len(classes_))):
        y_pred = np.array([classes_[i] for i in y_pred])

    # metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    shap_importance = {}
    try:
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, feature_names=list(X.columns))
        shap_values = booster.predict(dtrain, pred_contribs=True)
        shap_arr = np.array(shap_values)
        # Expected shape: (n_samples, n_classes, n_features + 1)
        if shap_arr.ndim == 3:
            feature_values = shap_arr[:, :, :-1]
            class_labels = getattr(model, "classes_", None)
            if class_labels is None:
                class_labels = np.unique(y)
            y_train_str = y_train.astype(str)
            pass_mask = y_train_str.str.upper() == "PASS"
            pass_means = X_train[pass_mask].mean(numeric_only=True)
            pass_std = X_train[pass_mask].std(numeric_only=True)
            for idx, cls in enumerate(class_labels):
                if idx >= feature_values.shape[1]:
                    break
                cls_mask = y_train_str == str(cls)
                cls_means = X_train[cls_mask].mean(numeric_only=True)
                class_vals = feature_values[:, idx, :]
                mean_abs = np.abs(class_vals).mean(axis=0)
                mean_signed = class_vals.mean(axis=0)
                total = mean_abs.sum()
                df = pd.DataFrame({"feature": list(X.columns), "importance": mean_abs})
                df["share_pct"] = np.where(total > 0, (df["importance"] / total) * 100, 0)
                df["direction"] = mean_signed
                df["failure_avg"] = df["feature"].map(cls_means.to_dict())
                df["pass_avg"] = df["feature"].map(pass_means.to_dict())
                df["pass_std"] = df["feature"].map(pass_std.to_dict())
                df = df.sort_values("importance", ascending=False)
                df["rank"] = np.arange(1, len(df) + 1)
                shap_importance[str(cls)] = df
    except Exception:
        shap_importance = {}

    # ROC (one-vs-rest)
    roc = []
    classes = np.unique(y_test)
    for i, cls in enumerate(classes):
        try:
            y_true = (y_test == cls).astype(int)
            # find the proba column for this class
            if classes_ is not None:
                cls_index = int(np.where(classes_ == cls)[0][0])
            else:
                cls_index = i
            y_score = y_prob[:, cls_index]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc.append(
                {"class": str(cls), "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
            )
        except Exception:
            pass

    return {
        "model_name": "XGBClassifier",
        "classification_report": report,
        "shap_importance": shap_importance,
        "roc": roc,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def run_analysis(data_path: str, config: dict) -> dict:
    # Load
    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)

    X, y = _prepare_xy(df)

    model_type = (config or {}).get("model_type", "CatBoost")

    if model_type == "XGBoost":
        res = _train_xgboost(X, y, config or {})
    else:  # default to CatBoost
        res = _train_catboost(X, y, config or {})

    # Build bins from the full label distribution (not just the test split)
    bcounts = pd.Series(y).value_counts().sort_index()
    bins = pd.DataFrame(
        {"bin_name": bcounts.index.astype(str), "count": bcounts.values}
    )
    bins["rate"] = bins["count"] / bins["count"].sum()

    artifacts = {
        "log": res["classification_report"],
        "models": {},  # attach pickled model bytes here if desired
    }

    return {
        "meta": {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "model": res["model_name"],
        },
        "shap_importance": res.get("shap_importance", {}),
        "bins": bins,
        "roc": res["roc"],
        "classification_report": res["classification_report"],
        "artifacts": artifacts,
        "dataframe": df.head(200),  # for the preview table
    }
