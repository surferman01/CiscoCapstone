# analysis.py
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
from sklearn.preprocessing import label_binarize

import re

_ID_NAME_RE = re.compile(
    r"(serial|serno|s/n|sn\b|lot|wafer|unit|device|die|barcode|uuid|guid|hash|mac|imei|imsi|ip\b|hostname|name\b|id\b|identifier|index)",
    re.IGNORECASE,
)

_DATE_NAME_RE = re.compile(r"(date|time|timestamp|datetime)", re.IGNORECASE)


def _weak_non_measurement_candidates(df: pd.DataFrame) -> list[str]:
    """
    Conservative heuristic:
    - Drop obvious identifier columns by name (serial, lot, uuid, etc.)
    - Drop columns that are *mostly unique* AND are non-numeric strings (typical IDs)
    - Drop obvious datetime columns
    Do NOT drop numeric columns unless they have an obvious ID-like name.
    """
    n = len(df)
    if n == 0:
        return []

    candidates: set[str] = set()

    for c in df.columns:
        name = str(c)

        # Name-based: very high confidence
        if _ID_NAME_RE.search(name):
            candidates.add(c)
            continue

        if _DATE_NAME_RE.search(name):
            candidates.add(c)
            continue

        s = df[c]

        # Datetime dtype -> metadata
        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.add(c)
            continue

        # Object-like columns that look like IDs:
        # mostly unique + mostly non-numeric => likely ID string
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.dropna()
            if len(non_null) < max(10, int(0.1 * n)):
                # too sparse, don't decide
                continue

            # uniqueness ratio
            uniq_ratio = non_null.nunique(dropna=True) / max(1, len(non_null))

            # fraction of values that can be parsed as numbers
            num_parse = pd.to_numeric(non_null, errors="coerce")
            numericish_ratio = float(num_parse.notna().mean())

            # If it's mostly unique AND mostly not numeric => likely an ID
            if uniq_ratio > 0.95 and numericish_ratio < 0.20:
                candidates.add(c)
                continue

    return sorted(candidates)


def _apply_exclusions_for_training(
    df: pd.DataFrame,
    target_col: str,
    recommended_targets: list[str] | None = None,
    explicit_excludes: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Drops safe-to-drop non-measurement columns *from features* while preserving the chosen target.
    Returns (df_filtered, meta_debug).
    """
    recommended_targets = recommended_targets or []
    explicit_excludes = explicit_excludes or []

    weak_meta = _weak_non_measurement_candidates(df)

    # Exclude all recommended targets except the selected one
    rec_excludes = [
        c for c in recommended_targets if c in df.columns and c != target_col
    ]

    # Combine drops (but never drop the chosen target)
    drop_cols = set(weak_meta) | set(rec_excludes) | set(explicit_excludes)
    drop_cols.discard(target_col)

    df2 = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    meta = {
        "dropped_weak_non_measurement": weak_meta,
        "dropped_recommended_targets": rec_excludes,
        "dropped_explicit": [c for c in explicit_excludes if c in df.columns],
        "total_dropped": len([c for c in drop_cols if c in df.columns]),
    }
    return df2, meta


# -------------------------
# Data loading
# -------------------------
def _load_df(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


# -------------------------
# Mild column dropping (SAFE)
# -------------------------
def _drop_obvious_id_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    MUCH safer than the previous aggressive dropping.

    Drops only:
    - common ID-like names (id/serial/uuid/etc.)
    - columns that are all-unique (one unique per row) => leakage/memorization
    Keeps everything else (including categoricals).
    """
    n = len(df)
    drop = set()

    name_hints = ("id", "serial", "sn", "uuid", "guid", "barcode", "ticket", "hash")
    exact_drop = {"index", "row_id"}

    for c in df.columns:
        if c == target_col:
            continue

        cl = c.strip().lower()

        # exact matches
        if cl in exact_drop:
            drop.add(c)
            continue

        # name-based
        if any(h in cl for h in name_hints) or cl.endswith("_id") or cl.endswith("id"):
            drop.add(c)
            continue

        # all-unique => strong identifier signal
        try:
            nunique = df[c].nunique(dropna=True)
            if n > 0 and nunique >= 0.98 * n:
                drop.add(c)
        except Exception:
            # if nunique fails, do not drop
            pass

    if drop:
        return df.drop(columns=[c for c in drop if c in df.columns])
    return df


# -------------------------
# Feature prep
# -------------------------
def _split_features(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    return X, y


def _fill_missing_for_catboost(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # numeric median
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))

    # categoricals: fill NA with string token
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype("object").where(~X[c].isna(), "NA")
    return X


def _prep_for_xgboost(X: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost expects numeric features. We'll one-hot encode categoricals,
    then force the final matrix to be pure numeric float.
    """
    X = X.copy()

    # Separate numeric vs categorical-ish
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Fill numeric NaNs with median
    if len(num_cols):
        X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))

    # Coerce categoricals to strings, fill NaNs
    for c in cat_cols:
        X[c] = X[c].astype("object").where(~X[c].isna(), "NA").astype(str)

    # One-hot encode categoricals
    X_oh = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # ✅ CRITICAL: force everything to numeric float
    # Any non-numeric garbage becomes NaN, then we fill with 0.
    X_oh = X_oh.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Ensure consistent dtype
    return X_oh.astype(np.float32)


# -------------------------
# Per-class SHAP-like tables
# -------------------------
def _build_per_class_importance_tables(
    X_num: pd.DataFrame,
    y_true_labels,
    shap_values,
    class_names,
):
    """
    Returns dict[class_lower] -> DataFrame with columns:
    rank, feature, importance, share_pct, direction, failure_avg, pass_avg, pass_std
    """
    X_vals = X_num.to_numpy(dtype=np.float64, copy=False)
    feats = list(X_num.columns)
    y_series = pd.Series(y_true_labels).astype(str)

    sv = np.asarray(shap_values)

    # Normalize to (N, K, F+1)
    if sv.ndim == 2:
        # binary single-output style => treat as K=1
        sv = sv[:, None, :]
    elif sv.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected SHAP values shape: {sv.shape}")

    N, K, Fp1 = sv.shape
    F = Fp1 - 1  # ignore bias term

    class_names = [str(c) for c in class_names]
    out = {}

    # If K==1 but we have 2 classes, build two tables (positive and negative)
    if K == 1 and len(class_names) == 2:
        sv_pos = sv[:, 0, :F]
        sv_neg = -sv_pos
        per_class = {class_names[1]: sv_pos, class_names[0]: sv_neg}
    else:
        per_class = {
            class_names[i]: sv[:, i, :F] for i in range(min(K, len(class_names)))
        }

    for cls_name, sv_cls in per_class.items():
        imp = np.mean(np.abs(sv_cls), axis=0)
        total = float(np.sum(imp)) if np.sum(imp) > 0 else 1.0
        share = (imp / total) * 100.0
        direction = np.mean(sv_cls, axis=0)

        mask = (y_series == str(cls_name)).values
        if mask.sum() == 0:
            fail_avg = np.full(F, np.nan)
            other_avg = np.full(F, np.nan)
            other_std = np.full(F, np.nan)
        else:
            fail_avg = np.nanmean(X_vals[mask, :], axis=0)
            other_avg = (
                np.nanmean(X_vals[~mask, :], axis=0)
                if (~mask).sum()
                else np.full(F, np.nan)
            )
            other_std = (
                np.nanstd(X_vals[~mask, :], axis=0)
                if (~mask).sum()
                else np.full(F, np.nan)
            )

        df = (
            pd.DataFrame(
                {
                    "feature": feats,
                    "importance": imp,
                    "share_pct": share,
                    "direction": direction,
                    "failure_avg": fail_avg,
                    "pass_avg": other_avg,
                    "pass_std": other_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        df.insert(0, "rank", np.arange(1, len(df) + 1))
        out[str(cls_name).strip().lower()] = df

    return out


# -------------------------
# ROC helper
# -------------------------
def _compute_roc(y_true_labels, y_prob, class_names):
    """
    Returns list[{class,fpr,tpr,auc}] (one-vs-rest for multiclass).
    """
    class_names = [str(c) for c in class_names]
    y_true = pd.Series(y_true_labels).astype(str)

    roc_list = []
    if len(class_names) == 2:
        # For binary: compute both classes curves (symmetry), but we can provide both like your GUI shows
        for i, cls in enumerate(class_names):
            y_bin = (y_true == cls).astype(int).values
            try:
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
                roc_list.append(
                    {"class": cls, "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
                )
            except Exception:
                pass
        return roc_list

    # multiclass OVR
    Y = label_binarize(y_true, classes=class_names)
    for i, cls in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
            roc_list.append(
                {"class": cls, "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
            )
        except Exception:
            pass
    return roc_list


# -------------------------
# Train: CatBoost
# -------------------------


def _apply_feature_excludes(
    df: pd.DataFrame, target_col: str, exclude_cols
) -> pd.DataFrame:
    """
    Drops exclude_cols from dataframe, but never drops the chosen target_col.
    """
    if not exclude_cols:
        return df
    drop = [c for c in exclude_cols if c in df.columns and c != target_col]
    if drop:
        return df.drop(columns=drop)
    return df


def _train_catboost(X: pd.DataFrame, y: pd.Series, cfg: dict):
    from catboost import CatBoostClassifier, Pool

    X_cb = _fill_missing_for_catboost(X)

    # Identify categorical feature indices for CatBoost
    num_cols = X_cb.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X_cb.columns if c not in num_cols]
    cat_idx = [X_cb.columns.get_loc(c) for c in cat_cols]

    y_series = pd.Series(y)
    y_nonnull = y_series.dropna().astype(str)

    classes = [str(c) for c in pd.unique(y_nonnull)]
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError(
            "Target must have at least 2 unique values (after dropping NaNs)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_cb,
        y_series.astype(str),
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 67),
        stratify=y_series.astype(str),
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_idx if cat_idx else None)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx if cat_idx else None)

    use_gpu = bool(cfg.get("use_gpu", True))
    loss = "Logloss" if num_classes == 2 else "MultiClass"

    params = dict(
        iterations=int(cfg.get("iterations", 500)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        depth=int(cfg.get("depth", 8)),
        loss_function=loss,
        eval_metric="Accuracy",
        random_seed=int(cfg.get("random_state", 67)),
        verbose=int(cfg.get("verbose", 100)),
    )

    if use_gpu:
        params.update({"task_type": "GPU"})
        # optional: cfg["devices"] if you want

    # train (fallback to CPU automatically if GPU fails)
    try:
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    except Exception:
        params.pop("task_type", None)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_pred = model.predict(X_test).flatten().astype(str)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=[str(c) for c in model.classes_])
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, zero_division=0
    )

    # bins (use model.classes_ order)
    bin_names = [str(c) for c in model.classes_]
    counts = pd.Series(y_test).value_counts().reindex(bin_names).fillna(0).astype(int)
    bins_df = pd.DataFrame({"bin_name": bin_names, "count": counts.values})

    # global feature importance
    fi_vals = model.get_feature_importance(train_pool, type="FeatureImportance")
    fi_df = pd.DataFrame(
        {"feature": list(X_cb.columns), "importance": fi_vals}
    ).sort_values("importance", ascending=False)

    # ROC
    roc_list = _compute_roc(y_test, y_prob, bin_names)

    # per-class SHAP-style importance tables
    shap_vals = model.get_feature_importance(
        test_pool, type="ShapValues"
    )  # (N,K,F+1) or (N,F+1)
    shap_importance = _build_per_class_importance_tables(
        X_train.reset_index(drop=True).astype(float, errors="ignore")
        if False
        else _prep_for_xgboost(X_test).reset_index(drop=True),
        # NOTE: We want numeric matrix for averages; easiest is use xgboost-style numeric features:
        # We'll instead compute on a numeric version of X_test:
        # (This matches what the tables display; CatBoost SHAP is computed on original features.)
        # Use a safe numeric encoding:
        y_test.reset_index(drop=True),
        shap_vals,
        class_names=bin_names,
    )

    return {
        "model_name": "CatBoostClassifier",
        "metrics": {"Accuracy": float(acc), "F1_weighted": float(f1w)},
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "feature_importance": fi_df,
        "roc": roc_list,
        "bins": bins_df,
        "shap_importance": shap_importance,
        "class_names": bin_names,
    }


# -------------------------
# Train: XGBoost
# -------------------------
def _train_xgboost(X: pd.DataFrame, y: pd.Series, cfg: dict):
    import numpy as np
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    # Encode target labels to ints for XGB
    y_series = pd.Series(y).astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    class_names = [str(c) for c in le.classes_]
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Target must have at least 2 unique values.")

    # One-hot encode categoricals
    X_num = _prep_for_xgboost(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_num,
        y_enc,
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 67),
        stratify=y_enc,
    )

    use_gpu_requested = bool(cfg.get("use_gpu", True))

    # objective / eval metric
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

    # Try GPU if requested; otherwise CPU.
    # NOTE: some xgboost builds don't support gpu_hist at all => fallback to hist.
    if use_gpu_requested:
        candidates = [
            {"tree_method": "gpu_hist"},  # modern GPU
            {"tree_method": "hist"},  # CPU fallback
        ]
    else:
        candidates = [{"tree_method": "hist"}]

    last_err = None
    model = None
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
            continue

    if model is None:
        raise RuntimeError(f"XGBoost training failed. Last error: {last_err}")

    # Predictions
    y_pred_enc = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test_labels, y_pred_labels)
    f1w = f1_score(y_test_labels, y_pred_labels, average="weighted")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)
    report_dict = classification_report(
        y_test_labels, y_pred_labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test_labels, y_pred_labels, zero_division=0
    )

    # bins
    counts = (
        pd.Series(y_test_labels)
        .value_counts()
        .reindex(class_names)
        .fillna(0)
        .astype(int)
    )
    bins_df = pd.DataFrame({"bin_name": class_names, "count": counts.values})

    # global feature importance
    fi_df = pd.DataFrame(
        {"feature": list(X_num.columns), "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    # ROC
    roc_list = _compute_roc(y_test_labels, y_prob, class_names)

    # SHAP-like contributions (built-in pred_contribs)
    dtest = xgb.DMatrix(X_test, feature_names=list(X_num.columns))
    contrib = model.get_booster().predict(dtest, pred_contribs=True)
    contrib = np.asarray(contrib)

    # Handle (N, K*(F+1)) flattened multiclass case
    if (
        contrib.ndim == 2
        and num_classes > 2
        and contrib.shape[1] == num_classes * (X_num.shape[1] + 1)
    ):
        contrib = contrib.reshape(contrib.shape[0], num_classes, X_num.shape[1] + 1)

    shap_importance = _build_per_class_importance_tables(
        X_test.reset_index(drop=True),
        pd.Series(y_test_labels).reset_index(drop=True),
        contrib,
        class_names=class_names,
    )

    return {
        "model_name": "XGBClassifier",
        "metrics": {"Accuracy": float(acc), "F1_weighted": float(f1w)},
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "feature_importance": fi_df,
        "roc": roc_list,
        "bins": bins_df,
        "shap_importance": shap_importance,
        "class_names": class_names,
    }


# -------------------------
# Public entrypoint
# -------------------------
def run_analysis(data_path: str, cfg: dict) -> dict:
    """
    End-to-end entrypoint called by the GUI worker.

    Responsibilities:
      - Load CSV/Parquet
      - Validate/choose target column (GUI should provide cfg["target_column"])
      - Apply conservative exclusion to avoid training on non-measurement metadata
        (IDs/serials, datetime, and other "recommended target" columns)
      - Train chosen model (CatBoost or XGBoost)
      - Build payload for GUI (metrics, bins, roc, shap_importance, etc.)

    Expected cfg keys (most are optional):
      - model_type: "CatBoost" | "XGBoost"
      - use_gpu: bool
      - target_column: str   (required)
      - recommended_targets: list[str]  (optional; from GUI suggestions)
      - exclude_columns: list[str]      (optional; explicit drops)
      - test_size, random_state, iterations, learning_rate, depth, etc.
    """
    cfg = cfg or {}
    model_type = str(cfg.get("model_type", "XGBoost")).strip()
    use_gpu = bool(cfg.get("use_gpu", True))

    # --- Load data
    df = _load_df(data_path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Loaded dataset is empty or invalid.")

    # --- Validate target column
    target_col = cfg.get("target_column")
    if not target_col or target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)[:50]}"
        )

    # --- Conservative exclusion: don't train on non-measurement/metadata
    recommended_targets = cfg.get("recommended_targets", []) or []
    explicit_excludes = cfg.get("exclude_columns", []) or []

    df_filtered, drop_meta = _apply_exclusions_for_training(
        df=df,
        target_col=target_col,
        recommended_targets=recommended_targets,
        explicit_excludes=explicit_excludes,
    )

    # --- Split features/target
    X, y = _split_features(df_filtered, target_col)

    # Basic sanity
    y_series = pd.Series(y).astype(str)
    class_names = sorted(y_series.dropna().unique().tolist())
    if len(class_names) < 2:
        raise ValueError(
            f"Target '{target_col}' must have at least 2 unique values. Found: {class_names}"
        )

    # --- Train model
    if model_type.lower() == "catboost":
        results = _train_catboost(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "CatBoostClassifier"
    elif model_type.lower() == "xgboost":
        results = _train_xgboost(X, y_series, {**cfg, "use_gpu": use_gpu})
        model_name = "XGBClassifier"
    else:
        raise ValueError("model_type must be 'CatBoost' or 'XGBoost'")

    # --- Attach common metadata + dataframe preview for GUI
    meta = results.get("meta", {}) or {}
    meta.update(
        {
            "model": model_name,
            "model_type": model_type,
            "use_gpu_requested": use_gpu,
            "target_column": target_col,
            "num_classes": len(class_names),
            "classes": class_names,
            "data_path": data_path,
            "n_rows": int(df.shape[0]),
            "n_cols_original": int(df.shape[1]),
            "n_cols_after_filter": int(df_filtered.shape[1]),
            "dropped_columns_info": drop_meta,
        }
    )
    results["meta"] = meta

    # Provide a dataframe for the Data Table tab (cap columns if needed)
    # Keep original columns (helps users inspect), but it's fine to show filtered too.
    # We'll show filtered to match training set, plus the target.
    results["dataframe"] = df_filtered.copy()

    return results
