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
from sklearn.impute import SimpleImputer
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


def _can_stratify(y: np.ndarray) -> bool:
    y = np.asarray(y)
    if y.size == 0:
        return False
    vals, cnt = np.unique(y, return_counts=True)
    return len(vals) > 1 and bool(np.all(cnt >= 2))


def _split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 67,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    holdout = float(val_size + test_size)
    frac_test = test_size / max(holdout, 1e-12)
    last_split = None

    for i in range(8):
        rs = int(random_state + i)
        strat1 = y if _can_stratify(y) else None
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X,
            y,
            test_size=holdout,
            random_state=rs,
            stratify=strat1,
        )
        strat2 = y_tmp if _can_stratify(y_tmp) else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp,
            y_tmp,
            test_size=frac_test,
            random_state=rs + 1,
            stratify=strat2,
        )
        last_split = (X_train, X_val, X_test, y_train, y_val, y_test)
        if np.unique(y_train).size >= 2:
            return last_split

    assert last_split is not None
    return last_split


def _class_weight_from_power(y: np.ndarray, power: float = 1.0) -> Tuple[np.ndarray, Dict[int, float]]:
    y = np.asarray(y).astype(int)
    vals, cnt = np.unique(y, return_counts=True)
    total = float(len(y))
    n_classes = float(len(vals))
    weights: Dict[int, float] = {}
    for cls, c in zip(vals, cnt):
        base = total / max(n_classes * float(c), 1.0)
        weights[int(cls)] = float(base ** float(power))
    sample_weight = np.array([weights[int(v)] for v in y], dtype=float)
    return sample_weight, weights


def _safe_sample_data(name: str, X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if name == "none":
        return X, y
    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE

        sampler = None
        if name == "random_over":
            sampler = RandomOverSampler(random_state=seed)
        elif name == "smote":
            sampler = SMOTE(random_state=seed, k_neighbors=3)
        elif name == "borderline_smote":
            sampler = BorderlineSMOTE(random_state=seed, k_neighbors=3, kind="borderline-1")
        elif name == "adasyn":
            sampler = ADASYN(random_state=seed, n_neighbors=3)
        elif name == "smoteenn":
            sampler = SMOTEENN(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=3))
        else:
            return X, y
        return sampler.fit_resample(X, y)
    except Exception:
        # Fallback keeps pipeline robust on tiny/edge-case datasets.
        return X, y


def _evaluate_encoded(
    y_true_enc: np.ndarray,
    y_pred_enc: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: List[str],
) -> Dict[str, Any]:
    y_true_labels = np.array([class_names[int(i)] for i in y_true_enc], dtype=str)
    y_pred_labels = np.array([class_names[int(i)] for i in y_pred_enc], dtype=str)

    report_dict = classification_report(
        y_true_labels, y_pred_labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)

    counts = (
        pd.Series(y_true_labels)
        .value_counts()
        .reindex(class_names)
        .fillna(0)
        .astype(int)
    )
    bins_df = pd.DataFrame({"bin_name": class_names, "count": counts.values})
    roc_list = _compute_roc(y_true_labels, y_prob, class_names)

    wavg = report_dict.get("weighted avg", {}) or {}
    return {
        "metrics": {
            "Accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
            "Precision_weighted": float(wavg.get("precision", 0.0)),
            "Recall_weighted": float(wavg.get("recall", 0.0)),
            "F1_weighted": float(f1_score(y_true_labels, y_pred_labels, average="weighted")),
            "F1_macro": float(f1_score(y_true_labels, y_pred_labels, average="macro")),
        },
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "roc": roc_list,
        "bins": bins_df,
        "y_true_labels": y_true_labels,
    }


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


def _derive_pass_mask(y_labels: pd.Series) -> np.ndarray:
    s = pd.Series(y_labels).astype(str)
    upper = s.str.upper().str.strip()

    pass_like = upper.str.contains("PASS", na=False)
    if pass_like.any():
        return pass_like.to_numpy()

    fail_like = upper.str.contains("FAIL", na=False)
    if fail_like.any():
        return (~fail_like).to_numpy()

    uniq = upper.dropna().unique().tolist()
    if len(uniq) == 2:
        pass_class = pd.Series(upper).value_counts().index[0]
        return (upper == pass_class).to_numpy()

    return np.zeros(len(s), dtype=bool)


def _select_distribution_groups(y_labels: pd.Series, max_groups: int = 2) -> List[str]:
    upper = pd.Series(y_labels).astype(str).str.upper().str.strip()
    if upper.empty:
        return []

    is_pass_fail = upper.str.contains("PASS|FAIL", regex=True, na=False)
    is_other = upper.str.contains("OTHER", na=False)
    preferred = upper[~is_pass_fail & ~is_other]

    counts = preferred.value_counts()
    groups = counts.index.tolist()[:max_groups]
    if len(groups) >= max_groups:
        return groups

    # Fallback: fill from all labels by frequency.
    all_counts = upper.value_counts()
    for g in all_counts.index.tolist():
        if g not in groups:
            groups.append(g)
        if len(groups) >= max_groups:
            break

    return groups


def _build_visual_plot_payload(
    X_eval: pd.DataFrame,
    y_true_labels: pd.Series,
    y_prob: Optional[np.ndarray],
    class_names: List[str],
    shap_importance: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    y_series = pd.Series(y_true_labels).astype(str).reset_index(drop=True)
    pass_mask = _derive_pass_mask(y_series)

    # Top 5 SHAP-ranked numeric features for PASS vs FAIL histograms
    feature_scores: Dict[str, float] = {}
    if isinstance(shap_importance, dict):
        for _, df_imp in shap_importance.items():
            if not isinstance(df_imp, pd.DataFrame) or df_imp.empty:
                continue
            use_cols = {"feature", "importance"}
            if not use_cols.issubset(df_imp.columns):
                continue
            for _, row in df_imp[["feature", "importance"]].head(50).iterrows():
                feat = str(row["feature"])
                try:
                    val = float(row["importance"])
                except Exception:
                    val = 0.0
                feature_scores[feat] = feature_scores.get(feat, 0.0) + max(val, 0.0)

    ranked = [k for k, _ in sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True)]

    top_feats: List[str] = []
    for f in ranked:
        if f in X_eval.columns:
            s = pd.to_numeric(X_eval[f], errors="coerce")
            if s.notna().sum() >= 10:
                top_feats.append(f)
        if len(top_feats) >= 5:
            break

    if len(top_feats) < 5:
        numeric_cols = X_eval.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            if c not in top_feats:
                top_feats.append(c)
            if len(top_feats) >= 5:
                break

    hist_items: List[Dict[str, Any]] = []
    for feat in top_feats[:5]:
        s = pd.to_numeric(X_eval[feat], errors="coerce")
        pass_vals = s[pass_mask].dropna()
        fail_vals = s[~pass_mask].dropna()

        # cap payload size
        if len(pass_vals) > 2000:
            pass_vals = pass_vals.sample(2000, random_state=67)
        if len(fail_vals) > 2000:
            fail_vals = fail_vals.sample(2000, random_state=67)

        hist_items.append(
            {
                "feature": str(feat),
                "pass_values": pass_vals.astype(float).tolist(),
                "fail_values": fail_vals.astype(float).tolist(),
            }
        )

    # Probability boxplot records by top class groups and PASS/FAIL
    prob_records: List[Dict[str, Any]] = []
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2 and len(y_prob) == len(y_series):
            labels = [str(c) for c in class_names]
            class_to_idx = {c: i for i, c in enumerate(labels)}
            y_upper = y_series.astype(str).str.upper().str.strip()
            group_candidates = _select_distribution_groups(y_upper, max_groups=2)

            pass_idx = None
            for i, cls in enumerate(labels):
                cu = cls.upper().strip()
                if cu == "PASS" or "PASS" in cu:
                    pass_idx = i
                    break

            pf_labels = np.where(pass_mask, "PASS", "FAIL")

            if pass_idx is not None and pass_idx < y_prob.shape[1]:
                p_pass = y_prob[:, pass_idx]
                p_used = np.where(pass_mask, p_pass, 1.0 - p_pass)
            else:
                idx_arr = y_series.map(lambda x: class_to_idx.get(str(x), -1)).to_numpy()
                p_used = np.array(
                    [
                        y_prob[i, j] if 0 <= j < y_prob.shape[1] else np.nan
                        for i, j in enumerate(idx_arr)
                    ],
                    dtype=float,
                )

            for i in range(len(y_series)):
                grp = y_upper.iat[i]
                pv = p_used[i]
                if grp not in group_candidates or not np.isfinite(pv):
                    continue
                prob_records.append(
                    {
                        "group": grp,
                        "pass_fail": str(pf_labels[i]),
                        "probability": float(np.clip(pv, 0.0, 1.0)),
                    }
                )

    return {
        "top_shap_hist": hist_items,
        "probability_box": prob_records,
    }


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
        "visual_plots": visual_plots,
    }


def _build_xgb_trial_params(trial, prefix: str = "") -> Dict[str, Any]:
    p = f"{prefix}_" if prefix else ""
    return {
        "n_estimators": trial.suggest_int(f"{p}n_estimators", 120, 700),
        "max_depth": trial.suggest_int(f"{p}max_depth", 3, 10),
        "learning_rate": trial.suggest_float(f"{p}learning_rate", 0.015, 0.25, log=True),
        "subsample": trial.suggest_float(f"{p}subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float(f"{p}colsample_bytree", 0.45, 1.0),
        "min_child_weight": trial.suggest_float(f"{p}min_child_weight", 0.5, 16.0),
        "gamma": trial.suggest_float(f"{p}gamma", 0.0, 6.0),
        "reg_alpha": trial.suggest_float(f"{p}reg_alpha", 1e-4, 12.0, log=True),
        "reg_lambda": trial.suggest_float(f"{p}reg_lambda", 1e-3, 30.0, log=True),
        "max_delta_step": trial.suggest_float(f"{p}max_delta_step", 0.0, 9.0),
    }


def _tune_binary_threshold(y_true_bin: np.ndarray, probs: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true_bin, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def _predict_with_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    adjusted = probs - thresholds[np.newaxis, :]
    eligible = probs >= thresholds[np.newaxis, :]
    adjusted[~eligible] = -np.inf
    from_eligible = np.argmax(adjusted, axis=1)
    fallback = np.argmax(probs, axis=1)
    return np.where(eligible.any(axis=1), from_eligible, fallback)


def _train_xgboost_mega_multiclass(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    import optuna
    from xgboost import XGBClassifier

    y_series = pd.Series(y).astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    class_names = [str(c) for c in le.classes_]
    if len(class_names) < 2:
        raise ValueError("Target must have at least 2 unique values.")

    X_num_df = _prep_for_xgboost(X)
    training_features = list(X_num_df.columns)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_train_val_test(
        X_num_df.to_numpy(dtype=np.float32),
        y_enc,
        random_state=int(cfg.get("random_state", 67)),
        val_size=float(cfg.get("val_size", 0.2)),
        test_size=float(cfg.get("test_size", 0.2)),
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    tree_method = "hist"

    def objective(trial) -> float:
        sampler = trial.suggest_categorical(
            "sampler", ["none", "random_over", "smote", "borderline_smote", "adasyn", "smoteenn"]
        )
        weight_power = trial.suggest_float("class_weight_power", 0.5, 2.0)
        params = _build_xgb_trial_params(trial)
        X_res, y_res = _safe_sample_data(sampler, X_train, y_train, seed=42)
        sample_weight, _ = _class_weight_from_power(y_res, power=weight_power)

        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            tree_method=tree_method,
            n_jobs=8,
            **params,
        )
        model.fit(
            X_res,
            y_res,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weight,
            verbose=False,
        )
        val_pred = model.predict(X_val)
        return float(f1_score(y_val, val_pred, average="macro"))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )
    study.optimize(objective, n_trials=int(cfg.get("mega_n_trials", 45)), show_progress_bar=False)
    best = study.best_trial

    best_sampler = str(best.params["sampler"])
    best_weight_power = float(best.params["class_weight_power"])
    best_params = {k: v for k, v in best.params.items() if k not in {"sampler", "class_weight_power"}}
    X_res, y_res = _safe_sample_data(best_sampler, X_train, y_train, seed=42)
    sample_weight, _ = _class_weight_from_power(y_res, power=best_weight_power)

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method=tree_method,
        n_jobs=8,
        **best_params,
    )
    model.fit(
        X_res,
        y_res,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weight,
        verbose=False,
    )

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    y_val_prob = model.predict_proba(X_val)
    y_test_prob = model.predict_proba(X_test)
    val_eval = _evaluate_encoded(y_val, y_val_pred, y_val_prob, class_names)
    test_eval = _evaluate_encoded(y_test, y_test_pred, y_test_prob, class_names)

    X_test_df = pd.DataFrame(X_test, columns=training_features)
    visual_plots = _build_visual_plot_payload(
        X_test_df.reset_index(drop=True),
        pd.Series(test_eval["y_true_labels"]).reset_index(drop=True),
        y_test_prob,
        class_names,
        {},
    )
    return {
        "model_name": "Mega Multiclass XGBoost",
        "trained_model": {
            "type": "mega_multiclass_xgb",
            "model": model,
            "imputer": imputer,
            "training_features": training_features,
            "label_classes": class_names,
        },
        "training_features": training_features,
        "label_classes": class_names,
        "class_names": class_names,
        "metrics": test_eval["metrics"],
        "confusion_matrix": test_eval["confusion_matrix"],
        "classification_report": test_eval["classification_report"],
        "classification_report_text": test_eval["classification_report_text"],
        "roc": test_eval["roc"],
        "bins": test_eval["bins"],
        "shap_importance": {},
        "visual_plots": visual_plots,
        "meta": {
            "mega_pipeline": "multiclass",
            "n_trials": int(len(study.trials)),
            "best_sampler": best_sampler,
            "best_weight_power": best_weight_power,
            "best_params": best_params,
            "val_f1_macro": float(val_eval["metrics"]["F1_macro"]),
        },
    }


def _train_xgboost_mega_ovr(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    import optuna
    from xgboost import XGBClassifier

    y_series = pd.Series(y).astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    class_names = [str(c) for c in le.classes_]
    if len(class_names) < 2:
        raise ValueError("Target must have at least 2 unique values.")

    X_num_df = _prep_for_xgboost(X)
    training_features = list(X_num_df.columns)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_train_val_test(
        X_num_df.to_numpy(dtype=np.float32),
        y_enc,
        random_state=int(cfg.get("random_state", 67)),
        val_size=float(cfg.get("val_size", 0.2)),
        test_size=float(cfg.get("test_size", 0.2)),
    )
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    tree_method = "hist"

    def train_ovr(
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
        params: Dict[str, Any],
        sampler_name: str,
        weight_power: float,
        scale_multiplier: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_classes = len(class_names)
        val_probs = np.zeros((len(X_va), n_classes), dtype=float)
        thresholds = np.zeros(n_classes, dtype=float)
        best_iterations = np.zeros(n_classes, dtype=int)
        for cls_idx in range(n_classes):
            y_tr_bin = (y_tr == cls_idx).astype(int)
            y_va_bin = (y_va == cls_idx).astype(int)
            X_res, y_res = _safe_sample_data(sampler_name, X_tr, y_tr_bin, seed=42 + cls_idx)
            sample_weight, _ = _class_weight_from_power(y_res, power=weight_power)

            pos = int(np.sum(y_res == 1))
            neg = int(np.sum(y_res == 0))
            scale_pos_weight = float((neg / max(pos, 1)) * scale_multiplier)

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42 + cls_idx,
                tree_method=tree_method,
                n_jobs=8,
                scale_pos_weight=scale_pos_weight,
                **params,
            )
            model.fit(
                X_res,
                y_res,
                eval_set=[(X_va, y_va_bin)],
                sample_weight=sample_weight,
                verbose=False,
            )
            cls_prob = model.predict_proba(X_va)[:, 1]
            val_probs[:, cls_idx] = cls_prob
            thresholds[cls_idx] = _tune_binary_threshold(y_va_bin, cls_prob)
            best_iterations[cls_idx] = int(getattr(model, "best_iteration", -1))
        return val_probs, thresholds, best_iterations

    def train_ovr_and_predict_test(
        params: Dict[str, Any],
        sampler_name: str,
        weight_power: float,
        scale_multiplier: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Any]]:
        val_probs, thresholds, best_iterations = train_ovr(
            X_train, y_train, X_val, y_val, params, sampler_name, weight_power, scale_multiplier
        )
        models = []
        test_probs = np.zeros((len(X_test), len(class_names)), dtype=float)
        for cls_idx in range(len(class_names)):
            y_tr_bin = (y_train == cls_idx).astype(int)
            X_res, y_res = _safe_sample_data(sampler_name, X_train, y_tr_bin, seed=42 + cls_idx)
            sample_weight, _ = _class_weight_from_power(y_res, power=weight_power)

            pos = int(np.sum(y_res == 1))
            neg = int(np.sum(y_res == 0))
            scale_pos_weight = float((neg / max(pos, 1)) * scale_multiplier)

            n_estimators = int(params["n_estimators"])
            if int(best_iterations[cls_idx]) >= 0:
                n_estimators = max(40, int(best_iterations[cls_idx]) + 1)

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42 + cls_idx,
                tree_method=tree_method,
                n_jobs=8,
                n_estimators=n_estimators,
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=float(params["subsample"]),
                colsample_bytree=float(params["colsample_bytree"]),
                min_child_weight=float(params["min_child_weight"]),
                gamma=float(params["gamma"]),
                reg_alpha=float(params["reg_alpha"]),
                reg_lambda=float(params["reg_lambda"]),
                max_delta_step=float(params["max_delta_step"]),
                scale_pos_weight=scale_pos_weight,
            )
            model.fit(X_res, y_res, sample_weight=sample_weight, verbose=False)
            models.append(model)
            test_probs[:, cls_idx] = model.predict_proba(X_test)[:, 1]
        return val_probs, test_probs, thresholds, best_iterations, models

    def objective(trial) -> float:
        sampler = trial.suggest_categorical(
            "sampler", ["none", "random_over", "smote", "borderline_smote", "adasyn", "smoteenn"]
        )
        weight_power = trial.suggest_float("class_weight_power", 0.5, 2.0)
        scale_multiplier = trial.suggest_float("scale_pos_weight_multiplier", 0.5, 3.0)
        params = _build_xgb_trial_params(trial)
        val_probs, thresholds, _ = train_ovr(
            X_train, y_train, X_val, y_val, params, sampler, weight_power, scale_multiplier
        )
        val_pred = _predict_with_thresholds(val_probs, thresholds)
        return float(f1_score(y_val, val_pred, average="macro"))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=6),
    )
    study.optimize(objective, n_trials=int(cfg.get("mega_n_trials", 30)), show_progress_bar=False)
    best = study.best_trial
    best_sampler = str(best.params["sampler"])
    best_weight_power = float(best.params["class_weight_power"])
    best_scale_multiplier = float(best.params["scale_pos_weight_multiplier"])
    best_params = {
        k: v
        for k, v in best.params.items()
        if k not in {"sampler", "class_weight_power", "scale_pos_weight_multiplier"}
    }

    val_probs, test_probs, thresholds, best_iterations, models = train_ovr_and_predict_test(
        best_params, best_sampler, best_weight_power, best_scale_multiplier
    )
    y_val_pred = _predict_with_thresholds(val_probs, thresholds)
    y_test_pred = _predict_with_thresholds(test_probs, thresholds)
    val_eval = _evaluate_encoded(y_val, y_val_pred, val_probs, class_names)
    test_eval = _evaluate_encoded(y_test, y_test_pred, test_probs, class_names)

    X_test_df = pd.DataFrame(X_test, columns=training_features)
    visual_plots = _build_visual_plot_payload(
        X_test_df.reset_index(drop=True),
        pd.Series(test_eval["y_true_labels"]).reset_index(drop=True),
        test_probs,
        class_names,
        {},
    )
    return {
        "model_name": "Mega OVR XGBoost",
        "trained_model": {
            "type": "mega_ovr_xgb",
            "models": models,
            "thresholds": thresholds.astype(float),
            "imputer": imputer,
            "training_features": training_features,
            "label_classes": class_names,
        },
        "training_features": training_features,
        "label_classes": class_names,
        "class_names": class_names,
        "metrics": test_eval["metrics"],
        "confusion_matrix": test_eval["confusion_matrix"],
        "classification_report": test_eval["classification_report"],
        "classification_report_text": test_eval["classification_report_text"],
        "roc": test_eval["roc"],
        "bins": test_eval["bins"],
        "shap_importance": {},
        "visual_plots": visual_plots,
        "meta": {
            "mega_pipeline": "ovr",
            "n_trials": int(len(study.trials)),
            "best_sampler": best_sampler,
            "best_weight_power": best_weight_power,
            "best_scale_multiplier": best_scale_multiplier,
            "best_params": best_params,
            "best_iterations_per_class": [int(x) for x in best_iterations.tolist()],
            "class_thresholds": [float(x) for x in thresholds.tolist()],
            "val_f1_macro": float(val_eval["metrics"]["F1_macro"]),
        },
    }


def _train_xgboost_mega_hierarchical(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    import optuna
    from xgboost import XGBClassifier

    y_series = pd.Series(y).astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    class_names = [str(c) for c in le.classes_]
    if len(class_names) < 2:
        raise ValueError("Target must have at least 2 unique values.")

    X_num_df = _prep_for_xgboost(X)
    training_features = list(X_num_df.columns)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_train_val_test(
        X_num_df.to_numpy(dtype=np.float32),
        y_enc,
        random_state=int(cfg.get("random_state", 67)),
        val_size=float(cfg.get("val_size", 0.2)),
        test_size=float(cfg.get("test_size", 0.2)),
    )
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    tree_method = "hist"

    values, counts = np.unique(y_train, return_counts=True)
    pass_idx = int(values[np.argmax(counts)])
    for i, cls in enumerate(class_names):
        if str(cls).strip().upper() == "PASS":
            pass_idx = int(i)
            break
    fail_class_indices = [i for i in range(len(class_names)) if i != pass_idx]
    fail_remap = {cls: i for i, cls in enumerate(fail_class_indices)}

    y_train_s1 = (y_train != pass_idx).astype(int)
    y_val_s1 = (y_val != pass_idx).astype(int)
    fail_train_mask = y_train != pass_idx
    fail_val_mask = y_val != pass_idx
    X_train_s2_base = X_train[fail_train_mask]
    y_train_s2 = np.array([fail_remap[int(v)] for v in y_train[fail_train_mask]], dtype=int)
    X_val_s2 = X_val[fail_val_mask]
    y_val_s2 = np.array([fail_remap[int(v)] for v in y_val[fail_val_mask]], dtype=int)

    def combine_probs(prob_fail: np.ndarray, fail_probs: np.ndarray) -> np.ndarray:
        full = np.zeros((len(prob_fail), len(class_names)), dtype=float)
        full[:, pass_idx] = 1.0 - prob_fail
        for i, cls_idx in enumerate(fail_class_indices):
            full[:, cls_idx] = prob_fail * fail_probs[:, i]
        return full

    def predict_hier(prob_fail: np.ndarray, fail_probs: np.ndarray, threshold: float) -> np.ndarray:
        fail_choice = np.argmax(fail_probs, axis=1)
        fail_pred = np.array([fail_class_indices[i] for i in fail_choice], dtype=int)
        return np.where(prob_fail >= threshold, fail_pred, pass_idx)

    def best_threshold(y_true: np.ndarray, prob_fail: np.ndarray, fail_probs: np.ndarray) -> float:
        best_t, best_score = 0.5, -1.0
        for t in np.linspace(0.15, 0.85, 29):
            pred = predict_hier(prob_fail, fail_probs, float(t))
            score = f1_score(y_true, pred, average="macro")
            if score > best_score:
                best_score = float(score)
                best_t = float(t)
        return best_t

    def objective(trial) -> float:
        s1_sampler = trial.suggest_categorical(
            "stage1_sampler", ["none", "random_over", "smote", "borderline_smote", "adasyn", "smoteenn"]
        )
        s2_sampler = trial.suggest_categorical(
            "stage2_sampler", ["none", "random_over", "smote", "borderline_smote", "adasyn", "smoteenn"]
        )
        s1_weight_power = trial.suggest_float("stage1_weight_power", 0.5, 2.0)
        s2_weight_power = trial.suggest_float("stage2_weight_power", 0.5, 2.0)
        s1_scale_mult = trial.suggest_float("stage1_scale_multiplier", 0.5, 3.0)
        s1_params = _build_xgb_trial_params(trial, prefix="s1")
        s2_params = _build_xgb_trial_params(trial, prefix="s2")

        X_s1, y_s1 = _safe_sample_data(s1_sampler, X_train, y_train_s1, seed=42)
        s1_weight, _ = _class_weight_from_power(y_s1, power=s1_weight_power)
        pos = int(np.sum(y_s1 == 1))
        neg = int(np.sum(y_s1 == 0))
        s1_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            tree_method=tree_method,
            n_jobs=8,
            scale_pos_weight=float((neg / max(pos, 1)) * s1_scale_mult),
            **s1_params,
        )
        s1_model.fit(X_s1, y_s1, eval_set=[(X_val, y_val_s1)], sample_weight=s1_weight, verbose=False)
        prob_fail_val = s1_model.predict_proba(X_val)[:, 1]

        if len(fail_class_indices) <= 1 or len(y_train_s2) == 0:
            fail_probs_val = np.ones((len(X_val), max(len(fail_class_indices), 1)), dtype=float)
        else:
            X_s2, y_s2 = _safe_sample_data(s2_sampler, X_train_s2_base, y_train_s2, seed=42)
            s2_weight, _ = _class_weight_from_power(y_s2, power=s2_weight_power)
            s2_model = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                tree_method=tree_method,
                n_jobs=8,
                num_class=len(fail_class_indices),
                **s2_params,
            )
            eval_set = [(X_val_s2, y_val_s2)] if len(y_val_s2) > 0 else None
            fit_kwargs = {"sample_weight": s2_weight, "verbose": False}
            if eval_set is not None:
                fit_kwargs["eval_set"] = eval_set
            s2_model.fit(X_s2, y_s2, **fit_kwargs)
            fail_probs_val = s2_model.predict_proba(X_val)

        t = best_threshold(y_val, prob_fail_val, fail_probs_val)
        pred = predict_hier(prob_fail_val, fail_probs_val, t)
        return float(f1_score(y_val, pred, average="macro"))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=6),
    )
    study.optimize(objective, n_trials=int(cfg.get("mega_n_trials", 35)), show_progress_bar=False)
    best = study.best_trial

    s1_sampler = str(best.params["stage1_sampler"])
    s2_sampler = str(best.params["stage2_sampler"])
    s1_weight_power = float(best.params["stage1_weight_power"])
    s2_weight_power = float(best.params["stage2_weight_power"])
    s1_scale_mult = float(best.params["stage1_scale_multiplier"])
    s1_params = {k.replace("s1_", ""): v for k, v in best.params.items() if k.startswith("s1_")}
    s2_params = {k.replace("s2_", ""): v for k, v in best.params.items() if k.startswith("s2_")}

    X_s1, y_s1 = _safe_sample_data(s1_sampler, X_train, y_train_s1, seed=42)
    s1_weight, _ = _class_weight_from_power(y_s1, power=s1_weight_power)
    pos = int(np.sum(y_s1 == 1))
    neg = int(np.sum(y_s1 == 0))
    stage1 = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method=tree_method,
        n_jobs=8,
        scale_pos_weight=float((neg / max(pos, 1)) * s1_scale_mult),
        **s1_params,
    )
    stage1.fit(X_s1, y_s1, eval_set=[(X_val, y_val_s1)], sample_weight=s1_weight, verbose=False)

    stage2 = None
    if len(fail_class_indices) > 1 and len(y_train_s2) > 0:
        X_s2, y_s2 = _safe_sample_data(s2_sampler, X_train_s2_base, y_train_s2, seed=42)
        s2_weight, _ = _class_weight_from_power(y_s2, power=s2_weight_power)
        stage2 = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            tree_method=tree_method,
            n_jobs=8,
            num_class=len(fail_class_indices),
            **s2_params,
        )
        eval_set = [(X_val_s2, y_val_s2)] if len(y_val_s2) > 0 else None
        fit_kwargs = {"sample_weight": s2_weight, "verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        stage2.fit(X_s2, y_s2, **fit_kwargs)

    prob_fail_val = stage1.predict_proba(X_val)[:, 1]
    prob_fail_test = stage1.predict_proba(X_test)[:, 1]
    if stage2 is None:
        fail_probs_val = np.ones((len(X_val), max(len(fail_class_indices), 1)), dtype=float)
        fail_probs_test = np.ones((len(X_test), max(len(fail_class_indices), 1)), dtype=float)
    else:
        fail_probs_val = stage2.predict_proba(X_val)
        fail_probs_test = stage2.predict_proba(X_test)

    threshold = best_threshold(y_val, prob_fail_val, fail_probs_val)
    y_val_pred = predict_hier(prob_fail_val, fail_probs_val, threshold)
    y_test_pred = predict_hier(prob_fail_test, fail_probs_test, threshold)
    y_val_prob = combine_probs(prob_fail_val, fail_probs_val)
    y_test_prob = combine_probs(prob_fail_test, fail_probs_test)
    val_eval = _evaluate_encoded(y_val, y_val_pred, y_val_prob, class_names)
    test_eval = _evaluate_encoded(y_test, y_test_pred, y_test_prob, class_names)

    X_test_df = pd.DataFrame(X_test, columns=training_features)
    visual_plots = _build_visual_plot_payload(
        X_test_df.reset_index(drop=True),
        pd.Series(test_eval["y_true_labels"]).reset_index(drop=True),
        y_test_prob,
        class_names,
        {},
    )
    return {
        "model_name": "Mega Hierarchical XGBoost",
        "trained_model": {
            "type": "mega_hierarchical_xgb",
            "stage1": stage1,
            "stage2": stage2,
            "pass_idx": int(pass_idx),
            "fail_class_indices": [int(x) for x in fail_class_indices],
            "threshold": float(threshold),
            "imputer": imputer,
            "training_features": training_features,
            "label_classes": class_names,
        },
        "training_features": training_features,
        "label_classes": class_names,
        "class_names": class_names,
        "metrics": test_eval["metrics"],
        "confusion_matrix": test_eval["confusion_matrix"],
        "classification_report": test_eval["classification_report"],
        "classification_report_text": test_eval["classification_report_text"],
        "roc": test_eval["roc"],
        "bins": test_eval["bins"],
        "shap_importance": {},
        "visual_plots": visual_plots,
        "meta": {
            "mega_pipeline": "hierarchical",
            "n_trials": int(len(study.trials)),
            "best_stage1_sampler": s1_sampler,
            "best_stage2_sampler": s2_sampler,
            "best_stage1_weight_power": s1_weight_power,
            "best_stage2_weight_power": s2_weight_power,
            "best_stage1_scale_multiplier": s1_scale_mult,
            "best_stage1_params": s1_params,
            "best_stage2_params": s2_params,
            "best_fail_threshold": float(threshold),
            "anchor_class": class_names[int(pass_idx)],
            "val_f1_macro": float(val_eval["metrics"]["F1_macro"]),
        },
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
            },
        }

    elif model_type.lower() in {
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
        X_eval = imputer.transform(X_num.to_numpy(dtype=np.float32)) if imputer is not None else X_num.to_numpy(dtype=np.float32)

        y_prob = None
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
        counts = (
            pd.Series(y_true).value_counts().reindex(label_classes).fillna(0).astype(int)
        )
        bins_df = pd.DataFrame({"bin_name": label_classes, "count": counts.values})
        roc_list = _compute_roc(y_true, y_prob, label_classes)
        visual_plots = _build_visual_plot_payload(
            pd.DataFrame(X_eval, columns=feat_order).reset_index(drop=True),
            pd.Series(y_true).reset_index(drop=True),
            y_prob,
            label_classes,
            {},
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
            "shap_importance": {},
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
            },
        }

    else:
        raise ValueError(f"Unknown model_type in artifact: {model_type}")
