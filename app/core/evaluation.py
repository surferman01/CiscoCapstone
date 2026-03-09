from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


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


def _class_weight_from_power(
    y: np.ndarray, power: float = 1.0
) -> Tuple[np.ndarray, Dict[int, float]]:
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


def _safe_sample_data(
    name: str, X: np.ndarray, y: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    if name == "none":
        return X, y
    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import (
            ADASYN,
            BorderlineSMOTE,
            RandomOverSampler,
            SMOTE,
        )

        sampler = None
        if name == "random_over":
            sampler = RandomOverSampler(random_state=seed)
        elif name == "smote":
            sampler = SMOTE(random_state=seed, k_neighbors=3)
        elif name == "borderline_smote":
            sampler = BorderlineSMOTE(
                random_state=seed, k_neighbors=3, kind="borderline-1"
            )
        elif name == "adasyn":
            sampler = ADASYN(random_state=seed, n_neighbors=3)
        elif name == "smoteenn":
            sampler = SMOTEENN(
                random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=3)
            )
        else:
            return X, y
        return sampler.fit_resample(X, y)
    except Exception:
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
            "F1_weighted": float(
                f1_score(y_true_labels, y_pred_labels, average="weighted")
            ),
            "F1_macro": float(
                f1_score(y_true_labels, y_pred_labels, average="macro")
            ),
        },
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "roc": roc_list,
        "bins": bins_df,
        "y_true_labels": y_true_labels,
    }


def _compute_roc(
    y_true_labels: np.ndarray, y_prob: Optional[np.ndarray], class_names: List[str]
) -> List[Dict[str, Any]]:
    if y_prob is None:
        return []
    y_true = np.asarray(y_true_labels).astype(str)
    out = []
    for i, cls in enumerate(class_names):
        y_bin = (y_true == str(cls)).astype(int)
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


def _build_per_class_importance_tables(
    X_num: pd.DataFrame,
    y_true_labels: pd.Series,
    contrib: np.ndarray,
    class_names: List[str],
) -> Dict[str, pd.DataFrame]:
    X_vals = X_num.to_numpy(dtype=np.float64, copy=False)
    feats = list(X_num.columns)
    y_series = pd.Series(y_true_labels).astype(str).reset_index(drop=True)

    baseline = "PASS" if "PASS" in [str(c) for c in class_names] else None

    contrib = np.asarray(contrib)
    if contrib.ndim == 2:
        shap_by_class = {str(cls): contrib[:, :-1] for cls in class_names}
    elif contrib.ndim == 3:
        shap_by_class = {}
        for i, cls in enumerate(class_names):
            shap_by_class[str(cls)] = contrib[:, i, :-1]
    else:
        return {}

    out: Dict[str, pd.DataFrame] = {}

    for cls in class_names:
        cls_str = str(cls)
        shap_vals = shap_by_class.get(cls_str)
        if shap_vals is None:
            continue

        mask_cls = (y_series == cls_str).to_numpy()
        if mask_cls.sum() == 0:
            continue

        imp = np.nanmean(np.abs(shap_vals[mask_cls, :]), axis=0)
        if not np.isfinite(imp).any():
            continue

        direction = np.nanmean(shap_vals[mask_cls, :], axis=0)
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


def _select_distribution_groups(
    y_labels: pd.Series, max_groups: int = 2
) -> List[str]:
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

    ranked = [
        k for k, _ in sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True)
    ]

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
