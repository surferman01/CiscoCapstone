from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_ID_NAME_RE = re.compile(
    r"(serial|serno|s/n|sn\b|lot|wafer|unit|device|die|barcode|uuid|guid|hash|mac|imei|imsi|ip\b|hostname|name\b|id\b|identifier|index)",
    re.IGNORECASE,
)
_DATE_NAME_RE = re.compile(r"(date|time|timestamp|datetime)", re.IGNORECASE)


def _weak_non_measurement_candidates(df: pd.DataFrame) -> List[str]:
    n = len(df)
    if n == 0:
        return []

    drops: set[str] = set()
    for c in df.columns:
        name = str(c)

        if _ID_NAME_RE.search(name) or _DATE_NAME_RE.search(name):
            drops.add(c)
            continue

        s = df[c]

        if pd.api.types.is_datetime64_any_dtype(s):
            drops.add(c)
            continue

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


def suggest_drop_columns_weak(df: pd.DataFrame) -> list[str]:
    return _weak_non_measurement_candidates(df)


def _apply_exclusions_for_training(
    df: pd.DataFrame,
    target_col: str,
    recommended_targets: Optional[List[str]] = None,
    explicit_excludes: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    recommended_targets = recommended_targets or []
    explicit_excludes = explicit_excludes or []

    weak_meta = _weak_non_measurement_candidates(df)
    rec_excludes = [c for c in recommended_targets if c in df.columns and c != target_col]

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


def _prep_for_xgboost(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X.columns if c not in num_cols]

    if len(num_cols):
        X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))

    for c in cat_cols:
        X[c] = X[c].astype("object").where(~X[c].isna(), "NA").astype(str)

    X_oh = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X_oh = X_oh.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_oh.astype(np.float32)


def _align_features_to_training(
    X_new: pd.DataFrame, training_features: List[str]
) -> pd.DataFrame:
    X_new = X_new.copy()
    for c in training_features:
        if c not in X_new.columns:
            X_new[c] = 0.0
    return X_new[training_features]
