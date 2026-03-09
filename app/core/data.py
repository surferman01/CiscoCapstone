from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd


def _load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _split_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target_col]
    X = df.drop(columns=[target_col], errors="ignore")
    return X, y


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
