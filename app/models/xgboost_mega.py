from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from core.evaluation import (
    _build_xgb_contrib_importance_tables,
    _build_visual_plot_payload,
    _build_xgb_trial_params,
    _class_weight_from_power,
    _evaluate_encoded,
    _predict_with_thresholds,
    _safe_sample_data,
    _split_train_val_test,
    _tune_binary_threshold,
)
from core.preprocessing import _prep_for_xgboost


def _train_xgboost_mega_multiclass(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
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

    best_sampler = None
    best_weight_power = None
    best_params = None
    used_preset = False
    study_trials = 0

    if isinstance(cfg.get("best_params"), dict) and cfg.get("best_sampler") is not None:
        best_sampler = str(cfg.get("best_sampler"))
        best_weight_power = float(cfg.get("best_weight_power", 1.0))
        best_params = dict(cfg.get("best_params") or {})
        used_preset = True
    else:
        import optuna

        def objective(trial) -> float:
            sampler = trial.suggest_categorical(
                "sampler",
                ["none", "random_over", "smote", "borderline_smote", "adasyn", "smoteenn"],
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
        study.optimize(
            objective, n_trials=int(cfg.get("mega_n_trials", 45)), show_progress_bar=False
        )
        best = study.best_trial
        best_sampler = str(best.params["sampler"])
        best_weight_power = float(best.params["class_weight_power"])
        best_params = {
            k: v for k, v in best.params.items() if k not in {"sampler", "class_weight_power"}
        }
        study_trials = int(len(study.trials))

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
    shap_importance = _build_xgb_contrib_importance_tables(
        model,
        X_test_df,
        pd.Series(test_eval["y_true_labels"]),
        class_names,
    )
    visual_plots = _build_visual_plot_payload(
        X_test_df.reset_index(drop=True),
        pd.Series(test_eval["y_true_labels"]).reset_index(drop=True),
        y_test_prob,
        class_names,
        shap_importance,
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
        "shap_importance": shap_importance,
        "visual_plots": visual_plots,
        "meta": {
            "mega_pipeline": "multiclass",
            "n_trials": int(study_trials),
            "tuning_source": "preset" if used_preset else "optuna",
            "best_sampler": best_sampler,
            "best_weight_power": best_weight_power,
            "best_params": best_params,
            "val_f1_macro": float(val_eval["metrics"]["F1_macro"]),
            "feature_importance_source": (
                "xgboost_pred_contribs" if shap_importance else "unavailable"
            ),
        },
    }


def _train_xgboost_mega_ovr(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
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
            X_res, y_res = _safe_sample_data(
                sampler_name, X_tr, y_tr_bin, seed=42 + cls_idx
            )
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
            X_res, y_res = _safe_sample_data(
                sampler_name, X_train, y_tr_bin, seed=42 + cls_idx
            )
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

    best_sampler = None
    best_weight_power = None
    best_scale_multiplier = None
    best_params = None
    used_preset = False
    study_trials = 0

    if (
        isinstance(cfg.get("best_params"), dict)
        and cfg.get("best_sampler") is not None
        and cfg.get("best_scale_multiplier") is not None
    ):
        best_sampler = str(cfg.get("best_sampler"))
        best_weight_power = float(cfg.get("best_weight_power", 1.0))
        best_scale_multiplier = float(cfg.get("best_scale_multiplier"))
        best_params = dict(cfg.get("best_params") or {})
        used_preset = True
    else:
        import optuna

        def objective(trial) -> float:
            sampler = trial.suggest_categorical(
                "sampler",
                ["none", "random_over", "smote", "borderline_smote", "adasyn", "smoteenn"],
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
        study.optimize(
            objective, n_trials=int(cfg.get("mega_n_trials", 30)), show_progress_bar=False
        )
        best = study.best_trial
        best_sampler = str(best.params["sampler"])
        best_weight_power = float(best.params["class_weight_power"])
        best_scale_multiplier = float(best.params["scale_pos_weight_multiplier"])
        best_params = {
            k: v
            for k, v in best.params.items()
            if k not in {"sampler", "class_weight_power", "scale_pos_weight_multiplier"}
        }
        study_trials = int(len(study.trials))

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
            "n_trials": int(study_trials),
            "tuning_source": "preset" if used_preset else "optuna",
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
    y_train_s2 = np.array(
        [fail_remap[int(v)] for v in y_train[fail_train_mask]], dtype=int
    )
    X_val_s2 = X_val[fail_val_mask]
    y_val_s2 = np.array([fail_remap[int(v)] for v in y_val[fail_val_mask]], dtype=int)

    def combine_probs(prob_fail: np.ndarray, fail_probs: np.ndarray) -> np.ndarray:
        full = np.zeros((len(prob_fail), len(class_names)), dtype=float)
        full[:, pass_idx] = 1.0 - prob_fail
        for i, cls_idx in enumerate(fail_class_indices):
            full[:, cls_idx] = prob_fail * fail_probs[:, i]
        return full

    def predict_hier(
        prob_fail: np.ndarray, fail_probs: np.ndarray, threshold: float
    ) -> np.ndarray:
        fail_choice = np.argmax(fail_probs, axis=1)
        fail_pred = np.array([fail_class_indices[i] for i in fail_choice], dtype=int)
        return np.where(prob_fail >= threshold, fail_pred, pass_idx)

    def best_threshold(
        y_true: np.ndarray, prob_fail: np.ndarray, fail_probs: np.ndarray
    ) -> float:
        best_t, best_score = 0.5, -1.0
        for t in np.linspace(0.15, 0.85, 29):
            pred = predict_hier(prob_fail, fail_probs, float(t))
            score = f1_score(y_true, pred, average="macro")
            if score > best_score:
                best_score = float(score)
                best_t = float(t)
        return best_t

    used_preset = False
    study_trials = 0

    if (
        isinstance(cfg.get("best_stage1_params"), dict)
        and isinstance(cfg.get("best_stage2_params"), dict)
        and cfg.get("best_stage1_sampler") is not None
        and cfg.get("best_stage2_sampler") is not None
    ):
        s1_sampler = str(cfg.get("best_stage1_sampler"))
        s2_sampler = str(cfg.get("best_stage2_sampler"))
        s1_weight_power = float(cfg.get("best_stage1_weight_power", 1.0))
        s2_weight_power = float(cfg.get("best_stage2_weight_power", 1.0))
        s1_scale_mult = float(cfg.get("best_stage1_scale_multiplier", 1.0))
        s1_params = dict(cfg.get("best_stage1_params") or {})
        s2_params = dict(cfg.get("best_stage2_params") or {})
        used_preset = True
    else:
        import optuna

        def objective(trial) -> float:
            s1_sampler = trial.suggest_categorical(
                "stage1_sampler",
                ["none", "random_over", "smote", "borderline_smote", "adasyn", "smoteenn"],
            )
            s2_sampler = trial.suggest_categorical(
                "stage2_sampler",
                ["none", "random_over", "smote", "borderline_smote", "adasyn", "smoteenn"],
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
            s1_model.fit(
                X_s1,
                y_s1,
                eval_set=[(X_val, y_val_s1)],
                sample_weight=s1_weight,
                verbose=False,
            )
            prob_fail_val = s1_model.predict_proba(X_val)[:, 1]

            if len(fail_class_indices) <= 1 or len(y_train_s2) == 0:
                fail_probs_val = np.ones((len(X_val), max(len(fail_class_indices), 1)), dtype=float)
            else:
                X_s2, y_s2 = _safe_sample_data(
                    s2_sampler, X_train_s2_base, y_train_s2, seed=42
                )
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
        study.optimize(
            objective, n_trials=int(cfg.get("mega_n_trials", 35)), show_progress_bar=False
        )
        best = study.best_trial

        s1_sampler = str(best.params["stage1_sampler"])
        s2_sampler = str(best.params["stage2_sampler"])
        s1_weight_power = float(best.params["stage1_weight_power"])
        s2_weight_power = float(best.params["stage2_weight_power"])
        s1_scale_mult = float(best.params["stage1_scale_multiplier"])
        s1_params = {
            k.replace("s1_", ""): v for k, v in best.params.items() if k.startswith("s1_")
        }
        s2_params = {
            k.replace("s2_", ""): v for k, v in best.params.items() if k.startswith("s2_")
        }
        study_trials = int(len(study.trials))

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
    stage1.fit(
        X_s1,
        y_s1,
        eval_set=[(X_val, y_val_s1)],
        sample_weight=s1_weight,
        verbose=False,
    )

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
        fail_probs_test = np.ones(
            (len(X_test), max(len(fail_class_indices), 1)), dtype=float
        )
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
            "n_trials": int(study_trials),
            "tuning_source": "preset" if used_preset else "optuna",
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
