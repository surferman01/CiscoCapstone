
import io
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

def _train_catboost(X, y, cfg):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_state", 67), stratify=y
    )
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    params = dict(
        iterations=cfg.get("iterations", 500),
        learning_rate=cfg.get("learning_rate", 0.05),
        depth=cfg.get("depth", 8),
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=cfg.get("random_state", 67),
        verbose=cfg.get("verbose", 100),
    )

    # Try GPU; fall back to CPU if not available
    use_gpu = cfg.get("use_gpu", True)
    if use_gpu:
        params.update({"task_type": "GPU"})
        # 'devices' optional; many envs don't require it
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

    # Predictions & metrics
    y_pred = model.predict(X_test).flatten()
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Feature importance
    fi_vals = model.get_feature_importance(train_pool, type='FeatureImportance')
    fi = pd.DataFrame({'feature': list(X.columns), 'importance': fi_vals}).sort_values('importance', ascending=False)

    # ROC curves (One-vs-Rest)
    roc = []
    classes = np.unique(y_test)
    for i, cls in enumerate(classes):
        # y_score is column i of y_prob if CatBoost uses class order as classes
        try:
            y_true = (y_test == cls).astype(int)
            y_score = y_prob[:, i]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc.append({'class': str(cls), 'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)})
        except Exception:
            pass

    return {
        'model': model,
        'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob,
        'metrics': {'Accuracy': float(acc), 'F1_weighted': float(f1w)},
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_importance': fi,
        'roc': roc,
    }

def run_analysis(data_path: str, config: dict) -> dict:
    # Load
    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)

    # Expect columns: id, test_bin, (optional) test_result, features...
    df = df.copy()
    if 'test_result' in df.columns:
        df = df.drop(columns=['test_result'])

    y = df['test_bin']
    X = df.drop(columns=[c for c in ['id','test_bin'] if c in df.columns])

    # numeric handling per user's snippet
    X = X.fillna(X.median(numeric_only=True))

    # Train
    res = _train_catboost(X, y, config or {})

    # Bins = class distribution on test set
    bcounts = pd.Series(res['y_test']).value_counts().sort_index()
    bins = pd.DataFrame({'bin_name': bcounts.index.astype(str), 'count': bcounts.values})
    bins['rate'] = bins['count'] / bins['count'].sum()

    artifacts = {
        'log': res['classification_report'],
        'models': {}  # you could pickle and attach here if desired
    }

    return {
        'meta': {'rows': int(len(df)), 'cols': int(len(df.columns)), 'model': 'CatBoostClassifier'},
        'metrics': res['metrics'],
        'feature_importance': res['feature_importance'],
        'bins': bins,
        'roc': res['roc'],
        'confusion_matrix': res['confusion_matrix'],
        'artifacts': artifacts,
        'dataframe': df.head(200)
    }
