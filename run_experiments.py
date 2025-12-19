import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data.csv'
TEST_PATH = BASE_DIR / 'test.csv'
EXPERIMENT_LOG_CSV = BASE_DIR / 'experiment_log.csv'
EXPERIMENT_LOG_JSON = BASE_DIR / 'experiment_log.json'


COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

CV_FOLDS = 3
SCORING = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}


def load_dataset(path: Path, is_test: bool = False) -> pd.DataFrame:
    if is_test:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            first = f.readline().strip()
        skip = 1 if 'Cross validator' in first else 0
    else:
        skip = 0

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=skip,
    )
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    df['income'] = df['income'].str.replace('.', '', regex=False)
    df = df.replace('?', np.nan)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['capital-gain'] = pd.to_numeric(df['capital-gain'], errors='coerce')
    df['capital-loss'] = pd.to_numeric(df['capital-loss'], errors='coerce')
    df['cap_gain_log'] = np.log1p(df['capital-gain'])
    df['cap_loss_log'] = np.log1p(df['capital-loss'])
    df['cap_gain_gt0'] = (df['capital-gain'] > 0).astype(int)
    df['cap_loss_gt0'] = (df['capital-loss'] > 0).astype(int)
    return df


def data_version(paths: list[Path]) -> str:
    parts = []
    for path in paths:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        parts.append(f"{path.name}:{h.hexdigest()[:12]}")
    return '; '.join(parts)


def confusion_interpretation(tn: int, fp: int, fn: int, tp: int, recall: float, precision: float) -> str:
    if fn > fp:
        base = 'More false negatives than false positives; the model misses >50K cases.'
    elif fp > fn:
        base = 'More false positives than false negatives; the model flags <=50K as >50K.'
    else:
        base = 'False positives and false negatives are balanced.'

    extra = []
    if recall < 0.7:
        extra.append('Recall is moderate; consider class weighting or threshold tuning.')
    if precision < 0.7:
        extra.append('Precision is moderate; consider feature pruning or threshold tuning.')
    return ' '.join([base] + extra)


def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop',
    )


def search_and_evaluate(name: str, model, param_grid, preprocess, x_train, y_train, x_test, y_test) -> dict:
    pipeline = Pipeline(
        steps=[
            ('preprocess', preprocess),
            ('model', model),
        ]
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit='f1',
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(x_test)
    if hasattr(best, 'predict_proba'):
        y_score = best.predict_proba(x_test)[:, 1]
    else:
        y_score = best.decision_function(x_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_score)),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    cm = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    interpretation = confusion_interpretation(
        tn, fp, fn, tp, metrics['recall'], metrics['precision']
    )
    best_idx = grid.best_index_
    cv_metrics = {
        metric: float(grid.cv_results_[f"mean_test_{metric}"][best_idx])
        for metric in SCORING
    }
    return {
        'name': name,
        'best_params': grid.best_params_,
        'cv_metrics': cv_metrics,
        'metrics': metrics,
        'confusion_matrix': cm,
        'interpretation': interpretation,
        'pipeline': best,
    }


def main() -> None:
    train = load_dataset(DATA_PATH)
    test = load_dataset(TEST_PATH, is_test=True)

    train = add_features(train)
    test = add_features(test)

    y_train = (train['income'] == '>50K').astype(int)
    y_test = (test['income'] == '>50K').astype(int)
    x_train = train.drop(columns=['income'])
    x_test = test.drop(columns=['income'])

    categorical_cols = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
    ]
    numeric_cols = [
        'age',
        'fnlwgt',
        'education-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'cap_gain_log',
        'cap_loss_log',
        'cap_gain_gt0',
        'cap_loss_gt0',
    ]

    preprocess = build_preprocessor(categorical_cols, numeric_cols)

    models = [
        {
            'experiment_id': 'E01',
            'name': 'Logistic Regression (baseline)',
            'justification': 'Simple, interpretable baseline for binary classification.',
            'estimator': LogisticRegression(
                solver='saga',
                max_iter=1000,
                class_weight='balanced',
                n_jobs=1,
                random_state=42,
            ),
            'param_grid': {
                'model__C': [0.3, 1.0, 3.0],
            },
        },
        {
            'experiment_id': 'E02',
            'name': 'Random Forest (tree-based)',
            'justification': 'Captures nonlinear interactions and handles mixed feature types.',
            'estimator': RandomForestClassifier(
                class_weight='balanced',
                n_jobs=1,
                random_state=42,
            ),
            'param_grid': {
                'model__n_estimators': [200, 300],
                'model__max_depth': [12, 16],
                'model__min_samples_leaf': [1, 2],
            },
        },
        {
            'experiment_id': 'E03',
            'name': 'HistGradientBoosting (advanced)',
            'justification': 'Boosted trees often outperform single trees and capture complex patterns.',
            'estimator': HistGradientBoostingClassifier(
                random_state=42,
            ),
            'param_grid': {
                'model__max_iter': [200, 300],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 6],
            },
        },
    ]

    experiments = []
    version = data_version([DATA_PATH, TEST_PATH])
    timestamp = datetime.now().isoformat(timespec='seconds')

    for model_info in models:
        result = search_and_evaluate(
            model_info['name'],
            model_info['estimator'],
            model_info['param_grid'],
            preprocess,
            x_train,
            y_train,
            x_test,
            y_test,
        )
        metrics = result['metrics']
        cv_metrics = result['cv_metrics']
        best_params = result['best_params']
        observations = (
            'Baseline performance established.'
            if 'baseline' in model_info['name'].lower()
            else 'Model trained and evaluated on test set.'
        )
        if metrics['recall'] < 0.7:
            observations = 'Recall is moderate; the model misses some >50K cases.'
        next_steps = 'Tune hyperparameters and consider threshold adjustment.'
        if metrics['precision'] < 0.7:
            next_steps = 'Improve precision with feature pruning or threshold tuning.'

        experiments.append(
            {
                'experiment_id': model_info['experiment_id'],
                'datetime': timestamp,
                'name': model_info['name'],
                'justification': model_info['justification'],
                'hyperparameters': json.dumps(best_params, sort_keys=True),
                'search_space': model_info['param_grid'],
                'data_version': version,
                'metrics': metrics,
                'cv_metrics': cv_metrics,
                'confusion_matrix': result['confusion_matrix'],
                'interpretation': result['interpretation'],
                'observations': observations,
                'next_steps': next_steps,
            }
        )

    EXPERIMENT_LOG_JSON.write_text(json.dumps(experiments, indent=2), encoding='utf-8')

    with EXPERIMENT_LOG_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'experiment_id',
                'datetime',
                'model',
                'justification',
                'hyperparameters',
                'cv_accuracy',
                'cv_precision',
                'cv_recall',
                'cv_f1',
                'cv_roc_auc',
                'data_version',
                'accuracy',
                'precision',
                'recall',
                'f1',
                'roc_auc',
                'tn',
                'fp',
                'fn',
                'tp',
                'interpretation',
                'observations',
                'next_steps',
            ]
        )
        for exp in experiments:
            cm = exp['confusion_matrix']
            m = exp['metrics']
            cvm = exp['cv_metrics']
            writer.writerow(
                [
                    exp['experiment_id'],
                    exp['datetime'],
                    exp['name'],
                    exp['justification'],
                    exp['hyperparameters'],
                    f"{cvm['accuracy']:.4f}",
                    f"{cvm['precision']:.4f}",
                    f"{cvm['recall']:.4f}",
                    f"{cvm['f1']:.4f}",
                    f"{cvm['roc_auc']:.4f}",
                    exp['data_version'],
                    f"{m['accuracy']:.4f}",
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                    f"{m['roc_auc']:.4f}",
                    cm['tn'],
                    cm['fp'],
                    cm['fn'],
                    cm['tp'],
                    exp['interpretation'],
                    exp['observations'],
                    exp['next_steps'],
                ]
            )

    print(f"Wrote: {EXPERIMENT_LOG_CSV}")
    print(f"Wrote: {EXPERIMENT_LOG_JSON}")


if __name__ == '__main__':
    main()
