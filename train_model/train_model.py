from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bank_distress_bot import (BASE_DIR, DEFAULT_CSV, DEFAULT_MODEL, TorchMLPClassifier, build_features, file_digest,
                               make_labels, read_dataframe)

DEEP = ("deep_mlp", "deep_mlp_wide")


def candidates(seed: int, epochs: int = 150) -> dict:
    return {
        "deep_mlp": TorchMLPClassifier(hidden=(128, 64, 32), dropout=0.15, max_epochs=epochs, random_state=seed),
        "deep_mlp_wide": TorchMLPClassifier(hidden=(256, 128, 64, 32), dropout=0.25, lr=1e-3, weight_decay=1e-3,
                                            max_epochs=epochs, random_state=seed),
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=250,
                                                                 class_weight="balanced", random_state=seed),
    }


def best_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, proba)
    return float(t[np.argmax(2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_CSV)
    ap.add_argument("--outcomes", help="optional CSV: Customer ID, distress_outcome (0/1)")
    ap.add_argument("--model-out", default=DEFAULT_MODEL)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=150, help="maximum training epochs per neural network")
    ap.add_argument("--allow-classical", action="store_true", help="let classical models be selected too")
    a = ap.parse_args()

    try:
        import torch
        torch.manual_seed(a.seed)
    except ImportError:
        torch = None
        if not a.allow_classical:
            sys.exit("PyTorch is required for the deep-learning models. Install it with: pip install -r requirements.txt\n"
                     "(or add --allow-classical to train the classical models only)")
        print("PyTorch is not installed - training the classical models only.")

    data = Path(a.data) if Path(a.data).exists() else BASE_DIR / a.data
    if not data.exists():
        sys.exit(f"Data file not found: {data}")
    out = Path(a.model_out) if Path(a.model_out).is_absolute() else BASE_DIR / a.model_out
    out.parent.mkdir(parents=True, exist_ok=True)

    df = read_dataframe(str(data))
    X = build_features(df)
    if a.outcomes:
        y = df["Customer ID"].map(pd.read_csv(a.outcomes).set_index("Customer ID")["distress_outcome"])
        if y.isna().any():
            sys.exit(f"--outcomes is missing {int(y.isna().sum())} customers from the data file.")
        y, source = y.astype(int), f"real outcomes from {a.outcomes}"
    else:
        y, source = make_labels(df), "rule-engine flags (EARLY WARNING or worse)"
    ids = df["Customer ID"].astype(int)
    print(f"Rows: {len(df):,} | features: {X.shape[1]} | positives: {int(y.sum())} ({y.mean():.1%}) | labels: {source}")

    X_tr, X_te, y_tr, _, id_tr, id_te = train_test_split(X, y, ids, test_size=a.test_size, stratify=y,
                                                         random_state=a.seed)
    print(f"Train: {len(X_tr):,} rows | Test (held out, untouched): {len(X_te):,} rows")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    scores, cv_proba = {}, {}
    for name, m in candidates(a.seed, a.max_epochs).items():
        if torch is None and name in DEEP:
            continue
        cv_proba[name] = cross_val_predict(m, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        scores[name] = float(average_precision_score(y_tr, cv_proba[name]))
        print(f"  CV average precision  {name:<24s} {scores[name]:.4f}   [{'deep' if name in DEEP else 'baseline'}]")
    eligible = {k: v for k, v in scores.items() if a.allow_classical or k in DEEP}
    best = max(eligible, key=eligible.get)
    print(f"Selected: {best}" + ("" if a.allow_classical else "  (deep-learning only; --allow-classical to widen)"))

    threshold = best_threshold(y_tr.to_numpy(), cv_proba[best])
    model = candidates(a.seed, a.max_epochs)[best].fit(X_tr, y_tr)
    if best in DEEP:
        print(f"Neural network stopped at epoch {model.n_epochs_} (best validation loss {model.val_loss_:.4f})")
    print(f"Decision threshold (max F1 on cross-validated train predictions): {threshold:.3f}")

    sha = file_digest(str(data))
    joblib.dump(dict(model=model, feature_columns=list(X.columns), threshold=threshold, meta=dict(
        trained_at=datetime.now().isoformat(timespec="seconds"), model_name=best, label_source=source,
        sklearn_version=sklearn.__version__, data_file=data.name, data_sha256=sha, seed=a.seed, test_size=a.test_size,
        cv_average_precision=scores, model_family="deep learning (PyTorch)" if best in DEEP else "classical",
        torch_version=torch.__version__ if torch else None, epochs_trained=getattr(model, "n_epochs_", None),
        n_train=len(X_tr), n_test=len(X_te))), out)
    (out.parent / "split.json").write_text(json.dumps(dict(
        seed=a.seed, label_source=source, data_sha256=sha,
        train_ids=sorted(map(int, id_tr)), test_ids=sorted(map(int, id_te)))))
    if a.outcomes:
        y.rename("label").to_frame().assign(**{"Customer ID": ids}).to_csv(out.parent / "labels.csv", index=False)
    reports = BASE_DIR / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "training_summary.json").write_text(json.dumps(dict(
        selected_model=best, model_family="deep learning (PyTorch)" if best in DEEP else "classical",
        decision_threshold=round(threshold, 4), cv_average_precision={k: round(v, 4) for k, v in scores.items()},
        epochs_trained=getattr(model, "n_epochs_", None), n_features=int(X.shape[1]), n_train=len(X_tr), n_test=len(X_te),
        positives=int(y.sum()), label_source=source), indent=2))
    print(f"Saved training summary -> {reports / 'training_summary.json'}")
    print(f"Saved model -> {out}\nSaved split -> {out.parent / 'split.json'}\nNext: python test_distress.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
