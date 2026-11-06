from __future__ import annotations

import argparse
import json
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

from bank_distress_bot import BASE_DIR, DEFAULT_CSV, build_features, make_labels, read_dataframe, resolve_path
from train_model import best_threshold


def search_spaces(seed: int, epochs: int, with_deep: bool) -> dict:
    spaces = {
        "random_forest": (RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
                          RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
                          dict(n_estimators=[100, 200, 300, 500], max_depth=[None, 6, 10, 16],
                               min_samples_leaf=[1, 2, 4, 8], max_features=["sqrt", 0.5, 0.8])),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
                                   HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
                                   dict(max_depth=[2, 3, 4, 6, None], learning_rate=[0.02, 0.05, 0.08, 0.15],
                                        max_iter=[100, 250, 400], l2_regularization=[0.0, 0.1, 1.0], min_samples_leaf=[10, 20, 40])),
    }
    if with_deep:
        from bank_distress_bot import TorchMLPClassifier
        spaces["deep_mlp"] = (TorchMLPClassifier(max_epochs=epochs, random_state=seed),
                              TorchMLPClassifier(max_epochs=epochs, random_state=seed),
                              dict(hidden=[(64, 32), (128, 64, 32), (256, 128, 64, 32)], dropout=[0.0, 0.15, 0.3],
                                   lr=[5e-4, 1e-3, 2e-3, 4e-3], weight_decay=[1e-5, 1e-4, 1e-3], batch_size=[128, 256]))
    return spaces


def score(model, X_te, y_te, threshold: float) -> dict:
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= threshold).astype(int)
    return dict(roc_auc=float(roc_auc_score(y_te, p)), average_precision=float(average_precision_score(y_te, p)),
                precision=float(precision_score(y_te, pred, zero_division=0)), recall=float(recall_score(y_te, pred)),
                f1=float(f1_score(y_te, pred)), threshold=round(float(threshold), 4))


def run_tune(a) -> int:
    data = resolve_path(a.data)
    if not data.exists():
        raise SystemExit(f"Data file not found: {data}")
    out_dir = BASE_DIR / a.reports
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataframe(str(data))
    X, y = build_features(df), make_labels(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=a.test_size, stratify=y, random_state=a.seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)

    try:
        import torch
        deep = not a.skip_deep
    except ImportError:
        deep = False
        print("PyTorch not installed - tuning classical models only (pip install -r requirements.txt for the network).")

    results = {}
    for name, (base, default, space) in search_spaces(a.seed, a.max_epochs, deep).items():
        t0 = time.time()
        d_cv = cross_val_predict(default, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        d_thr = best_threshold(y_tr.to_numpy(), d_cv)
        before = score(default.fit(X_tr, y_tr), X_te, y_te, d_thr)
        before["cv_average_precision"] = float(average_precision_score(y_tr, d_cv))

        rs = RandomizedSearchCV(base, space, n_iter=a.n_iter, scoring="average_precision", cv=cv, random_state=a.seed,
                                n_jobs=1, refit=True).fit(X_tr, y_tr)
        t_cv = cross_val_predict(rs.best_estimator_, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        t_thr = best_threshold(y_tr.to_numpy(), t_cv)
        after = score(rs.best_estimator_, X_te, y_te, t_thr)
        after["cv_average_precision"] = float(rs.best_score_)
        results[name] = dict(before=before, after=after, best_params={k: (list(v) if isinstance(v, tuple) else v)
                                                                       for k, v in rs.best_params_.items()},
                             seconds=round(time.time() - t0, 1))
        print(f"\n{name}  ({results[name]['seconds']}s)  best params: {results[name]['best_params']}")
        print(pd.DataFrame({"default": before, "tuned": after}).round(4).to_string())

    (out_dir / "tuning_report.json").write_text(json.dumps(results, indent=2))
    metrics = ["roc_auc", "average_precision", "precision", "recall", "f1"]
    fig, ax = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.2), squeeze=False)
    for x, (name, r) in zip(ax[0], results.items()):
        w = np.arange(len(metrics))
        x.bar(w - 0.2, [r["before"][m] for m in metrics], 0.4, label="default")
        x.bar(w + 0.2, [r["after"][m] for m in metrics], 0.4, label="tuned")
        x.set_xticks(w); x.set_xticklabels(metrics, rotation=25); x.set_ylim(min(0.8, min(r["before"][m] for m in metrics) - 0.05), 1.01)
        x.set_title(f"{name}: default vs tuned (test set)"); x.legend()
    fig.tight_layout(); fig.savefig(out_dir / "tuning_comparison.png", dpi=130); plt.close(fig)
    print(f"\nSaved tuning_report.json and tuning_comparison.png to {out_dir}")
    print("Reminder: labels come from the rule engine, so near-perfect scores show the model re-learns the rules - "
          "see the label-leakage risk in the report.")
    return 0


def prepare(df: pd.DataFrame) -> np.ndarray:
    X = build_features(df).drop(columns=["Anomaly"]).to_numpy(dtype=float)
    X = np.sign(X) * np.log1p(np.abs(X))
    return np.clip(StandardScaler().fit_transform(X), -6, 6)


def torch_autoencoder_scores(X: np.ndarray, seed: int, epochs: int, bottleneck: int):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    n = X.shape[1]
    net = nn.Sequential(nn.Linear(n, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, bottleneck), nn.SiLU(),
                        nn.Linear(bottleneck, 16), nn.SiLU(), nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, n))
    data = torch.from_numpy(X.astype(np.float32))
    opt, loss_fn = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5), nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(data), generator=gen)
        total = 0.0
        for i in range(0, len(order), 128):
            b = data[order[i:i + 128]]
            opt.zero_grad()
            loss = loss_fn(net(b), b)
            loss.backward(); opt.step()
            total += float(loss) * len(b)
        history.append(total / len(data))
    net.eval()
    with torch.no_grad():
        err = ((net(data) - data) ** 2).mean(dim=1).numpy()
    return err, history, "PyTorch autoencoder"


def sklearn_autoencoder_scores(X: np.ndarray, seed: int, epochs: int, bottleneck: int):
    from sklearn.neural_network import MLPRegressor
    m = MLPRegressor(hidden_layer_sizes=(32, 16, bottleneck, 16, 32), activation="relu", max_iter=epochs, random_state=seed,
                     early_stopping=False).fit(X, X)
    err = ((m.predict(X) - X) ** 2).mean(axis=1)
    return err, list(m.loss_curve_), "scikit-learn MLPRegressor autoencoder (PyTorch not installed)"


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    k = int(y_true.sum())
    top = np.argsort(-score)[:k]
    return dict(roc_auc=float(roc_auc_score(y_true, score)), average_precision=float(average_precision_score(y_true, score)),
                precision_at_k=float(y_true[top].mean()), k=k, base_rate=float(y_true.mean()))


def run_autoencoder(a) -> int:
    data = resolve_path(a.data)
    if not data.exists():
        raise SystemExit(f"Data file not found: {data}")
    out_dir = BASE_DIR / a.reports
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataframe(str(data))
    X, y = prepare(df), (df["Anomaly"].to_numpy() == -1).astype(int)
    try:
        import torch
        err, hist, kind = torch_autoencoder_scores(X, a.seed, a.epochs, a.bottleneck)
    except ImportError:
        err, hist, kind = sklearn_autoencoder_scores(X, a.seed, a.epochs, a.bottleneck)
    iso = -IsolationForest(n_estimators=300, random_state=a.seed).fit(X).score_samples(X)

    res = {"autoencoder": evaluate(y, err), "isolation_forest": evaluate(y, iso), "model": kind,
           "final_training_loss": float(hist[-1]), "epochs": len(hist)}
    print(f"{kind} | features {X.shape[1]} | bank-flagged anomalies {int(y.sum())} ({y.mean():.1%})")
    print(pd.DataFrame({k: res[k] for k in ("autoencoder", "isolation_forest")}).round(4).to_string())
    print("ROC-AUC near 0.5 means the account fields carry no information about the bank's anomaly indicator.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(hist); ax[0].set_title("Autoencoder training loss"); ax[0].set_xlabel("epoch")
    ax[1].hist([err[y == 0], err[y == 1]], bins=40, density=True, label=["normal", "bank-flagged anomaly"], color=["#1f77b4", "#d62728"])
    ax[1].set_yscale("log"); ax[1].set_title("Reconstruction error by bank label"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_dir / "autoencoder_scores.png", dpi=130); plt.close(fig)
    (out_dir / "autoencoder_report.json").write_text(json.dumps(res, indent=2))
    print(f"Saved autoencoder_report.json and autoencoder_scores.png to {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Model tuning and autoencoder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tune", "autoencoder"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=DEFAULT_CSV)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--reports", default="reports")
        if name == "tune":
            p.add_argument("--test-size", type=float, default=0.20)
            p.add_argument("--n-iter", type=int, default=8, help="random-search candidates per model")
            p.add_argument("--max-epochs", type=int, default=80)
            p.add_argument("--skip-deep", action="store_true")
        else:
            p.add_argument("--epochs", type=int, default=60)
            p.add_argument("--bottleneck", type=int, default=6)
    a = ap.parse_args()
    return run_tune(a) if a.cmd == "tune" else run_autoencoder(a)


if __name__ == "__main__":
    raise SystemExit(main())
