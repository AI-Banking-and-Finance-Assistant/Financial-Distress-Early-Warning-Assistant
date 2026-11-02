from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bank_distress_bot import (BASE_DIR, DEFAULT_CSV, DEFAULT_MODEL, POSITIVE_STATUSES, ModelAdvisor, assess,
                               read_dataframe)


def predict_dataframe(df: pd.DataFrame, advisor: ModelAdvisor, split: dict | None = None) -> pd.DataFrame:
    proba = advisor.predict(df)
    rules = [assess(r) for r in df.to_dict("records")]
    out = pd.DataFrame({"Customer ID": df["Customer ID"].astype(int).to_numpy(),
                        "model_probability": proba.round(4),
                        "model_flag": (proba >= advisor.threshold).astype(int),
                        "rule_status": [r["status"] for r in rules],
                        "rule_score": [r["score"] for r in rules]})
    model, rule = out.model_flag == 1, out.rule_status.isin(POSITIVE_STATUSES)
    out["agreement"] = np.select([model & ~rule, ~model & rule], ["model only", "rules only"], "agree")
    ids = out["Customer ID"]
    out["split"] = np.select([ids.isin(split["train_ids"]), ids.isin(split["test_ids"])], ["train", "test"],
                             "new") if split else "new"
    return out.sort_values("model_probability", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_CSV, help="CSV to predict (default: the training database)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--customer", type=int, nargs="+", help="one or more customer IDs")
    ap.add_argument("--all", action="store_true", help="predict every customer in the file")
    ap.add_argument("--out", default="predictions.csv", help="where to save results for --all / --data")
    ap.add_argument("--top", type=int, default=0, help="print the N highest-probability customers")
    a = ap.parse_args()

    model_path = Path(a.model) if Path(a.model).is_absolute() else BASE_DIR / a.model
    if not model_path.exists():
        sys.exit(f"No trained model at {model_path}. Run: python train_model.py")
    data = Path(a.data) if Path(a.data).exists() else BASE_DIR / a.data
    if not data.exists():
        sys.exit(f"Data file not found: {data}")
    if not (a.customer or a.all or a.top or a.data != DEFAULT_CSV):
        sys.exit("Choose what to predict: --customer ID [ID ...], --all, --top N, or --data FILE.csv")

    advisor = ModelAdvisor.load(model_path)
    if advisor is None:
        return 1
    split_file = model_path.parent / "split.json"
    split = json.loads(split_file.read_text()) if split_file.exists() and data.name == advisor.meta.get("data_file") else None

    df = read_dataframe(str(data))
    if a.customer:
        missing = sorted(set(a.customer) - set(df["Customer ID"].astype(int)))
        if missing:
            print(f"Not found in {data.name}: {missing}")
        df = df[df["Customer ID"].astype(int).isin(a.customer)]
        if df.empty:
            return 1
    res = predict_dataframe(df, advisor, split)

    print(f"Model: {advisor.meta.get('model_name')} | threshold {advisor.threshold:.3f} | {len(res):,} customer(s) | "
          f"labels at training: {advisor.meta.get('label_source')}\n")
    show = res if a.customer else res.head(a.top) if a.top else None
    if show is not None:
        print(show.to_string(index=False))
    if a.all or not a.customer:
        out = Path(a.out) if Path(a.out).is_absolute() else BASE_DIR / a.out
        res.to_csv(out, index=False)
        print(f"\nSaved {len(res):,} predictions -> {out}")
        print("Agreement summary:\n" + res.groupby("split")["agreement"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
