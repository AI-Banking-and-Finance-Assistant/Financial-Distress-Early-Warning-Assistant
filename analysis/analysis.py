from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bank_distress_bot import (BASE_DIR, DEFAULT_CSV, DATE_COLS, POSITIVE_STATUSES, assess_all, build_features,
                               read_dataframe, resolve_path)


DICTIONARY = {
    "Customer ID": ("identifier", "Unique customer number (never used as a model input)"),
    "First Name": ("personal / text", "Customer first name (display only)"),
    "Last Name": ("personal / text", "Customer last name (display only)"),
    "Age": ("demographic / numeric", "Age in years (fairness audit only, never scored)"),
    "Gender": ("demographic / categorical", "Gender (fairness audit only, never scored)"),
    "Address": ("personal / text", "Street address (display only)"),
    "City": ("demographic / categorical", "City (fairness audit only, never scored)"),
    "Contact Number": ("personal / numeric", "Phone number (display only)"),
    "Email": ("personal / text", "Email address (display only)"),
    "Account Type": ("account / categorical", "Current or savings account"),
    "Account Balance": ("account / numeric", "Current account balance"),
    "Date Of Account Opening": ("account / date", "When the account was opened"),
    "Last Transaction Date": ("account / date", "Date of the most recent transaction"),
    "TransactionID": ("transaction / identifier", "Transaction number"),
    "Transaction Date": ("transaction / date", "Date of the recorded transaction (2023) - the time axis"),
    "Transaction Type": ("transaction / categorical", "Deposit, withdrawal or transfer"),
    "Transaction Amount": ("transaction / numeric", "Amount of the recorded transaction"),
    "Account Balance After Transaction": ("transaction / numeric", "Balance straight after that transaction"),
    "Branch ID": ("account / identifier", "Branch number"),
    "Loan ID": ("loan / identifier", "Loan number"),
    "Loan Amount": ("loan / numeric", "Loan principal"),
    "Loan Type": ("loan / categorical", "Mortgage, auto, personal, ..."),
    "Interest Rate": ("loan / numeric", "Loan interest rate (%)"),
    "Loan Term": ("loan / numeric", "Loan term (months)"),
    "Approval/Rejection Date": ("loan / date", "Date the loan decision was made"),
    "Loan Status": ("loan / categorical", "Approved, rejected, closed"),
    "CardID": ("card / identifier", "Card number"),
    "Card Type": ("card / categorical", "Card network"),
    "Credit Limit": ("card / numeric", "Card credit limit"),
    "Credit Card Balance": ("card / numeric", "Amount owed on the card"),
    "Minimum Payment Due": ("card / numeric", "Minimum card payment due"),
    "Payment Due Date": ("card / date", "Card payment due date"),
    "Last Credit Card Payment Date": ("card / date", "Date of last card payment"),
    "Rewards Points": ("card / numeric", "Reward points balance"),
    "Feedback ID": ("feedback / identifier", "Feedback record number"),
    "Feedback Date": ("feedback / date", "When feedback was given"),
    "Feedback Type": ("feedback / categorical", "Suggestion, complaint or praise"),
    "Resolution Status": ("feedback / categorical", "Pending or resolved"),
    "Resolution Date": ("feedback / date", "When feedback was resolved"),
    "Anomaly": ("flag / numeric", "Bank-supplied anomaly indicator (-1 = anomalous, 1 = normal)"),
}


def build_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        form, meaning = DICTIONARY.get(c, ("other", ""))
        rows.append(dict(column=c, data_form=form, dtype=str(df[c].dtype), missing=int(df[c].isna().sum()),
                         unique=int(df[c].nunique()), example=str(df[c].iloc[0]), meaning=meaning))
    return pd.DataFrame(rows)


def quality_report(df: pd.DataFrame) -> dict:
    out = dict(rows=int(len(df)), columns=int(df.shape[1]), missing_cells=int(df.isna().sum().sum()),
               duplicate_rows=int(df.duplicated().sum()), duplicate_customer_ids=int(df["Customer ID"].duplicated().sum()),
               negative_balances=int((df["Account Balance"] < 0).sum()),
               zero_credit_limits=int((df["Credit Limit"] <= 0).sum()),
               card_balance_over_limit=int((df["Credit Card Balance"] > df["Credit Limit"]).sum()),
               transaction_date_range=[str(df["Transaction Date"].min().date()), str(df["Transaction Date"].max().date())])
    out["date_columns_parsed"] = [c for c in DATE_COLS if pd.api.types.is_datetime64_any_dtype(df[c])]
    return out


def run_eda(a) -> int:
    data = resolve_path(a.data)
    if not data.exists():
        raise SystemExit(f"Data file not found: {data}")
    out_dir = BASE_DIR / a.reports
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataframe(str(data))
    q = quality_report(df)
    num_cols = ["Account Balance", "Transaction Amount", "Credit Limit", "Credit Card Balance", "Loan Amount", "Age"]
    summary = dict(quality=q, numeric=df[num_cols].describe().round(2).to_dict(),
                   categorical={c: df[c].value_counts().to_dict() for c in
                                ["Account Type", "Transaction Type", "Loan Type", "Loan Status", "Card Type",
                                 "Feedback Type", "Resolution Status", "Gender"]},
                   anomaly=df["Anomaly"].value_counts().to_dict())
    status = pd.Series({k: v["status"] for k, v in assess_all(df).items()}).value_counts()
    summary["rule_status_counts"] = status.to_dict()
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    dd = build_dictionary(df)
    dd.to_csv(out_dir / "data_dictionary.csv", index=False)

    print("DATA QUALITY")
    for k, v in q.items():
        print(f"  {k:<28s} {v}")
    print("\nRULE-ENGINE STATUS COUNTS\n" + status.to_string())
    print("\nDATA FORMS\n" + dd.groupby("data_form").size().to_string())

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    df["Account Balance"].plot.hist(bins=40, ax=ax[0], color="#1f77b4"); ax[0].set_title("Account balance")
    (df["Credit Card Balance"] / df["Credit Limit"].where(df["Credit Limit"] > 0)).clip(0, 5).plot.hist(
        bins=40, ax=ax[1], color="#ff7f0e"); ax[1].set_title("Card utilisation (clipped at 500%)")
    df["Transaction Type"].value_counts().plot.bar(ax=ax[2], color="#2ca02c"); ax[2].set_title("Transaction type")
    df["Loan Status"].value_counts().plot.bar(ax=ax[3], color="#d62728"); ax[3].set_title("Loan status")
    df.groupby(df["Transaction Date"].dt.to_period("M")).size().plot(ax=ax[4], marker="o"); ax[4].set_title("Transactions per month")
    status.reindex(["NO FLAG", "EARLY WARNING", "FINANCIAL DISTRESS"]).plot.bar(ax=ax[5], color=["#2ca02c", "#ff7f0e", "#d62728"])
    ax[5].set_title("Rule-engine status")
    for x in ax:
        x.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out_dir / "eda_overview.png", dpi=130); plt.close(fig)
    print(f"Saved {out_dir / 'eda_overview.png'}")

    X = build_features(df)
    corr = X.loc[:, X.nunique() > 2].corr()
    fig, x = plt.subplots(figsize=(9, 8))
    im = x.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    x.set_xticks(range(len(corr))); x.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    x.set_yticks(range(len(corr))); x.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=x); x.set_title("Correlation of model features")
    fig.tight_layout(); fig.savefig(out_dir / "eda_correlation.png", dpi=130); plt.close(fig)
    print(f"Saved {out_dir / 'eda_correlation.png'}")
    print(f"\nSaved reports to {out_dir}")
    return 0


ORDER = ["NO FLAG", "EARLY WARNING", "FINANCIAL DISTRESS"]
COLOURS = ["#2ca02c", "#ff7f0e", "#d62728"]
MIN_GROUP = 30


def attach(df: pd.DataFrame) -> pd.DataFrame:
    a = assess_all(df)
    out = df.copy()
    ids = out["Customer ID"].astype(int)
    out["status"] = ids.map(lambda i: a[i]["status"])
    out["score"] = ids.map(lambda i: a[i]["score"])
    out["flagged"] = out["status"].isin(POSITIVE_STATUSES).astype(int)
    out["signal_codes"] = ids.map(lambda i: [s["code"] for s in a[i]["signals"]])
    out["age_group"] = pd.cut(out["Age"], [0, 24, 34, 44, 54, 64, 120],
                              labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
    return out


def group_rates(d: pd.DataFrame, col: str, top: int | None = None) -> pd.DataFrame:
    g = d.groupby(col, observed=True)["flagged"].agg(rate="mean", n="size").reset_index()
    if top:
        g = g.sort_values("n", ascending=False).head(top).sort_values("rate", ascending=False)
    return g


def chi_square_p(d: pd.DataFrame, col: str) -> float | None:
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return None
    return float(chi2_contingency(pd.crosstab(d[col], d["flagged"]))[1])


def run_charts(a) -> int:
    data = resolve_path(a.data)
    if not data.exists():
        raise SystemExit(f"Data file not found: {data}")
    out_dir = BASE_DIR / a.reports
    out_dir.mkdir(parents=True, exist_ok=True)
    d = attach(read_dataframe(str(data)))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    counts = d["status"].value_counts().reindex(ORDER)
    bars = ax[0].bar(ORDER, counts, color=COLOURS); ax[0].set_title("Customers by rule-engine status")
    for b, v in zip(bars, counts):
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(d):.1%})", ha="center", va="bottom", fontsize=9)
    ax[0].set_ylim(0, counts.max() * 1.18); ax[0].tick_params(axis="x", labelsize=8)
    d["score"].value_counts().sort_index().plot.bar(ax=ax[1], color="#1f77b4"); ax[1].set_title("Risk score distribution")
    ax[1].set_xlabel("score (4+ = early warning, 7+ = distress)")
    d["Anomaly"].map({1: "normal", -1: "anomalous"}).value_counts().plot.bar(ax=ax[2], color=["#7f7f7f", "#9467bd"])
    ax[2].set_title("Bank anomaly indicator"); ax[2].tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(out_dir / "class_distribution.png", dpi=130); plt.close(fig)

    sig = pd.Series([c for codes in d["signal_codes"] for c in codes]).value_counts()
    fig, x = plt.subplots(figsize=(8, 4.5))
    (sig / len(d) * 100).sort_values().plot.barh(ax=x, color="#ff7f0e"); x.set_xlabel("% of customers")
    x.set_title("How often each warning signal fires")
    fig.tight_layout(); fig.savefig(out_dir / "signal_frequency.png", dpi=130); plt.close(fig)

    audit = {}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    overall = d["flagged"].mean()
    for x, col, top in zip(ax, ["age_group", "Gender", "City"], [None, None, 10]):
        g = group_rates(d, col, top)
        colours = ["#1f77b4" if n >= MIN_GROUP else "#c7d7e8" for n in g["n"]]
        x.bar(g[col].astype(str), g["rate"] * 100, color=colours); x.axhline(overall * 100, ls="--", c="grey")
        x.set_title(f"Flag rate by {col.replace('_', ' ').lower()}"); x.set_ylabel("% flagged")
        x.tick_params(axis="x", rotation=40, labelsize=8)
        audit[col] = dict(chi_square_p=chi_square_p(d, col), rates={str(k): round(float(v), 4) for k, v in zip(g[col], g["rate"])},
                          min_group_size=int(g["n"].min()))
    fig.suptitle(f"Fairness audit - dashed line = overall rate {overall:.1%}; faded bars = fewer than {MIN_GROUP} customers", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "demographics_audit.png", dpi=130); plt.close(fig)
    (out_dir / "fairness_audit.json").write_text(json.dumps(audit, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ft = pd.crosstab(d["status"], d["Feedback Type"], normalize="index").reindex(ORDER) * 100
    ft.plot.bar(ax=ax[0], stacked=True); ax[0].set_title("Feedback type by status (%)"); ax[0].tick_params(axis="x", rotation=15)
    rs = pd.crosstab(d["status"], d["Resolution Status"], normalize="index").reindex(ORDER) * 100
    rs.plot.bar(ax=ax[1], stacked=True); ax[1].set_title("Feedback resolution by status (%)"); ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(out_dir / "feedback_analysis.png", dpi=130); plt.close(fig)

    print(f"Overall flag rate {overall:.2%}")
    for col, v in audit.items():
        p = v["chi_square_p"]
        verdict = "n/a (scipy missing)" if p is None else ("differences larger than chance - investigate" if p < 0.05
                                                         else "no evidence of difference")
        print(f"  {col:<10s} chi-square p = {p if p is None else round(p, 4)}  -> {verdict}")
    print(f"Saved charts and fairness_audit.json to {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Data exploration and charts")
    ap.add_argument("which", nargs="?", default="all", choices=["all", "eda", "charts"])
    ap.add_argument("--data", default=DEFAULT_CSV)
    ap.add_argument("--reports", default="reports")
    a = ap.parse_args()
    rc = 0
    if a.which in ("all", "eda"):
        rc |= run_eda(a)
    if a.which in ("all", "charts"):
        rc |= run_charts(a)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
