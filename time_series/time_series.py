from __future__ import annotations

import argparse
import json
import sqlite3

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bank_distress_bot import BASE_DIR, DEFAULT_CSV, POSITIVE_STATUSES, assess_all, read_dataframe, resolve_path


def customer_frame(df: pd.DataFrame) -> pd.DataFrame:
    a = assess_all(df)
    out = pd.DataFrame({"date": df["Transaction Date"].dt.normalize(),
                        "flagged": [int(a[int(i)]["status"] in POSITIVE_STATUSES) for i in df["Customer ID"]],
                        "distress": [int(a[int(i)]["status"] == "FINANCIAL DISTRESS") for i in df["Customer ID"]],
                        "score": [a[int(i)]["score"] for i in df["Customer ID"]],
                        "balance": df["Account Balance"].to_numpy(),
                        "volume": df["Transaction Amount"].to_numpy()})
    return out


def daily_series(cf: pd.DataFrame) -> pd.DataFrame:
    full = pd.date_range(cf["date"].min(), cf["date"].max(), freq="D")
    g = cf.groupby("date").agg(customers=("flagged", "size"), flagged=("flagged", "sum"), distress=("distress", "sum"),
                               avg_score=("score", "mean"), avg_balance=("balance", "mean"),
                               volume=("volume", "sum")).reindex(full)
    g[["customers", "flagged", "distress", "volume"]] = g[["customers", "flagged", "distress", "volume"]].fillna(0)
    g["flag_rate"] = (g["flagged"] / g["customers"].where(g["customers"] > 0)).astype(float)
    return g


def weekly_series(daily: pd.DataFrame) -> pd.DataFrame:
    block = (np.arange(len(daily)) // 7)
    g = daily.assign(block=block).groupby("block")
    w = pd.DataFrame({"start": g.apply(lambda x: x.index[0], include_groups=False), "days": g.size(),
                      "customers": g["customers"].sum(), "flagged": g["flagged"].sum(), "distress": g["distress"].sum(),
                      "volume": g["volume"].sum()})
    w["flag_rate"] = w["flagged"] / w["customers"]
    avg = daily.assign(block=block, ws=daily["avg_score"] * daily["customers"], wb=daily["avg_balance"] * daily["customers"])
    a = avg.groupby("block")[["ws", "wb"]].sum()
    w["avg_score"], w["avg_balance"] = a["ws"] / w["customers"], a["wb"] / w["customers"]
    return w[w["days"] == 7].set_index("start")


def decompose(y: pd.Series, period: int = 7) -> pd.DataFrame:
    trend = y.rolling(period, center=True, min_periods=period).mean()
    detr = y - trend
    prof = detr.groupby(np.arange(len(y)) % period).mean()
    seasonal = pd.Series(prof.reindex(np.arange(len(y)) % period).to_numpy() - prof.mean(), index=y.index)
    return pd.DataFrame({"observed": y, "trend": trend, "seasonal": seasonal, "residual": y - trend - seasonal})


def acf(y: np.ndarray, nlags: int = 12) -> np.ndarray:
    y = np.asarray(y, float) - np.mean(y)
    d = float(np.dot(y, y)) or 1.0
    return np.array([np.dot(y[:len(y) - k], y[k:]) / d for k in range(nlags + 1)])


def f_naive(train: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(train[-1], h)


def f_mean(train: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(train.mean(), h)


def f_moving_average(train: np.ndarray, h: int, window: int = 4) -> np.ndarray:
    return np.repeat(train[-window:].mean(), h)


def holt_fit(train: np.ndarray) -> tuple[float, float, float, float]:
    best = (np.inf, 0.5, 0.1, train[0], 0.0)
    for al in np.linspace(0.05, 0.95, 19):
        for be in np.linspace(0.01, 0.5, 10):
            lvl, tr, sse = train[0], train[1] - train[0], 0.0
            for v in train[1:]:
                sse += (v - (lvl + tr)) ** 2
                new = al * v + (1 - al) * (lvl + tr)
                tr = be * (new - lvl) + (1 - be) * tr
                lvl = new
            if sse < best[0]:
                best = (sse, al, be, lvl, tr)
    return best[1:]


def f_holt(train: np.ndarray, h: int) -> np.ndarray:
    _, _, lvl, tr = holt_fit(train)
    return lvl + tr * np.arange(1, h + 1)


def f_arima(train: np.ndarray, h: int) -> np.ndarray | None:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return None
    try:
        return np.asarray(ARIMA(train, order=(1, 0, 1)).fit().forecast(h))
    except Exception:
        return None


MODELS = {"naive (last value)": f_naive, "mean of training weeks": f_mean, "moving average (4 wks)": f_moving_average,
          "Holt linear trend": f_holt, "ARIMA(1,0,1)": f_arima}


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))),
                MAPE=float(np.mean(np.abs(err) / np.where(actual == 0, np.nan, np.abs(actual))) * 100))


def compare_models(y: np.ndarray, holdout: int) -> pd.DataFrame:
    train, test = y[:-holdout], y[-holdout:]
    rows = []
    for name, fn in MODELS.items():
        pred = fn(train, holdout)
        if pred is not None:
            rows.append(dict(model=name, **metrics(test, pred)))
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def snapshot_history(db_path: str) -> pd.DataFrame:
    path = resolve_path(db_path)
    if not path.exists():
        return pd.DataFrame()
    con = sqlite3.connect(path)
    try:
        return pd.read_sql_query(
            "SELECT s.snapshot_id, s.taken_at, COUNT(*) AS customers, "
            "SUM(c.status IN ('EARLY WARNING','FINANCIAL DISTRESS')) AS flagged, "
            "SUM(c.status = 'FINANCIAL DISTRESS') AS distress, AVG(c.score) AS avg_score "
            "FROM snapshots s JOIN snapshot_customers c USING(snapshot_id) GROUP BY s.snapshot_id ORDER BY s.snapshot_id", con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Time series analysis of flagged customers")
    ap.add_argument("--data", default=DEFAULT_CSV)
    ap.add_argument("--db", default="distress_bot.db")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--holdout", type=int, default=8, help="final weeks kept back to test the forecasts")
    ap.add_argument("--horizon", type=int, default=8, help="weeks to forecast beyond the data")
    a = ap.parse_args()

    data = resolve_path(a.data)
    if not data.exists():
        raise SystemExit(f"Data file not found: {data}")
    out_dir = BASE_DIR / a.reports
    out_dir.mkdir(parents=True, exist_ok=True)

    cf = customer_frame(read_dataframe(str(data)))
    daily, weekly = daily_series(cf), None
    weekly = weekly_series(daily)
    y = weekly["flag_rate"].to_numpy()
    print(f"Daily points: {len(daily)} | complete weeks: {len(weekly)} | mean weekly flag rate {y.mean():.1%} "
          f"(sd {y.std(ddof=1):.1%})")

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    weekly["flagged"].plot(ax=ax[0, 0], marker="o", color="#d62728", label="flagged"); weekly["distress"].plot(
        ax=ax[0, 0], marker="s", color="#7f0000", label="distress"); ax[0, 0].set_title("Customers flagged per week"); ax[0, 0].legend()
    weekly["avg_score"].plot(ax=ax[0, 1], color="#ff7f0e"); ax[0, 1].set_title("Average risk score per week")
    weekly["avg_balance"].plot(ax=ax[1, 0], color="#1f77b4"); ax[1, 0].set_title("Average account balance per week")
    weekly["volume"].plot(ax=ax[1, 1], color="#2ca02c"); ax[1, 1].set_title("Total transaction volume per week")
    fig.tight_layout(); fig.savefig(out_dir / "ts_overview.png", dpi=130); plt.close(fig)

    dec = decompose(daily["flagged"].astype(float), 7)
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for x, c in zip(ax, dec.columns):
        dec[c].plot(ax=x); x.set_ylabel(c)
    ax[0].set_title("Decomposition of daily flagged customers (period = 7 days)")
    fig.tight_layout(); fig.savefig(out_dir / "ts_decomposition.png", dpi=130); plt.close(fig)
    seasonal_share = float(dec["seasonal"].var() / dec["observed"].var())

    ac, band = acf(y, 12), 1.96 / np.sqrt(len(y))
    fig, x = plt.subplots(figsize=(8, 4))
    x.bar(range(len(ac)), ac); x.axhline(band, ls="--", c="grey"); x.axhline(-band, ls="--", c="grey")
    x.set_title("Autocorrelation of weekly flag rate (dashed = 95% noise band)"); x.set_xlabel("lag (weeks)")
    fig.tight_layout(); fig.savefig(out_dir / "ts_acf.png", dpi=130); plt.close(fig)
    significant = [int(k) for k in range(1, len(ac)) if abs(ac[k]) > band]

    table = compare_models(y, a.holdout)
    print("\nForecast comparison on the last", a.holdout, "weeks (weekly flag rate):")
    print(table.assign(MAE=lambda d: d.MAE * 100, RMSE=lambda d: d.RMSE * 100).round(2).to_string(index=False))
    print("(MAE/RMSE in percentage points, MAPE in %)")
    best = table.iloc[0]["model"]
    fc = MODELS[best](y, a.horizon)
    resid_sd = float(np.std(y[-a.holdout:] - MODELS[best](y[:-a.holdout], a.holdout), ddof=1))
    idx = pd.date_range(weekly.index[-1] + pd.Timedelta(days=7), periods=a.horizon, freq="7D")
    fig, x = plt.subplots(figsize=(12, 4.5))
    weekly["flag_rate"].plot(ax=x, marker="o", label="actual weekly flag rate")
    pd.Series(fc, index=idx).plot(ax=x, marker="s", color="#d62728", label=f"forecast ({best})")
    x.fill_between(idx, fc - 1.96 * resid_sd, fc + 1.96 * resid_sd, color="#d62728", alpha=0.15, label="approx. 95% range")
    x.axvline(weekly.index[-a.holdout], ls=":", c="grey"); x.set_title("Weekly flag rate with forecast"); x.legend()
    fig.tight_layout(); fig.savefig(out_dir / "ts_forecast.png", dpi=130); plt.close(fig)

    snaps = snapshot_history(a.db)
    if len(snaps) >= 2:
        print("\nSnapshots in the audit database:\n" + snaps.to_string(index=False))
    else:
        print("\nSnapshots: fewer than 2 stored - load a new data extract in the bot to start trend tracking.")

    report = dict(weeks=int(len(weekly)), mean_flag_rate=float(y.mean()), sd_flag_rate=float(y.std(ddof=1)),
                  weekday_seasonality_share_of_variance=seasonal_share, significant_acf_lags=significant,
                  best_forecast_model=str(best), forecast_comparison=table.to_dict("records"),
                  forecast=[round(float(v), 4) for v in fc], snapshots=len(snaps))
    (out_dir / "time_series_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWeekday seasonality explains {seasonal_share:.1%} of daily variance; significant ACF lags: {significant or 'none'}")
    print(f"Saved charts and time_series_report.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
