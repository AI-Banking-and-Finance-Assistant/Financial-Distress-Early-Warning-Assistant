from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import threading
import unittest
import urllib.error
import urllib.request
from datetime import timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

import nlp_intent
import time_series as ts
from bank_distress_bot import (BASE_DIR, CANONICAL, DEFAULT_CSV, DEFAULT_MODEL, Bot, Store, TorchMLPClassifier, assess,
                               build_features, file_digest, make_labels, parse_router_reply, read_dataframe)

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

MIN_ROC_AUC, MIN_RECALL, MIN_PRECISION = 0.95, 0.85, 0.80
T0 = pd.Timestamp("2023-06-01")
CCB, BAL, BAL_AFTER = "Credit Card Balance", "Account Balance", "Account Balance After Transaction"
DATA = BASE_DIR / DEFAULT_CSV


def record(**kw) -> dict:
    base = {"Customer ID": 1, "First Name": "Test", "Last Name": "User", "Contact Number": 12345678901,
            "Email": "t@example.com", "Address": "1 Main St", "City": "Town", "Anomaly": 1,
            BAL: 5000.0, BAL_AFTER: 5200.0, "Transaction Type": "Deposit", "Transaction Amount": 200.0,
            "Transaction Date": T0, "Credit Limit": 10000.0, CCB: 1000.0, "Minimum Payment Due": 50.0,
            "Payment Due Date": T0, "Last Credit Card Payment Date": T0 - timedelta(days=3),
            "Loan Status": "Rejected", "Loan Amount": 0.0, "Loan Type": "Personal"}
    return {**base, **kw}


class TestScoring(unittest.TestCase):
    def codes(self, **kw):
        return {s["code"] for s in assess(record(**kw))["signals"]}

    def test_healthy_customer_has_no_flag(self):
        a = assess(record())
        self.assertEqual((a["status"], a["score"], a["signals"]), ("NO FLAG", 0, []))

    def test_utilisation_points(self):
        pts = lambda v: next(s["points"] for s in assess(record(**{CCB: v}))["signals"] if s["code"] == "HIGH_UTILISATION")
        self.assertEqual((pts(7500), pts(9000), pts(10001)), (1, 2, 3))
        self.assertNotIn("HIGH_UTILISATION", self.codes(**{CCB: 7400}))

    def test_late_payment_points(self):
        late = lambda d: assess(record(**{"Last Credit Card Payment Date": T0 + timedelta(days=d)}))["signals"]
        self.assertEqual(late(10)[0]["points"], 1)
        self.assertEqual(late(31)[0]["points"], 2)
        self.assertEqual(late(0), [])

    def test_low_balance_and_overdrawn(self):
        self.assertIn("LOW_BALANCE", self.codes(**{BAL: 900.0}))
        self.assertIn("OVERDRAWN", self.codes(**{BAL_AFTER: -5.0}))
        self.assertIn("CANNOT_COVER_MIN", self.codes(**{BAL: 20.0}))

    def test_loan_burden_only_counts_approved_loans(self):
        self.assertIn("LOAN_BURDEN", self.codes(**{"Loan Status": "Approved", "Loan Amount": 60000.0}))
        self.assertNotIn("LOAN_BURDEN", self.codes(**{"Loan Status": "Rejected", "Loan Amount": 60000.0}))

    def test_status_thresholds(self):
        early = assess(record(**{CCB: 10500.0, BAL: 900.0, BAL_AFTER: 900.0}))
        self.assertEqual((early["score"], early["status"]), (4, "EARLY WARNING"))
        dist = assess(record(**{CCB: 10500.0, BAL: 100.0, BAL_AFTER: -10.0}))
        self.assertEqual(dist["status"], "FINANCIAL DISTRESS")

    def test_zero_credit_limit_does_not_crash(self):
        self.assertEqual(assess(record(**{"Credit Limit": 0.0}))["metrics"]["util"], 0.0)

    def test_demographics_never_change_the_result(self):
        a = assess(record(**{CCB: 9500.0}))
        b = assess(record(**{CCB: 9500.0, "Age": 80, "Gender": "Female", "City": "Elsewhere"}))
        self.assertEqual((a["status"], a["score"]), (b["status"], b["score"]))


class TestFeaturesAndRouter(unittest.TestCase):
    def test_features_exclude_personal_attributes(self):
        words = set(re.findall(r"[a-z]+", " ".join(build_features(read_dataframe(str(DATA)).head(50)).columns).lower()))
        for banned in ("age", "gender", "city", "address", "email", "first", "last", "name", "contact", "customer", "id"):
            self.assertNotIn(banned, words, f"personal attribute '{banned}' leaked into model features")

    def test_features_are_numeric_and_finite(self):
        X = build_features(read_dataframe(str(DATA)).head(200))
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertTrue(all(np.issubdtype(t, np.number) for t in X.dtypes))

    def test_router_reply_parsing(self):
        for reply, expected in [('{"index": 2, "customer_id": null}', (2, None)),
                                ('noise {"index": 1, "customer_id": 77} noise', (1, 77)),
                                ('{"index": 999, "customer_id": true}', (None, None)),
                                ("not json", (None, None))]:
            self.assertEqual(parse_router_reply(reply), expected)


class TestBotWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [record(**{"Customer ID": 1, CCB: 10500.0, BAL: 900.0}),
                record(**{"Customer ID": 2, CCB: 10500.0, BAL: 100.0, BAL_AFTER: -10.0}),
                record(**{"Customer ID": 3})]
        cls.assessments = {r["Customer ID"]: assess(r) for r in rows}

    def setUp(self):
        self.bot = Bot(self.assessments, Store(":memory:"))
        self.s = self.bot.new_session("E1234")

    def say(self, text, session=None):
        return self.bot.handle(text, session or self.s)

    def test_check_and_not_found(self):
        self.assertIn("EARLY WARNING", self.say("Check customer 1"))
        self.assertIn("was not found", self.say("Check customer 999"))

    def test_flag_lists_sorted_by_priority(self):
        out = self.say("Find customers at risk")
        self.assertLess(out.index("Customer 2"), out.index("Customer 1"))
        self.assertNotIn("Customer 3", out)

    def test_resolve_requires_note_and_hides_flag(self):
        self.assertIn("short review note", self.say("resolve flag for 1 by E1234"))
        self.assertIn("marked as resolved", self.say("resolve flag for 1 by E1234: called customer, plan agreed"))
        self.assertNotIn("Customer 1 ", self.say("Find customers at risk"))

    def test_distress_needs_a_different_supervisor(self):
        self.assertIn("supervisor approval", self.say("resolve flag for 2 by E1234: spoke with customer today"))
        self.assertIn("valid supervisor", self.say("approve 2 by E1234"))
        self.assertIn("approved by SUP002", self.say("approve 2 by SUP002"))

    def test_supervisor_cannot_approve_own_request(self):
        s = self.bot.new_session("SUP001")
        self.say("resolve flag for 2 by SUP001: reviewed accounts with customer", s)
        self.assertIn("different person", self.say("approve 2 by SUP001", s))

    def test_resolution_expires_if_customer_worsens(self):
        self.say("resolve flag for 1 by E1234: called customer, plan agreed")
        worse = dict(self.assessments[1], status="FINANCIAL DISTRESS", score=8)
        bot2 = Bot({**self.assessments, 1: worse}, self.bot.store)
        self.assertIn("Customer 1", bot2.handle("Find customers at risk", bot2.new_session("E1234")))

    def test_audit_log_written(self):
        self.say("Check customer 3")
        self.assertEqual(self.bot.store.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0], 1)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed (pip install -r requirements.txt)")
class TestDeepLearningModel(unittest.TestCase):
    @staticmethod
    def make():
        return TorchMLPClassifier(hidden=(16, 8), max_epochs=80, patience=10, random_state=0)

    @classmethod
    def setUpClass(cls):
        cls.X = np.random.RandomState(0).randn(500, 6) * np.array([1, 1, 1, 1, 1, 5000.0])
        cls.y = (cls.X[:, 0] + 0.5 * cls.X[:, 1] > 0).astype(int)
        cls.model = cls.make().fit(cls.X, cls.y)

    def test_probabilities_are_valid(self):
        p = self.model.predict_proba(self.X)
        self.assertEqual(p.shape, (500, 2))
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_learns_a_simple_pattern(self):
        self.assertGreater((self.model.predict(self.X) == self.y).mean(), 0.9)

    def test_same_seed_gives_same_model(self):
        np.testing.assert_allclose(self.make().fit(self.X, self.y).predict_proba(self.X),
                                   self.model.predict_proba(self.X), atol=1e-5)

    def test_pickle_round_trip(self):
        clone = pickle.loads(pickle.dumps(self.model))
        np.testing.assert_allclose(clone.predict_proba(self.X), self.model.predict_proba(self.X), atol=1e-6)

    def test_works_inside_sklearn_clone(self):
        from sklearn.base import clone
        c = clone(self.model)
        self.assertEqual(c.get_params()["hidden"], (16, 8))
        self.assertFalse(hasattr(c, "state_"))


class TestTimeSeries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = read_dataframe(str(DATA)).head(1500)
        cls.cf = ts.customer_frame(cls.df)
        cls.daily = ts.daily_series(cls.cf)

    def test_daily_counts_add_up(self):
        self.assertEqual(int(self.daily["customers"].sum()), len(self.df))
        self.assertTrue((self.daily["flagged"] <= self.daily["customers"]).all())

    def test_weekly_blocks_are_complete(self):
        w = ts.weekly_series(self.daily)
        self.assertTrue((w["days"] == 7).all())
        self.assertTrue(w["flag_rate"].between(0, 1).all())

    def test_decomposition_reconstructs_series(self):
        y = pd.Series(np.sin(np.arange(70) * 2 * np.pi / 7) + np.arange(70) * 0.1)
        d = ts.decompose(y, 7).dropna()
        np.testing.assert_allclose((d.trend + d.seasonal + d.residual).to_numpy(), d.observed.to_numpy(), atol=1e-9)

    def test_acf_lag_zero_is_one(self):
        self.assertAlmostEqual(ts.acf(np.random.RandomState(0).randn(100), 5)[0], 1.0)

    def test_holt_follows_a_straight_line(self):
        y = np.arange(30, dtype=float) * 2 + 5
        np.testing.assert_allclose(ts.f_holt(y, 3), [65, 67, 69], atol=0.5)

    def test_metrics_and_comparison(self):
        m = ts.metrics(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
        self.assertAlmostEqual(m["MAE"], 1.0)
        t = ts.compare_models(np.random.RandomState(1).rand(40), 6)
        self.assertGreaterEqual(len(t), 4)
        self.assertEqual(list(t["RMSE"]), sorted(t["RMSE"]))


class TestNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = nlp_intent.make_dataset(0)
        cls.model = nlp_intent.build_pipeline().fit([nlp_intent.normalise(s) for s in X], y)

    def test_every_intent_has_phrases_and_valid_index(self):
        self.assertTrue(all(0 <= i < len(CANONICAL) and p for i, p in nlp_intent.PHRASES.items()))
        self.assertEqual(set(nlp_intent.PHRASES), set(range(len(CANONICAL))))

    def test_clear_questions_are_classified(self):
        for text, expected in [("who paid late", 4), ("how does flagging work", 13), ("why was customer 12 flagged", 1)]:
            self.assertEqual(nlp_intent.classify(self.model, text)[0], expected)

    def test_customer_id_is_extracted(self):
        self.assertEqual(nlp_intent.classify(self.model, "check customer 4312")[2], 4312)

    def test_nonsense_is_rejected(self):
        self.assertIsNone(nlp_intent.classify(self.model, "what's the weather like in Paris")[0])


class TestWebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_app
        from bank_distress_bot import Backend
        cls.be = Backend(DEFAULT_CSV, ":memory:", "models/does_not_exist.joblib", "E1234")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), web_app.make_handler(web_app.App(cls.be)))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def post(self, body, ctype="application/json", host=None):
        h = {"Content-Type": ctype, **({"Host": host} if host else {})}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/ask", data=body, headers=h)
        return urllib.request.urlopen(req)

    def test_chat_round_trip(self):
        r = json.loads(self.post(json.dumps({"text": "How does flagging work?"}).encode()).read())
        self.assertIn("reply", r)

    def test_rejects_non_json_and_foreign_host(self):
        with self.assertRaises(urllib.error.HTTPError) as c:
            self.post(b"text=hi", ctype="text/plain")
        self.assertEqual(c.exception.code, 415)
        with self.assertRaises(urllib.error.HTTPError) as c:
            self.post(json.dumps({"text": "hi"}).encode(), host="evil.example.com")
        self.assertEqual(c.exception.code, 403)

    def test_page_served(self):
        self.assertIn(b"Early-Warning", urllib.request.urlopen(f"http://127.0.0.1:{self.port}/").read())

    def test_clear_resets_conversation_but_keeps_employee(self):
        self.be.session.employee_id = "E1234"
        self.post(json.dumps({"text": "Check customer 1"}).encode())
        self.assertIsNotNone(self.be.session.last_customer)
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/clear", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        self.assertTrue(json.loads(urllib.request.urlopen(req).read())["ok"])
        self.assertIsNone(self.be.session.last_customer)
        self.assertEqual(self.be.session.employee_id, "E1234")


def evaluate_model(data_path: Path, model_path: Path, out_dir: Path) -> dict:
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, average_precision_score,
                                 classification_report, confusion_matrix, f1_score, precision_score, recall_score,
                                 roc_auc_score)

    bundle = joblib.load(model_path)
    split = json.loads((model_path.parent / "split.json").read_text())
    assert split["data_sha256"] == file_digest(str(data_path)), "Data file changed since training - retrain first."
    assert not set(split["train_ids"]) & set(split["test_ids"]), "Train/test leakage: IDs overlap."

    df = read_dataframe(str(data_path))
    test = df[df["Customer ID"].astype(int).isin(split["test_ids"])].reset_index(drop=True)
    labels_file = model_path.parent / "labels.csv"
    if labels_file.exists():
        y = test["Customer ID"].map(pd.read_csv(labels_file).set_index("Customer ID")["label"]).astype(int)
    else:
        y = make_labels(test)
    X = build_features(test).reindex(columns=bundle["feature_columns"], fill_value=0.0)
    model, thr = bundle["model"], bundle["threshold"]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    rep = dict(n_test=int(len(y)), positives=int(y.sum()), threshold=round(thr, 4), model=bundle["meta"]["model_name"],
               label_source=bundle["meta"]["label_source"], roc_auc=float(roc_auc_score(y, proba)),
               average_precision=float(average_precision_score(y, proba)), precision=float(precision_score(y, pred)),
               recall=float(recall_score(y, pred)), f1=float(f1_score(y, pred)),
               confusion_matrix=confusion_matrix(y, pred).tolist())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_report.json").write_text(json.dumps(rep, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    RocCurveDisplay.from_predictions(y, proba, ax=ax[0]); ax[0].set_title("ROC curve (held-out test set)")
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax[1]); ax[1].set_title("Precision-recall curve")
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["No flag", "Flagged"], ax=ax[2], colorbar=False)
    ax[2].set_title(f"Confusion matrix (threshold {thr:.2f})")
    fig.tight_layout(); fig.savefig(out_dir / "test_evaluation.png", dpi=130); plt.close(fig)

    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        imp = permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring="average_precision").importances_mean
    fig, ax = plt.subplots(figsize=(7, 5))
    pd.Series(imp, index=bundle["feature_columns"]).sort_values().tail(12).plot.barh(ax=ax)
    ax.set_title("Top feature importances")
    fig.tight_layout(); fig.savefig(out_dir / "feature_importance.png", dpi=130); plt.close(fig)

    print("\n" + "=" * 70 + "\nHELD-OUT TEST SET EVALUATION\n" + "=" * 70)
    print(f"Model: {rep['model']} | labels: {rep['label_source']}\nTest rows: {rep['n_test']} (positives: {rep['positives']})")
    print(classification_report(y, pred, target_names=["No flag", "Flagged"], digits=3))
    print(f"ROC-AUC {rep['roc_auc']:.4f} | Average precision {rep['average_precision']:.4f}")
    print(f"Saved: {out_dir / 'test_report.json'}, test_evaluation.png, feature_importance.png")
    return rep


class TestHeldOutModel(unittest.TestCase):
    report: dict = {}

    def test_model_quality_on_unseen_customers(self):
        r = type(self).report
        self.assertGreaterEqual(r["roc_auc"], MIN_ROC_AUC)
        self.assertGreaterEqual(r["recall"], MIN_RECALL, "Model misses too many flagged customers.")
        self.assertGreaterEqual(r["precision"], MIN_PRECISION, "Model raises too many false alarms.")


def main() -> int:
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_CSV)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--skip-model", action="store_true", help="run unit tests only")
    a = ap.parse_args()
    DATA = Path(a.data) if Path(a.data).exists() else BASE_DIR / a.data
    model_path = Path(a.model) if Path(a.model).is_absolute() else BASE_DIR / a.model

    load = unittest.defaultTestLoader.loadTestsFromTestCase
    suite = unittest.TestSuite(load(c) for c in (TestScoring, TestFeaturesAndRouter, TestBotWorkflow, TestDeepLearningModel,
                                                           TestTimeSeries, TestNLP, TestWebApp))
    if not a.skip_model:
        if not model_path.exists():
            print(f"No trained model at {model_path} - run `python train_model.py` first (or use --skip-model).")
            return 2
        TestHeldOutModel.report = evaluate_model(DATA, model_path, BASE_DIR / a.reports)
        suite.addTests(load(TestHeldOutModel))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
