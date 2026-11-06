from __future__ import annotations

import argparse
import json
import random
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline

from bank_distress_bot import BASE_DIR, CANONICAL

INTENT_MODEL = "models/intent_model.joblib"
MIN_CONFIDENCE = 0.45

PHRASES: dict[int, list[str]] = {
    0: ["check customer {id}", "look up customer {id}", "show me customer {id}", "what is the status of customer {id}",
        "how is customer {id} doing", "pull up account {id}", "status for client {id}", "give me a summary of customer {id}",
        "is customer {id} flagged", "review customer {id}", "customer {id} details"],
    1: ["why was customer {id} flagged", "why is customer {id} on the list", "explain the flag for customer {id}",
        "what triggered the warning for customer {id}", "reason for flag on {id}", "why did customer {id} get flagged",
        "what are the warning signs for customer {id}", "how come customer {id} is at risk"],
    2: ["find customers at risk", "who is at risk", "show me flagged customers", "list customers with warnings",
        "which customers need attention", "show all early warning customers", "who might be struggling financially",
        "list the at risk accounts", "give me customers that need review"],
    3: ["find customers already in financial distress", "who is in distress", "show distress customers",
        "list customers in financial distress", "which customers are in serious trouble", "show the most urgent cases",
        "customers with the highest risk", "who is in severe financial difficulty"],
    4: ["find customers with late payments", "who paid late", "show overdue card payments", "customers behind on payments",
        "list late payers", "who missed their credit card payment", "which customers are delinquent",
        "who is behind on card payments", "show customers with overdue payments"],
    5: ["find customers with high credit utilisation", "who has maxed out their cards", "show customers over their credit limit",
        "list high card usage", "customers close to their credit limit", "who is using too much of their credit",
        "show high credit card utilization", "customers with maxed cards"],
    6: ["find customers with low balance", "who has a low account balance", "show customers with little money",
        "list accounts with insufficient funds", "customers running out of funds", "show low balance accounts",
        "who has almost no money in their account", "accounts with very low funds"],
    7: ["find customers overdrawn", "who is overdrawn", "show customers with negative balance", "list overdraft accounts",
        "which accounts went below zero", "customers in overdraft", "who has an overdrawn account"],
    8: ["show customers with multiple warning signs", "who has several warnings", "customers with more than one flag",
        "list customers with many risk signals", "which customers have multiple issues", "show accounts with several red flags",
        "customers triggering many rules"],
    9: ["is customer {id} improving", "is customer {id} getting better", "has customer {id} recovered",
        "is the situation better for customer {id}", "customer {id} progress", "is {id} doing better than before",
        "any improvement for customer {id}"],
    10: ["what changed for customer {id}", "what is different for customer {id}", "show changes for customer {id}",
         "compare customer {id} with last time", "what has changed since the last snapshot for customer {id}",
         "customer {id} changes", "how has customer {id} moved since last data"],
    11: ["what support could be relevant", "what help can we offer", "suggest support options", "how can we assist this customer",
         "what assistance is available", "recommend support for them", "what can we do to help",
         "which support should we recommend"],
    12: ["show contact details for customer {id}", "what is the phone number of customer {id}", "how do I contact customer {id}",
         "get the email for customer {id}", "contact info for {id}", "customer {id} phone and email", "address of customer {id}"],
    13: ["how does flagging work", "how are customers scored", "explain the rules", "what are the flag thresholds",
         "how do you decide who is flagged", "what rules do you use", "how is the risk score calculated",
         "explain the scoring system"],
    14: ["help", "what can you do", "show me examples", "what can I ask", "I need help", "commands", "how do I use this",
         "what questions do you understand"],
}

UNSEEN = [("which accounts have gone negative", 7), ("tell me about customer 4312", 0), ("any customers in real trouble", 3),
          ("who hasn't paid their card on time", 4), ("accounts that are nearly empty", 6), ("why did 77 get a flag", 1),
          ("customers whose cards are almost maxed", 5), ("whats the phone for client 912", 12), ("has 55 turned a corner", 9),
          ("anyone showing several red flags", 8), ("what should we offer them", 11), ("how are the thresholds set", 13),
          ("who needs a review today", 2), ("what changed on account 3001", 10), ("pls help", 14)]


def typo(text: str, rng: random.Random) -> str:
    if len(text) < 6:
        return text
    i = rng.randrange(1, len(text) - 1)
    return rng.choice([text[:i] + text[i + 1:], text[:i] + text[i + 1] + text[i] + text[i + 2:], text[:i] + text[i] + text[i:]])


def make_dataset(seed: int = 42, per_phrase: int = 4):
    rng = random.Random(seed)
    prefixes, suffixes = ["", "", "please ", "can you ", "could you ", "hey, ", "I need to "], ["", "", " please", " now", " for me", "?"]
    X, y = [], []
    for idx, phrases in PHRASES.items():
        for p in phrases:
            for _ in range(per_phrase):
                s = rng.choice(prefixes) + p.format(id=rng.randint(1, 5000)) + rng.choice(suffixes)
                X.append(typo(s, rng) if rng.random() < 0.25 else s)
                y.append(idx)
    return X, np.array(y)


def build_pipeline():
    feats = FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
                          ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True))])
    return make_pipeline(feats, LogisticRegression(max_iter=3000, C=10.0))


def normalise(text: str) -> str:
    return re.sub(r"\d+", " ", text.lower())


def classify(model, text: str):
    proba = model.predict_proba([normalise(text)])[0]
    k = int(np.argmax(proba))
    m = re.search(r"\b(\d{1,6})\b", text)
    cid = int(m.group(1)) if m else None
    return (int(model.classes_[k]) if proba[k] >= MIN_CONFIDENCE else None), float(proba[k]), cid


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline intent classifier for the chatbot")
    ap.add_argument("--ask", help="classify one sentence using the saved model")
    ap.add_argument("--model-out", default=INTENT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    path = BASE_DIR / a.model_out

    if a.ask:
        if not path.exists():
            raise SystemExit("No saved intent model - run `python nlp_intent.py` first.")
        idx, conf, cid = classify(joblib.load(path), a.ask)
        print(f"intent: {CANONICAL[idx][0] if idx is not None else 'not understood'} | confidence {conf:.2f} | customer id: {cid}")
        return 0

    X, y = make_dataset(a.seed)
    Xn = [normalise(s) for s in X]
    Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.25, stratify=y, random_state=a.seed)
    model = build_pipeline().fit(Xtr, ytr)
    pred = model.predict(Xte)
    names = [CANONICAL[i][0].replace("{id}", "N")[:38] for i in sorted(PHRASES)]
    print(f"Synthetic sentences: {len(X)} | train {len(Xtr)} | test {len(Xte)} | intents {len(PHRASES)}")
    print(f"Held-out synthetic test accuracy {accuracy_score(yte, pred):.3f} | macro-F1 {f1_score(yte, pred, average='macro'):.3f}")
    print("NOTE: test sentences share templates with training sentences, so this number is optimistic.\n")

    ux, uy = [normalise(s) for s, _ in UNSEEN], np.array([i for _, i in UNSEEN])
    up = model.predict(ux)
    conf = model.predict_proba(ux).max(axis=1)
    print(f"Unseen hand-written questions: top-1 accuracy {accuracy_score(uy, up):.3f} ({int((up == uy).sum())}/{len(uy)})")
    answered = conf >= MIN_CONFIDENCE
    print(f"With the {MIN_CONFIDENCE:.2f} confidence cut-off the classifier answers {int(answered.sum())}/{len(uy)}; "
          f"{int(((up == uy) & answered).sum())} of those are correct, {int((~answered).sum())} fall back to the normal bot reply.")
    for (s, i), p, c in zip(UNSEEN, up, conf):
        print(f"  {'OK ' if p == i else 'MISS'} {s!r:<45} -> {CANONICAL[p][0][:40]!r} ({c:.2f})")
    print("\n" + classification_report(yte, pred, target_names=names, digits=3, zero_division=0))

    final = build_pipeline().fit(Xn, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, path)
    (BASE_DIR / "reports").mkdir(exist_ok=True)
    (BASE_DIR / "reports" / "nlp_report.json").write_text(json.dumps(dict(
        synthetic_test_accuracy=float(accuracy_score(yte, pred)), unseen_accuracy=float(accuracy_score(uy, up)), unseen_answered=int(answered.sum()),
        unseen_correct_and_answered=int(((up == uy) & answered).sum()), unseen_total=len(uy),
        n_sentences=len(X), n_intents=len(PHRASES), min_confidence=MIN_CONFIDENCE), indent=2))
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
