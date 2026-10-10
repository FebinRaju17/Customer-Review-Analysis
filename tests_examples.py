import re
import joblib
import pandas as pd
from transformers import pipeline
from preprocess import clean_text

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def light_clean(text):
    return re.sub(r"<.*?>", " ", str(text)).strip()

# (review, what a human would say)
EXAMPLES = [
    ("Absolutely loved it, will buy again!", "Positive"),
    ("Terrible quality, broke in two days. Waste of money.", "Negative"),
    ("It's okay, nothing special.", "Neutral"),
    ("Not bad at all.", "Positive"),                                   # negation
    ("Great, another box arrived crushed. Just what I needed.", "Negative"),   # sarcasm
    ("The taste is amazing but the packaging was damaged.", "Neutral"),        # mixed
    ("Delivery was late, but the product is worth the wait.", "Positive"),     # mixed
    ("I expected more for this price.", "Negative"),
    ("Average product. Does the job.", "Neutral"),
    ("Not worth it. Would not recommend.", "Negative"),
    ("Best coffee I have ever had!!!", "Positive"),
    ("Product arrived on time. Packaging was fine.", "Neutral"),
]

baseline = joblib.load("models/baseline_model.joblib")
clf = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME,
               truncation=True, max_length=512, device=-1)

rows = []
for text, truth in EXAMPLES:
    b = baseline.predict([clean_text(text)])[0]
    r = clf(light_clean(text))[0]["label"].capitalize()
    rows.append({"review": text[:55], "expected": truth, "baseline": b, "roberta": r,
                 "base_ok": b == truth, "rob_ok": r == truth})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
print("\nBaseline correct:", out["base_ok"].sum(), "/", len(out))
print("RoBERTa  correct:", out["rob_ok"].sum(), "/", len(out))
out.to_csv("models/hand_test_results.csv", index=False)