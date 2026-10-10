import re
import time
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

LABELS = ["Negative", "Neutral", "Positive"]
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
N_PER_CLASS = 200   # lower to 100 if your laptop is slow


def light_clean(text):
    """Transformers need natural text, so only strip HTML tags."""
    return re.sub(r"<.*?>", " ", str(text)).strip()


def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


if __name__ == "__main__":
    # 1. Balanced evaluation subset from the SAME test set
    test_df = pd.read_csv("data/test_set.csv")
    parts = [test_df[test_df["sentiment"] == l].sample(N_PER_CLASS, random_state=42)
             for l in LABELS]
    sub = pd.concat(parts).reset_index(drop=True)
    print("Evaluation subset:", sub.shape)

    # 2. Baseline predictions on the same subset
    baseline = joblib.load("models/baseline_model.joblib")
    sub["baseline_pred"] = baseline.predict(sub["clean_review"])

    # 3. Transformer predictions (first run downloads ~500 MB)
    print("\nLoading transformer model...")
    clf = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME,
                   truncation=True, max_length=512, device=-1)

    texts = [light_clean(t) for t in sub["review"]]
    print("Running transformer on", len(texts), "reviews...")
    start = time.time()
    results = clf(texts, batch_size=16)
    print(f"Done in {time.time() - start:.0f} seconds")

    sub["transformer_pred"] = [r["label"].capitalize() for r in results]
    sub["transformer_conf"] = [round(r["score"], 4) for r in results]

    # 4. Metrics for both models
    base_m = get_metrics(sub["sentiment"], sub["baseline_pred"])
    tran_m = get_metrics(sub["sentiment"], sub["transformer_pred"])

    comparison = pd.DataFrame([
        {"Model": "TF-IDF + Logistic Regression", **base_m},
        {"Model": "RoBERTa (pretrained)", **tran_m},
    ]).round(4)

    print("\n===== COMPARISON TABLE =====")
    print(comparison.to_string(index=False))

    print("\nTransformer per-class report:\n")
    print(classification_report(sub["sentiment"], sub["transformer_pred"],
                                labels=LABELS, digits=3))

    # 5. Confusion matrix image for the report
    cm = confusion_matrix(sub["sentiment"], sub["transformer_pred"], labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Greens")
    plt.title("RoBERTa (pretrained)")
    plt.tight_layout()
    plt.savefig("models/transformer_confusion_matrix.png", dpi=150)

    # 6. Misclassified examples (for the viva)
    wrong = sub[sub["sentiment"] != sub["transformer_pred"]].head(5)
    print("\n===== 5 TRANSFORMER MISTAKES =====")
    for _, row in wrong.iterrows():
        print(f"\nActual: {row['sentiment']} | Predicted: {row['transformer_pred']} "
              f"({row['transformer_conf']})")
        print("Review:", light_clean(row["review"])[:250])

    # 7. Save
    comparison.to_csv("models/comparison_table.csv", index=False)
    sub.to_csv("data/transformer_predictions.csv", index=False)
    print("\nSaved: models/comparison_table.csv, models/transformer_confusion_matrix.png,")
    print("       data/transformer_predictions.csv")