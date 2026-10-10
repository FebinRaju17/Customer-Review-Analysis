import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

LABELS = ["Negative", "Neutral", "Positive"]

# 1. Load processed data
df = pd.read_csv("data/reviews_processed.csv")
df = df.dropna(subset=["clean_review"])
print("Data shape:", df.shape)

# 2. Stratified 80/20 split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["sentiment"]
)
print("Train:", len(train_df), "| Test:", len(test_df))

# 3. Pipeline: TF-IDF (unigrams + bigrams) -> Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=3)),
    ("clf", LogisticRegression(max_iter=1000, C=2.0)),
])

print("\nTraining...")
model.fit(train_df["clean_review"], train_df["sentiment"])

# 4. Evaluate
y_true = test_df["sentiment"]
y_pred = model.predict(test_df["clean_review"])

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {prec:.4f} (macro)")
print(f"Recall   : {rec:.4f} (macro)")
print(f"F1-score : {f1:.4f} (macro)")
print("\nPer-class report:\n")
print(classification_report(y_true, y_pred, labels=LABELS, digits=3))

# 5. Confusion matrix (printed + saved as image for the report)
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
print("Confusion matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
plt.title("Baseline: TF-IDF + Logistic Regression")
plt.tight_layout()
plt.savefig("models/baseline_confusion_matrix.png", dpi=150)

# 6. Save everything for later steps
joblib.dump(model, "models/baseline_model.joblib")
test_df.to_csv("data/test_set.csv", index=False)
with open("models/baseline_metrics.json", "w") as f:
    json.dump({"model": "TF-IDF + Logistic Regression", "accuracy": acc,
               "precision": prec, "recall": rec, "f1": f1}, f, indent=2)

print("\nSaved: models/baseline_model.joblib, models/baseline_metrics.json,")
print("       models/baseline_confusion_matrix.png, data/test_set.csv")