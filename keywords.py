import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Aspect keyword lists (words are in lemmatized form, matching clean_review)
ASPECTS = {
    "Price":     ["price", "cost", "expensive", "cheap", "overpriced", "worth", "value", "money", "pay"],
    "Quality":   ["quality", "fresh", "stale", "defective", "durable", "cheaply", "poor", "excellent"],
    "Taste":     ["taste", "flavor", "flavour", "delicious", "bland", "sweet", "bitter", "smell"],
    "Delivery":  ["ship", "shipping", "delivery", "deliver", "arrive", "late", "delay", "slow", "fast"],
    "Packaging": ["package", "packaging", "box", "damage", "broken", "leak", "seal", "bag"],
    "Support":   ["customer service", "refund", "return", "replace", "support", "seller", "contact"],
}


def _vectorizer(n_docs, ngram_range):
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=1 if n_docs < 50 else 3,
        max_features=5000,
    )


def top_terms(texts, ngram_range=(1, 1), top_n=15):
    """Most important terms in a group of reviews (mean TF-IDF score)."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < 2:
        return pd.DataFrame(columns=["term", "score"])
    vec = _vectorizer(len(texts), ngram_range)
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    idx = scores.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[idx], "score": scores[idx].round(4)})


def distinctive_terms(df, target, other, ngram_range=(2, 2), top_n=15):
    """Terms that stand out in `target` reviews compared with `other` reviews."""
    sub = df[df["sentiment"].isin([target, other])]
    texts = sub["clean_review"].fillna("")
    if len(sub) < 2:
        return pd.DataFrame(columns=["term", "score"])
    vec = _vectorizer(len(sub), ngram_range)
    X = vec.fit_transform(texts)
    is_target = (sub["sentiment"] == target).values
    diff = np.asarray(X[is_target].mean(axis=0)).ravel() - np.asarray(X[~is_target].mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    idx = diff.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[idx], "score": diff[idx].round(4)})


def aspect_summary(df):
    """How often each aspect is mentioned, split by sentiment."""
    rows = []
    for aspect, words in ASPECTS.items():
        pattern = r"\b(" + "|".join(words) + r")\b"
        mask = df["clean_review"].fillna("").str.contains(pattern, regex=True)
        counts = df[mask]["sentiment"].value_counts()
        pos, neu, neg = (int(counts.get(s, 0)) for s in ["Positive", "Neutral", "Negative"])
        total = pos + neu + neg
        rows.append({
            "aspect": aspect, "mentions": total,
            "positive": pos, "neutral": neu, "negative": neg,
            "negative_%": round(100 * neg / total, 1) if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values("mentions", ascending=False).reset_index(drop=True)


def make_wordcloud(text, colormap="viridis"):
    """Returns a matplotlib figure with a word cloud (None if no text)."""
    if not text or not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap=colormap, max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = pd.read_csv("data/reviews_processed.csv")
    pos = df[df["sentiment"] == "Positive"]["clean_review"]
    neg = df[df["sentiment"] == "Negative"]["clean_review"]

    print("===== TOP POSITIVE KEYWORDS =====")
    print(top_terms(pos, (1, 1)).to_string(index=False))
    print("\n===== TOP NEGATIVE KEYWORDS =====")
    print(top_terms(neg, (1, 1)).to_string(index=False))

    print("\n===== TOP NEGATIVE BIGRAMS =====")
    print(top_terms(neg, (2, 2)).to_string(index=False))

    print("\n===== TOP COMPLAINTS (stand out in Negative vs Positive) =====")
    print(distinctive_terms(df, "Negative", "Positive").to_string(index=False))

    print("\n===== TOP PRAISED FEATURES (stand out in Positive vs Negative) =====")
    print(distinctive_terms(df, "Positive", "Negative").to_string(index=False))

    print("\n===== ASPECT SUMMARY =====")
    summary = aspect_summary(df)
    print(summary.to_string(index=False))
    summary.to_csv("models/aspect_summary.csv", index=False)

    make_wordcloud(" ".join(pos), "Greens").savefig("models/wordcloud_positive.png", dpi=150)
    make_wordcloud(" ".join(neg), "Reds").savefig("models/wordcloud_negative.png", dpi=150)
    print("\nSaved: models/aspect_summary.csv, wordcloud_positive.png, wordcloud_negative.png")