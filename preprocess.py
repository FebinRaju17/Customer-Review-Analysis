import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

# Keep negations: "not good" and "good" mean opposite things
NEGATIONS = {"not", "no", "nor", "never"}
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS

# Contractions -> full forms (done BEFORE removing punctuation)
CONTRACTIONS = [
    (r"won't", "will not"),
    (r"can't", "can not"),
    (r"n't", " not"),
    (r"'re", " are"),
    (r"'m", " am"),
    (r"'ve", " have"),
    (r"'ll", " will"),
    (r"'d", " would"),
    (r"'s", ""),
]


def clean_text(text):
    """Raw review -> cleaned, lemmatized string."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase + normalise curly apostrophes
    text = text.lower().replace("’", "'")

    # 2. Remove HTML tags (<br />) and URLs
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 3. Expand contractions
    for pattern, repl in CONTRACTIONS:
        text = re.sub(pattern, repl, text)

    # 4. Keep only letters (removes punctuation, numbers, emojis)
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Tokenize
    tokens = word_tokenize(text)

    # 6. Remove stop-words (negations kept) and 1-letter tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # 7. Lemmatize (verb first, then noun: "loved" -> "love", "dogs" -> "dog")
    tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(t, pos="v")) for t in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    df = pd.read_csv("data/reviews_clean.csv")

    print("Cleaning", len(df), "reviews... (takes ~1 min)")
    df["clean_review"] = df["review"].apply(clean_text)

    # Drop rows that became empty after cleaning
    df = df[df["clean_review"].str.strip() != ""].reset_index(drop=True)

    df.to_csv("data/reviews_processed.csv", index=False)
    print("Saved data/reviews_processed.csv | shape:", df.shape)

    # Before/after samples for the report
    print("\n--- BEFORE / AFTER SAMPLES ---")
    for i in range(3):
        print(f"\nRaw    : {df['review'][i][:200]}")
        print(f"Cleaned: {df['clean_review'][i][:200]}")
        print(f"Label  : {df['sentiment'][i]}")