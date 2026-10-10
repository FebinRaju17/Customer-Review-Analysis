import pandas as pd

RAW_PATH = "data/Reviews.csv"
OUT_PATH = "data/reviews_clean.csv"
TEXT_COL = "Text"
RATING_COL = "Score"
PER_CLASS = 7000   # max reviews per sentiment class

# 1. Load only needed columns
df = pd.read_csv(RAW_PATH, usecols=["ProductId", "Time", RATING_COL, TEXT_COL])
print("Raw shape:", df.shape)

# 2. Rename for consistency across the project
df = df.rename(columns={TEXT_COL: "review", RATING_COL: "rating", "ProductId": "product"})

# 3. Clean basics
df = df.dropna(subset=["review", "rating"])
df = df.drop_duplicates(subset=["review"])
df = df[df["review"].str.len() >= 15]          # drop tiny/useless reviews
df["date"] = pd.to_datetime(df["Time"], unit="s")
df = df.drop(columns=["Time"])

# 4. Map ratings to 3 sentiment classes
def to_label(r):
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    return "Positive"

df["sentiment"] = df["rating"].apply(to_label)
print("\nClass counts before sampling:\n", df["sentiment"].value_counts())

# 5. Balanced sample: take up to PER_CLASS rows from each class
parts = []
for label in ["Negative", "Neutral", "Positive"]:
    subset = df[df["sentiment"] == label]
    parts.append(subset.sample(n=min(len(subset), PER_CLASS), random_state=42))

df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print("\nClass counts after sampling:\n", df["sentiment"].value_counts())
print("\nFinal shape:", df.shape)
print(df.head())

# 6. Save
df.to_csv(OUT_PATH, index=False)
print("\nSaved to", OUT_PATH)