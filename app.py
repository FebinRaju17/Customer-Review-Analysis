import re
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from preprocess import clean_text
from keywords import top_terms, distinctive_terms, aspect_summary, make_wordcloud

st.set_page_config(page_title="Customer Review Analyser", page_icon="📝", layout="wide")

LABELS = ["Negative", "Neutral", "Positive"]
COLORS = {"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#2ecc71"}
ROBERTA = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BASELINE_NAME = "Baseline (TF-IDF + Logistic Regression)"
ROBERTA_NAME = "RoBERTa (pretrained, slower)"
ROBERTA_MAX_ROWS = 300   # RoBERTa is slow on CPU, so cap bulk runs


# ---------- Model loading (cached so they load only once) ----------
@st.cache_resource
def load_baseline():
    return joblib.load("models/baseline_model.joblib")


@st.cache_resource
def load_roberta():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=ROBERTA, tokenizer=ROBERTA,
                    truncation=True, max_length=512, device=-1)


# ---------- Prediction helpers ----------
def strip_html(text):
    return re.sub(r"<.*?>", " ", str(text)).strip()


def predict_baseline(cleaned_texts):
    model = load_baseline()
    proba = model.predict_proba(cleaned_texts)
    classes = list(model.classes_)
    preds = [classes[i] for i in proba.argmax(axis=1)]
    conf = proba.max(axis=1).round(4).tolist()
    return preds, conf


def predict_roberta(raw_texts):
    clf = load_roberta()
    results = clf([strip_html(t) for t in raw_texts], batch_size=16)
    preds = [r["label"].capitalize() for r in results]
    conf = [round(r["score"], 4) for r in results]
    return preds, conf


def predict(raw_texts, cleaned_texts, model_choice):
    if model_choice == ROBERTA_NAME:
        return predict_roberta(raw_texts)
    return predict_baseline(cleaned_texts)


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.radio("Choose model", [BASELINE_NAME, ROBERTA_NAME])

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data")
uploaded = st.sidebar.file_uploader("Upload reviews CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (data/test_set.csv)")

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Could not read the file: {e}")
elif use_sample:
    df = pd.read_csv("data/test_set.csv")

review_col = None
if df is not None:
    if df.empty:
        st.sidebar.error("The file is empty.")
        df = None
    else:
        guess = next((c for c in df.columns if c.lower() in ("review", "text", "reviews", "review_text")), df.columns[0])
        review_col = st.sidebar.selectbox("Review text column", df.columns, index=list(df.columns).index(guess))


# ---------- Main page ----------
st.title("📝 Customer Review Analyser")
st.caption("Sentiment analysis of customer reviews using NLP")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Review", "📊 Bulk Results", "🔑 Keywords & Insights", "🏷️ Category-wise"])

# ----- Tab 1: single review -----
with tab1:
    st.subheader("Analyse one review")
    review = st.text_area("Type or paste a review", height=150,
                          placeholder="e.g. The taste was great but the packaging was damaged.")
    if st.button("Analyse sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = clean_text(review)
            if not cleaned and model_choice == BASELINE_NAME:
                st.warning("Nothing left after cleaning. Try a longer review.")
            else:
                with st.spinner("Analysing..."):
                    preds, conf = predict([review], [cleaned], model_choice)
                label, score = preds[0], conf[0]
                emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}[label]
                c1, c2 = st.columns(2)
                c1.metric("Sentiment", f"{emoji} {label}")
                c2.metric("Confidence", f"{score * 100:.1f}%")
                st.progress(float(score))
                with st.expander("See cleaned text (after preprocessing)"):
                    st.write(cleaned if cleaned else "(empty)")

# ----- Tab 2: bulk results -----
with tab2:
    st.subheader("Bulk analysis")
    if df is None:
        st.info("Upload a CSV or tick 'Use sample dataset' in the sidebar to begin.")
    else:
        st.write(f"**{len(df):,} rows loaded.** Using column: `{review_col}`")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("Run analysis", type="primary"):
            work = df.dropna(subset=[review_col]).copy()
            if work.empty:
                st.error("That column has no text. Pick another review column in the sidebar.")
                st.stop()
            if pd.api.types.is_numeric_dtype(work[review_col]):
                st.error("That column is numeric, not review text. Pick the text column in the sidebar.")
                st.stop()
            work[review_col] = work[review_col].astype(str)
            if model_choice == ROBERTA_NAME and len(work) > ROBERTA_MAX_ROWS:
                st.warning(f"RoBERTa is slow on CPU, so only a random {ROBERTA_MAX_ROWS} reviews are analysed.")
                work = work.sample(ROBERTA_MAX_ROWS, random_state=42)

            with st.spinner("Cleaning and analysing reviews... please wait"):
                texts = work[review_col].tolist()
                cleaned = [clean_text(t) for t in texts]
                preds, conf = predict(texts, cleaned, model_choice)
                work["clean_review"] = cleaned
                work["predicted_sentiment"] = preds
                work["confidence"] = conf
            st.session_state["results"] = work.reset_index(drop=True)
            st.session_state["review_col"] = review_col
            st.session_state["model_used"] = model_choice

        if "results" in st.session_state:
            res = st.session_state["results"]
            st.success(f"Done! Model used: {st.session_state['model_used']}")

            counts = res["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0)
            m1, m2, m3 = st.columns(3)
            m1.metric("😊 Positive", int(counts["Positive"]))
            m2.metric("😐 Neutral", int(counts["Neutral"]))
            m3.metric("😞 Negative", int(counts["Negative"]))

            c1, c2 = st.columns(2)
            pie = px.pie(names=counts.index, values=counts.values, color=counts.index,
                         color_discrete_map=COLORS, title="Sentiment distribution", hole=0.4)
            c1.plotly_chart(pie, use_container_width=True)
            bar = px.bar(x=counts.index, y=counts.values, color=counts.index,
                         color_discrete_map=COLORS, title="Review count per sentiment",
                         labels={"x": "Sentiment", "y": "Reviews"})
            c2.plotly_chart(bar, use_container_width=True)

            st.dataframe(res, use_container_width=True)
            st.download_button("⬇️ Download results CSV",
                               res.to_csv(index=False).encode("utf-8"),
                               file_name="sentiment_results.csv", mime="text/csv")

# ----- Tabs 3 and 4: coming in Part 2 -----
def safe_terms(fn, *args, **kwargs):
    """Small datasets can give an 'empty vocabulary' error, so return an empty table instead."""
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return pd.DataFrame(columns=["term", "score"])


def keyword_chart(terms_df, title, color):
    fig = px.bar(terms_df, x="score", y="term", orientation="h", title=title,
                 color_discrete_sequence=[color])
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
    return fig


# ----- Tab 3: keywords & insights -----
with tab3:
    st.subheader("Keywords & business insights")
    if "results" not in st.session_state:
        st.info("Run the analysis in the 'Bulk Results' tab first.")
    else:
        res = st.session_state["results"]
        data = (res.drop(columns=["sentiment"], errors="ignore")
                   .rename(columns={"predicted_sentiment": "sentiment"}))
        data = data[data["clean_review"].fillna("").str.strip() != ""]

        if len(data) < 30:
            st.warning("Need at least 30 reviews for meaningful keywords. Analyse more data.")
        else:
            # --- Overall satisfaction insight ---
            pos_pct = 100 * (data["sentiment"] == "Positive").mean()
            neg_pct = 100 * (data["sentiment"] == "Negative").mean()
            net = pos_pct - neg_pct
            level = "Excellent" if net >= 50 else "Good" if net >= 20 else "Mixed" if net >= 0 else "Poor"

            m1, m2, m3 = st.columns(3)
            m1.metric("Positive reviews", f"{pos_pct:.1f}%")
            m2.metric("Negative reviews", f"{neg_pct:.1f}%")
            m3.metric("Overall satisfaction", level, f"net {net:+.1f}")

            pos_text = data[data["sentiment"] == "Positive"]["clean_review"]
            neg_text = data[data["sentiment"] == "Negative"]["clean_review"]

            # --- Top keywords ---
            st.markdown("### 🔑 Top keywords")
            top_n = st.slider("Number of keywords", 5, 25, 15)
            c1, c2 = st.columns(2)
            pos_kw = safe_terms(top_terms, pos_text, (1, 1), top_n)
            neg_kw = safe_terms(top_terms, neg_text, (1, 1), top_n)
            if not pos_kw.empty:
                c1.plotly_chart(keyword_chart(pos_kw, "Positive reviews", "#2ecc71"), use_container_width=True)
            if not neg_kw.empty:
                c2.plotly_chart(keyword_chart(neg_kw, "Negative reviews", "#e74c3c"), use_container_width=True)

            # --- Word clouds ---
            st.markdown("### ☁️ Word clouds")
            w1, w2 = st.columns(2)
            fig_pos = make_wordcloud(" ".join(pos_text), "Greens")
            fig_neg = make_wordcloud(" ".join(neg_text), "Reds")
            if fig_pos:
                w1.pyplot(fig_pos)
            else:
                w1.info("No positive reviews to show.")
            if fig_neg:
                w2.pyplot(fig_neg)
            else:
                w2.info("No negative reviews to show.")

            # --- Complaints and praises ---
            st.markdown("### 📌 Top complaints and praised features")
            st.caption("Phrases that stand out in one group compared with the other.")
            p1, p2 = st.columns(2)
            complaints = safe_terms(distinctive_terms, data, "Negative", "Positive", (2, 2), top_n)
            praises = safe_terms(distinctive_terms, data, "Positive", "Negative", (2, 2), top_n)
            p1.markdown("**Top complaints**")
            p1.dataframe(complaints, use_container_width=True, hide_index=True)
            p2.markdown("**Top praised features**")
            p2.dataframe(praises, use_container_width=True, hide_index=True)

            # --- Aspect summary ---
            st.markdown("### 🏷️ Aspect-wise summary")
            asp = aspect_summary(data)
            if asp["mentions"].sum() == 0:
                st.info("No aspect keywords found in these reviews.")
            else:
                fig = px.bar(asp, x="aspect", y=["positive", "neutral", "negative"],
                             barmode="stack", title="Aspect mentions by sentiment",
                             color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
                             labels={"value": "Mentions", "variable": "Sentiment"})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(asp, use_container_width=True, hide_index=True)
                worst = asp[asp["mentions"] >= 20].sort_values("negative_%", ascending=False)
                if not worst.empty:
                    st.warning(f"⚠️ Most complaint-heavy aspect: **{worst.iloc[0]['aspect']}** "
                               f"({worst.iloc[0]['negative_%']}% of its mentions are negative)")

# ----- Tab 4: category-wise sentiment -----
with tab4:
    st.subheader("Category-wise sentiment")
    if "results" not in st.session_state:
        st.info("Run the analysis in the 'Bulk Results' tab first.")
    else:
        res = st.session_state["results"]
        skip = {st.session_state["review_col"], "clean_review", "predicted_sentiment", "confidence", "sentiment"}
        options = [c for c in res.columns if c not in skip]

        if not options:
            st.info("Your CSV has no extra columns (like product or category) to group by.")
        else:
            cat = st.selectbox("Group by column", options)
            top_k = st.slider("Show top N categories (by review count)", 3, 15, 8)

            top_cats = res[cat].value_counts().head(top_k).index
            sub = res[res[cat].isin(top_cats)]
            share = (pd.crosstab(sub[cat].astype(str), sub["predicted_sentiment"], normalize="index") * 100)
            share = share.reindex(columns=LABELS, fill_value=0).round(1)

            fig = px.bar(share, x=share.index, y=LABELS, barmode="stack",
                         color_discrete_map=COLORS, title=f"Sentiment share by {cat} (%)",
                         labels={"value": "% of reviews", "x": cat, "variable": "Sentiment"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(share, use_container_width=True)

            # --- Trend over time (optional) ---
            st.markdown("### 📈 Sentiment trend over time")
            date_col = st.selectbox("Date column (optional)", ["None"] + options)
            if date_col != "None":
                dates = pd.to_datetime(res[date_col], errors="coerce")
                ok = res[dates.notna()].copy()
                ok["_d"] = dates[dates.notna()]
                if ok.empty:
                    st.warning("That column doesn't look like dates.")
                else:
                    span_days = (ok["_d"].max() - ok["_d"].min()).days
                    freq = "M" if span_days <= 730 else "Y"
                    ok["period"] = ok["_d"].dt.to_period(freq).astype(str)
                    trend = pd.crosstab(ok["period"], ok["predicted_sentiment"]).reindex(columns=LABELS, fill_value=0)
                    fig = px.line(trend, x=trend.index, y=LABELS, markers=True,
                                  color_discrete_map=COLORS, title="Reviews per period by sentiment",
                                  labels={"value": "Reviews", "x": "Period", "variable": "Sentiment"})
                    st.plotly_chart(fig, use_container_width=True)