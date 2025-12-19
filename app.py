import streamlit as st
from joblib import load

MODEL_PATH = "models/sentiment.joblib"
LOW_CONF_MIN = 0.45
LOW_CONF_MAX = 0.55

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")


@st.cache_resource
def load_model():
    return load(MODEL_PATH)


def as_percent(x: float) -> str:
    return f"{x * 100:.1f}%"


st.title("💬 Sentiment Analysis")
st.caption("Paste comments below (one per line), then click Predict.")

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model from: {MODEL_PATH}")
    st.exception(e)
    st.stop()

texts_raw = st.text_area(
    "Comments (one per line)",
    placeholder="Example:\nI would definitely recommend this.\nNot worth watching.",
    height=240,
)

if st.button("Predict", use_container_width=True):
    texts = [t.strip() for t in texts_raw.splitlines() if t.strip()]
    if not texts:
        st.error("Please enter at least one comment.")
        st.stop()

    preds = model.predict(texts)
    probs = model.predict_proba(texts)[:, 1]

    st.subheader("Results")

    for text, label, prob in zip(texts, preds, probs):
        sentiment = "Positive" if label == 1 else "Negative"
        color = "🟢" if label == 1 else "🔴"
        low_conf = LOW_CONF_MIN <= prob <= LOW_CONF_MAX

        with st.container(border=True):
            st.write(f"**{color} {sentiment}**")
            st.progress(prob)
            st.caption(f"Positive probability: {as_percent(prob)}")

            if low_conf:
                st.warning("⚠️ Low confidence prediction")

            st.write(text)

with st.expander("ℹ️ About this model"):
    st.write(
        """
        This app uses a classical NLP pipeline:
        **TF-IDF (word + character n-grams) + Logistic Regression**.

        It is trained on a small, curated dataset and performs well on
        clear sentiment. However, it may struggle with:
        - sarcasm
        - mixed opinions
        - subtle or implicit sentiment

        Predictions with probabilities close to 50% are flagged as low confidence.
        """
    )