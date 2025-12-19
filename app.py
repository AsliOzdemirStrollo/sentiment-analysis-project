import os

import streamlit as st
from joblib import load

MODEL_PATH = "models/sentiment.joblib"

LOW_CONF_MIN = 0.45
LOW_CONF_MAX = 0.55

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")


@st.cache_resource
def load_model(model_path: str, model_mtime: float):
    # model_mtime is only used to invalidate the cache when the file changes
    return load(model_path)


def as_percent(x: float) -> str:
    return f"{x * 100:.1f}%"


st.title("💬 Sentiment Analysis")
st.caption("Paste comments below (one per line), then click Predict.")

# Load model (with cache invalidation based on file modified time)
try:
    model_mtime = os.path.getmtime(MODEL_PATH)
    model = load_model(MODEL_PATH, model_mtime)
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
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

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)[:, 1]

    st.subheader("Results")

    for i, text in enumerate(texts):
        label_int = int(preds[i])
        sentiment = "Positive" if label_int == 1 else "Negative"
        icon = "🟢" if label_int == 1 else "🔴"

        prob_pos = None if probs is None else float(probs[i])
        low_conf = prob_pos is not None and (LOW_CONF_MIN <= prob_pos <= LOW_CONF_MAX)

        with st.container(border=True):
            st.write(f"**{icon} {sentiment}**")

            if prob_pos is None:
                st.caption("No probability available for this model.")
            else:
                st.progress(prob_pos)
                st.caption(f"Positive probability: {as_percent(prob_pos)}")

                if low_conf:
                    st.warning("⚠️ Low confidence prediction")

            st.write(text)

with st.expander("ℹ️ About this model"):
    st.write(
        """
        This app uses a classical NLP pipeline:
        **TF-IDF (word + character n-grams) + Logistic Regression**.

        It performs well on clear sentiment, but may struggle with:
        - sarcasm
        - mixed opinions
        - subtle or implicit sentiment

        Predictions with probabilities close to 50% are flagged as low confidence.
        """
    )

st.markdown("---")
st.markdown(
    "Made with ❤️ by **Asli Ozdemir Strollo**  \n"
    "[GitHub](https://github.com/AsliOzdemirStrollo) · "
    "[LinkedIn](https://www.linkedin.com/in/asliozdemirstrollo/)"
)
