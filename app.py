import streamlit as st
from joblib import load

MODEL_PATH = "models/sentiment.joblib"

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")


@st.cache_resource
def load_model():
    return load(MODEL_PATH)


def as_percent(x: float) -> str:
    return f"{x * 100:.1f}%"


st.title("💬 Sentiment Analysis")
st.caption("Paste comments below (one per line), then click Predict.")

# Load model early so errors show immediately
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model from: {MODEL_PATH}")
    st.exception(e)
    st.stop()

examples = [
    "I absolutely love this product!",
    "This is terrible. Waste of money.",
    "Not bad, but could be better.",
    "Absolutely fantastic service.",
]

texts_raw = st.text_area(
    "Comments (one per line)",
    placeholder="Example:\nI love this product\nThis is terrible",
    height=240,
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("Predict", use_container_width=True)
with col2:
    fill_examples = st.button("Paste examples", use_container_width=True)

if fill_examples:
    st.session_state["bulk_texts"] = "\n".join(examples)
    st.rerun()

# If user hasn't typed anything yet, allow the example paste to populate the box
if "bulk_texts" in st.session_state and not texts_raw:
    texts_raw = st.session_state["bulk_texts"]

if predict_clicked:
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

        if probs is None:
            st.write(f"**{sentiment}** — {text}")
        else:
            prob_pos = float(probs[i])
            with st.container(border=True):
                st.write(f"**{sentiment}**")
                st.progress(prob_pos)
                st.caption(f"Positive probability: {as_percent(prob_pos)}")
                st.write(text)
