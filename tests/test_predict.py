from pathlib import Path

import pytest

from src.predict import load_model, predict_texts


@pytest.fixture(scope="session")
def model():
    """Load the trained sentiment model once for the whole test session."""
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "sentiment.joblib"
    return load_model(str(model_path))


@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("I love this movie, it was fantastic and inspiring!", 1),  # positive
        ("The service was terrible and the food was awful.", 0),  # negative
    ],
)
def test_sentiment_sanity_predictions(model, text, expected_label):
    """Sanity-check: obvious texts should get the expected label."""
    preds, probs = predict_texts(model, [text])

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert preds[0] in (0, 1)

    # Main sanity check
    assert preds[0] == expected_label

    # Optional: if probabilities exist, check they're valid
    if probs[0] is not None:
        assert 0.0 <= probs[0] <= 1.0
