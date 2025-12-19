import argparse
import os

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5


def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV and ensures it has the required columns.
    """
    df = pd.read_csv(data_path)

    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")

    # Basic cleanup: drop missing rows, ensure strings
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)

    return df


def split_data(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing sets.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["label"],
        )
    except ValueError:
        # Fallback if stratification fails (e.g., extremely small / skewed data)
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    """
    Builds an sklearn Pipeline using a FeatureUnion of:
    - word-level TF-IDF (unigrams + bigrams)
    - character-level TF-IDF (3-5 char ngrams within word boundaries)
    This is often stronger and more robust on small datasets.
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words=None,
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )

    feats = FeatureUnion([("word", word_vec), ("char", char_vec)])

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    return Pipeline([("feats", feats), ("model", model)])


def evaluate_holdout(clf: Pipeline, X_test: pd.Series, y_test: pd.Series) -> None:
    """
    Prints evaluation metrics on the holdout set.
    """
    acc = clf.score(X_test, y_test)
    print(f"\nHoldout test accuracy: {acc:.3f}")

    y_pred = clf.predict(X_test)
    print("\nClassification report (holdout):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix (holdout):")
    print(confusion_matrix(y_test, y_pred))


def evaluate_cv(pipeline: Pipeline, df: pd.DataFrame) -> None:
    """
    Runs stratified cross-validation to get a more stable performance estimate.
    """
    try:
        cv = StratifiedKFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        scores = cross_val_score(
            pipeline,
            df["text"],
            df["label"],
            cv=cv,
            scoring="f1_macro",
        )
        print("\nCross-validation (f1_macro):")
        print("Scores:", [round(float(s), 3) for s in scores])
        print("Mean:", round(float(scores.mean()), 3))
    except ValueError as e:
        print("\nCross-validation skipped:", e)


def save_model(model: Pipeline, model_path: str) -> None:
    """
    Saves the trained model to a file.
    """
    model_dir = os.path.dirname(model_path) or "."
    os.makedirs(model_dir, exist_ok=True)
    dump(model, model_path)
    print(f"\nSaved model to {model_path}")


def main(data_path: str, model_path: str) -> None:
    """
    Main workflow to load, train, evaluate, and save the model.
    """
    df = load_and_validate_data(data_path)

    # Quick visibility into dataset properties
    print("Dataset size:", len(df))
    print("Label counts:\n", df["label"].value_counts(), "\n")

    pipeline = build_pipeline()

    # More stable evaluation on small data
    evaluate_cv(pipeline, df)

    # Holdout evaluation (noisy on small datasets but still useful)
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline.fit(X_train, y_train)
    evaluate_holdout(pipeline, X_test, y_test)

    # Fit on full data for the artifact we ship to Streamlit
    pipeline.fit(df["text"], df["label"])
    save_model(pipeline, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sentiments.csv")
    parser.add_argument("--out", default="models/sentiment.joblib")

    args: argparse.Namespace = parser.parse_args()
    main(data_path=args.data, model_path=args.out)