# 💬 Sentiment Analysis Pipeline (CI/CD Edition)

An end-to-end **sentiment analysis application** that predicts whether a given piece of text expresses **positive** or **negative** sentiment.

This project demonstrates a complete ML workflow: data preparation, model training, evaluation, CI/CD automation, and live deployment using **Streamlit**.

---

## 🚀 Live Demo

👉 **Try the app here:**  
**https://YOUR-STREAMLIT-APP-URL.streamlit.app**

Paste one or more sentences (one per line) and click **Predict** to see:
- sentiment label (positive / negative)
- probability score
- **low-confidence warnings** for uncertain predictions

---

## ✨ Features

- Interactive **Streamlit web app**
- Sentiment prediction (positive / negative)
- Probability score for positive sentiment
- **Low-confidence warning** for uncertain predictions
- Bulk input support (one sentence per line)
- Optional command-line interface
- Fully Dockerised application
- Automated CI/CD pipeline with GitHub Actions

---

## 🧠 Model Overview

This model uses **TF-IDF (word + character n-grams) with Logistic Regression**, trained on a **small, curated, and balanced dataset**.

Using TF-IDF (word + character n-grams) with Logistic Regression, the model achieves a **macro-F1 of ~0.72 under cross-validation** on a balanced dataset.

The model performs well on **clear sentiment expressions**, but may struggle with:
- sarcasm
- mixed or contradictory wording
- subtle sentiment cues

Predictions close to 50% probability are explicitly flagged as **low confidence** in the UI.

---

## 📊 Dataset

- **Total samples:** 120  
- **Positive:** 62  
- **Negative:** 58  

The dataset was expanded iteratively using a **data-centric approach**, where misclassifications were corrected by adding **balanced counter-examples**.

---

## 📈 Model Performance

Evaluation was performed using both cross-validation and a holdout test set.

- **5-fold cross-validation (macro F1):** ~**0.72**
- **Holdout test accuracy:** ~**0.75**
- **Holdout macro F1:** ~**0.75**

Macro F1 is reported to ensure balanced performance across sentiment classes.

---

## 🧠 Example Prediction Output (CLI)

Example CLI output:

    1    0.546    I love this product
    0    0.508    This is terrible

Where:
- `1` = positive sentiment (`0` = negative)
- Probability represents likelihood of positive sentiment

---

## 🧰 Tech Stack

Component        | Technology
-----------------|----------------
UI               | Streamlit / CLI
Language         | Python 3.11
ML               | scikit-learn
Containerisation | Docker
CI/CD            | GitHub Actions
Testing          | pytest
Linting          | Ruff

---

## 🛠️ Local Development (Optional)

Most users should use the **live Streamlit app** above.  
The steps below are only required for local development or experimentation.

Clone the repository:

    git clone https://github.com/AsliOzdemirStrollo/sentiment-analysis-project.git
    cd sentiment-analysis-project

Install dependencies:

    pip install -r requirements.txt

Train the model:

    python src/train.py --data data/sentiments.csv --out models/sentiment.joblib

Run the app locally:

    streamlit run app.py

---

## 🐳 Docker Usage (Optional)

    docker pull aslistr/sentiment-analysis:latest
    docker run --rm aslistr/sentiment-analysis:latest

---

## 📁 Project Structure

    sentiment-analysis-project/
    ├── .github/workflows/
    │   └── ci.yml
    ├── data/
    │   └── sentiments.csv
    ├── models/
    │   └── sentiment.joblib
    ├── src/
    │   ├── train.py
    │   └── predict.py
    ├── tests/
    │   ├── test_predict.py
    │   └── __init__.py
    ├── app.py
    ├── Dockerfile
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── pyproject.toml
    └── README.md

---

## ✅ Status

The project is fully functional.

Every push to the `main` branch triggers:
- linting
- testing
- model validation via CI

The Streamlit app automatically redeploys on updates.

---

## 👤 Author

Made with ❤️ by **Asli Ozdemir Strollo**  
GitHub: https://github.com/AsliOzdemirStrollo  
LinkedIn: https://www.linkedin.com/in/asliozdemirstrollo/