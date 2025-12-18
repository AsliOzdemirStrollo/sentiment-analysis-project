# 💬 Sentiment Analysis Pipeline (CI/CD Edition)

A machine learning sentiment analysis application that predicts whether a given piece of text expresses positive or negative sentiment. Built using Python, scikit-learn, Docker, and GitHub Actions, this project demonstrates how to take an ML model from local development to a fully automated, production-ready pipeline.

---

## ✨ Features

- Sentiment prediction (positive / negative)
- Probability score for positive sentiment
- Command-line interface for predictions
- Fully Dockerised application
- Automated CI/CD pipeline with GitHub Actions
- Automatic Docker image publishing to Docker Hub

---

## 🧠 Example Prediction Output

1    0.546    I love this product  
0    0.508    This is terrible  

Where:
- 1 = positive sentiment (0 = negative)
- Probability represents likelihood of positive sentiment

---

## 🧰 Tech Stack

Component        Technology  
UI               Command Line Interface  
Language         Python 3.11  
ML               scikit-learn  
Containerisation Docker  
CI/CD            GitHub Actions  
Testing          pytest  
Linting          Ruff  

---

## 🛠️ Local Setup & Installation

### 1. Clone the repository
git clone https://github.com/AsliOzdemirStrollo/sentiment-analysis-project.git  
cd sentiment-analysis-project  

### 2. Create a virtual environment
python -m venv .venv  
source .venv/bin/activate  # macOS/Linux  
.venv\Scripts\activate     # Windows  

### 3. Install dependencies
pip install -r requirements.txt  

### 4. Train the model
python src/train.py --data data/sentiments.csv --out models/sentiment.joblib  

### 5. Run predictions
python src/predict.py "I absolutely loved it" "That was awful"  

---

## 🐳 Docker Usage

### Pull the image
docker pull aslistr/sentiment-analysis:latest  

### Run the container
docker run --rm aslistr/sentiment-analysis:latest  

### Run with custom input
docker run --rm aslistr/sentiment-analysis:latest python src/predict.py "I love this product"  

---

## 📁 Project Structure

sentiment-analysis-project/
├── src/
│   ├── train.py
│   └── predict.py
├── data/
│   └── sentiments.csv
├── models/
│   └── sentiment.joblib
├── tests/
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md

---

## ✅ Status

The project is fully functional. Every push to the main branch triggers linting, testing, Docker image build, and automatic publication to Docker Hub.

---

## 👤 Author

Made with ❤️ by Asli Ozdemir Strollo  
GitHub: https://github.com/AsliOzdemirStrollo  
LinkedIn: https://www.linkedin.com/in/asliozdemirstrollo/